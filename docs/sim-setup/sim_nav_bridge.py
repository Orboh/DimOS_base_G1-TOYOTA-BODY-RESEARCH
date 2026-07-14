#!/usr/bin/env python3
"""sim_nav_bridge.py — Isaac 収穫シーン(chinou+g1bag)を dimos 標準ナビへつなぐ "目と車輪"（isaac-sim env）。

Unity 抜きの (b)。Isaac 側は **dimos 非依存＝生 LCM のみ**。`.venv` の `nav_isaac_adapter.py` が
これを nav_stack 形式へ翻訳する。本ブリッジの役割:
  - sub  isaac/cmd_vel (<f32 x3> vx,vy,wz) → base をキネマティック積分（g1bag は base-fix なので
         set_world_pose で台ごとテレポート）
  - pub  isaac/lidar  (<u32 N><f32 N*3>)   ← **簡易レイキャスト Lidar**（PhysX で周囲に光線、当たった
         world 点群）。RTX Lidar への差し替えは後日（API が重いのでまず簡易版で nav ループ成立を確認）。
  - pub  isaac/odom   (<f32 x7> x,y,z,qx,qy,qz,qw) ← 現在の base 姿勢

前提: 部屋(chinou_center.usd)に collider があること（build_chinou_phys.py）。lo マルチキャスト設定済み。
実行例:
  PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES SIM_LOAD_ROOM=1 SIM_HEADLESS=0 \
    ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/sim_nav_bridge.py
"""
from __future__ import annotations

import math
import os
import struct
import sys
import time

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

REPO = os.getenv("SIM_REPO", "/home/kota-ueda/Desktop/dimos-hackathon")
ROOM_USD = os.getenv("SIM_ROOM_USD", f"{REPO}/usd_file/chinou_center.usd")
G1_USD = os.getenv("SIM_G1_USD", f"{REPO}/usd_file/g1-29dof-dex1-base-fix-usd/g1bag.usd")

# 生 LCM チャンネル（nav_isaac_adapter.py と一致）
CH_LIDAR = "isaac/lidar"
CH_ODOM = "isaac/odom"
CH_CMDVEL = "isaac/cmd_vel"

# 簡易 Lidar パラメータ
LIDAR_H = float(os.getenv("SIM_LIDAR_H", "0.9"))          # 床からのセンサ高 [m]
LIDAR_MAXR = float(os.getenv("SIM_LIDAR_MAXR", "10.0"))   # 最大測距 [m]
LIDAR_AZ_STEP = float(os.getenv("SIM_LIDAR_AZ_STEP", "2.0"))  # 方位角刻み [deg]
LIDAR_ELEVS_DEG = [float(x) for x in os.getenv("SIM_LIDAR_ELEVS", "-10,0,10,20").split(",")]
LIDAR_EVERY = int(os.getenv("SIM_LIDAR_EVERY", "5"))      # 何 step ごとにスキャン発行するか
MAX_SPEED = float(os.getenv("SIM_NAV_MAX_SPEED", "1.5"))  # base 速度の安全上限 [m/s, rad/s]


def main() -> None:
    headless = os.getenv("SIM_HEADLESS", "1") != "0"
    load_room = os.getenv("SIM_LOAD_ROOM", "1") == "1"

    import lcm as lcm_mod  # 生 LCM（dimos 非依存）

    from isaacsim import SimulationApp

    sim_app = SimulationApp({"headless": headless})

    import carb
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage, open_stage
    import omni.usd
    from omni.physx import get_physx_scene_query_interface

    try:
        from isaacsim.core.prims import SingleArticulation as ArtCls
    except Exception:  # noqa: BLE001
        from isaacsim.core.api.articulations import Articulation as ArtCls  # type: ignore

    # --- シーン ---
    if load_room:
        open_stage(ROOM_USD)
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    try:
        world.get_physics_context().set_gravity(0.0)  # キネマティック移動なので重力OFF
    except Exception as _e:  # noqa: BLE001
        print(f"[nav-bridge] set_gravity warn: {_e}", flush=True)

    add_reference_to_stage(usd_path=G1_USD, prim_path="/G1")
    g1 = stage.GetPrimAtPath("/G1")
    bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    lift = -float(bbc.ComputeWorldBound(g1).ComputeAlignedRange().GetMin()[2])
    UsdGeom.XformCommonAPI(g1).SetTranslate(Gf.Vec3d(0.0, 0.0, lift))

    art_root = None
    for p in Usd.PrimRange(g1):
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_root = p.GetPath().pathString
            break
    robot = ArtCls(prim_path=art_root or "/G1", name="g1")
    world.scene.add(robot)

    world.reset()
    for _ in range(20):
        world.step(render=not headless)

    # --- LCM ---
    lc = lcm_mod.LCM()
    cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "t": time.time()}

    def on_cmd(_ch, data):
        if len(data) >= 12:
            vx, vy, wz = struct.unpack_from("<3f", data, 0)
            cmd["vx"] = max(-MAX_SPEED, min(MAX_SPEED, vx))
            cmd["vy"] = max(-MAX_SPEED, min(MAX_SPEED, vy))
            cmd["wz"] = max(-MAX_SPEED, min(MAX_SPEED, wz))
            cmd["t"] = time.time()

    lc.subscribe(CH_CMDVEL, on_cmd)

    # 方位角・仰角テーブル
    azs = np.deg2rad(np.arange(0.0, 360.0, LIDAR_AZ_STEP))
    elevs = np.deg2rad(np.array(LIDAR_ELEVS_DEG))
    dirs = []
    for el in elevs:
        ce, se = math.cos(el), math.sin(el)
        for az in azs:
            dirs.append((ce * math.cos(az), ce * math.sin(az), se))
    print(f"[nav-bridge] lidar rays/scan = {len(dirs)} (elevs={LIDAR_ELEVS_DEG}, az_step={LIDAR_AZ_STEP})", flush=True)

    pxq = get_physx_scene_query_interface()

    def scan(ox: float, oy: float, oz: float) -> "np.ndarray":
        pts = []
        for dx, dy, dz in dirs:
            hit = pxq.raycast_closest(carb.Float3(ox, oy, oz), carb.Float3(dx, dy, dz), LIDAR_MAXR)
            if hit and hit.get("hit"):
                p = hit["position"]
                pts.append((float(p[0]), float(p[1]), float(p[2])))
        return np.asarray(pts, dtype="<f4").reshape(-1, 3)

    # --- base 状態（world: x,y,yaw, z=lift 固定）---
    x = y = yaw = 0.0
    dt = 1.0 / 60.0
    step = 0
    print("[nav-bridge] loop start（Ctrl-C で終了）", flush=True)
    while True:
        # cmd_vel 受信
        lc.handle_timeout(0)
        # 速度ゼロ化（指令が途切れたら止まる）
        if time.time() - cmd["t"] > 0.5:
            cmd["vx"] = cmd["vy"] = cmd["wz"] = 0.0
        # キネマティック積分（body→world）
        cy, sy = math.cos(yaw), math.sin(yaw)
        x += (cmd["vx"] * cy - cmd["vy"] * sy) * dt
        y += (cmd["vx"] * sy + cmd["vy"] * cy) * dt
        yaw += cmd["wz"] * dt
        qw, qz = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        try:
            robot.set_world_pose(position=np.array([x, y, lift]),
                                 orientation=np.array([qw, 0.0, 0.0, qz]))  # (w,x,y,z)
        except Exception as _e:  # noqa: BLE001
            if step % 300 == 0:
                print(f"[nav-bridge] set_world_pose warn: {_e}", flush=True)

        world.step(render=not headless)
        step += 1

        # odom 発行（毎ステップ）
        lc.publish(CH_ODOM, struct.pack("<7f", x, y, LIDAR_H, 0.0, 0.0, qz, qw))

        # lidar 発行（間引き）
        if step % LIDAR_EVERY == 0:
            pts = scan(x, y, LIDAR_H)
            n = int(pts.shape[0])
            lc.publish(CH_LIDAR, struct.pack("<I", n) + (pts.tobytes() if n else b""))
            if step % (LIDAR_EVERY * 60) == 0:
                print(f"[nav-bridge] step={step} pose=({x:.2f},{y:.2f},{math.degrees(yaw):.0f}deg) "
                      f"scan_pts={n} cmd=({cmd['vx']:.2f},{cmd['vy']:.2f},{cmd['wz']:.2f})", flush=True)


if __name__ == "__main__":
    main()
