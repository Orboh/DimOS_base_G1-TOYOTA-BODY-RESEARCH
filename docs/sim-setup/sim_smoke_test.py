#!/usr/bin/env python3
"""Isaac Sim 4.5 オクラ収穫 sim 第1歩スモークテスト（DoD 2-5）。

  room.usd を *stage 全体* として開く（壁・床はルート直下にあり defaultPrim 外なので
  add_reference では落ちる → open_stage で全部読む）
  → カスタムG1 USD を room 内に配置（fix_base:true。原点=骨盤なので足が床に来るよう持ち上げ）
  → 物理を回して発散しないことを確認 → 右腕1関節を目標角へ動かす
  → viewport / Camera センサで RGB 保存

8GB VRAM 機向けに headless + 低解像度で実行。GUI で見たい場合は --gui。

実行:
  cd <repo>
  PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES \
    ~/miniconda3/envs/isaac-sim/bin/python sim_smoke_test.py [--gui]
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
ROOM_USD = f"{REPO}/usd_file/room.usd"
G1_USD = f"{REPO}/usd_file/g1-29dof-dex1-base-fix-usd/g1_29dof_with_dex1_base_fix1.usd"
OUT_DIR = "/tmp/claude-1000/-home-kota-ueda-Desktop-dimos-hackathon/1487315e-e890-48a9-8754-b1a3f80b5f18/scratchpad/sim_out"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--cam-w", type=int, default=960)
    ap.add_argument("--cam-h", type=int, default=540)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    from isaacsim import SimulationApp

    sim_app = SimulationApp({"headless": not args.gui})

    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage, open_stage
    from isaacsim.core.utils.viewports import set_camera_view
    import omni.usd

    try:
        from isaacsim.core.prims import SingleArticulation as ArtCls
    except Exception:  # noqa: BLE001
        from isaacsim.core.api.articulations import Articulation as ArtCls  # type: ignore
    from isaacsim.core.utils.types import ArticulationAction

    # --- 部屋全体を stage として開く（ルート直下の壁/床も含めて読む） ---
    print(f"[smoke] open full room stage: {ROOM_USD}", flush=True)
    open_stage(ROOM_USD)

    print("[smoke] World ...", flush=True)
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # --- G1 をルートに配置（room の /World と衝突させない） ---
    print(f"[smoke] add G1: {G1_USD}", flush=True)
    add_reference_to_stage(usd_path=G1_USD, prim_path="/G1")
    g1_prim = stage.GetPrimAtPath("/G1")

    # G1 原点は骨盤位置（bbox 下端が負）→ 足を床(z=0)へ
    _bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    _min_z = float(_bbc.ComputeWorldBound(g1_prim).ComputeAlignedRange().GetMin()[2])
    _lift = -_min_z
    UsdGeom.XformCommonAPI(g1_prim).SetTranslate(Gf.Vec3d(0.0, 0.0, _lift))
    print(f"[smoke] G1 bbox min_z={_min_z:.3f} -> lift +{_lift:.3f} m", flush=True)

    # Articulation root（ArticulationRootAPI 保持 prim）を /G1 配下で探索
    art_root = None
    for prim in Usd.PrimRange(g1_prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_root = prim.GetPath().pathString
            break
    print(f"[smoke] articulation root = {art_root}", flush=True)
    robot = ArtCls(prim_path=art_root or "/G1", name="g1")
    world.scene.add(robot)

    # PDF(§6.2) 記載の正規 spawn: pos(-0.15,-0.47895,0.76) rot=(0.7071,0,0,0.7071)=Z軸90°
    SPAWN_POS = np.array([-0.15, -0.47895, _lift])  # z は足接地のため実測 lift を使用
    SPAWN_ORI = np.array([0.7071, 0.0, 0.0, 0.7071])  # (w,x,y,z) Z軸90°

    # 壁で囲われた本体(~3x3m, 原点付近)を俯瞰
    set_camera_view(eye=np.array([3.2, -3.2, 2.2]), target=np.array([0.0, 0.0, 0.4]))

    print("[smoke] reset ...", flush=True)
    world.reset()
    # fix_base が z=0 に再固定するので、reset 後に PDF の spawn 姿勢へ
    try:
        robot.set_world_pose(position=SPAWN_POS, orientation=SPAWN_ORI)
    except Exception as e:  # noqa: BLE001
        print(f"[smoke] set_world_pose failed: {e}", flush=True)
    for _ in range(20):
        world.step(render=True)

    dof_names = list(robot.dof_names)
    print(f"[smoke] num_dof = {robot.num_dof}", flush=True)
    print(f"[smoke] dof_names = {dof_names}", flush=True)

    target_joint = None
    for key in ("right_shoulder_pitch", "right_elbow", "right_shoulder", "right_wrist"):
        for j in dof_names:
            if key in j.lower():
                target_joint = j
                break
        if target_joint:
            break
    if target_joint is None:
        target_joint = next((j for j in dof_names if "right" in j.lower()), dof_names[0])
    jidx = dof_names.index(target_joint)
    print(f"[smoke] target joint = {target_joint} (idx {jidx})", flush=True)
    q0 = np.asarray(robot.get_joint_positions(), dtype=float)

    try:
        bp, _ = robot.get_world_pose()
        print(f"[smoke] base world pos after lift = {np.asarray(bp)}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[smoke] base pose n/a: {e}", flush=True)

    import omni.kit.viewport.utility as vpu

    def capture_viewport(tag: str) -> None:
        for _ in range(8):
            world.step(render=True)
        vp = vpu.get_active_viewport()
        path = f"{OUT_DIR}/{tag}.png"
        vpu.capture_viewport_to_file(vp, path)
        for _ in range(12):
            sim_app.update()
        print(f"[smoke] viewport -> {path}", flush=True)

    cam = None
    try:
        from isaacsim.sensors.camera import Camera

        persp = stage.GetPrimAtPath("/OmniverseKit_Persp")
        tf = UsdGeom.Xformable(persp).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        tr = tf.ExtractTranslation()
        q = tf.ExtractRotationQuat()
        pos = np.array([tr[0], tr[1], tr[2]])
        ori = np.array([q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]])
        cam = Camera(prim_path="/smoke_cam", resolution=(args.cam_w, args.cam_h))
        cam.initialize()
        cam.set_world_pose(pos, ori, camera_axes="usd")
        for _ in range(30):
            world.step(render=True)
    except Exception as e:  # noqa: BLE001
        print(f"[smoke] camera sensor unavailable: {e}", flush=True)

    def save_sensor_rgb(tag: str) -> None:
        if cam is None:
            return
        for _ in range(5):
            world.step(render=True)
        rgba = cam.get_rgba()
        if rgba is None or getattr(rgba, "size", 0) == 0:
            print(f"[smoke] sensor rgba empty at {tag}", flush=True)
            return
        from PIL import Image

        Image.fromarray(np.asarray(rgba)[:, :, :3].astype("uint8")).save(f"{OUT_DIR}/{tag}.png")
        print(f"[smoke] sensor rgb -> {OUT_DIR}/{tag}.png", flush=True)

    capture_viewport("01_standing")
    save_sensor_rgb("01_standing_sensor")

    target_angle = float(q0[jidx]) + 0.6
    cmd = q0.copy()
    cmd[jidx] = target_angle
    print(f"[smoke] command {target_joint}: {q0[jidx]:.3f} -> {target_angle:.3f} rad", flush=True)
    robot.apply_action(ArticulationAction(joint_positions=cmd))
    for i in range(args.steps):
        world.step(render=True)
        if i % 50 == 0:
            q = np.asarray(robot.get_joint_positions(), dtype=float)
            print(f"[smoke] step {i}: q[{jidx}]={q[jidx]:.3f}", flush=True)

    qf = np.asarray(robot.get_joint_positions(), dtype=float)
    print(f"[smoke] final q[{jidx}]={qf[jidx]:.3f} moved={abs(qf[jidx]-q0[jidx]):.3f} rad", flush=True)

    capture_viewport("02_joint_moved")
    save_sensor_rgb("02_joint_moved_sensor")

    print("[smoke] DONE", flush=True)
    sim_app.close()


if __name__ == "__main__":
    main()
