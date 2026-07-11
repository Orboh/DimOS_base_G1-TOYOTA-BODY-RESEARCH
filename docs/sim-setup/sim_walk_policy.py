#!/usr/bin/env python3
"""歩行ポリシー単体検証（A方式・段階1）: base-free g1bag を unitree_rl_lab の velocity policy で
「立つ→cmd_velで歩く」を確認する最小 Isaac スクリプト（収穫は無し）。

実装の正本は sim_walk_lib.py（floating base 化 / SDK順 gains / per-term obs / PolicyWalker）。
6大要点（欠くと転倒/NaN）もそちらの docstring とメモリ g1-isaac-policy-walk-floating-base 参照。

実績（2026-07-11）: 立位 20s 転倒0 / 前進 vx=0.3 で 20s 6.9m（実測 0.35m/s, 横ズレ 0.3m）
                   GUI でも 20s 5.5m 転倒0。

実行:
  PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES \
    WALK_VX=0.3 WALK_SECS=20 WALK_BASE_Z=0.80 WALK_SETTLE_STEPS=5 \
    ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/sim_walk_policy.py [--gui]
  WALK_ROOM=0 で部屋なし（地面のみ）。WALK_VX=0 で立位のみ。
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
ROOM = f"{REPO}/usd_file/chinou_center.usd"
G1 = f"{REPO}/usd_file/g1-29dof-dex1-base-fix-usd/g1bag.usd"
POLICY = f"{REPO}/usd_file/walk_policy/policy.onnx"
DEPLOY_YAML = f"{REPO}/usd_file/walk_policy/deploy.yaml"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # sim_walk_lib（同ディレクトリ）

import numpy as np

import sim_walk_lib as wl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()
    vx = float(os.getenv("WALK_VX", "0.0"))    # [m/s] 前進指令
    vy = float(os.getenv("WALK_VY", "0.0"))    # [m/s] 左指令
    wz = float(os.getenv("WALK_WZ", "0.0"))    # [rad/s] 旋回指令
    secs = float(os.getenv("WALK_SECS", "8"))  # [s] 実行時間（wall-clock）

    dp = wl.load_deploy(DEPLOY_YAML)
    print(f"[walk] policy obs hist={dp['hist']} step_dt={dp['step_dt']} cmd=({vx},{vy},{wz})", flush=True)

    from isaacsim import SimulationApp
    sim_app = SimulationApp({"headless": not args.gui})
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage, open_stage
    from isaacsim.core.utils.types import ArticulationAction
    import omni.usd

    if os.getenv("WALK_ROOM", "1") == "1":
        open_stage(ROOM)
    else:
        print("[walk] WALK_ROOM=0: 部屋なし（ground plane のみ）", flush=True)
    # physics_dt=1/200: policy 50Hz の 1/4。既定 1/60 のままだと ≈15Hz になり破綻（要点）。
    world = World(stage_units_in_meters=1.0, physics_dt=wl.PHYS_DT, rendering_dt=4 * wl.PHYS_DT)
    stage = omni.usd.get_context().get_stage()
    try:
        world.scene.add_default_ground_plane(z_position=0.0)
        print("[walk] ground plane @z=0 追加（足場保証）", flush=True)
    except Exception as _e:  # noqa: BLE001
        print(f"[walk] ground plane warn: {_e}", flush=True)

    add_reference_to_stage(usd_path=G1, prim_path="/G1")
    g1 = stage.GetPrimAtPath("/G1")

    # floating base 化（root=pelvis, solver 32/4, selfColl OFF, armature 0.01）
    pelvis_path = wl.make_floating_base(
        stage, g1, armature=float(os.getenv("WALK_ARMATURE", "0.01")))
    if pelvis_path is None:
        print("[walk] ERROR: pelvis が見つからない", flush=True)
        sim_app.close()
        return 1

    robot = SingleArticulation(prim_path=pelvis_path, name="g1")
    world.scene.add(robot)
    world.reset()  # ★reset 後は step せず先に配置（authored 姿勢は足が床を踏み抜いている）

    imap = wl.motor_to_isaac_map(list(robot.dof_names))
    n_mapped = sum(1 for i in imap if i is not None)
    print(f"[walk] num_dof={robot.num_dof} mapped {n_mapped}/29", flush=True)

    wl.apply_sdk_gains(robot, dp, imap, kp_scale=float(os.getenv("WALK_KP_SCALE", "1.0")))

    walker = wl.PolicyWalker(POLICY, dp, imap)
    walker.place_upright(robot, world, render=args.gui,
                         settle=int(os.getenv("WALK_SETTLE_STEPS", "5")),
                         base_z=float(os.getenv("WALK_BASE_Z", str(wl.BASE_Z0))))

    cmd = np.array([vx, vy, wz], dtype=np.float32)
    print(f"[walk] loop start（{secs}s, GUI={args.gui}）", flush=True)
    import time as _t
    t0 = _t.time()
    step = 0
    while _t.time() - t0 < secs and sim_app.is_running():
        tgt = walker.tick(robot, cmd)
        robot.apply_action(ArticulationAction(joint_positions=tgt))
        # 1 tick = 20ms。GUI: step(render=True) は 4substep 進む→1回 / headless: 5ms×4回（要点）
        if args.gui:
            world.step(render=True)
        else:
            for _ in range(4):
                world.step(render=False)
        step += 1
        if step < 5:  # 立ち上がり確認: grav≈[0,0,-1] / angvel≈0 / |action| 有界なら健全
            d = walker.dbg
            print(f"[diag] step{step - 1}: grav={np.round(d['grav'], 2).tolist()} "
                  f"|angvel|={float(np.linalg.norm(d['angvel'])):.2f} "
                  f"|action|max={float(np.abs(d['action']).max()):.2f}", flush=True)
        if step % 25 == 0:
            pos, _ = robot.get_world_pose()
            px, py, pz = (float(v) for v in np.asarray(pos).reshape(-1)[:3])
            print(f"[walk] t={_t.time() - t0:.1f}s pos=({px:+.2f},{py:+.2f},{pz:.3f})"
                  f"（立=z~0.79, 転=低z）", flush=True)

    print("[walk] done", flush=True)
    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
