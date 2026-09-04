#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Smoke-test the generated MuJoCo scene against the Pinocchio IK it exists to serve.

This is the gate that makes every later sim result meaningful. If sim FK and IK FK
disagree, a "successful" reach in sim says nothing about the hardware. Four checks:

  1. JOINT ORDER  -- MJCF hinge order == canonical make_humanoid_joints("g1") 29-DOF order.
                     Every 14-vec arm_target and 29-vec motor_states in the pipeline
                     indexes by position, so a permutation here is silent corruption.
  2. FK AGREEMENT -- gripper-tip position from MuJoCo (body xpos/xmat + wrist-frame tip
                     offset), expressed in torso_link, vs right_arm_model's fk_tip. These
                     must match to well under a millimetre.
  3. TRACKING     -- IK-solve for the okra at the pre-grasp standoff, feed the solution to
                     the position actuators, step, and confirm the tip settles onto the
                     commanded point (this is what IkReachBridge will do at run time).
  4. CAMERAS      -- all three cameras render, and the okra is actually visible in the
                     chest view (otherwise there is nothing to click on).

Run:
    MUJOCO_GL=glfw .venv/bin/python oda/mujoco_sim/smoke_sim_arm.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

# Must be set BEFORE mujoco creates any GL context. EGL is the only backend that stayed
# correct on this laptop (RTX 3070, driver 580.173.02) when several Renderers are alive at
# once: with MUJOCO_GL=glfw the 2nd and 3rd renderer return uninitialised GPU memory that
# *looks* like a plausible image (nonzero mean) but is pure noise. See gl_backend() below.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dimos.control.components import make_humanoid_joints
from dimos.robot.unitree.g1.ik_reach.right_arm_model import load_g1_right_arm_ik
from oda.mujoco_sim.build_g1_scene import _OUT, HOME_Q, OKRA_IN_TORSO

# Same tip offset the okra blueprints use (Dex1-1 jaw): wrist -> fingertip, WRIST frame.
TIP_OFFSET = [float(v) for v in os.getenv("OKRA_TIP_OFFSET_XYZ", "0.1845,-0.003,0.0").split(",")]
STANDOFF_M = float(os.getenv("OKRA_STANDOFF_M", "0.05"))

_RIGHT_SLICE = slice(22, 29)
FK_TOL_M = 1e-4
TRACK_TOL_M = 0.01


class SimRig:
    """Thin wrapper: load the scene, expose torso-frame tip FK and arm actuation."""

    def __init__(self, scene: Path) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        mujoco.mj_forward(self.model, self.data)
        self.joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        self._torso = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self._wrist = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link"
        )
        self._okra = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "okra")

    def torso_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.data.xpos[self._torso], dtype=float),
            np.asarray(self.data.xmat[self._torso], dtype=float).reshape(3, 3),
        )

    def to_torso(self, p_world: np.ndarray) -> np.ndarray:
        pos, rot = self.torso_pose()
        return rot.T @ (np.asarray(p_world, dtype=float) - pos)

    def tip_world(self, tip_offset: np.ndarray) -> np.ndarray:
        """Gripper tip in world coords: wrist body origin + tip offset rotated by the wrist."""
        pos = np.asarray(self.data.xpos[self._wrist], dtype=float)
        rot = np.asarray(self.data.xmat[self._wrist], dtype=float).reshape(3, 3)
        return pos + rot @ np.asarray(tip_offset, dtype=float)

    def tip_torso(self, tip_offset: np.ndarray) -> np.ndarray:
        return self.to_torso(self.tip_world(tip_offset))

    def okra_torso(self) -> np.ndarray:
        return self.to_torso(np.asarray(self.data.xpos[self._okra], dtype=float))

    def command_right_arm(self, q_right: np.ndarray) -> None:
        self.data.ctrl[_RIGHT_SLICE] = np.asarray(q_right, dtype=float)

    def settle(self, seconds: float) -> None:
        for _ in range(int(seconds / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)

    def q_right(self) -> np.ndarray:
        return np.asarray(self.data.qpos[_RIGHT_SLICE], dtype=float).copy()


def _check_joint_order(rig: SimRig) -> list[str]:
    canonical = [n.split("/")[-1] for n in make_humanoid_joints("g1")]
    sim = [n.removesuffix("_joint") for n in rig.joint_names]
    fails = []
    if len(sim) != 29:
        fails.append(f"sim has {len(sim)} joints, expected 29")
    elif sim != canonical:
        bad = [(i, s, c) for i, (s, c) in enumerate(zip(sim, canonical, strict=False)) if s != c]
        fails.append(f"joint order mismatch at {bad[:4]}")
    print(f"[1] joint order      : {'OK' if not fails else 'FAIL'} ({len(sim)} joints)")
    if fails:
        print(f"    sim      : {sim}")
        print(f"    canonical: {canonical}")
    return fails


def _check_fk(rig: SimRig, arm) -> list[str]:  # type: ignore[no-untyped-def]
    """MuJoCo tip vs Pinocchio tip, in torso frame, at several arm configurations."""
    fails = []
    configs = {
        "home": np.array(HOME_Q[22:29]),
        "zeros": np.zeros(7),
        "reachish": np.array([0.246, -0.215, -0.001, 0.949, -0.064, -0.079, -0.002]),
        "wristy": np.array([0.10, -0.35, 0.25, 0.80, 0.40, -0.30, 0.55]),
    }
    print("[2] FK agreement (MuJoCo vs Pinocchio, torso frame)")
    for label, q in configs.items():
        rig.data.qpos[_RIGHT_SLICE] = q
        rig.data.qvel[:] = 0.0
        mujoco.mj_forward(rig.model, rig.data)
        mj_tip = rig.tip_torso(np.array(TIP_OFFSET))
        pin_tip = np.asarray(arm.root_to_torso_pose(arm.fk_tip(q)).translation, dtype=float)
        err = float(np.linalg.norm(mj_tip - pin_tip))
        status = "OK" if err < FK_TOL_M else "FAIL"
        if err >= FK_TOL_M:
            fails.append(f"FK mismatch at {label}: {err * 1000:.3f} mm")
        print(
            f"    {label:9s} mj={np.round(mj_tip, 5).tolist()} "
            f"pin={np.round(pin_tip, 5).tolist()} err={err * 1000:.4f} mm {status}"
        )
    return fails


def _check_tracking(rig: SimRig, arm) -> list[str]:  # type: ignore[no-untyped-def]
    """IK to the okra at standoff, drive the actuators, confirm the tip arrives."""
    fails = []
    key = mujoco.mj_name2id(rig.model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(rig.model, rig.data, key)
    mujoco.mj_forward(rig.model, rig.data)

    okra_t = rig.okra_torso()
    expected_t = np.asarray(OKRA_IN_TORSO, dtype=float)
    okra_err = float(np.linalg.norm(okra_t - expected_t))
    if okra_err > 1e-3:
        fails.append(f"okra torso pos {okra_t} != designed {expected_t} ({okra_err * 1000:.1f} mm)")

    # Same target shaping IkReachBridge applies: standoff along torso -X only.
    target_t = okra_t - np.array([STANDOFF_M, 0.0, 0.0])
    target_root = arm.torso_to_root(target_t)
    import pinocchio

    target_se3 = pinocchio.SE3(np.eye(3), target_root)
    q_seed = rig.q_right()
    q_sol, converged, err = arm.ik.solve(target_se3, q_seed)

    print("[3] tracking (IK -> actuators -> settle)")
    print(f"    okra   (torso) : {np.round(okra_t, 4).tolist()}  (designed {expected_t.tolist()})")
    print(f"    target (torso) : {np.round(target_t, 4).tolist()}  standoff={STANDOFF_M} m")
    print(f"    IK             : converged={converged} err={err:.6f} m")
    print(f"    q_sol          : {np.round(q_sol, 4).tolist()}")
    if not converged:
        fails.append(f"IK did not converge to the okra standoff target (err={err:.4f})")
    if not arm.clamp_ok(q_sol):
        fails.append("IK solution violates joint limits")

    rig.command_right_arm(q_sol)
    rig.settle(3.0)
    tip_t = rig.tip_torso(np.array(TIP_OFFSET))
    track_err = float(np.linalg.norm(tip_t - target_t))
    q_err_deg = np.degrees(rig.q_right() - q_sol)
    print(
        f"    tip after 3 s  : {np.round(tip_t, 4).tolist()}  |tip-target|={track_err * 1000:.2f} mm"
    )
    print(f"    worst joint err: {np.abs(q_err_deg).max():.3f} deg")
    if track_err > TRACK_TOL_M:
        fails.append(f"arm did not track: {track_err * 1000:.1f} mm > {TRACK_TOL_M * 1000:.0f} mm")
    return fails


def _check_cameras(rig: SimRig, out_dir: Path) -> list[str]:
    fails = []
    print("[4] cameras")
    import cv2

    for cam, (w, h) in (
        ("chest_cam", (640, 360)),
        ("wrist_cam", (320, 240)),
        ("spectator", (640, 480)),
    ):
        renderer = None
        try:
            renderer = mujoco.Renderer(rig.model, height=h, width=w)
            renderer.update_scene(rig.data, camera=cam)
            px = renderer.render().copy()
        except Exception as exc:
            fails.append(f"{cam} render failed: {exc}")
            print(f"    {cam:10s} FAIL {exc}")
            continue
        finally:
            if renderer is not None:
                renderer.close()
        arr = np.asarray(px)
        # GARBAGE GUARD. A broken GL context hands back uninitialised GPU memory, which has
        # a perfectly plausible mean brightness but tens of thousands of distinct colours;
        # a real render of this flat-shaded scene has ~1-4k. Without this check the camera
        # test passes on pure noise -- it did exactly that under MUJOCO_GL=glfw, where the
        # 2nd and 3rd live Renderer return junk.
        uniq = len(np.unique(arr.reshape(-1, 3), axis=0))
        # "Green enough to be the okra": G clearly dominant over both R and B. The scene's
        # greys and the gradient sky have no channel dominance, so this isolates pod+stem.
        r, g, b = (arr[:, :, i].astype(int) for i in range(3))
        greenish = int(np.sum(((g - r) > 25) & ((g - b) > 25)))
        print(f"    {cam:10s} {arr.shape} mean={arr.mean():.1f} uniq={uniq} green_px={greenish}")
        if uniq > 8000:
            fails.append(f"{cam} render looks like GPU noise ({uniq} unique colours)")
        if arr.mean() < 1.0:
            fails.append(f"{cam} rendered an all-black frame (GL backend problem?)")
        if cam == "chest_cam" and greenish < 50:
            fails.append(f"chest_cam sees only {greenish} okra pixels -- nothing to click on")
        cv2.imwrite(str(out_dir / f"smoke_{cam}.png"), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    print(f"    wrote {out_dir}/smoke_*.png")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default=str(_OUT))
    args = ap.parse_args()

    scene = Path(args.scene)
    if not scene.exists():
        print(f"scene missing: {scene}\nrun: .venv/bin/python oda/mujoco_sim/build_g1_scene.py")
        return 2

    rig = SimRig(scene)
    arm = load_g1_right_arm_ik(gripper_offset_xyz=TIP_OFFSET)
    if not arm.order_matches_canonical:
        print(f"FAIL: reduced IK model order {arm.joint_names} is not canonical")
        return 1

    fails: list[str] = []
    fails += _check_joint_order(rig)
    fails += _check_fk(rig, arm)
    fails += _check_tracking(rig, arm)
    fails += _check_cameras(rig, scene.parent)

    print()
    if fails:
        print(f"SMOKE_SIM_ARM FAIL ({len(fails)})")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("SMOKE_SIM_ARM OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
