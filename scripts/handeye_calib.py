#!/usr/bin/env python3
# Copyright 2025-2026 Dimensional Inc.
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

"""Interactive hand-eye calibration for the G1 okra-harvest head D435i.

The gripper tip lands a few cm off the clicked okra. The gripper tip is on the
wrist axis (URDF-correct), so the suspect is the camera->torso MOUNT extrinsic
(pitch / Z) — URDF nominal vs the real mount. This tool measures it WITHOUT
needing to know where torso_link physically is: it compares, for the same
physical point, where the ARM thinks its tip is (FK = truth) vs where the CAMERA
maps a click (current extrinsic). It then fits a corrected mount (pitch + xyz).

It runs ALONGSIDE the okra-harvest app (which drives the arm, point cloud and
viewer). This tool only LISTENS on the LCM bus (/g1/motor_states, /clicked_point)
and walks you through each measurement pair, capturing on Enter.

Per pair (the tool prompts; you act, then press Enter):
  A) Position the arm (click a spot in the viewer; let it settle). Bring a marker
     (or the okra) to TOUCH the gripper tip. Press Enter -> captures P_arm = the
     measured tip in torso (arm FK; this equals the marker's true position).
  B) Click that SAME marker in the point cloud. Press Enter -> captures the raw
     click and P_cam (current extrinsic). Delta = P_cam - P_arm is the error.

Run (laptop, SAME LCM bus as the harvest app):
  term1:  OKRA_TARGET_Z_OFFSET=0 OKRA_ACT_HANDOFF=0 bash .../start_okra_harvest.sh --live
  term2:  LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1' \
          .venv/bin/python scripts/handeye_calib.py --pairs 5
  selftest (no robot): .venv/bin/python scripts/handeye_calib.py --selftest
"""

from __future__ import annotations

import argparse
import os
import threading

import numpy as np
import pinocchio

from dimos.robot.unitree.g1.act.ik_reach_bridge import (
    _D435_RPY,
    _D435_XYZ,
    _OPTICAL_WXYZ,
    _default_torso_from_optical,
)
from dimos.robot.unitree.g1.ik_reach.right_arm_model import load_g1_right_arm_ik

LCM_URL = os.getenv("LCM_DEFAULT_URL", "udpm://239.255.76.67:7667?ttl=1")
# Match the bridge's gripper tip (palm 0.0415 + Dex1 0.143; Y from palm_joint).
GRIPPER_OFFSET_XYZ = [0.1845, -0.003, 0.0]
_RIGHT_SLICE = slice(22, 29)  # right-arm q in the 29-DOF motor vector
_ARM_END = 29


def _fit_mount(p_opts: np.ndarray, p_arms: np.ndarray) -> dict:
    """Fit the torso<-d435 mount (pitch about Y + translation) to map the raw
    optical clicks onto the arm-truth points.

    Model: p_torso = Ry(pitch) @ (R_opt @ p_opt) + t   (d435<-optical has 0 translation)
    1-D grid over pitch; closed-form least-squares translation per pitch.
    """
    r_opt = pinocchio.Quaternion(*_OPTICAL_WXYZ).toRotationMatrix()
    p_d435 = (r_opt @ p_opts.T).T  # N x 3, optical mapped into the d435 frame

    def residual(pitch: float) -> tuple[float, np.ndarray]:
        ry = pinocchio.rpy.rpyToMatrix(0.0, pitch, 0.0)
        d = (ry @ p_d435.T).T  # N x 3
        t = (p_arms - d).mean(axis=0)  # best translation for this pitch
        err = p_arms - (d + t)
        return float(np.sqrt(np.mean(np.sum(err**2, axis=1)))), t

    nominal = float(_D435_RPY[1])
    best = None
    for pitch in np.arange(nominal - 0.25, nominal + 0.25, 0.001):
        rms, t = residual(pitch)
        if best is None or rms < best["rms"]:
            best = {"rms": rms, "pitch": float(pitch), "t": t}
    nom_rms, _ = residual(nominal)
    best["nominal_rms"] = nom_rms
    return best


def _report(p_opts: np.ndarray, p_arms: np.ndarray) -> None:
    T_cur = _default_torso_from_optical()
    p_cam = np.array([np.asarray(T_cur.act(po)) for po in p_opts])
    delta = p_cam - p_arms  # camera - arm, per pair
    print("\n================ RESULT ================")
    np.set_printoptions(precision=4, suppress=True)
    for i, (pa, pc, d, po) in enumerate(zip(p_arms, p_cam, delta, p_opts, strict=False)):
        depth = float(po[2])  # optical Z = depth
        print(f"#{i + 1} depth={depth:.3f}  P_arm={pa}  P_cam={pc}  Δ(cam-arm)={d}")
    print(f"\nmean Δ (cam-arm) = {delta.mean(axis=0)}  | std = {delta.std(axis=0)}")
    # depth dependence of the vertical error => pitch; ~constant => Z translation
    depths = p_opts[:, 2]
    if len(depths) >= 3 and depths.std() > 1e-3:
        cz = float(np.corrcoef(depths, delta[:, 2])[0, 1])
        print(f"corr(depth, Δz) = {cz:+.2f}  ( |corr|→1 = pitch error; ~0 = constant Z )")
    fit = _fit_mount(p_opts, p_arms)
    dp = np.degrees(fit["pitch"] - float(_D435_RPY[1]))
    print(
        f"\nfit RMS = {fit['rms'] * 1000:.1f} mm   (nominal RMS = {fit['nominal_rms'] * 1000:.1f} mm)"
    )
    print("---- corrected camera mount (torso<-d435) ----")
    print(f"  xyz : {np.array(_D435_XYZ)}  ->  {fit['t']}")
    print(f"  pitch(rad) : {float(_D435_RPY[1]):.6f}  ->  {fit['pitch']:.6f}   (Δ {dp:+.2f} deg)")
    print("\nApply to BOTH (keep them in sync):")
    print(
        f'  g1.urdf d435_joint:  <origin xyz="{fit["t"][0]:.5f} {fit["t"][1]:.5f} {fit["t"][2]:.5f}"'
        f' rpy="0 {fit["pitch"]:.10f} 0"/>'
    )
    print(
        f"  ik_reach_bridge.py:  _D435_XYZ={list(np.round(fit['t'], 5))}  _D435_RPY=[0.0, {fit['pitch']:.10f}, 0.0]"
    )
    print("========================================\n")


def _selftest() -> int:
    """Plant a known mount error, synthesize pairs, confirm the fit recovers it."""
    rng_pitch = float(_D435_RPY[1]) + np.radians(4.0)  # plant +4 deg pitch error
    rng_t = np.array(_D435_XYZ) + np.array([0.0, 0.0, -0.03])  # and -3 cm Z
    r_opt = pinocchio.Quaternion(*_OPTICAL_WXYZ).toRotationMatrix()
    ry = pinocchio.rpy.rpyToMatrix(0.0, rng_pitch, 0.0)
    p_opts = np.array(
        [
            [0.05, -0.10, 0.45],
            [-0.04, 0.06, 0.55],
            [0.12, 0.02, 0.62],
            [-0.08, -0.05, 0.50],
            [0.0, 0.10, 0.58],
        ]
    )
    p_arms = np.array([ry @ (r_opt @ po) + rng_t for po in p_opts])  # "truth"
    fit = _fit_mount(p_opts, p_arms)
    ok = abs(fit["pitch"] - rng_pitch) < np.radians(0.3) and np.allclose(fit["t"], rng_t, atol=2e-3)
    print(f"[selftest] planted pitch={rng_pitch:.5f} t={rng_t}")
    print(f"[selftest] fitted  pitch={fit['pitch']:.5f} t={fit['t']}  rms={fit['rms'] * 1e3:.2f}mm")
    print(f"[selftest] {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=int, default=5, help="number of calibration pairs")
    ap.add_argument("--selftest", action="store_true", help="offline fit self-test (no robot)")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()

    from dimos.msgs.geometry_msgs.PointStamped import PointStamped
    from dimos.msgs.sensor_msgs.JointState import JointState
    from dimos.protocol.pubsub.impl.lcmpubsub import LCM, Topic

    arm = load_g1_right_arm_ik(gripper_offset_xyz=GRIPPER_OFFSET_XYZ)
    diag = arm.ik.model.createData()
    lock = threading.Lock()
    state = {"q": None, "click": None, "click_n": 0}

    def on_state(msg, _t):  # type: ignore[no-untyped-def]
        pos = list(msg.position)
        if len(pos) >= _ARM_END:
            with lock:
                state["q"] = np.array([float(x) for x in pos[_RIGHT_SLICE]])

    def on_click(msg, _t):  # type: ignore[no-untyped-def]
        with lock:
            state["click"] = np.array([float(msg.x), float(msg.y), float(msg.z)])
            state["click_n"] += 1

    lc = LCM(url=LCM_URL)
    lc.start()
    lc.subscribe(Topic("/g1/motor_states", JointState), on_state)
    lc.subscribe(Topic("/clicked_point", PointStamped), on_click)
    print(f"[calib] listening on {LCM_URL} (/g1/motor_states, /clicked_point). {args.pairs} pairs.")
    print("[calib] Run the okra-harvest app LIVE with OKRA_ACT_HANDOFF=0 in the other terminal.\n")

    def tip_torso(q: np.ndarray) -> np.ndarray:
        pinocchio.forwardKinematics(arm.ik.model, diag, q)
        pinocchio.updateFramePlacements(arm.ik.model, diag)
        return np.asarray(arm.root_to_torso_pose(diag.oMf[arm.tip_frame_id]).translation)

    p_opts: list[np.ndarray] = []
    p_arms: list[np.ndarray] = []
    try:
        for k in range(1, args.pairs + 1):
            input(
                f"--- Pair {k}/{args.pairs} (A): position arm, TOUCH the marker to the gripper tip, "
                f"then press Enter to capture P_arm > "
            )
            with lock:
                q = None if state["q"] is None else state["q"].copy()
            if q is None:
                print("  no motor_states yet — is the harvest app running? skipping.")
                continue
            p_arm = tip_torso(q)
            with lock:
                n0 = state["click_n"]
            print(f"  P_arm (tip in torso) = {np.round(p_arm, 4)}")
            input(
                f"--- Pair {k}/{args.pairs} (B): now CLICK that same marker in the viewer, "
                f"then press Enter > "
            )
            with lock:
                n1, click = (
                    state["click_n"],
                    None if state["click"] is None else state["click"].copy(),
                )
            if click is None or n1 == n0:
                print("  no NEW click detected since (A) — discarding this pair.")
                continue
            p_cam = np.asarray(_default_torso_from_optical().act(click))
            print(
                f"  click(optical)={np.round(click, 4)}  P_cam={np.round(p_cam, 4)}  "
                f"Δ(cam-arm)={np.round(p_cam - p_arm, 4)}"
            )
            p_opts.append(click)
            p_arms.append(p_arm)
    except (KeyboardInterrupt, EOFError):
        print("\n[calib] interrupted.")
    finally:
        try:
            lc.stop()
        except Exception:
            pass

    if len(p_arms) >= 3:
        _report(np.array(p_opts), np.array(p_arms))
    else:
        print(f"\n[calib] only {len(p_arms)} pairs — need >=3 to fit. Nothing written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
