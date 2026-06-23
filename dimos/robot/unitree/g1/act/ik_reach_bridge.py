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

"""IkReachBridge: human-clicked 3D point -> right-arm IK reach (ACT-independent).

A human clicks a point on the okra in the point-cloud viewer; ``RerunWebSocketServer``
emits it as a ``PointStamped``. This Module transforms the point into the right-arm
IK root frame, solves a 7-DOF reach with :class:`PinocchioIK`, and publishes a
14-joint ``arm_target`` (left arm held, right arm = IK solution) to
``G1ArmSdkConnection`` — which applies the proven 250 Hz clip-to-measured / weight
ramp. It is a structural clone of :class:`ActBridge` with the ZMQ ACT call replaced
by a one-shot IK solve.

Safety design (each gate from the plan's adversarial review; see
``.claude/plans/...`` and ``IK_REACH_PLAN.md``):
- Transform math is done entirely in pinocchio SE3 (no Pose.__add__ operand-order
  trap). click-frame -> torso uses a STATIC SE3 (camera is rigidly mounted), NOT a
  cross-host live TF lookup; empty/blank frame_id is rejected outright.
- The reach is gated on: IK convergence, per-joint delta from the measured pose,
  absolute joint-limit compliance, and a torso-frame workspace box. Any failure
  publishes NOTHING (G1ArmSdkConnection then holds the last pose).
- One-shot: each accepted click publishes a single arm_target. A debounce interval
  and click/state freshness checks prevent chained / stale motions.
- DRY-RUN is the safe default: with ``log_only=True`` the computed target is only
  logged. Driving the robot also requires G1ArmSdkConnection.publish_cmd=True.
"""

from __future__ import annotations

import threading
from threading import Thread
import time
from typing import Any

import numpy as np
import pinocchio

from dimos.control.components import make_humanoid_joints
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.manipulation.planning.kinematics.pinocchio_ik import (
    check_joint_delta,
    get_worst_joint_delta,
)
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.Bool import Bool
from dimos.robot.unitree.g1.ik_reach.right_arm_model import (
    DEFAULT_URDF,
    load_g1_right_arm_ik,
)
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# Canonical 29-DOF G1 joint vector: arms are 15-28 (left 15-21, right 22-28).
_ARM_START = 15
_NUM_ARM = 14
_LEFT_SLICE = slice(15, 22)
_RIGHT_SLICE = slice(22, 29)
_G1_JOINTS = make_humanoid_joints("g1")
_ARM_JOINT_NAMES = _G1_JOINTS[_ARM_START : _ARM_START + _NUM_ARM]

# d435_joint extrinsic from g1.urdf:564-568 (torso_link -> d435_link).
_D435_XYZ = np.array([0.0576235, 0.01753, 0.42987])
_D435_RPY = np.array([0.0, 0.8307767239493009, 0.0])
# REP-103 optical-frame rotation (dimos OPTICAL_ROTATION = Quaternion xyzw -0.5,0.5,-0.5,0.5;
# pinocchio.Quaternion takes (w,x,y,z)).
_OPTICAL_WXYZ = (0.5, -0.5, 0.5, -0.5)


def _default_torso_from_optical() -> pinocchio.SE3:
    """Static SE3 torso_link <- camera_color_optical_frame (finalize in R1).

    Composition: torso->d435_link (URDF d435_joint) then d435_link->color_optical
    (REP-103 optical rotation). The small color-vs-depth baseline is ignored; R1
    must confirm the actual click frame and pin the exact transform.
    """
    t_torso_d435 = pinocchio.SE3(pinocchio.rpy.rpyToMatrix(*_D435_RPY), _D435_XYZ.copy())
    r_opt = pinocchio.Quaternion(*_OPTICAL_WXYZ).toRotationMatrix()
    t_d435_optical = pinocchio.SE3(r_opt, np.zeros(3))
    return t_torso_d435 * t_d435_optical


class IkReachBridgeConfig(ModuleConfig):
    urdf_path: str = str(DEFAULT_URDF)
    # DRY-RUN safe default: log the target, publish nothing. Set False to drive arm_sdk.
    log_only: bool = True
    # Reach target shaping (torso_link frame).
    approach_offset_xyz: list[float] = [0.0, 0.0, 0.0]  # +X fwd, +Y left, +Z up [m]
    # Gripper-tip offset from the wrist (right_wrist_yaw_joint), in the WRIST/EE frame
    # [m]. IK drives THIS tip onto the clicked target. Because the tip is a frame
    # rigidly attached to the wrist, wrist-yaw rotation is accounted for exactly each
    # solve iteration (not a frozen pre-offset). Change this per gripper shape.
    # Gripper = Unitree Dex1-1:
    #   G1 part  (g1.urdf right_hand_palm_joint, attached directly to right_wrist_yaw_link,
    #             no intermediate link/plate): yaw-motor -> hand-mount = [0.0415, -0.003, 0]
    #   Dex1-1   mount-face -> fingertip = 0.143 m (Unitree spec: 143x78x67mm body,
    #             jaw length 80mm). NOTE the dex1_1_service URDF joint-origin sum (~0.097)
    #             undercounts — it stops at the last link ORIGIN, missing the jaw/fingertip
    #             mesh extent; the 143mm spec is the real mount-to-tip.
    #   total = [0.0415 + 0.143, -0.003, 0] = [0.1845, -0.003, 0]
    # Only remaining assumption: the Dex1-1 finger axis is mounted along wrist +X (same way
    # the rubber hand pointed). Verify/fine-tune from the live CALIB log (hand_tip torso pos
    # vs the real okra). A cutting attachment, if fitted, extends this further. Load-time.
    gripper_offset_xyz: list[float] = [0.1845, -0.003, 0.0]
    # Handoff standoff [m] along the EE approach axis (= normalize(gripper_offset_xyz)).
    # IK stops the hand TIP this far SHORT of the clicked okra, leaving a pre-grasp pose
    # for ACT to close the last few cm (design: ① IK reach → handoff → ② ACT grasp).
    # 0.0 = drive the tip exactly onto the okra (today's validated behavior). Load-time.
    standoff_m: float = 0.05   # placeholder: tune to ACT's handoff distance once known
    # Fixed EE orientation as quaternion xyzw in the IK ROOT frame; empty = hold the
    # current EE orientation (position-only reach; safest for R3).
    fixed_orientation_xyzw: list[float] = []
    # Expected click frame_id (== the Rerun entity_path of the okra point cloud, e.g.
    # "world/camera/pointcloud"). The static SE3 assumes clicks arrive in the camera
    # color-optical frame; a click on any OTHER entity would be silently mis-transformed.
    # Empty = not yet pinned: dry-run accepts any non-empty frame_id but logs it loudly
    # (so R1 can pin it); LIVE REFUSES to move until this is set. MUST be set for LIVE.
    expected_click_frame: str = ""
    # Safety gates.
    # 90° one-shot cap: a reach from the rest pose to an okra at the edge of the
    # wrist's ~0.45 m reach needs ~66° of shoulder-pitch swing (measured 2026-06-19).
    max_joint_delta_deg: float = 90.0          # gross one-shot sanity gate
    require_converged: bool = True             # never publish a solve worse than max_reach_pos_err_m
    # Best-effort tolerance: an okra at the reach edge converges to ~24 mm, not <eps.
    # Reaching to within this distance is an acceptable "reach toward" for the PoC.
    max_reach_pos_err_m: float = 0.05
    # Torso-frame workspace box [m]: reject targets outside a plausible right-arm reach.
    # Re-fit to the measured palm-tip reach envelope with the 20 cm gripper_offset_xyz
    # (200k uniform-joint samples in torso frame: x[-0.57,0.58] y[-0.71,0.43] z[-0.32,0.83])
    # plus margin. Two human limits kept tighter than kinematics on purpose: x lower
    # (forward-only — okra are in front, never behind/into the torso) and y upper
    # (don't reach across the body to the left; the right arm lives at -Y).
    ws_x: list[float] = [0.05, 0.65]
    ws_y: list[float] = [-0.75, 0.20]          # right arm lives at -Y
    ws_z: list[float] = [-0.35, 0.85]          # raised: 20 cm tip reaches higher/lower
    reach_min_interval_s: float = 2.0          # debounce: ignore clicks during/just after a reach
    max_click_age_s: float = 5.0               # reject stale clicks (laptop-local receive time)
    max_state_age_s: float = 1.0               # reject reach if measured motor_states is stale
    # IK->ACT handoff: after an accepted reach, fire reach_done once the measured
    # right-arm pose has converged to the IK solution (the arm has physically
    # settled at the pre-grasp), so ACT starts from the real pre-grasp pose (not
    # a mid-slew transitional pose). In DRY (log_only) the arm is not driven, so
    # reach_done fires immediately (see _reach) — this only gates the LIVE wait.
    # IK->ACT handoff settle gate. arm_sdk's clip-to-measured leaves a POSE-
    # DEPENDENT steady-state residual (often > several deg, sometimes > 5°,
    # measured on real hw 2026-06-23), so judging "within tol of q_sol" is
    # unreliable. PRIMARY trigger is "the arm STOPPED MOVING" (reached steady
    # state); the residual is then arm_sdk's limit and ACT corrects the rest.
    settle_tol_deg: float = 5.0    # fast-path: fire at once if this close to q_sol
    settle_move_deg: float = 0.5   # per-poll motion below this counts as "not moving"
    settle_stable_s: float = 0.4   # low motion sustained this long = settled (stopped)
    settle_timeout_s: float = 5.0  # fire anyway after this (arm is at steady state by now)
    log_every_n: int = 1


class IkReachBridge(Module):
    """Clicked 3D point -> right-arm IK -> 14-joint arm_target (one-shot reach)."""

    config: IkReachBridgeConfig

    clicked_point: In[PointStamped]      # human click in the viewer (frame=entity_path)
    motor_states: In[JointState]         # full 29-DOF measured state (IK warm-start)
    arm_target: Out[JointState]          # 14 arm targets -> G1ArmSdkConnection
    reach_done: Out[Bool]                # fired once after the arm settles -> ActBridge

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._arm = load_g1_right_arm_ik(
            self.config.urdf_path,
            gripper_offset_xyz=self.config.gripper_offset_xyz,
            standoff_m=self.config.standoff_m,
        )
        # FAIL CLOSED: every downstream index (warm-start pos[22:29], q_sol, the
        # 14-vec, the delta-gate message) assumes the canonical right-arm order.
        # A non-canonical reduced order would silently map shoulder targets onto
        # wrist joints with no per-index gate able to catch it — refuse to construct.
        if not self._arm.order_matches_canonical:
            raise RuntimeError(
                f"reduced right-arm order {self._arm.joint_names} != canonical "
                "RIGHT_ARM_JOINTS; the index mapping would be silently wrong. "
                "Implement an explicit permutation before using this bridge."
            )
        self._T_torso_click = _default_torso_from_optical()
        self._lock = threading.Lock()
        self._latest_state: JointState | None = None
        self._pending_click: PointStamped | None = None
        self._click_recv_t: float = 0.0  # laptop-local receive time (clock-skew-safe freshness)
        self._click_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: Thread | None = None
        self._last_reach_t: float = 0.0
        self._count = 0

    @rpc
    def start(self) -> None:
        super().start()
        from reactivex.disposable import Disposable

        self.register_disposable(Disposable(self.clicked_point.subscribe(self._on_click)))
        self.register_disposable(Disposable(self.motor_states.subscribe(self._on_state)))
        self._stop_event.clear()
        self._thread = Thread(target=self._reach_loop, daemon=True, name="ik-reach-bridge")
        self._thread.start()
        logger.info(
            "IkReachBridge started",
            log_only=self.config.log_only,
            ee_joint_id=self._arm.ee_joint_id,
            approach_offset=self.config.approach_offset_xyz,
        )

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        self._click_event.set()  # wake the worker so it can exit
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        super().stop()

    def _on_state(self, state: JointState) -> None:
        with self._lock:
            self._latest_state = state

    def _on_click(self, pt: PointStamped) -> None:
        recv = time.time()
        with self._lock:
            if self._pending_click is not None:
                logger.debug("IkReachBridge: superseding an unprocessed click (latest wins).")
            self._pending_click = pt
            self._click_recv_t = recv  # stamp on the consumer side (clock-skew-safe)
        self._click_event.set()

    def _reach_loop(self) -> None:
        while not self._stop_event.is_set():
            self._click_event.wait()
            if self._stop_event.is_set():
                break
            self._click_event.clear()
            with self._lock:
                click = self._pending_click
                recv_t = self._click_recv_t
                state = self._latest_state
                self._pending_click = None
            if click is not None and state is not None:
                try:
                    self._reach(click, state, recv_t)
                except Exception as e:  # never let the worker die on one bad click
                    logger.warning(f"IkReachBridge reach failed: {e!r}")

    def _reach(self, click: PointStamped, state: JointState, recv_t: float) -> None:
        now = time.time()
        # --- input gates -------------------------------------------------------
        if not click.frame_id:
            logger.warning("IkReachBridge: click has empty frame_id; rejecting (no silent guess).")
            return
        # frame_id is the Rerun entity_path; the static SE3 is only valid for the okra
        # cloud's optical entity. Require an exact match once pinned (R1); refuse LIVE
        # motion until pinned; in dry-run, accept-but-log so the operator can pin it.
        expected = self.config.expected_click_frame
        if expected:
            if click.frame_id != expected:
                logger.warning(
                    f"IkReachBridge: click frame {click.frame_id!r} != expected "
                    f"{expected!r}; rejecting (wrong entity / frame)."
                )
                return
        elif not self.config.log_only:
            logger.error(
                "IkReachBridge: expected_click_frame is unset while LIVE; refusing to move. "
                "Pin it from R1 (the okra cloud entity_path) before driving the arm."
            )
            return
        else:
            logger.warning(
                f"IkReachBridge: expected_click_frame unset (dry-run). Click frame is "
                f"{click.frame_id!r} — PIN THIS in config before LIVE."
            )
        # Freshness uses the laptop-local RECEIVE time, not the viewer's clock (skew-safe).
        if recv_t and (now - recv_t) > self.config.max_click_age_s:
            logger.warning(f"IkReachBridge: stale click ({now - recv_t:.1f}s old); rejecting.")
            return
        if (now - self._last_reach_t) < self.config.reach_min_interval_s:
            logger.info("IkReachBridge: within debounce interval; ignoring click.")
            return

        # Reject if the measured state is stale (warm-start / orientation / delta-gate
        # baseline all depend on a fresh measured pose).
        state_ts = float(getattr(state, "ts", 0.0) or 0.0)
        if state_ts and (now - state_ts) > self.config.max_state_age_s:
            logger.warning(f"IkReachBridge: stale motor_states ({now - state_ts:.1f}s old); rejecting.")
            return

        pos = list(state.position)
        if len(pos) < _ARM_START + _NUM_ARM:
            logger.warning(f"motor_states has {len(pos)} joints; expected >= 29; skipping.")
            return
        q_left = np.array([float(x) for x in pos[_LEFT_SLICE]])
        q_right = np.array([float(x) for x in pos[_RIGHT_SLICE]])
        if not (np.all(np.isfinite(q_left)) and np.all(np.isfinite(q_right))):
            logger.warning("IkReachBridge: measured arm pose has non-finite values; rejecting.")
            return

        # --- click point -> torso -> IK ROOT frame (static SE3; blocker #1) ----
        p_click = np.array([float(click.x), float(click.y), float(click.z)])
        p_torso = np.asarray(self._T_torso_click.act(p_click)) + np.array(
            self.config.approach_offset_xyz, dtype=float
        )
        # CALIB: raw click (cloud/optical frame) -> torso. Compare torso z against the
        # real object height relative to torso_link (waist) to validate the transform.
        logger.warning(
            "IkReachBridge[CALIB] frame=%s click_optical=[%.3f %.3f %.3f] -> torso=[%.3f %.3f %.3f]",
            click.frame_id,
            float(click.x),
            float(click.y),
            float(click.z),
            float(p_torso[0]),
            float(p_torso[1]),
            float(p_torso[2]),
        )

        # workspace box in torso frame (blocker #11): reject implausible targets.
        if not (
            self.config.ws_x[0] <= p_torso[0] <= self.config.ws_x[1]
            and self.config.ws_y[0] <= p_torso[1] <= self.config.ws_y[1]
            and self.config.ws_z[0] <= p_torso[2] <= self.config.ws_z[1]
        ):
            logger.warning(
                f"IkReachBridge: torso target {np.round(p_torso, 3)} outside workspace box; rejecting."
            )
            return

        p_root = self._arm.torso_to_root(p_torso)

        # orientation: fixed (ROOT frame) or hold current EE orientation.
        if self.config.fixed_orientation_xyzw:
            qx, qy, qz, qw = self.config.fixed_orientation_xyzw
            rot = pinocchio.Quaternion(qw, qx, qy, qz).normalized().toRotationMatrix()
        else:
            rot = self._arm.fk_root(q_right).rotation

        target = pinocchio.SE3(rot, np.asarray(p_root, dtype=float))

        # --- solve + safety gates ---------------------------------------------
        q_sol, converged, err = self._arm.ik.solve(target, q_right)
        q_sol = np.asarray(q_sol, dtype=float).flatten()

        if self.config.require_converged and not converged and err > self.config.max_reach_pos_err_m:
            logger.warning(
                f"IkReachBridge: IK err={err:.4f} m exceeds tol {self.config.max_reach_pos_err_m} m; rejecting."
            )
            return
        if not converged:
            logger.warning(
                f"IkReachBridge: accepting best-effort reach (err={err:.4f} m ≤ "
                f"{self.config.max_reach_pos_err_m} m tol)."
            )
        if not check_joint_delta(q_sol, q_right, self.config.max_joint_delta_deg):
            wi, wd = get_worst_joint_delta(q_sol, q_right)
            logger.warning(
                f"IkReachBridge: rejecting — joint {self._arm.joint_names[wi]} delta "
                f"{wd:.1f}° exceeds {self.config.max_joint_delta_deg}°."
            )
            return
        if not self._arm.clamp_ok(q_sol):
            logger.warning(f"IkReachBridge: q_sol {np.round(q_sol, 3)} violates joint limits; rejecting.")
            return

        # --- accepted: arm the debounce identically in DRY and LIVE -----------
        self._last_reach_t = now
        self._count += 1
        arm14 = np.concatenate([q_left, q_sol])  # left7 hold + right7 IK (canonical order)
        if self.config.log_every_n and self._count % self.config.log_every_n == 0:
            tag = "DRY" if self.config.log_only else "LIVE->arm_sdk"
            # Real hand tip (torso frame): with standoff_m>0 it lands short of p_torso.
            tip_torso = self._arm.root_to_torso_pose(self._arm.fk_tip(q_sol)).translation
            logger.info(
                f"[{tag}] reach #{self._count}: click({click.frame_id}) "
                f"-> torso{np.round(p_torso, 3)} | hand_tip(torso){np.round(np.asarray(tip_torso), 3)} "
                f"standoff={self.config.standoff_m} | q_right={np.round(q_sol, 3)} "
                f"converged={converged} err={err:.4f}"
            )
        if not self.config.log_only:
            self.arm_target.publish(
                JointState(
                    name=list(_ARM_JOINT_NAMES),
                    position=[float(x) for x in arm14],
                    velocity=[0.0] * _NUM_ARM,
                    effort=[0.0] * _NUM_ARM,
                )
            )

        # --- IK->ACT handoff: fire reach_done once the arm has settled --------
        # DRY: the arm is NOT driven (no arm_target published), so it can never
        # converge to q_sol — fire immediately so the wiring is testable without
        # the robot. LIVE: wait until the measured right arm reaches q_sol.
        if self.config.log_only:
            logger.info("IkReachBridge[DRY]: arm not driven; firing reach_done immediately (no settle wait).")
            self.reach_done.publish(Bool(data=True))
            return
        # Fire when the arm SETTLES: reached q_sol (fast path), OR stopped moving
        # (steady state — residual is arm_sdk's clip-to-measured limit), OR timed
        # out (arm is at steady state by now). Never silently strand ACT.
        tol = float(np.deg2rad(self.config.settle_tol_deg))
        move_eps = float(np.deg2rad(self.config.settle_move_deg))
        poll_dt = 0.05
        stable_needed = max(1, int(self.config.settle_stable_s / poll_dt))
        deadline = now + self.config.settle_timeout_s
        prev: np.ndarray | None = None
        stable = 0
        while not self._stop_event.is_set():
            with self._lock:
                st = self._latest_state
            meas_right: np.ndarray | None = None
            if st is not None:
                pos_now = list(st.position)
                if len(pos_now) >= _ARM_START + _NUM_ARM:
                    cand = np.array([float(x) for x in pos_now[_RIGHT_SLICE]])
                    if np.all(np.isfinite(cand)):
                        meas_right = cand
            timed_out = time.time() > deadline
            if meas_right is not None:
                err = float(np.max(np.abs(meas_right - q_sol)))
                moved = float(np.max(np.abs(meas_right - prev))) if prev is not None else float("inf")
                prev = meas_right
                stable = stable + 1 if moved < move_eps else 0
                reached = err < tol
                stopped = stable >= stable_needed
                if reached or stopped or timed_out:
                    reason = "reached" if reached else ("stopped" if stopped else "timeout")
                    logger.info(
                        f"IkReachBridge: pre-grasp settled ({reason}; worst joint err "
                        f"{err:.4f} rad); firing reach_done."
                    )
                    self.reach_done.publish(Bool(data=True))
                    return
            elif timed_out:
                logger.warning(
                    "IkReachBridge: settle timeout with no fresh measured state; firing reach_done anyway."
                )
                self.reach_done.publish(Bool(data=True))
                return
            self._stop_event.wait(poll_dt)


__all__ = ["IkReachBridge", "IkReachBridgeConfig"]
