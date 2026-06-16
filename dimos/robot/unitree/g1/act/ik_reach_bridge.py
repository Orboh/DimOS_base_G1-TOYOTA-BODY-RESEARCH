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
    max_joint_delta_deg: float = 60.0          # gross one-shot sanity gate
    require_converged: bool = True             # never publish a non-converged solve
    # Torso-frame workspace box [m]: reject targets outside a plausible right-arm reach.
    ws_x: list[float] = [0.05, 0.85]
    ws_y: list[float] = [-0.85, 0.20]          # right arm lives at -Y
    ws_z: list[float] = [-0.45, 0.55]
    reach_min_interval_s: float = 2.0          # debounce: ignore clicks during/just after a reach
    max_click_age_s: float = 5.0               # reject stale clicks (laptop-local receive time)
    max_state_age_s: float = 1.0               # reject reach if measured motor_states is stale
    log_every_n: int = 1


class IkReachBridge(Module):
    """Clicked 3D point -> right-arm IK -> 14-joint arm_target (one-shot reach)."""

    config: IkReachBridgeConfig

    clicked_point: In[PointStamped]      # human click in the viewer (frame=entity_path)
    motor_states: In[JointState]         # full 29-DOF measured state (IK warm-start)
    arm_target: Out[JointState]          # 14 arm targets -> G1ArmSdkConnection

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._arm = load_g1_right_arm_ik(self.config.urdf_path)
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

        if self.config.require_converged and not converged:
            logger.warning(f"IkReachBridge: IK did not converge (err={err:.4f}); rejecting.")
            return
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
            logger.info(
                f"[{tag}] reach #{self._count}: click({click.frame_id}) "
                f"-> torso{np.round(p_torso, 3)} | q_right={np.round(q_sol, 3)} "
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


__all__ = ["IkReachBridge", "IkReachBridgeConfig"]
