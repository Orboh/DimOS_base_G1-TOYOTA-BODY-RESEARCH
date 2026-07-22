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
from dimos.robot.unitree.g1.act.two_click_confirm import TwoClickConfirm
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


def _torso_from_optical(xyz: Any, rpy: Any) -> pinocchio.SE3:
    """Static SE3 torso_link <- camera_color_optical_frame for a rigid camera mount.

    Composition: torso->camera_mount (xyz+rpy, URDF convention: +X fwd, +Y left,
    +Z up, positive pitch = camera nose down) then camera_mount->color_optical
    (REP-103 optical rotation — camera-model independent). The small
    color-vs-depth baseline is ignored.
    """
    t_torso_mount = pinocchio.SE3(
        pinocchio.rpy.rpyToMatrix(*np.asarray(rpy, dtype=float)),
        np.asarray(xyz, dtype=float).copy(),
    )
    r_opt = pinocchio.Quaternion(*_OPTICAL_WXYZ).toRotationMatrix()
    t_mount_optical = pinocchio.SE3(r_opt, np.zeros(3))
    return t_torso_mount * t_mount_optical


def _default_torso_from_optical() -> pinocchio.SE3:
    """Static SE3 torso_link <- color_optical for the HEAD D435i (URDF d435_joint)."""
    return _torso_from_optical(_D435_XYZ, _D435_RPY)


class IkReachBridgeConfig(ModuleConfig):
    urdf_path: str = str(DEFAULT_URDF)
    # DRY-RUN safe default: log the target, publish nothing. Set False to drive arm_sdk.
    log_only: bool = True
    # Camera mount pose torso_link <- camera_mount as [x, y, z, roll, pitch, yaw]
    # ([m], [rad]; URDF convention: +X fwd, +Y left, +Z up, positive pitch = camera
    # nose down). Describes the camera BODY placement — the REP-103 optical rotation
    # is composed internally, same as the D435i path. Empty (default) = the head
    # D435i URDF constants (_D435_XYZ/_D435_RPY), preserving existing behavior for
    # every current blueprint. Set this when the click source is a DIFFERENT,
    # rigidly-mounted camera (e.g. the chest ZED Mini).
    camera_mount_xyzrpy: list[float] = []
    # Frame convention of the INCOMING click coordinates. False (default): clicks are
    # raw optical-frame points (X=image right, Y=image down, Z=depth) — true for the
    # D435i pipeline, whose Jetson publisher sends no TF, so the viewer returns the
    # cloud's raw coordinates. True: clicks arrive already rotated into the camera
    # BODY frame (X fwd, Y left, Z up at the camera mount) — true when the camera
    # module (e.g. in-process ZEDCamera) publishes TF: the Rerun viewer parents the
    # cloud under the optical TF chain and resolves click positions into that root,
    # applying the optical rotation BEFORE we see the point (verified empirically
    # 2026-07-16: a point 40cm in front of the ZED clicked as [0.374,-0.006,0.002]).
    # With True, the optical rotation must NOT be composed again (it would be
    # double-applied) — the mount xyz+rpy alone maps click -> torso.
    click_in_camera_body_frame: bool = False
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
    # Handoff standoff [m] along TORSO -X only (Y,Z of the clicked okra preserved).
    # The tip is driven to (click_x - standoff_m, click_y, click_z), leaving a pre-grasp
    # pose ACT closes by advancing +X (design: ① IK reach → handoff → ② ACT grasp).
    # NOT along the gripper approach axis; applied in _reach as a torso-frame offset.
    # 0.0 = drive the tip exactly onto the okra. Effective per-reach (not load-time).
    standoff_m: float = 0.05   # placeholder: tune to ACT's handoff distance once known
    # Approach-from-above two-phase reach (2026-07-21): >0 = first reach a waypoint
    # this many meters DIRECTLY ABOVE the target (same x,y), then descend vertically
    # onto it. Motivation: the one-shot joint-space slew from the low rest pose sweeps
    # UP INTO the plant and knocks the okra; a vertical last leg avoids that and is
    # the natural cutting direction. 0.0 = legacy single-phase reach (default,
    # behavior unchanged). Waypoint infeasible (workspace/IK/limits) -> falls back
    # to the direct reach with a warning. Wired from env OKRA_APPROACH_ABOVE_M.
    approach_above_m: float = 0.0
    # Cartesian streaming of the U-path legs: waypoint spacing [m] and publish
    # cadence [s]. Tip speed ~= path_step_m / path_cadence_s (~0.19 m/s default).
    # Only used when approach_above_m > 0.
    path_step_m: float = 0.035
    path_cadence_s: float = 0.18
    # Two-click confirm guard (2026-07-22): viewer camera-drags emit phantom clicks
    # that drive the arm ("クリックしてないのに勝手に動く"). When True, a reach fires
    # only when a SECOND click lands within confirm_radius_m of the first inside
    # confirm_window_s (the first click only arms + logs). Drag artifacts scatter
    # spatially and never confirm. Default False = legacy single-click behavior.
    # Wired from env OKRA_CONFIRM_CLICK.
    confirm_click: bool = False
    confirm_radius_m: float = 0.03
    confirm_window_s: float = 2.5
    # Minimum gap between the two clicks (2026-07-22 hardening): a slow camera-drag
    # can emit two NEARBY points, defeating the radius check alone. Clicks arriving
    # faster than this re-arm instead of firing, so drag bursts never confirm while
    # a deliberate "click ... click" still does. Wired from OKRA_CONFIRM_MIN_GAP_S.
    confirm_min_gap_s: float = 0.35
    # IK->ACT handoff switch. True (default) = fire reach_done after the reach so ACT
    # takes over. False = do the IK reach and HOLD the pre-grasp (no reach_done, ACT
    # never starts) — for inspecting the reach/standoff without ACT immediately moving
    # the arm. The okra-harvest blueprint wires this from env OKRA_ACT_HANDOFF.
    fire_reach_done: bool = True
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
    # IK->ACT handoff: after an accepted reach we publish q_sol ONCE to arm_sdk
    # (which slews there via clip-to-measured) and fire reach_done after an
    # OPEN-LOOP timed wait — we do NOT read motor state to judge completion.
    # Inferring "settled" from measured convergence false-fired ACT mid-slew
    # (handed off 24-32° short of q_sol, 2026-06-23), and arm_sdk leaves a
    # pose-dependent steady-state residual no within-tol/plateau gate reads
    # reliably. Instead give the arm a duration estimated from the joint travel
    # (delta / nominal speed + margin), clamped, then hand off. nominal_speed is
    # biased SLOW: too fast re-introduces mid-slew firing; too slow only delays
    # the grasp. CAVEAT: open-loop can't detect a stalled/blocked arm — the
    # e-stop operator is the stall detector for this PoC. Tune on hardware.
    reach_nominal_speed_rad_s: float = 1.0   # effective slew speed for the wait estimate [rad/s]
    reach_margin_s: float = 0.5              # additive settle margin on top of travel time [s]
    reach_min_wait_s: float = 0.8            # floor (latency + tiny moves) [s]
    reach_max_wait_s: float = 3.0            # ceiling before handoff [s]
    reach_dry_wait_s: float = 0.1            # DRY: short fixed wait (arm not driven) [s]
    log_every_n: int = 1
    # Hand-eye calibration diagnostic: log the MEASURED gripper tip in torso every N
    # motor_states (0 = off). With this on, position the tip at a marker the head camera
    # can see, read this P_arm, click the same tip in the cloud (P_cam from CALIB), and
    # Δ = P_cam - P_arm is the camera->torso extrinsic error (no need to locate torso).
    tip_log_every_n: int = 0


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
        mount = self.config.camera_mount_xyzrpy
        if mount:
            # FAIL CLOSED: a malformed mount would silently mis-transform every click.
            if len(mount) != 6 or not np.all(np.isfinite(np.asarray(mount, dtype=float))):
                raise ValueError(
                    f"camera_mount_xyzrpy must be 6 finite floats [x,y,z,roll,pitch,yaw], "
                    f"got {mount!r}"
                )
            if self.config.click_in_camera_body_frame:
                # Clicks already rotated to the camera BODY frame by the viewer's TF
                # resolution (see config docstring): mount xyz+rpy alone, NO optical.
                self._T_torso_click = pinocchio.SE3(
                    pinocchio.rpy.rpyToMatrix(*np.asarray(mount[3:], dtype=float)),
                    np.asarray(mount[:3], dtype=float),
                )
            else:
                self._T_torso_click = _torso_from_optical(mount[:3], mount[3:])
        elif self.config.click_in_camera_body_frame:
            raise ValueError(
                "click_in_camera_body_frame=True requires camera_mount_xyzrpy (the "
                "D435i default constants are optical-frame only)."
            )
        else:
            self._T_torso_click = _default_torso_from_optical()
        self._lock = threading.Lock()
        # Two-click confirm gate (shared logic with GripperGraspOnReach so the arm
        # and the jaw always agree on which click fired).
        self._confirm = TwoClickConfirm(
            radius_m=self.config.confirm_radius_m,
            window_s=self.config.confirm_window_s,
            min_gap_s=self.config.confirm_min_gap_s,
        )
        self._latest_state: JointState | None = None
        self._pending_click: PointStamped | None = None
        self._click_recv_t: float = 0.0  # laptop-local receive time (clock-skew-safe freshness)
        self._click_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: Thread | None = None
        self._last_reach_t: float = 0.0
        self._count = 0
        # Separate pinocchio data buffer for the measured-tip diagnostic FK, so it never
        # races the reach thread's ik.solve (which writes self._arm.ik._data).
        self._diag_data = self._arm.ik.model.createData()
        self._state_count = 0

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
            camera_mount=self.config.camera_mount_xyzrpy or "default (head D435i)",
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
        # Hand-eye diagnostic: throttled log of the MEASURED gripper tip in torso.
        n = self.config.tip_log_every_n
        if n <= 0:
            return
        self._state_count += 1
        if self._state_count % n != 0:
            return
        try:
            pos = list(state.position)
            if len(pos) < _ARM_START + _NUM_ARM:
                return
            qr = np.array([float(x) for x in pos[_RIGHT_SLICE]])
            if not np.all(np.isfinite(qr)):
                return
            m = self._arm.ik.model
            pinocchio.forwardKinematics(m, self._diag_data, qr)
            pinocchio.updateFramePlacements(m, self._diag_data)
            tip_root = self._diag_data.oMf[self._arm.tip_frame_id]
            tip_torso = np.asarray(self._arm.root_to_torso_pose(tip_root).translation)
            logger.info(
                "IkReachBridge[TIP] measured gripper tip (torso) = [%.3f %.3f %.3f]",
                float(tip_torso[0]), float(tip_torso[1]), float(tip_torso[2]),
            )
        except Exception as e:  # diagnostic only — never disturb the reach path
            logger.debug(f"tip-log failed: {e!r}")

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

        # --- two-click confirm guard (phantom-click protection) ----------------
        if self.config.confirm_click:
            if not self._confirm.feed(float(click.x), float(click.y), float(click.z), now):
                logger.info(
                    "IkReachBridge: click ARMED (confirm mode) — click the SAME point again "
                    f"within {self.config.confirm_window_s:.1f}s "
                    f"(>= {self.config.confirm_min_gap_s:.2f}s later, "
                    f"<= {self.config.confirm_radius_m * 100:.0f} cm) to fire the reach."
                )
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

        # Pre-grasp standoff along TORSO -X only: stop the tip standoff_m short of the
        # clicked okra in X, preserving its Y and Z (ACT then advances +X to grasp).
        # This is a fixed torso-frame offset, independent of the gripper's approach
        # direction (NOT baked into a wrist frame).
        p_torso = p_torso - np.array([self.config.standoff_m, 0.0, 0.0])

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

        # --- approach-from-above (0 = legacy direct). U-shaped 3-phase path:
        # (1) LIFT straight up at the current tip x,y (near the body / clear of the
        #     plant), (2) TRANSIT horizontally at altitude to above the target,
        # (3) DESCEND vertically onto it. Each phase falls back gracefully if its
        # waypoint is infeasible (2026-07-21: the 2-phase version still swept the
        # plant on the way UP — "上に上げるときにすでに当たる").
        # Each leg is STREAMED as a dense IK waypoint sequence (path_step_m spacing at
        # path_cadence_s) so the TIP follows the straight line — a single endpoint
        # publish only fixes the ends; the joint-space slew between them arcs forward
        # into the plant ("結局下からカーブする", 2026-07-21).
        q_start = q_right
        approach = float(self.config.approach_above_m)
        if approach > 0.0 and not self.config.log_only:
            step_m = max(float(self.config.path_step_m), 0.005)
            cadence = max(float(self.config.path_cadence_s), 0.05)

            def _stream_leg(p_from: np.ndarray, p_to: np.ndarray, q_seed: np.ndarray, label: str):
                """Stream a straight Cartesian leg as dense IK targets. -> (q_next, ok, abort)."""
                q_cur_ = q_seed
                seg = np.asarray(p_to, dtype=float) - np.asarray(p_from, dtype=float)
                n = max(1, int(np.ceil(float(np.linalg.norm(seg)) / step_m)))
                for i in range(1, n + 1):
                    p_i = np.asarray(p_from, dtype=float) + seg * (i / n)
                    if not (
                        self.config.ws_x[0] <= p_i[0] <= self.config.ws_x[1]
                        and self.config.ws_y[0] <= p_i[1] <= self.config.ws_y[1]
                        and self.config.ws_z[0] <= p_i[2] <= self.config.ws_z[1]
                    ):
                        logger.warning(f"IkReachBridge: {label} step {i}/{n} outside workspace; leg stopped.")
                        return q_cur_, False, False
                    tgt = pinocchio.SE3(rot, np.asarray(self._arm.torso_to_root(p_i), dtype=float))
                    qn, cv, er = self._arm.ik.solve(tgt, q_cur_)
                    qn = np.asarray(qn, dtype=float).flatten()
                    if not (
                        (cv or er <= self.config.max_reach_pos_err_m)
                        and check_joint_delta(qn, q_cur_, self.config.max_joint_delta_deg)
                        and self._arm.clamp_ok(qn)
                    ):
                        logger.warning(f"IkReachBridge: {label} step {i}/{n} infeasible; leg stopped.")
                        return q_cur_, False, False
                    self.arm_target.publish(
                        JointState(
                            name=list(_ARM_JOINT_NAMES),
                            position=[float(x) for x in np.concatenate([q_left, qn])],
                            velocity=[0.0] * _NUM_ARM,
                            effort=[0.0] * _NUM_ARM,
                        )
                    )
                    q_cur_ = qn
                    if self._stop_event.wait(cadence):
                        return q_cur_, False, True  # shutting down: abort the whole reach
                logger.info(f"IkReachBridge: {label} leg done ({n} steps -> torso{np.round(p_to, 3)})")
                return q_cur_, True, False

            tip_now = np.asarray(
                self._arm.root_to_torso_pose(self._arm.fk_tip(q_right)).translation
            ).flatten()
            z_transit = max(float(p_torso[2]) + approach, float(tip_now[2]))
            p_lift = np.array([tip_now[0], tip_now[1], z_transit])
            p_over = np.array([p_torso[0], p_torso[1], z_transit])
            q_cur = q_right
            ok = True
            abort = False
            # (1) LIFT: straight up at the current tip x,y (clear of the plant).
            if float(tip_now[2]) < z_transit - 0.02:
                q_cur, ok, abort = _stream_leg(tip_now, p_lift, q_cur, "lift")
                if abort:
                    return
            # (2) TRANSIT: horizontal at altitude to directly above the target.
            if ok:
                q_cur, ok, abort = _stream_leg(p_lift, p_over, q_cur, "transit")
                if abort:
                    return
            # (3) DESCEND: vertical drop onto the target.
            if ok:
                q_cur, ok, abort = _stream_leg(p_over, np.asarray(p_torso, dtype=float), q_cur, "descend")
                if abort:
                    return
            # Adopt the streamed endpoint solution when the path completed; on a
            # stopped leg fall through to the direct final publish from wherever
            # the arm is now (q_start=q_cur keeps the handoff wait honest).
            if q_cur is not q_right:
                if ok:
                    q_sol = q_cur
                else:
                    q_d, conv_d, err_d = self._arm.ik.solve(target, q_cur)
                    q_d = np.asarray(q_d, dtype=float).flatten()
                    if (
                        (conv_d or err_d <= self.config.max_reach_pos_err_m)
                        and check_joint_delta(q_d, q_cur, self.config.max_joint_delta_deg)
                        and self._arm.clamp_ok(q_d)
                    ):
                        q_sol, converged, err = q_d, bool(conv_d), float(err_d)
                q_start = q_cur

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

        # --- IK->ACT handoff: OPEN-LOOP timed wait, then fire reach_done ------
        # We do NOT read motor state to decide completion (that false-fired ACT
        # mid-slew). Estimate how long the arm needs from the joint travel it must
        # cover (q_right = measured start pose, rad), wait that long, then hand off.
        if self.config.log_only:
            # DRY: arm not driven; short fixed wait keeps wiring testable + ordering.
            delta = 0.0
            wait_s = self.config.reach_dry_wait_s
        else:
            delta = float(np.max(np.abs(q_sol - q_start)))
            wait_s = delta / max(self.config.reach_nominal_speed_rad_s, 1e-3) + self.config.reach_margin_s
            wait_s = min(max(wait_s, self.config.reach_min_wait_s), self.config.reach_max_wait_s)
        if self._stop_event.wait(wait_s):
            return  # stopping: never hand off to ACT on a shutting-down bridge
        # Diagnostic ONLY (never a gate): worst-joint err at handoff, to tune the
        # wait. If this is large/decreasing, the wait fired mid-slew -> lower
        # reach_nominal_speed_rad_s or raise reach_margin_s.
        fire_err = float("nan")
        with self._lock:
            st = self._latest_state
        if st is not None:
            pos_now = list(st.position)
            if len(pos_now) >= _ARM_START + _NUM_ARM:
                m = np.array([float(x) for x in pos_now[_RIGHT_SLICE]])
                if np.all(np.isfinite(m)):
                    fire_err = float(np.max(np.abs(m - q_sol)))
        if not self.config.fire_reach_done:
            # ACT handoff disabled: hold the IK pre-grasp so the reach can be inspected
            # (arm_sdk keeps commanding q_sol). ACT never starts (no reach_done on the bus).
            logger.info(
                f"IkReachBridge: ACT handoff OFF — holding pre-grasp (delta={delta:.3f} rad, "
                f"worst-joint err at hold={fire_err:.4f} rad); NOT firing reach_done."
            )
            return
        logger.info(
            f"IkReachBridge: open-loop wait {wait_s:.2f}s done (delta={delta:.3f} rad, "
            f"nominal={self.config.reach_nominal_speed_rad_s} margin={self.config.reach_margin_s}); "
            f"firing reach_done. [diag worst-joint err at fire={fire_err:.4f} rad - NOT a gate]"
        )
        self.reach_done.publish(Bool(data=True))


__all__ = ["IkReachBridge", "IkReachBridgeConfig"]
