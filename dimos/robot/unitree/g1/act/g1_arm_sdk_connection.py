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

"""G1 upper-body arm control via the Unitree ``rt/arm_sdk`` DDS interface.

Publishes arm joint targets to ``rt/arm_sdk`` (LowCmd_, unitree_hg) WITHOUT
releasing sport mode — the onboard controller keeps the legs balancing while
arm_sdk overrides the upper body, blended by the ``weight`` register at
``motor_cmd[29]`` (0→1). This is a faithful port of unitree_lerobot's
``G1_29_ArmController`` (the verified Stage A path that picked okra on the real
robot on 2026-06-11), restructured as a dimos Module.

What it drives (canonical 29-DOF G1 order):
* arms 15-28 (14 joints) → track the ACT target on ``arm_target``;
* waist 12-14 → held at the STARTUP pose with the stiff gains (kp=300), exactly
  as the reference does, so the torso does not go limp under arm_sdk authority;
* legs 0-11 → left untouched: arm_sdk in motion-control mode ignores them and
  the onboard locomotion controller keeps the robot balancing.

ANTI-DRIFT (the bug that caused the earlier drift):
The command is clipped TOWARD the target relative to the **measured** current
arm pose every 250 Hz cycle (``clip_arm_q_target``), limited to
``arm_velocity_limit`` rad/s — identical to the reference. This guarantees the
command never runs more than one step ahead of reality, so the closed-loop
observation cannot go out of distribution (the previous slew-from-last-command
at 0.5 rad/s made the arm lag the target → OOD → runaway).

SAFETY (first real motion):
- ``q_target`` is initialised to the CURRENT arm pose (hold) until arm_target
  messages arrive; if they stop, the last target is held.
- ``weight`` ramps 0→1 over ``weight_ramp_s`` (the reference snaps it to 1.0,
  which is safe because the clip-from-measured start has zero delta; the short
  ramp here is a conservative extra).
- on stop, ``weight`` ramps 1→0 to hand the arms back to the onboard controller.
- gripper joints are NOT touched here (Dex1 is a separate module/path).
"""

from __future__ import annotations

import threading
from threading import Thread
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import Field
from reactivex.disposable import Disposable

if TYPE_CHECKING:
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    from unitree_sdk2py.utils.crc import CRC

from dimos.control.components import make_humanoid_joints
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.Bool import Bool
from dimos.robot.unitree.g1.act.dds_init import channel_lock, ensure_channel_factory
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_NUM_MOTORS = 29
_WEIGHT_IDX = 29  # kNotUsedJoint0: arm_sdk authority weight (0..1)
_MODE_MACHINE_WAIT_S = 10.0

# Canonical 29-DOF G1 order (matches Unitree G1_29_JointIndex):
_WAIST_IDX = [12, 13, 14]            # waist yaw/roll/pitch — held at startup pose
_ARM_IDX = list(range(15, 29))       # left arm 15-21, right arm 22-28 (14 joints)
_WRIST_IDX = {19, 20, 21, 26, 27, 28}
_G1_JOINTS = make_humanoid_joints("g1")
_ARM_JOINT_NAMES = _G1_JOINTS[15:29]


class G1ArmSdkConnectionConfig(ModuleConfig):
    network_interface: str = Field(default="")
    publish_rate_hz: float = 250.0   # reference control_dt = 1/250
    # Proven arm gains (unitree_lerobot G1_29_ArmController): shoulder/elbow vs wrist.
    kp_arm: float = 80.0
    kd_arm: float = 3.0
    kp_wrist: float = 40.0
    kd_wrist: float = 1.5
    # Stiff gains used by the reference to hold the waist at its startup pose.
    kp_waist: float = 300.0
    kd_waist: float = 3.0
    # SAFETY knobs.
    weight_ramp_s: float = 2.0        # 0->1 authority handover time [s]
    arm_velocity_limit: float = 20.0  # per-cycle clip toward target [rad/s] (reference value)
    motor_states_rate_hz: float = 50.0
    frame_id: str = "g1_pelvis"
    # Throttle for the tracking-error log [cycles]; 250 == ~1/s at 250 Hz. This
    # reports max|target - measured| so B1 can confirm the arm follows the ACT
    # target without drifting (the failure mode this module was built to fix).
    log_track_err_every_n: int = 250
    # Optional 14-vector start pose [rad] the arms slew to before any ACT target
    # arrives, mirroring eval_g1.py (which moves to the dataset's recorded first
    # pose so the policy starts IN-distribution). Empty = hold the current pose.
    initial_arm_pose: list[float] = Field(default_factory=list)
    # DRY-RUN: when False, the loop still reads rt/lowstate and publishes
    # motor_states (so this can be the observation source) but writes NOTHING to
    # rt/arm_sdk — the arms do not move. Used by the dry-run blueprint.
    publish_cmd: bool = True
    # OPERATOR DISCONNECT (2-stage stop). When True, subscribe a `disconnect` Bool:
    # on True the loop ramps weight ->0 over weight_ramp_s (hand the upper body back
    # to the onboard controller) and HOLDS the arm at its measured pose, while the
    # process stays alive (so the operator cuts G1 transmission first, then quits the
    # program). Wired by the okra-harvest blueprint to /g1/arm_sdk_disconnect.
    enable_disconnect: bool = False
    # KINESTHETIC COLLECTION mode (default off = no behavior change for deploy).
    # When True: subscribe reach_done; after a reach completes, the RIGHT arm (22-28)
    # goes compliant (kp ramped to 0 + feedforward gravity tau) so a human hand-guides
    # it while LEFT arm + waist stay stiff. A new arm_target re-stiffens the right arm
    # (next reach). Gravity g(q) from the reduced right-arm model (pinocchio rnea, v=a=0),
    # same as xr_teleoperate robot_arm_ik.py. Verified on hw 2026-06-24 (Step 0/0b PASS).
    collection_mode: bool = False
    compliant_kp_ramp_s: float = 1.5   # right-arm kp 80->0 ramp time on going compliant [s]
    gravity_tau_scale: float = 1.0     # scale on the gravity feedforward tau
    urdf_path: str = ""                # gravity model URDF (empty = right_arm_model DEFAULT_URDF)
    kd_compliant: float = 1.0          # small joint damping while compliant (kp=0)
    # STIFF LEFT-ARM GRAVITY FEEDFORWARD (default OFF).  This is deliberately
    # independent from collection_mode: it retains position gains and only adds
    # a bounded torque estimate to the explicitly selected left-arm joints.
    # A dedicated gate must use a small scale and finite tau limit before any
    # LIVE trial; the reduced URDF model is an estimate, not a torque calibration.
    stiff_gravity_compensation_left: bool = False
    stiff_gravity_left_joint_indices: list[int] = Field(default_factory=list)
    stiff_gravity_tau_scale: float = 1.0
    stiff_gravity_tau_limit_nm: float = 0.0
    stiff_gravity_ramp_s: float = 5.0
    # STIFF RIGHT-ARM GRAVITY FEEDFORWARD (default OFF, added 2026-09-04).
    # Same design as the left-arm gate above, for the arm IkReachBridge drives.
    # Motivation (measured 2026-09-03): the right arm settles BELOW the commanded
    # pose and the shortfall scales with reach — shoulder_pitch gravity torque goes
    # 1.03 N*m at 0.30 m forward to 7.67 N*m at 0.55 m (7.5x). Because a position
    # loop only makes torque from error (tau = kp * dq), that droop cannot be
    # removed by gains: 1 deg at the far pose needs kp_arm ~294 (3.7x default) and
    # 0.5 deg needs ~587. Raising kp to 2x was the practical ceiling on hardware
    # (louder gearbox noise on the return move) and still left 19-29 mm. Cancelling
    # g(q) in the feedforward term removes the cause instead of fighting it.
    # SCALE: full compensation (scale=1.0) on all 7 joints is the intended
    # operating point, not a stretch goal. The same g(q), from the same calibrated
    # URDF, was validated on hardware via the collection_mode hold test
    # (feat/dex1-official-urdf-gravity-test): with kp ramped to ZERO the right arm
    # held its pose on gravity feedforward alone. Adding it here while KEEPING the
    # position gains is a strictly easier condition than that test.
    # SAFETY: opt-in, explicit joint list, finite tau clip, a slow ramp, and a
    # non-finite guard in the loop (np.clip passes NaN straight through). The clip
    # is a runaway backstop, not an operating limit: measured |g(q)| peaks at
    # 8.50 N*m (shoulder_pitch) over reachable poses and 8.67 N*m over the full
    # joint range, so the 12.0 N*m default never binds in normal use.
    stiff_gravity_compensation_right: bool = False
    stiff_gravity_right_joint_indices: list[int] = Field(default_factory=list)
    # Gravity-model URDF for the RIGHT stiff path. Empty = fall back to urdf_path,
    # then to the stock g1.urdf. With a Dex1-1 fitted prefer
    # ``dimos/robot/unitree/g1/g1_dex1_1_calibrated_550g.urdf``: it models the hand
    # as base + 2 fingers calibrated to the 550 g spec (546 g measured) instead of
    # one lumped link, which the stock URDF places ~3.7 mm short — worth ~11 % of
    # the shoulder torque at a far reach.
    stiff_gravity_right_urdf_path: str = ""


class G1ArmSdkConnection(Module):
    """Upper-body DDS control via rt/arm_sdk (legs stay on the onboard controller)."""

    config: G1ArmSdkConnectionConfig

    arm_target: In[JointState]      # 14 arm joint targets (left 7, right 7) [rad]
    motor_states: Out[JointState]   # full 29-DOF state, for the ACT observation
    reach_done: In[Bool]            # (collection_mode) IK settled -> right arm goes compliant
    disconnect: In[Bool]            # (enable_disconnect) operator cut: ramp weight->0, hold, stay alive

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Kinesthetic-collection compliant state (only used when collection_mode).
        self._compliant = False          # right arm hand-guidable (kp->0 + gravity tau)
        self._comp_t0 = 0.0              # kp-ramp start time
        self._grav_model = None
        self._grav_data = None
        self._stiff_left_grav_model = None
        self._stiff_right_grav_model = None
        self._stiff_right_nonfinite_logged = False
        self._stiff_left_grav_data = None
        self._stiff_left_grav_indices: frozenset[int] = frozenset()
        if self.config.collection_mode:
            import pinocchio  # ensure available; used in the loop

            from dimos.robot.unitree.g1.ik_reach.right_arm_model import (
                DEFAULT_URDF,
                load_g1_right_arm_ik,
            )
            urdf = self.config.urdf_path or str(DEFAULT_URDF)
            _arm = load_g1_right_arm_ik(urdf)
            self._grav_model = _arm.ik.model      # reduced 7-DOF right arm (q[0..6] = motors 22-28)
            self._grav_data = self._grav_model.createData()
        if self.config.stiff_gravity_compensation_left:
            selected = self.config.stiff_gravity_left_joint_indices
            if not selected:
                raise ValueError(
                    "stiff_gravity_compensation_left requires one or more left-arm joint indices"
                )
            if any(index < 0 or index >= 7 for index in selected):
                raise ValueError(
                    "stiff_gravity_left_joint_indices must be in the range 0..6"
                )
            if self.config.stiff_gravity_tau_limit_nm <= 0.0:
                raise ValueError(
                    "stiff gravity feedforward requires a positive tau limit"
                )
            import pinocchio  # noqa: F401  (used in the control loop)

            from dimos.robot.unitree.g1.ik_reach.left_arm_gravity_model import (
                load_g1_left_arm_gravity_model,
            )
            urdf = self.config.urdf_path or ""
            self._stiff_left_grav_model = load_g1_left_arm_gravity_model(urdf) if urdf else load_g1_left_arm_gravity_model()
            self._stiff_left_grav_data = self._stiff_left_grav_model.createData()
            self._stiff_left_grav_indices = frozenset(selected)
        if self.config.stiff_gravity_compensation_right:
            selected = self.config.stiff_gravity_right_joint_indices
            if not selected:
                raise ValueError(
                    "stiff_gravity_compensation_right requires one or more right-arm joint indices"
                )
            if any(index < 0 or index >= 7 for index in selected):
                raise ValueError(
                    "stiff_gravity_right_joint_indices must be in the range 0..6"
                )
            if self.config.stiff_gravity_tau_limit_nm <= 0.0:
                raise ValueError(
                    "stiff gravity feedforward requires a positive tau limit"
                )
            if self.config.collection_mode:
                # collection_mode ramps the right arm's kp to 0 and drives its own
                # gravity tau for hand-guiding. Running both would fight over the
                # same joints' tau with two different intents — refuse.
                raise ValueError(
                    "stiff_gravity_compensation_right cannot be combined with collection_mode "
                    "(collection_mode releases position control on the same arm)"
                )
            import pinocchio  # noqa: F401  (used in the control loop)

            from dimos.robot.unitree.g1.ik_reach.right_arm_gravity_model import (
                load_g1_right_arm_gravity_model,
            )
            urdf = self.config.stiff_gravity_right_urdf_path or self.config.urdf_path or ""
            self._stiff_right_grav_model = (
                load_g1_right_arm_gravity_model(urdf) if urdf else load_g1_right_arm_gravity_model()
            )
            self._stiff_right_grav_data = self._stiff_right_grav_model.createData()
            self._stiff_right_grav_indices = frozenset(selected)
            logger.warning(
                "STIFF RIGHT-ARM GRAVITY FEEDFORWARD ENABLED: joints=%s scale=%.3f "
                "limit=%.3f Nm ramp=%.1fs urdf=%s",
                sorted(selected), self.config.stiff_gravity_tau_scale,
                self.config.stiff_gravity_tau_limit_nm, self.config.stiff_gravity_ramp_s,
                urdf or "(default g1.urdf)",
            )
        self._publisher: ChannelPublisher | None = None
        self._subscriber: ChannelSubscriber | None = None
        self._low_cmd: LowCmd_ | None = None
        self._low_state: Any = None
        self._crc: CRC | None = None
        self._mode_machine: int | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Thread | None = None
        self._target_q: np.ndarray | None = None  # 14-vector ACT arm target [rad]
        self._t_start: float = 0.0
        # Operator-disconnect (2-stage stop) state.
        self._disconnect = False         # True => ramp weight->0 + hold, stay alive
        self._disc_t0 = 0.0              # disconnect-ramp start time
        self._weight_at_disc = 1.0       # weight value when disconnect fired (ramp from here)
        self._disc_logged = False        # log "transmission cut" once weight reaches 0
        self._last_weight = 0.0          # last weight commanded (so stop() ramps from here, not 1.0)

    @rpc
    def start(self) -> None:
        super().start()
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        # Serialise unitree DDS channel creation vs sibling DDS modules (the Dex1
        # gripper): concurrent cyclonedds type registration raises "Failed to
        # encode union ... DDS.XTypes.TypeObject".
        with channel_lock:
            ensure_channel_factory(self.config.network_interface)
            self._publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
            self._publisher.Init()
            self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
            self._subscriber.Init(self._on_low_state, 10)
        self._crc = CRC()

        self._low_cmd = unitree_hg_msg_dds__LowCmd_()
        self._low_cmd.mode_pr = 0
        # arm joints carry the tracking gains; waist carries the stiff hold gains.
        # Legs (0-11) are left at their defaults (mode 0, zero gain): arm_sdk ignores
        # them and the onboard locomotion controller keeps them. weight starts at 0.
        for i in _ARM_IDX:
            self._low_cmd.motor_cmd[i].mode = 1
            is_wrist = i in _WRIST_IDX
            self._low_cmd.motor_cmd[i].kp = self.config.kp_wrist if is_wrist else self.config.kp_arm
            self._low_cmd.motor_cmd[i].kd = self.config.kd_wrist if is_wrist else self.config.kd_arm
        for i in _WAIST_IDX:
            self._low_cmd.motor_cmd[i].mode = 1
            self._low_cmd.motor_cmd[i].kp = self.config.kp_waist
            self._low_cmd.motor_cmd[i].kd = self.config.kd_waist
        self._low_cmd.motor_cmd[_WEIGHT_IDX].q = 0.0

        # Wait for the first LowState: capture mode_machine + current arm/waist pose.
        logger.info("Waiting for first LowState (mode_machine + current upper-body pose)...")
        t0 = time.time()
        while time.time() - t0 < _MODE_MACHINE_WAIT_S:
            with self._lock:
                if self._mode_machine is not None and self._low_state is not None:
                    break
            time.sleep(0.05)
        with self._lock:
            if self._mode_machine is None or self._low_state is None:
                raise RuntimeError("No LowState received; cannot start arm_sdk safely")
            arm_q = np.array([float(self._low_state.motor_state[i].q) for i in _ARM_IDX])
            init = self.config.initial_arm_pose
            if init and len(init) == len(_ARM_IDX):
                # Slew (at the safe velocity limit) to the dataset start pose so the
                # policy begins in-distribution, instead of holding the current pose.
                self._target_q = np.array([float(x) for x in init])
                logger.info(f"arm_sdk: slewing to configured initial_arm_pose (max move "
                            f"{float(np.max(np.abs(self._target_q - arm_q))):.3f} rad)")
            else:
                self._target_q = arm_q.copy()  # hold current pose until ACT sends targets
            # Pin the waist command to the current pose ONCE; it is held thereafter.
            for i in _WAIST_IDX:
                self._low_cmd.motor_cmd[i].q = float(self._low_state.motor_state[i].q)
                self._low_cmd.motor_cmd[i].dq = 0.0
                self._low_cmd.motor_cmd[i].tau = 0.0
        logger.info(f"arm_sdk ready (mode_machine={self._mode_machine}); holding current upper-body pose")

        self.register_disposable(Disposable(self.arm_target.subscribe(self._on_arm_target)))
        if self.config.collection_mode:
            self.register_disposable(Disposable(self.reach_done.subscribe(self._on_reach_done)))
        if self.config.enable_disconnect:
            self.register_disposable(Disposable(self.disconnect.subscribe(self._on_disconnect)))
        self._t_start = time.perf_counter()
        self._stop_event.clear()
        self._thread = Thread(target=self._control_loop, name="g1-arm-sdk", daemon=True)
        self._thread.start()
        logger.info(
            "G1ArmSdkConnection started",
            rate_hz=self.config.publish_rate_hz,
            weight_ramp_s=self.config.weight_ramp_s,
            arm_velocity_limit=self.config.arm_velocity_limit,
        )

    @rpc
    def stop(self) -> None:
        # Ramp weight back to 0 to hand the arms back to the onboard controller.
        # Collection: clear compliant first so the loop re-stiffens the right arm at its
        # measured pose before the weight ramp-down (never hand back a limp right arm).
        with self._lock:
            self._compliant = False
        if self.config.collection_mode:
            time.sleep(2.0 * (1.0 / float(self.config.publish_rate_hz)) + 0.05)  # let a few cycles re-stiffen
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            if (
                self.config.publish_cmd
                and self._publisher is not None
                and self._low_cmd is not None
                and self._crc is not None
            ):
                # Ramp from the LAST commanded weight (≈0 already if the operator hit
                # 'd' / disconnect), so a clean shutdown never re-engages arm_sdk at 1.0.
                w_start = float(max(0.0, min(1.0, self._last_weight)))
                for w in np.linspace(w_start, 0.0, 101):
                    with self._lock:
                        # Re-check inside the lock: a concurrent stop() (worker +
                        # coordinator both call stop) may have torn these down already.
                        if self._low_cmd is None or self._publisher is None or self._crc is None:
                            break
                        self._low_cmd.motor_cmd[_WEIGHT_IDX].q = float(w)
                        if self._mode_machine is not None:
                            self._low_cmd.mode_machine = self._mode_machine
                        self._low_cmd.crc = self._crc.Crc(self._low_cmd)
                        self._publisher.Write(self._low_cmd)
                    time.sleep(0.02)
        except Exception as e:
            logger.warning(f"weight ramp-down on stop failed: {e}")
        if self._subscriber is not None:
            try:
                self._subscriber.Close()
            except (OSError, RuntimeError):
                pass
        if self._publisher is not None:
            try:
                self._publisher.Close()
            except (OSError, RuntimeError):
                pass
        self._publisher = self._subscriber = self._low_cmd = self._low_state = self._crc = None
        self._mode_machine = None
        logger.info("G1ArmSdkConnection disconnected")
        super().stop()

    def _on_low_state(self, msg: Any) -> None:
        with self._lock:
            self._low_state = msg
            if self._mode_machine is None:
                self._mode_machine = msg.mode_machine

    def _on_reach_done(self, _msg: Bool) -> None:
        # Collection: the IK reach settled at the pre-grasp -> make the RIGHT arm
        # compliant so the operator hand-guides the grasp (left arm + waist stay stiff).
        with self._lock:
            if not self._compliant:
                self._compliant = True
                self._comp_t0 = time.perf_counter()
        logger.info("G1ArmSdkConnection: reach_done -> RIGHT arm compliant (hand-guide; support it).")

    def _on_disconnect(self, msg: Bool) -> None:
        # Operator pressed 'd': cut G1 transmission. Ramp weight->0 from its current
        # value (hand the upper body back to the onboard controller) and hold the arm
        # at its measured pose; the process stays alive so the operator can then quit.
        if not bool(getattr(msg, "data", True)):
            return
        with self._lock:
            if self._disconnect:
                return
            self._disconnect = True
            self._disc_t0 = time.perf_counter()
            self._weight_at_disc = self._last_weight
        logger.warning(
            "G1ArmSdkConnection: DISCONNECT -> ramping weight->0 over "
            f"{self.config.weight_ramp_s}s; arm returns to the onboard controller (then quit safely)."
        )

    def _on_arm_target(self, msg: JointState) -> None:
        pos = list(msg.position)
        if len(pos) < len(_ARM_IDX):
            logger.warning(f"arm_target has {len(pos)} joints; expected {len(_ARM_IDX)}; ignoring")
            return
        # A new reach target re-stiffens the right arm (it must slew to the next pre-grasp).
        if self.config.collection_mode:
            with self._lock:
                self._compliant = False
        target = np.array([float(x) for x in pos[: len(_ARM_IDX)]])
        if not np.all(np.isfinite(target)):
            logger.warning("arm_target contains non-finite values; ignoring")
            return
        with self._lock:
            self._target_q = target

    def _clip_to_measured(self, target_q: np.ndarray, measured_q: np.ndarray) -> np.ndarray:
        """Scale (target - measured) so the largest joint step <= vel_limit * dt.

        Faithful port of G1_29_ArmController.clip_arm_q_target: clips relative to
        the MEASURED pose every cycle, so the command never runs ahead of reality.
        """
        dt = 1.0 / float(self.config.publish_rate_hz)
        delta = target_q - measured_q
        max_step = self.config.arm_velocity_limit * dt
        motion_scale = np.max(np.abs(delta)) / max_step if max_step > 0 else np.inf
        return measured_q + delta / max(motion_scale, 1.0)

    def _control_loop(self) -> None:
        period = 1.0 / float(self.config.publish_rate_hz)
        ms_period = 1.0 / float(self.config.motor_states_rate_hz)
        next_tick = time.perf_counter()
        last_ms = 0.0
        cycle = 0

        while not self._stop_event.is_set():
            now = time.perf_counter()
            weight = min(1.0, (now - self._t_start) / max(1e-3, self.config.weight_ramp_s))
            cycle += 1

            with self._lock:
                if self._low_cmd is None or self._crc is None or self._publisher is None:
                    break
                low_state = self._low_state
                target = self._target_q
                if low_state is not None and target is not None:
                    measured = np.array([float(low_state.motor_state[i].q) for i in _ARM_IDX])
                    if self._disconnect:
                        # Operator cut: ramp weight from its value at disconnect -> 0
                        # and HOLD the arm at measured (no ACT tracking, no compliant).
                        frac = (now - self._disc_t0) / max(1e-3, self.config.weight_ramp_s)
                        weight = max(0.0, self._weight_at_disc * (1.0 - frac))
                        clipped = measured
                        if weight <= 0.0 and not self._disc_logged:
                            self._disc_logged = True
                            logger.warning(
                                "G1ArmSdkConnection: G1 transmission CUT (weight=0; upper body "
                                "back on the onboard controller). Safe to quit the program ('q')."
                            )
                    else:
                        clipped = self._clip_to_measured(target, measured)
                    # Collection: RIGHT arm (motors 22-28 == _ARM_IDX[7:14]) compliant.
                    comp = self._compliant and not self._disconnect
                    g = None
                    if comp and self._grav_model is not None:
                        a_ramp = min(1.0, (now - self._comp_t0) / max(1e-3, self.config.compliant_kp_ramp_s))
                        import pinocchio
                        g = pinocchio.computeGeneralizedGravity(
                            self._grav_model, self._grav_data, measured[7:14].astype(np.float64)
                        )
                    stiff_left_g = None
                    stiff_left_scale = 0.0
                    if self._stiff_left_grav_model is not None:
                        import pinocchio
                        stiff_left_g = pinocchio.computeGeneralizedGravity(
                            self._stiff_left_grav_model,
                            self._stiff_left_grav_data,
                            measured[:7].astype(np.float64),
                        )
                        stiff_left_scale = float(self.config.stiff_gravity_tau_scale) * min(
                            1.0,
                            (now - self._t_start) / max(1e-3, self.config.stiff_gravity_ramp_s),
                        )
                    stiff_right_g = None
                    stiff_right_scale = 0.0
                    if self._stiff_right_grav_model is not None:
                        import pinocchio
                        # measured[7:14] == motors 22-28 == the reduced right-arm q.
                        stiff_right_g = pinocchio.computeGeneralizedGravity(
                            self._stiff_right_grav_model,
                            self._stiff_right_grav_data,
                            measured[7:14].astype(np.float64),
                        )
                        stiff_right_scale = float(self.config.stiff_gravity_tau_scale) * min(
                            1.0,
                            (now - self._t_start) / max(1e-3, self.config.stiff_gravity_ramp_s),
                        )
                        # np.clip passes NaN/inf straight through, so a broken model
                        # evaluation would reach the motors as a non-finite tau. Drop
                        # the feedforward for this cycle instead (kp still holds).
                        if not np.all(np.isfinite(stiff_right_g)):
                            if not self._stiff_right_nonfinite_logged:
                                self._stiff_right_nonfinite_logged = True
                                logger.error(
                                    "right stiff gravity ff: non-finite g(q)=%s at q=%s — "
                                    "feedforward disabled for these cycles (position gains still active)",
                                    stiff_right_g, measured[7:14],
                                )
                            stiff_right_g = None
                    for k, i in enumerate(_ARM_IDX):
                        is_wrist = i in _WRIST_IDX
                        base_kp = self.config.kp_wrist if is_wrist else self.config.kp_arm
                        if comp and i >= 22:  # right-arm joint -> compliant (hand-guided)
                            self._low_cmd.motor_cmd[i].kp = float(base_kp * (1.0 - a_ramp))
                            self._low_cmd.motor_cmd[i].kd = float(self.config.kd_compliant)
                            self._low_cmd.motor_cmd[i].q = float(measured[k])  # follow measured (no position fight)
                            self._low_cmd.motor_cmd[i].dq = 0.0
                            self._low_cmd.motor_cmd[i].tau = float(g[k - 7]) * self.config.gravity_tau_scale
                        else:                 # stiff position track (left arm always; right when not compliant)
                            if i >= 22:       # restore right-arm gains after compliant
                                self._low_cmd.motor_cmd[i].kp = float(base_kp)
                                self._low_cmd.motor_cmd[i].kd = float(self.config.kd_wrist if is_wrist else self.config.kd_arm)
                            self._low_cmd.motor_cmd[i].q = float(clipped[k])
                            self._low_cmd.motor_cmd[i].dq = 0.0
                            tau = 0.0
                            if i < 22 and stiff_left_g is not None and k in self._stiff_left_grav_indices:
                                raw_tau = float(stiff_left_g[k]) * stiff_left_scale
                                tau = float(np.clip(
                                    raw_tau,
                                    -self.config.stiff_gravity_tau_limit_nm,
                                    self.config.stiff_gravity_tau_limit_nm,
                                ))
                            elif (
                                i >= 22
                                and stiff_right_g is not None
                                and (k - 7) in self._stiff_right_grav_indices
                            ):
                                # k is the 0..13 arm index; the reduced right model is 0..6.
                                raw_tau = float(stiff_right_g[k - 7]) * stiff_right_scale
                                tau = float(np.clip(
                                    raw_tau,
                                    -self.config.stiff_gravity_tau_limit_nm,
                                    self.config.stiff_gravity_tau_limit_nm,
                                ))
                            self._low_cmd.motor_cmd[i].tau = tau
                    self._low_cmd.motor_cmd[_WEIGHT_IDX].q = weight
                    self._last_weight = weight  # so stop() ramps from here, not always 1.0
                    if self._mode_machine is not None:
                        self._low_cmd.mode_machine = self._mode_machine
                    self._low_cmd.crc = self._crc.Crc(self._low_cmd)
                    if self.config.publish_cmd:  # dry-run: never write to rt/arm_sdk
                        self._publisher.Write(self._low_cmd)
                    # Drift watch: max joint error between the ACT target and the
                    # measured pose. Should stay small (≈ within one slew step once
                    # the arm catches up); a growing value signals the arm is not
                    # following → observation goes OOD → the old runaway.
                    if cycle % max(1, self.config.log_track_err_every_n) == 1:
                        track_err = float(np.max(np.abs(target - measured)))
                        logger.info(
                            f"arm track: max|target-measured|={track_err:.3f} rad "
                            f"weight={weight:.2f} {'LIVE' if self.config.publish_cmd else 'DRY'}"
                        )
                        if stiff_left_g is not None:
                            selected_tau = {
                                _ARM_JOINT_NAMES[k]: float(np.clip(
                                    float(stiff_left_g[k]) * stiff_left_scale,
                                    -self.config.stiff_gravity_tau_limit_nm,
                                    self.config.stiff_gravity_tau_limit_nm,
                                ))
                                for k in sorted(self._stiff_left_grav_indices)
                            }
                            logger.info(
                                "left stiff gravity ff: "
                                f"scale={stiff_left_scale:.3f} tau_nm={selected_tau} "
                                f"limit={self.config.stiff_gravity_tau_limit_nm:.3f}"
                            )
                        if stiff_right_g is not None:
                            limit = self.config.stiff_gravity_tau_limit_nm
                            selected_tau = {
                                _ARM_JOINT_NAMES[k + 7]: float(np.clip(
                                    float(stiff_right_g[k]) * stiff_right_scale, -limit, limit
                                ))
                                for k in sorted(self._stiff_right_grav_indices)
                            }
                            # raw = uncapped estimate; if |raw| > limit the clip is binding
                            # and the joint is still under-compensated.
                            raw_tau = {
                                _ARM_JOINT_NAMES[k + 7]: round(
                                    float(stiff_right_g[k]) * stiff_right_scale, 3
                                )
                                for k in sorted(self._stiff_right_grav_indices)
                            }
                            logger.info(
                                "right stiff gravity ff: "
                                f"scale={stiff_right_scale:.3f} tau_nm={selected_tau} "
                                f"raw_nm={raw_tau} limit={limit:.3f}"
                            )

                # Publish motor_states for the ACT observation (downsampled).
                if low_state is not None and (now - last_ms) >= ms_period:
                    last_ms = now
                    names = list(_G1_JOINTS)
                    pos = [float(low_state.motor_state[i].q) for i in range(_NUM_MOTORS)]
                    vel = [float(low_state.motor_state[i].dq) for i in range(_NUM_MOTORS)]
                    js = JointState(name=names, position=pos, velocity=vel, effort=[0.0] * _NUM_MOTORS)
                    js.frame_id = self.config.frame_id
                    self.motor_states.publish(js)

            next_tick += period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                self._stop_event.wait(sleep_for)
            else:
                next_tick = time.perf_counter()


__all__ = ["G1ArmSdkConnection", "G1ArmSdkConnectionConfig"]
