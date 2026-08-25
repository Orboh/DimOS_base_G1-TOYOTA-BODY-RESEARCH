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
    # Caps how fast the published target may slew toward a distant goal [rad/s]. This is
    # ALSO the speed at which the arm snaps back after the robot briefly takes the arm away
    # (see the runaway guard below): at the reference 20.0 a 0.9 rad displacement is undone
    # in ~45 ms (~1150 deg/s), which is what smashed a hand attachment into the leg on
    # 2026-08-24. 4.0 still covers a full IK reach (0.86 rad in ~215 ms) inside the 1.3 s
    # open-loop wait, with no functional loss.
    arm_velocity_limit: float = 4.0   # per-cycle clip toward target [rad/s]
    # Runaway guard: if the measured pose departs this far from the commanded target, the
    # arm is no longer ours (the robot's own controller moved it). Fighting it back is what
    # causes the collision, so re-anchor the target onto the measured pose and hold.
    runaway_track_err_rad: float = 0.35   # ~20 deg
    # A large gap alone does NOT mean runaway: a fresh IK reach legitimately commands a
    # ~1 rad move, and the gap is large until the arm slews there. The discriminator is
    # DIRECTION -- while slewing to a new target the gap shrinks monotonically, whereas a
    # yank makes it grow. So runaway = gap over the limit AND still growing AND no new
    # target has arrived. (Getting this wrong once cancelled a real reach mid-flight.)
    runaway_growth_rad: float = 0.002     # per-cycle gap growth that counts as "moving away"
    runaway_release: bool = True
    # Spoken warning when the runaway guard trips (shares the one process-wide G1 speaker).
    voice: bool = False
    voice_nic: str = ""
    voice_volume: int = 100
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
    # GRAVITY FEEDFORWARD DURING STIFF POSITION TRACKING (default off = no behavior change).
    # Without it the right arm is a pure PD position loop (tau=0), so holding a pose against
    # gravity requires a permanent position error e = tau_gravity / kp — measured at 3-7 deg
    # (45-90 mm at the tip) on hw 2026-08-24 with kp_arm=80 plus the wrist GoPro/mount payload.
    # That droop is what stalls the UMI diffusion loop: the policy commands "measured + delta",
    # the arm settles "commanded - droop", and when delta ~= droop the tip never advances.
    # Feeding g(q) forward cancels the gravity term so the PD only handles the residual.
    # The model is the SAME reduced right-arm URDF used for IK, so it covers the bare arm only:
    # anything bolted on (GoPro, Media Mod, capture hardware, cabling) is NOT in it — trim with
    # gravity_tau_scale (>1 adds the missing payload). Start LOW (0.6-0.8) and raise while
    # watching `arm track`: too high pushes the arm up on its own.
    gravity_ff: bool = False
    # TIP PAYLOAD added to the gravity model (0 = model unchanged).
    # The reduced right-arm URDF ends in a 0.170 kg `right_rubber_hand` — the display hand
    # G1 ships with. Nothing that is ACTUALLY bolted on (Dex1-1 gripper, GoPro + Media Mod,
    # capture hardware, mount, cabling) exists in it, so g(q) covers roughly half the real
    # load and the arm keeps drooping even with gravity_ff on (measured 2026-08-24: model
    # 5.61 N*m vs 10.72 N*m implied by the residual position error).
    # This is the mass to ADD on top of what the URDF already models, i.e.
    #   tip_extra_mass_kg = (real payload) - 0.170   when the rubber hand was removed.
    # Adding the mass (rather than scaling g(q)) keeps the torque distributed correctly:
    # a tip payload loads the shoulder far more than the wrist, which a uniform
    # gravity_tau_scale cannot express.
    tip_extra_mass_kg: float = 0.0
    # Payload CoM in the WRIST-YAW joint frame [m]. Reference points along +X:
    #   0.0415 = wrist-yaw -> hand mount face, 0.1845 = wrist-yaw -> fingertip.
    # Default 0.113 ~= mid-gripper, a reasonable guess for a Dex1-1 + wrist-cam stack.
    tip_extra_com_xyz: list[float] = [0.113, -0.003, 0.0]
    kd_compliant: float = 1.0          # small joint damping while compliant (kp=0)


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
        # The same reduced right-arm model serves both consumers of g(q): the compliant
        # hand-guide (collection_mode) and the stiff-tracking feedforward (gravity_ff).
        if self.config.collection_mode or self.config.gravity_ff:
            import pinocchio  # noqa: F401  (ensure available; used in the loop)

            from dimos.robot.unitree.g1.ik_reach.right_arm_model import (
                DEFAULT_URDF,
                load_g1_right_arm_ik,
            )
            urdf = self.config.urdf_path or str(DEFAULT_URDF)
            _arm = load_g1_right_arm_ik(urdf)
            self._grav_model = _arm.ik.model      # reduced 7-DOF right arm (q[0..6] = motors 22-28)
            # Fold the real end-effector payload into the last link BEFORE createData():
            # pinocchio.Data caches inertia-derived quantities, so it must be built after.
            extra = float(self.config.tip_extra_mass_kg)
            if extra > 0.0:
                com = np.asarray(self.config.tip_extra_com_xyz, dtype=np.float64)
                if com.shape != (3,) or not np.all(np.isfinite(com)) or not np.isfinite(extra):
                    raise ValueError(
                        f"tip_extra_mass_kg / tip_extra_com_xyz malformed: {extra!r} {com!r}"
                    )
                last = self._grav_model.njoints - 1   # wrist-yaw: the frame the payload hangs off
                before = float(self._grav_model.inertias[last].mass)
                # Point mass: rotational inertia is irrelevant here — g(q) is RNEA with
                # v=a=0, where only mass and CoM lever contribute.
                self._grav_model.inertias[last] = self._grav_model.inertias[last] + pinocchio.Inertia(
                    extra, com, np.zeros((3, 3))
                )
                logger.warning(
                    "G1ArmSdkConnection: gravity model tip payload +%.3f kg at %s "
                    "(last link %.3f -> %.3f kg). URDF models a 0.170 kg rubber hand only.",
                    extra, list(com), before, float(self._grav_model.inertias[last].mass),
                )
            self._grav_data = self._grav_model.createData()
        self._publisher: ChannelPublisher | None = None
        self._subscriber: ChannelSubscriber | None = None
        self._low_cmd: LowCmd_ | None = None
        self._low_state: Any = None
        self._crc: CRC | None = None
        self._mode_machine: int | None = None
        self._mm_warned = False
        self._runaway = False
        self._target_epoch = 0    # bumped on every /g1/arm_target
        self._seen_epoch = 0
        self._prev_gap = 0.0
        from dimos.robot.unitree.g1.act.phase_voice import LogAnnouncer, PhaseVoice

        self._voice = PhaseVoice(LogAnnouncer())
        from dimos.robot.unitree.g1.act.phase_voice import LogAnnouncer, PhaseVoice

        self._voice = PhaseVoice(LogAnnouncer())
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
        if self.config.voice:
            from dimos.robot.unitree.g1.act.phase_voice import build_phase_voice

            self._voice = build_phase_voice(
                True, self.config.voice_nic, init_dds=False, volume=self.config.voice_volume
            )

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
            kp_arm=self.config.kp_arm,
            # Explicit in the log: gravity_ff silently doing nothing (model failed to load)
            # looks exactly like gravity_ff having no effect on the droop.
            gravity_ff=(
                f"ON x{self.config.gravity_tau_scale:g}"
                if (self.config.gravity_ff and self._grav_model is not None)
                else ("REQUESTED but NO MODEL" if self.config.gravity_ff else "off")
            ),
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
            elif msg.mode_machine != self._mode_machine and not self._mm_warned:
                # We latch mode_machine at startup and echo it in every LowCmd. If the robot
                # switches mode, our echoed value goes stale and the robot can stop honouring
                # rt/arm_sdk -- the arm reverts to the onboard controller and swings on its
                # own. Surfacing the change is how we tell that apart from a rogue publisher.
                self._mm_warned = True
                logger.error(
                    f"G1ArmSdkConnection: robot mode_machine CHANGED "
                    f"{self._mode_machine} -> {msg.mode_machine}. arm_sdk commands may be "
                    "ignored while this holds; the arm can move on its own. STOP and re-do "
                    "the startup sequence (L2+B, L2+Up, R1+Y)."
                )

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
            self._target_epoch += 1

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
                        # Runaway guard: a large target-vs-measured gap means the robot moved
                        # the arm itself (arm_sdk not being honoured). Slewing back into that
                        # gap is a high-speed swing through whatever is now in the way, so
                        # re-anchor onto where the arm actually is and hold there instead.
                        gap = float(np.max(np.abs(target - measured)))
                        if self._target_epoch != self._seen_epoch:
                            # A newly commanded pose. The gap it opens is intentional, so
                            # clear any latch and let the arm slew there.
                            self._seen_epoch = self._target_epoch
                            self._prev_gap = gap
                            if self._runaway:
                                self._runaway = False
                                logger.info(
                                    "G1ArmSdkConnection: new arm_target received; runaway latch "
                                    "cleared, tracking resumed."
                                )
                        growing = gap > self._prev_gap + self.config.runaway_growth_rad
                        self._prev_gap = gap
                        if (
                            self.config.runaway_release
                            and not self._runaway
                            and growing
                            and gap > self.config.runaway_track_err_rad
                        ):
                            if True:
                                self._runaway = True
                                logger.error(
                                    f"G1ArmSdkConnection: RUNAWAY -- measured pose is {gap:.3f} rad "
                                    f"({np.degrees(gap):.0f} deg) from the commanded target; the arm "
                                    "was moved by something other than us. Re-anchoring to the "
                                    "measured pose and HOLDING (not slewing back). Re-click to resume."
                                )
                                self._voice.say("腕の制御を失いました。その場で保持します")
                        if self._runaway:
                            # Latched until a new arm_target arrives -- re-anchoring makes the
                            # gap zero, so clearing on gap alone would unlatch instantly and the
                            # hold would never actually hold.
                            self._target_q = measured.copy()
                            self._prev_gap = 0.0
                            target = measured
                        clipped = self._clip_to_measured(target, measured)
                    # Collection: RIGHT arm (motors 22-28 == _ARM_IDX[7:14]) compliant.
                    comp = self._compliant and not self._disconnect
                    # Gravity feedforward during stiff tracking. Dropped while disconnecting:
                    # the upper body is being handed back to the onboard controller (weight
                    # ramping to 0), so injecting torque there would fight the handover.
                    grav_ff = self.config.gravity_ff and not self._disconnect
                    g = None
                    if comp:
                        a_ramp = min(1.0, (now - self._comp_t0) / max(1e-3, self.config.compliant_kp_ramp_s))
                    if (comp or grav_ff) and self._grav_model is not None:
                        import pinocchio
                        g = pinocchio.computeGeneralizedGravity(
                            self._grav_model, self._grav_data, measured[7:14].astype(np.float64)
                        )
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
                            # g(q) covers the RIGHT arm only (motors 22-28): it is the reduced
                            # 7-DOF model, and g[k-7] is only indexable for k >= 7. Left arm and
                            # waist keep tau=0 (they hold static poses, no droop problem).
                            self._low_cmd.motor_cmd[i].tau = (
                                float(g[k - 7]) * self.config.gravity_tau_scale
                                if (grav_ff and i >= 22 and g is not None)
                                else 0.0
                            )
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
