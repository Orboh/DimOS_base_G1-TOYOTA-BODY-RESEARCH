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
from dimos.robot.unitree.g1.act.dds_init import ensure_channel_factory
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


class G1ArmSdkConnection(Module):
    """Upper-body DDS control via rt/arm_sdk (legs stay on the onboard controller)."""

    config: G1ArmSdkConnectionConfig

    arm_target: In[JointState]      # 14 arm joint targets (left 7, right 7) [rad]
    motor_states: Out[JointState]   # full 29-DOF state, for the ACT observation

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
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

    @rpc
    def start(self) -> None:
        super().start()
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

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
                for w in np.linspace(1.0, 0.0, 101):
                    with self._lock:
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

    def _on_arm_target(self, msg: JointState) -> None:
        pos = list(msg.position)
        if len(pos) < len(_ARM_IDX):
            logger.warning(f"arm_target has {len(pos)} joints; expected {len(_ARM_IDX)}; ignoring")
            return
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
                    clipped = self._clip_to_measured(target, measured)
                    for k, i in enumerate(_ARM_IDX):
                        self._low_cmd.motor_cmd[i].q = float(clipped[k])
                        self._low_cmd.motor_cmd[i].dq = 0.0
                        self._low_cmd.motor_cmd[i].tau = 0.0
                    self._low_cmd.motor_cmd[_WEIGHT_IDX].q = weight
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
