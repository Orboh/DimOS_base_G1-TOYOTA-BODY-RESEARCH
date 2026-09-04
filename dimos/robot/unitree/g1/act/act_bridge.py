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

"""ActBridge: dimos <-> external ACT (lerobot) inference service over ZMQ.

The okra ACT policy runs in its own process/venv (lerobot+torch). This Module
keeps dimos dependency-clean: it subscribes to the head camera and joint states,
assembles the 16-dim observation the policy expects, ships it to the ACT service
over a neutral ZMQ/msgpack wire, and receives the 16-dim action chunk back.

Observation / action layout (confirmed from unitree_lerobot, identity-mapped to
dimos make_humanoid_joints("g1")):
    [0:7]   left arm   -> dimos motor index 15-21
    [7:14]  right arm  -> dimos motor index 22-28
    [14]    left gripper  (Dex1)   [15] right gripper (Dex1)

The G1 has only the RIGHT Dex1 installed and the okra dataset's left-gripper dim
is the constant 0, so state[14] is pinned to 0.0 and state[15] is the measured
right gripper q (from ``rt/dex1/right/state`` via the gripper module). On output
the 14 arm targets go to G1ArmSdkConnection and action[15] (right gripper) goes
to the gripper module; action[14] (left) is dropped.

DRY-RUN (default): the predicted action is only logged — nothing is published
downstream. Driving the robot is enabled with ``dry_run=False``.
"""

from __future__ import annotations

import threading
from threading import Thread
import time
from typing import Any

from reactivex.disposable import Disposable

from dimos.control.components import make_humanoid_joints
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.Bool import Bool
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# Arm slice within the canonical 29-DOF G1 joint vector (left 15-21, right 22-28).
_ARM_START = 15
_NUM_ARM = 14
_NUM_GRIPPER = 2
_STATE_DIM = _NUM_ARM + _NUM_GRIPPER  # 16
_LEFT_GRIP_IDX = _NUM_ARM  # state/action[14] = left gripper (constant 0, unused)
_RIGHT_GRIP_IDX = _NUM_ARM + 1  # state/action[15] = right gripper (the real Dex1)
_LEFT_SLICE = slice(_ARM_START, _ARM_START + 7)  # measured left-arm q (held; basket hand)
_RIGHT_SLICE = slice(_ARM_START + 7, _ARM_START + _NUM_ARM)  # measured right-arm q (22:29)
_NUM_ARM_HALF = _NUM_ARM // 2  # 7: action[0:7]=left arm, action[7:14]=right arm
# 8-DoF right-only layout (tree-right model): state/action = [right arm 7, right grip 1].
_RIGHT_ONLY_DIM = 8

_G1_JOINTS = make_humanoid_joints("g1")
_ARM_JOINT_NAMES = _G1_JOINTS[_ARM_START : _ARM_START + _NUM_ARM]
_RIGHT_GRIPPER_JOINT = "g1/right_gripper"


class ActBridgeConfig(ModuleConfig):
    act_endpoint: str = "tcp://127.0.0.1:5701"
    rate_hz: float = 30.0
    recv_timeout_ms: int = 2000
    log_every_n: int = 30  # throttle the per-action log (~1/s at 30 Hz)
    dry_run: bool = True  # log only; no motor command is published
    # Wait this long after start before the first inference, giving the arms time
    # to slew to G1ArmSdkConnection.initial_arm_pose first (mirrors eval_g1.py's
    # "move to start pose, sleep, then loop"). 0 = start inferring immediately.
    # In the IK->ACT pipeline this is 0: the arm is already at the IK pre-grasp.
    startup_delay_s: float = 0.0
    # IK->ACT trigger mode. False (default) = legacy free-run: the policy loop
    # runs continuously from start (unitree-g1-act-arm behavior, unchanged).
    # True = gated: the loop idles until a reach_done arrives, then runs for
    # grasp_duration_s and stops (one grasp per trigger). The okra-harvest
    # blueprint sets this True so IK owns the pre-grasp until it fires reach_done.
    trigger_mode: bool = False
    grasp_duration_s: float = 8.0
    # 8-DoF right-only policy (tree-right model). False (default) = legacy 16-dim
    # both-arms wire. True = state/action are [right arm 7, right grip 1]; the
    # head image goes as cam_high and right_wrist_image (if wired) as cam_right_wrist.
    right_only: bool = False
    # 7-DoF arm-only policy (kinesthetic wrist-only model, 2026-06-24). True =>
    # state/action are [right arm 7] ONLY — NO gripper dim. The single wrist image
    # is fed via color_image (the act_service resolves the model's lone cam_right_wrist
    # key from it); right_wrist_image / gripper are NOT used. ActBridge never publishes
    # gripper_target in this mode (the Dex1 is held open out-of-band). Takes precedence
    # over right_only. See model sotata/act-okura-kinesthetic-wrist-7d.
    arm_only: bool = False


class ActBridge(Module):
    """Bridges dimos observation streams to the external ACT service (dry-run)."""

    config: ActBridgeConfig

    color_image: In[Image]  # head / cam_high
    right_wrist_image: In[Image]  # cam_right_wrist (2-cam models; optional otherwise)
    motor_states: In[JointState]
    right_gripper_state: In[JointState]  # measured right Dex1 q (position[0])
    arm_target: Out[JointState]  # 14 arm targets -> G1ArmSdkConnection
    gripper_target: Out[JointState]  # right Dex1 target q (position[0]) -> gripper module
    reach_done: In[Bool]  # IK settled at pre-grasp -> start one grasp window

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._latest_image: Image | None = None
        self._latest_wrist_image: Image | None = None
        self._latest_state: JointState | None = None
        self._latest_gripper: float = 0.0  # measured right gripper q
        self._stop_event = threading.Event()
        self._thread: Thread | None = None
        # IK->ACT trigger gating (only meaningful when config.trigger_mode).
        self._active = threading.Event()  # set => the policy loop is running
        self._deadline = 0.0  # wall-clock end of the current grasp window
        self._reset_pending = True  # send reset=True on the first frame of a window

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.color_image.subscribe(self._on_image)))
        if self.config.right_only:
            self.register_disposable(
                Disposable(self.right_wrist_image.subscribe(self._on_wrist_image))
            )
        self.register_disposable(Disposable(self.motor_states.subscribe(self._on_state)))
        # arm_only (7-DoF) has no gripper dim, so the measured Dex1 q is not needed.
        if not self.config.arm_only:
            self.register_disposable(
                Disposable(self.right_gripper_state.subscribe(self._on_gripper_state))
            )
        if self.config.trigger_mode:
            self.register_disposable(Disposable(self.reach_done.subscribe(self._on_reach_done)))
        self._stop_event.clear()
        self._thread = Thread(target=self._act_loop, daemon=True, name="act-bridge")
        self._thread.start()
        logger.info(
            "ActBridge started",
            endpoint=self.config.act_endpoint,
            rate_hz=self.config.rate_hz,
            dry_run=self.config.dry_run,
        )

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        super().stop()

    def _on_image(self, image: Image) -> None:
        with self._lock:
            self._latest_image = image

    def _on_wrist_image(self, image: Image) -> None:
        with self._lock:
            self._latest_wrist_image = image

    def _on_state(self, state: JointState) -> None:
        with self._lock:
            self._latest_state = state

    def _on_gripper_state(self, state: JointState) -> None:
        pos = list(state.position)
        if not pos:
            return
        with self._lock:
            self._latest_gripper = float(pos[0])

    def _on_reach_done(self, _msg: Bool) -> None:
        """IK signalled the arm settled at pre-grasp: open one grasp window.

        Debounced: a reach_done arriving while a grasp is already running is
        ignored (it would otherwise restart the window mid-grasp). The window is
        closed by the policy loop once grasp_duration_s elapses.
        """
        if self._active.is_set():
            logger.info("ActBridge: reach_done during an active grasp; ignoring (debounce).")
            return
        self._deadline = time.time() + self.config.grasp_duration_s
        self._reset_pending = True
        self._active.set()
        logger.info(
            f"ActBridge: reach_done -> starting grasp window ({self.config.grasp_duration_s}s)."
        )

    def _build_state(self, state: JointState, right_grip: float) -> list[float] | None:
        """Assemble the policy state from a 29-DOF G1 JointState.

        arm_only=True    -> 7-dim  [right arm 7 (motor 22-28)] (NO gripper).
        right_only=True  -> 8-dim  [right arm 7 (motor 22-28), right grip 1].
        right_only=False -> 16-dim [arms 14 (15-28), left grip=0, right grip].
        """
        pos = list(state.position)
        if len(pos) < _ARM_START + _NUM_ARM:
            logger.warning(f"motor_states has {len(pos)} joints; expected >= 29; skipping")
            return None
        # Safety: confirm the slice really is the arms (names end with arm joints).
        if state.name and len(state.name) >= _ARM_START + _NUM_ARM:
            got = state.name[_ARM_START]
            if not str(got).endswith(_ARM_JOINT_NAMES[0].split("/")[-1]):
                logger.warning(
                    f"arm slice mismatch: index {_ARM_START} is {got!r}, "
                    f"expected ...{_ARM_JOINT_NAMES[0]}; check joint ordering"
                )
        if self.config.arm_only:
            right_arm = pos[_RIGHT_SLICE]  # motor 22-28
            return [float(x) for x in right_arm]  # 7-dim, no gripper
        if self.config.right_only:
            right_arm = pos[_RIGHT_SLICE]  # motor 22-28
            return [float(x) for x in right_arm] + [float(right_grip)]
        arms = pos[_ARM_START : _ARM_START + _NUM_ARM]
        grippers = [0.0, float(right_grip)]  # [left=const 0, right=measured Dex1 q]
        return [float(x) for x in arms] + grippers

    def _act_loop(self) -> None:
        import cv2
        import msgpack
        import numpy as np
        import zmq

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self.config.recv_timeout_ms)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self.config.act_endpoint)

        if self.config.startup_delay_s > 0:
            logger.info(
                f"ActBridge: holding inference {self.config.startup_delay_s}s "
                "while the arms slew to the start pose"
            )
            self._stop_event.wait(self.config.startup_delay_s)

        period = 1.0 / float(self.config.rate_hz)
        count = 0
        next_tick = time.perf_counter()
        # Legacy free-run (unitree-g1-act-arm): always active. Trigger mode
        # (okra-harvest): idle until reach_done opens a grasp window.
        if not self.config.trigger_mode:
            self._active.set()

        while not self._stop_event.is_set():
            # Trigger mode: close the grasp window after the fixed duration.
            if self.config.trigger_mode and self._active.is_set() and time.time() > self._deadline:
                self._active.clear()
                logger.info(f"ActBridge: grasp window ended ({self.config.grasp_duration_s}s).")

            if self._active.is_set():
                with self._lock:
                    image = self._latest_image
                    wrist_image = self._latest_wrist_image
                    state = self._latest_state
                    right_grip = self._latest_gripper

                # 2-cam (right_only) models need the wrist frame too; wait for it.
                have_inputs = (
                    image is not None
                    and state is not None
                    and (wrist_image is not None or not self.config.right_only)
                )
                if have_inputs:
                    state_vec = self._build_state(state, right_grip)
                    if state_vec is not None:
                        ok, jpeg = cv2.imencode(".jpg", image.to_opencv())
                        if ok:
                            req = {
                                "state": state_vec,
                                "image_jpeg": jpeg.tobytes(),
                                "reset": self._reset_pending,
                            }
                            if wrist_image is not None:
                                okw, wjpeg = cv2.imencode(".jpg", wrist_image.to_opencv())
                                if okw:
                                    req["image_right_wrist_jpeg"] = wjpeg.tobytes()
                            try:
                                sock.send(msgpack.packb(req, use_bin_type=True))
                                resp = msgpack.unpackb(sock.recv(), raw=False)
                                action = np.asarray(resp["action"], dtype=float)
                                self._reset_pending = False
                                count += 1
                                self._handle_action(action, count)
                            except zmq.error.Again:
                                logger.warning(
                                    "ACT service timeout; is act_service.py --serve running?"
                                )
                                sock.close()
                                sock = ctx.socket(zmq.REQ)
                                sock.setsockopt(zmq.RCVTIMEO, self.config.recv_timeout_ms)
                                sock.setsockopt(zmq.LINGER, 0)
                                sock.connect(self.config.act_endpoint)
                                self._reset_pending = True

            next_tick += period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                self._stop_event.wait(sleep_for)
            else:
                next_tick = time.perf_counter()

        sock.close()

    def _handle_action(self, action: Any, count: int) -> None:
        """Publish the 14 arm targets + right gripper target (or log only in dry-run).

        Left arm is HELD at its measured pose (the left hand holds the harvest
        basket; ACT drives only the right arm + right gripper -- 8-DoF spec). The
        policy still outputs 14 arm dims (action[0:7]=left, [7:14]=right); we
        ignore its left 7 and substitute the measured left 7 so the basket-holding
        arm never moves. Falls back to the policy's left if no state is available.
        """
        right_grip: float | None
        if self.config.arm_only:
            right_arm = [float(x) for x in action[0:_NUM_ARM_HALF]]  # 7-dim: action[0:7]=right arm
            right_grip = None  # no gripper dim (held open out-of-band)
        elif self.config.right_only:
            right_arm = [float(x) for x in action[0:_NUM_ARM_HALF]]  # 8-dim: action[0:7]=right arm
            right_grip = float(action[_RIGHT_ONLY_DIM - 1])  # action[7]=right grip
        else:
            right_arm = [float(x) for x in action[_NUM_ARM_HALF:_NUM_ARM]]  # 16-dim: action[7:14]
            right_grip = float(action[_RIGHT_GRIP_IDX])  # action[15]
        with self._lock:
            st = self._latest_state
        left_hold: list[float] | None = None
        if st is not None:
            pos = list(st.position)
            if len(pos) >= _ARM_START + _NUM_ARM:
                left_hold = [float(x) for x in pos[_LEFT_SLICE]]  # hold the basket arm at measured
        if left_hold is None:
            logger.warning("ActBridge: no measured state for left-arm hold; skipping this action.")
            return
        arm14 = left_hold + right_arm
        if not self.config.dry_run:
            self.arm_target.publish(
                JointState(
                    name=list(_ARM_JOINT_NAMES),
                    position=arm14,
                    velocity=[0.0] * _NUM_ARM,
                    effort=[0.0] * _NUM_ARM,
                )
            )
            # action[14] (left gripper) is dropped — only the right Dex1 is installed.
            # arm_only (7-DoF) has no gripper dim: leave the Dex1 to its held-open pose.
            if right_grip is not None:
                self.gripper_target.publish(
                    JointState(
                        name=[_RIGHT_GRIPPER_JOINT],
                        position=[right_grip],
                        velocity=[0.0],
                        effort=[0.0],
                    )
                )
        if count % self.config.log_every_n == 1:
            pairs = ", ".join(
                f"{n.split('/')[-1]}={v:.3f}" for n, v in zip(_ARM_JOINT_NAMES, arm14, strict=False)
            )
            tag = "dry-run" if self.config.dry_run else "LIVE→arm_sdk+dex1"
            layout = (
                "7d-arm"
                if self.config.arm_only
                else ("8d-right" if self.config.right_only else "16d")
            )
            grip_str = "held-open" if right_grip is None else f"{right_grip:.3f}"
            logger.info(
                f"[{tag}] ACT action #{count} ({layout}): "
                f"{pairs} | right_grip={grip_str} | left=HELD"
            )


__all__ = ["ActBridge", "ActBridgeConfig"]
