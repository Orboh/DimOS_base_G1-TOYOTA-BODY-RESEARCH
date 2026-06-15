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
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# Arm slice within the canonical 29-DOF G1 joint vector (left 15-21, right 22-28).
_ARM_START = 15
_NUM_ARM = 14
_NUM_GRIPPER = 2
_STATE_DIM = _NUM_ARM + _NUM_GRIPPER  # 16
_LEFT_GRIP_IDX = _NUM_ARM       # state/action[14] = left gripper (constant 0, unused)
_RIGHT_GRIP_IDX = _NUM_ARM + 1  # state/action[15] = right gripper (the real Dex1)

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
    startup_delay_s: float = 0.0


class ActBridge(Module):
    """Bridges dimos observation streams to the external ACT service (dry-run)."""

    config: ActBridgeConfig

    color_image: In[Image]
    motor_states: In[JointState]
    right_gripper_state: In[JointState]  # measured right Dex1 q (position[0])
    arm_target: Out[JointState]          # 14 arm targets -> G1ArmSdkConnection
    gripper_target: Out[JointState]      # right Dex1 target q (position[0]) -> gripper module

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._latest_image: Image | None = None
        self._latest_state: JointState | None = None
        self._latest_gripper: float = 0.0  # measured right gripper q
        self._stop_event = threading.Event()
        self._thread: Thread | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.color_image.subscribe(self._on_image)))
        self.register_disposable(Disposable(self.motor_states.subscribe(self._on_state)))
        self.register_disposable(
            Disposable(self.right_gripper_state.subscribe(self._on_gripper_state))
        )
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

    def _on_state(self, state: JointState) -> None:
        with self._lock:
            self._latest_state = state

    def _on_gripper_state(self, state: JointState) -> None:
        pos = list(state.position)
        if not pos:
            return
        with self._lock:
            self._latest_gripper = float(pos[0])

    def _build_state(self, state: JointState, right_grip: float) -> list[float] | None:
        """Assemble the 16-dim policy state from a 29-DOF G1 JointState.

        Arms are sliced by the canonical index range (left 15-21, right 22-28),
        verified against joint names. state[14] (left gripper) is the training
        constant 0.0; state[15] is the measured right Dex1 q.
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
        first = True
        count = 0
        next_tick = time.perf_counter()

        while not self._stop_event.is_set():
            with self._lock:
                image = self._latest_image
                state = self._latest_state
                right_grip = self._latest_gripper

            if image is not None and state is not None:
                state16 = self._build_state(state, right_grip)
                if state16 is not None:
                    bgr = image.to_opencv()
                    ok, jpeg = cv2.imencode(".jpg", bgr)
                    if ok:
                        req = {
                            "state": state16,
                            "image_jpeg": jpeg.tobytes(),
                            "reset": first,
                        }
                        try:
                            sock.send(msgpack.packb(req, use_bin_type=True))
                            resp = msgpack.unpackb(sock.recv(), raw=False)
                            action = np.asarray(resp["action"], dtype=float)
                            first = False
                            count += 1
                            self._handle_action(action, count)
                        except zmq.error.Again:
                            logger.warning("ACT service timeout; is act_service.py --serve running?")
                            sock.close()
                            sock = ctx.socket(zmq.REQ)
                            sock.setsockopt(zmq.RCVTIMEO, self.config.recv_timeout_ms)
                            sock.setsockopt(zmq.LINGER, 0)
                            sock.connect(self.config.act_endpoint)
                            first = True

            next_tick += period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                self._stop_event.wait(sleep_for)
            else:
                next_tick = time.perf_counter()

        sock.close()

    def _handle_action(self, action: Any, count: int) -> None:
        """Publish the 14 arm targets + right gripper target (or log only in dry-run)."""
        arms = action[:_NUM_ARM]
        right_grip = float(action[_RIGHT_GRIP_IDX])
        if not self.config.dry_run:
            self.arm_target.publish(
                JointState(
                    name=list(_ARM_JOINT_NAMES),
                    position=[float(x) for x in arms[:_NUM_ARM]],
                    velocity=[0.0] * _NUM_ARM,
                    effort=[0.0] * _NUM_ARM,
                )
            )
            # action[14] (left gripper) is dropped — only the right Dex1 is installed.
            self.gripper_target.publish(
                JointState(
                    name=[_RIGHT_GRIPPER_JOINT],
                    position=[right_grip],
                    velocity=[0.0],
                    effort=[0.0],
                )
            )
        if count % self.config.log_every_n == 1:
            grip = action[_NUM_ARM:_STATE_DIM]
            pairs = ", ".join(
                f"{n.split('/')[-1]}={v:.3f}" for n, v in zip(_ARM_JOINT_NAMES, arms, strict=False)
            )
            tag = "dry-run" if self.config.dry_run else "LIVE→arm_sdk+dex1"
            logger.info(f"[{tag}] ACT action #{count}: {pairs} | grip(L,R)={grip}")


__all__ = ["ActBridge", "ActBridgeConfig"]
