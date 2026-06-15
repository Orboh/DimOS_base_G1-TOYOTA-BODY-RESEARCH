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

"""G1 right Dex1 gripper control via the Unitree ``rt/dex1/right`` DDS topics.

Mirrors unitree_lerobot's ``Dex1_1_Gripper_Controller`` (the verified Stage A
path) for the RIGHT gripper only:

* command : ``rt/dex1/right/cmd``  (``MotorCmds_``, ``cmds[0].q``, kp=5.0/kd=0.05)
* state   : ``rt/dex1/right/state`` (``MotorStates_``, ``states[0].q``)

The real G1 has only the right Dex1 installed; ``rt/dex1/left/state`` is never
published (a blocking read on it would hang) and the okra dataset's left-gripper
dim is the constant 0 — so the left gripper is intentionally ignored here, which
matches both the hardware and the policy's training distribution.

SAFETY: the commanded q is initialised to the CURRENT measured gripper position
(hold) and is not published until the first valid state arrives, so the gripper
does not jerk on startup. The kp/kd are the proven soft gains; an optional clamp
guards against out-of-range targets without altering the verified behaviour (it
is disabled by default to stay faithful to Stage A).
"""

from __future__ import annotations

import math
import threading
from threading import Thread
import time
from typing import TYPE_CHECKING, Any

from pydantic import Field
from reactivex.disposable import Disposable

if TYPE_CHECKING:
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.unitree.g1.act.dds_init import ensure_channel_factory
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_RIGHT_GRIPPER_JOINT = "g1/right_gripper"
_STATE_WAIT_S = 10.0


class G1GripperConnectionConfig(ModuleConfig):
    network_interface: str = Field(default="")
    publish_rate_hz: float = 200.0  # matches Dex1_1_Gripper_Controller fps
    state_rate_hz: float = 50.0     # how often to republish measured state
    # Proven soft gains for the Dex1 gripper (unitree_lerobot).
    kp: float = 5.0
    kd: float = 0.05
    # Optional safety clamp on the commanded position [rad]. Disabled by default
    # to stay faithful to the verified Stage A path (which applies no clamp);
    # enable + tune if a target is ever seen driving the gripper past its stops.
    clamp_enabled: bool = False
    q_min: float = 0.0
    q_max: float = 9.0
    frame_id: str = "g1_right_gripper"
    # DRY-RUN: when False, the loop still reads rt/dex1/right/state and publishes
    # right_gripper_state, but writes NOTHING to rt/dex1/right/cmd — the gripper
    # does not move. Used by the dry-run blueprint.
    publish_cmd: bool = True


class G1GripperConnection(Module):
    """Right Dex1 gripper DDS control (command out, measured state in)."""

    config: G1GripperConnectionConfig

    gripper_target: In[JointState]        # position[0] = right gripper target q [rad]
    right_gripper_state: Out[JointState]  # position[0] = measured right gripper q [rad]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._publisher: ChannelPublisher | None = None
        self._subscriber: ChannelSubscriber | None = None
        self._cmd_msg: Any = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Thread | None = None
        self._measured_q: float | None = None
        self._target_q: float | None = None

    @rpc
    def start(self) -> None:
        super().start()
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

        ensure_channel_factory(self.config.network_interface)

        self._publisher = ChannelPublisher("rt/dex1/right/cmd", MotorCmds_)
        self._publisher.Init()
        self._subscriber = ChannelSubscriber("rt/dex1/right/state", MotorStates_)
        self._subscriber.Init(self._on_state, 10)

        # Single-motor gripper command, soft gains (kp=5/kd=0.05).
        self._cmd_msg = MotorCmds_()
        self._cmd_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]
        self._cmd_msg.cmds[0].dq = 0.0
        self._cmd_msg.cmds[0].tau = 0.0
        self._cmd_msg.cmds[0].kp = self.config.kp
        self._cmd_msg.cmds[0].kd = self.config.kd

        # Wait for the first measured state so we can hold the current position.
        logger.info("Waiting for first rt/dex1/right/state...")
        t0 = time.time()
        while time.time() - t0 < _STATE_WAIT_S:
            with self._lock:
                if self._measured_q is not None:
                    break
            time.sleep(0.05)
        with self._lock:
            if self._measured_q is None:
                raise RuntimeError(
                    "No rt/dex1/right/state received; cannot start gripper safely "
                    "(is the right Dex1 connected and teleimager/robot publishing?)"
                )
            self._target_q = self._measured_q  # hold current position until ACT sends a target
        logger.info(f"Dex1 right gripper ready; holding current q={self._target_q:.3f}")

        self.register_disposable(Disposable(self.gripper_target.subscribe(self._on_target)))
        self._stop_event.clear()
        self._thread = Thread(target=self._control_loop, name="g1-dex1-right", daemon=True)
        self._thread.start()
        logger.info(
            "G1GripperConnection started",
            rate_hz=self.config.publish_rate_hz,
            kp=self.config.kp,
            kd=self.config.kd,
        )

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None
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
        self._publisher = self._subscriber = self._cmd_msg = None
        logger.info("G1GripperConnection disconnected")
        super().stop()

    def _on_state(self, msg: Any) -> None:
        try:
            q = float(msg.states[0].q)
        except (AttributeError, IndexError, TypeError):
            return
        with self._lock:
            self._measured_q = q

    def _on_target(self, msg: JointState) -> None:
        pos = list(msg.position)
        if not pos:
            return
        q = float(pos[0])
        if not math.isfinite(q):
            logger.warning(f"gripper_target q={q} is not finite; ignoring")
            return
        if self.config.clamp_enabled:
            clamped = min(self.config.q_max, max(self.config.q_min, q))
            if clamped != q:
                logger.warning(
                    f"gripper target {q:.3f} out of [{self.config.q_min}, {self.config.q_max}]; "
                    f"clamped to {clamped:.3f}"
                )
            q = clamped
        with self._lock:
            self._target_q = q

    def _control_loop(self) -> None:
        period = 1.0 / float(self.config.publish_rate_hz)
        ms_period = 1.0 / float(self.config.state_rate_hz)
        next_tick = time.perf_counter()
        last_ms = 0.0

        while not self._stop_event.is_set():
            now = time.perf_counter()
            with self._lock:
                if self._cmd_msg is None or self._publisher is None:
                    break
                target = self._target_q
                measured = self._measured_q
                if target is not None and self.config.publish_cmd:  # dry-run: no cmd write
                    self._cmd_msg.cmds[0].q = float(target)
                    self._publisher.Write(self._cmd_msg)
                if measured is not None and (now - last_ms) >= ms_period:
                    last_ms = now
                    js = JointState(
                        name=[_RIGHT_GRIPPER_JOINT],
                        position=[float(measured)],
                        velocity=[0.0],
                        effort=[0.0],
                    )
                    js.frame_id = self.config.frame_id
                    self.right_gripper_state.publish(js)

            next_tick += period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                self._stop_event.wait(sleep_for)
            else:
                next_tick = time.perf_counter()


__all__ = ["G1GripperConnection", "G1GripperConnectionConfig"]
