#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
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

"""One deliberately tiny, rate-limited G1 upper-body commissioning motion.

This is not a general teleoperation or trajectory component.  It waits while
the arm-SDK hold gate settles, snapshots the *then-current* measured 14-arm
pose, and only offsets one configured joint.  All other arm targets remain at
that snapshot.  The input target is ramped slowly so that the position target
does not step.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_ARM_START = 15
_ARM_COUNT = 14


class G1SingleJointNudgeConfig(ModuleConfig):
    """Fixed-purpose target generator used only by the commissioning blueprint."""

    # Index in the 14-joint arm target order: left arm 0..6, right arm 7..13.
    joint_index: int = 4  # global joint 19: LeftWristRoll
    joint_name: str = "LeftWristRoll"
    delta_rad: float = 0.020
    rate_rad_s: float = 0.010
    settle_before_move_s: float = 6.5


class G1SingleJointNudge(Module):
    """After a settling hold, slowly offset exactly one measured arm joint."""

    config: G1SingleJointNudgeConfig

    motor_states: In[JointState]
    arm_target: Out[JointState]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not 0 <= self.config.joint_index < _ARM_COUNT:
            raise ValueError(f"joint_index must be in [0, {_ARM_COUNT - 1}]")
        if self.config.rate_rad_s <= 0.0:
            raise ValueError("rate_rad_s must be positive")
        self._first_state_t: float | None = None
        self._move_t: float | None = None
        self._baseline: np.ndarray | None = None
        self._complete_logged = False

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.motor_states.subscribe(self._on_motor_states)))
        logger.warning(
            "G1SingleJointNudge armed: it will wait %.1fs, then move arm-target index %d "
            "by %.3f rad at %.3f rad/s; all other arm targets are held at their "
            "measured pre-move pose.",
            self.config.settle_before_move_s,
            self.config.joint_index,
            self.config.delta_rad,
            self.config.rate_rad_s,
        )

    @rpc
    def stop(self) -> None:
        super().stop()

    def _on_motor_states(self, msg: JointState) -> None:
        pos = list(msg.position)
        if len(pos) < _ARM_START + _ARM_COUNT:
            logger.warning("G1SingleJointNudge: short motor-state message; ignoring")
            return
        measured = np.asarray(pos[_ARM_START : _ARM_START + _ARM_COUNT], dtype=float)
        if not np.all(np.isfinite(measured)):
            logger.warning("G1SingleJointNudge: non-finite motor state; ignoring")
            return

        now = time.perf_counter()
        if self._first_state_t is None:
            self._first_state_t = now
            logger.info("G1SingleJointNudge: receiving motor states; settling before motion")
            return
        if self._move_t is None:
            if now - self._first_state_t < self.config.settle_before_move_s:
                return
            # Take the baseline immediately before moving.  This prevents a target
            # produced at startup from correcting any unrelated joint drift.
            self._baseline = measured.copy()
            self._move_t = now
            logger.warning(
                "G1SingleJointNudge: BEGIN %s nudge from %.4f rad; "
                "physical STOP remains the primary abort.",
                self.config.joint_name,
                self._baseline[self.config.joint_index],
            )

        assert self._baseline is not None
        assert self._move_t is not None
        sign = 1.0 if self.config.delta_rad >= 0.0 else -1.0
        travelled = min(
            abs(self.config.delta_rad),
            self.config.rate_rad_s * (now - self._move_t),
        )
        target = self._baseline.copy()
        target[self.config.joint_index] += sign * travelled
        self.arm_target.publish(
            JointState(
                position=target.tolist(), velocity=[0.0] * _ARM_COUNT, effort=[0.0] * _ARM_COUNT
            )
        )
        if travelled >= abs(self.config.delta_rad) and not self._complete_logged:
            self._complete_logged = True
            logger.warning(
                "G1SingleJointNudge: target reached at %.4f rad; holding only this target "
                "until normal shutdown.",
                target[self.config.joint_index],
            )


__all__ = ["G1SingleJointNudge", "G1SingleJointNudgeConfig"]
