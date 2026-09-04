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

# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Real :class:`SafetyCheck`s for the harvest SafetyMonitor (§6).

These are the checks that gate real motion (the ActGraspModule reach / base
walking). They feed the same SafetyMonitor that can cancel a grasp mid-reach.

* :class:`FileEStop` — operator e-stop via a sentinel file (``touch`` to pause,
  ``rm`` to resume). Dead simple, always available, no extra UI.
* :class:`HumanEStop` — programmatic e-stop (trip/clear) for an HMI / RPC / key.
* :func:`make_torque_check` — auto-stop on unexpected arm joint torque (a crude
  contact/collision guard from ``motor_states`` effort; threshold needs tuning).

Failing/raising checks are treated as unsafe by the SafetyMonitor (fail-safe).
"""

from __future__ import annotations

from collections.abc import Callable
import os
import threading
from typing import Any

from dimos.robot.unitree.g1.harvest.safety import SafetyCheck
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_ARM_START = 15
_NUM_ARM = 14


class FileEStop:
    """Operator e-stop via a sentinel file: ``touch <path>`` pauses, ``rm`` resumes."""

    def __init__(self, path: str) -> None:
        self._path = path

    def is_clear(self) -> bool:
        return not os.path.exists(self._path)

    def as_check(self, name: str = "human_estop_file") -> SafetyCheck:
        return SafetyCheck(name, self.is_clear)


class HumanEStop:
    """Programmatic e-stop (HMI / RPC / keypress can call trip()/release())."""

    def __init__(self) -> None:
        self._clear = True
        self._lock = threading.Lock()

    def trip(self) -> None:
        with self._lock:
            self._clear = False
        logger.warning("HumanEStop: TRIPPED")

    def release(self) -> None:
        with self._lock:
            self._clear = True
        logger.info("HumanEStop: released")

    def is_clear(self) -> bool:
        return self._clear

    def as_check(self, name: str = "human_estop") -> SafetyCheck:
        return SafetyCheck(name, self.is_clear)


def make_torque_check(
    state_getter: Callable[[], Any],
    *,
    limit: float,
    arm_start: int = _ARM_START,
    arm_n: int = _NUM_ARM,
    name: str = "arm_torque",
) -> SafetyCheck:
    """SafetyCheck that flags unexpected arm joint torque (crude contact guard).

    Safe iff ``max |effort[arm]| < limit``. If no effort data is available it
    returns safe (can't judge). ``limit`` [N·m] must be tuned on the robot.
    """

    def is_safe() -> bool:
        state = state_getter()
        if state is None:
            return True
        effort = list(getattr(state, "effort", []) or [])
        arm = effort[arm_start : arm_start + arm_n]
        if not arm:
            return True
        peak = max(abs(float(e)) for e in arm)
        return peak < limit

    return SafetyCheck(name, is_safe)


__all__ = ["FileEStop", "HumanEStop", "make_torque_check"]
