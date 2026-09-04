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

"""Tests for the real safety checks (file e-stop, human e-stop, torque guard)."""

from __future__ import annotations

import os
import tempfile

from dimos.robot.unitree.g1.harvest.safety import SafetyMonitor
from dimos.robot.unitree.g1.harvest.safety_checks import (
    FileEStop,
    HumanEStop,
    make_torque_check,
)


def test_file_estop_pauses_and_resumes() -> None:
    path = os.path.join(tempfile.mkdtemp(), "okra_estop")
    estop = FileEStop(path)
    assert estop.is_clear() is True  # no file -> clear

    open(path, "w").close()  # operator "touch"
    assert estop.is_clear() is False  # file present -> stop
    os.remove(path)
    assert estop.is_clear() is True  # removed -> resume


def test_human_estop_trip_release() -> None:
    e = HumanEStop()
    assert e.is_clear() is True
    e.trip()
    assert e.is_clear() is False
    e.release()
    assert e.is_clear() is True


class _State:
    def __init__(self, effort):
        self.effort = effort
        self.position = [0.0] * 29


def test_torque_check_flags_high_arm_torque() -> None:
    safe = _State(effort=[0.0] * 29)
    spike = _State(effort=[0.0] * 29)
    spike.effort[20] = 50.0  # an arm joint (15..28) over the limit

    check_safe = make_torque_check(lambda: safe, limit=20.0)
    check_spike = make_torque_check(lambda: spike, limit=20.0)
    assert check_safe.is_safe() is True
    assert check_spike.is_safe() is False


def test_torque_check_safe_without_effort_data() -> None:
    check = make_torque_check(lambda: _State(effort=[]), limit=20.0)
    assert check.is_safe() is True  # can't judge -> safe
    assert make_torque_check(lambda: None, limit=20.0).is_safe() is True


def test_file_estop_drives_the_monitor() -> None:
    path = os.path.join(tempfile.mkdtemp(), "okra_estop")
    stops: list = []
    mon = SafetyMonitor([FileEStop(path).as_check()], on_pause=lambda r: stops.append(r))

    assert mon.step() == []  # clear
    open(path, "w").close()
    assert mon.step() == ["human_estop_file"]  # tripped
    assert mon.gate.is_paused() and stops  # on_pause fired (would stop the arm)
    os.remove(path)
    mon.step()
    assert not mon.gate.is_paused()  # resumed
