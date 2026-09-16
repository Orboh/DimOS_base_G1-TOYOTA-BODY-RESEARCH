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

"""Offline tests for return_to_rest.make_return_to_rest_fn (no robot, no sleep)."""

from __future__ import annotations

from dimos.robot.unitree.g1.harvest.return_to_rest import make_return_to_rest_fn


def _measured29(left7: list[float], right7: list[float]) -> list[float]:
    """29-DOF stand-in: only indices 15:22 (left arm) / 22:29 (right arm) matter."""
    q = [0.0] * 29
    q[15:22] = left7
    q[22:29] = right7
    return q


def test_interpolates_to_rest_pose_and_returns_true() -> None:
    """Walks smoothly from the measured pose to the captured rest pose."""
    sent: list[list[float]] = []
    slept: list[float] = []
    measured = _measured29([0.0] * 7, [1.0] * 7)
    rest14 = [0.0] * 7 + [0.0] * 7  # rest = arms at zero

    go_home = make_return_to_rest_fn(
        send_arm=lambda q: sent.append(q),
        get_measured=lambda: measured,
        rest_q_getter=lambda: rest14,
        speed_rad_s=0.5,
        rate_hz=50.0,
        sleep_fn=lambda s: slept.append(s),
    )
    assert go_home() is True
    assert len(sent) > 1  # multi-step interpolation, not a single jump
    assert sent[-1] == rest14  # final published target == the rest pose exactly
    assert all(s > 0 for s in slept)


def test_no_rest_pose_captured_skips_and_returns_false() -> None:
    """rest_q_getter() -> None (not captured yet, e.g. dummy mode) -> no-op."""
    sent: list[list[float]] = []
    go_home = make_return_to_rest_fn(
        send_arm=lambda q: sent.append(q),
        get_measured=lambda: _measured29([0.0] * 7, [0.0] * 7),
        rest_q_getter=lambda: None,
        sleep_fn=lambda s: None,
    )
    assert go_home() is False
    assert sent == []
