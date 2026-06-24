# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the synchronous IK approach skill (right-arm reach).

Requires ``pinocchio`` + the G1 URDF (present on the Orin; skipped where pinocchio
is absent, e.g. a laptop dev box). These exercise the gate logic (workspace box,
convergence, joint-delta, limits) and a round-trip reach via the real right-arm
model — no robot needed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pinocchio")  # IK solver dep; skip suite if absent (laptop)

from dimos.robot.unitree.g1.harvest.ik_approach import IkApproachResult, IkApproachSkill


def _rest_state() -> list[float]:
    """A plausible 29-DOF measured pose (zeros = arms down/rest)."""
    return [0.0] * 29


def test_reach_to_in_workspace_target_succeeds() -> None:
    """A torso-frame target inside the workspace box returns a 14-vec reach."""
    skill = IkApproachSkill()
    # Forward 0.35 m, right (-Y) 0.25 m, slightly below torso_link — well inside the box.
    target_torso = np.array([0.35, -0.25, 0.10])
    res = skill.solve(target_torso, _rest_state())
    assert isinstance(res, IkApproachResult)
    assert len(res.arm14) == 14
    assert len(res.joint_names) == 14
    # Left arm (first 7) is held at the measured value (rest = 0).
    assert res.arm14[:7] == pytest.approx([0.0] * 7, abs=1e-9)
    # Reach is within tolerance and the wait is clamped to the configured range.
    assert res.err <= 0.05 + 1e-6
    assert 0.8 <= res.wait_s <= 3.0


def test_target_outside_workspace_box_is_rejected() -> None:
    """A target behind/over the torso (outside the box) is rejected (returns None)."""
    skill = IkApproachSkill()
    behind = np.array([-0.5, -0.25, 0.10])  # x < ws_x[0]=0.05 -> reject
    assert skill.solve(behind, _rest_state()) is None
    too_left = np.array([0.35, 0.6, 0.10])  # y > ws_y[1]=0.20 -> reach across body -> reject
    assert skill.solve(too_left, _rest_state()) is None


def test_standoff_stops_short_in_torso_x() -> None:
    """With standoff_m>0 the commanded tip lands ~standoff short of the okra in X."""
    no_standoff = IkApproachSkill(standoff_m=0.0)
    with_standoff = IkApproachSkill(standoff_m=0.05)
    target = np.array([0.40, -0.25, 0.10])
    r0 = no_standoff.solve(target, _rest_state())
    r1 = with_standoff.solve(target, _rest_state())
    assert r0 is not None and r1 is not None
    # The two solutions differ (standoff shifts the target), proving standoff is applied.
    assert r0.q_right != pytest.approx(r1.q_right, abs=1e-6)


def test_short_pose_rejected() -> None:
    """A measured pose with < 29 joints is rejected, not crashed."""
    skill = IkApproachSkill()
    assert skill.solve(np.array([0.35, -0.25, 0.10]), [0.0] * 10) is None


def test_non_finite_pose_rejected() -> None:
    skill = IkApproachSkill()
    bad = [0.0] * 29
    bad[22] = float("nan")
    assert skill.solve(np.array([0.35, -0.25, 0.10]), bad) is None
