# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the okra-harvest LangGraph.

Covers the verify/retry recovery and the 3D §5 movement: approach a fruit that
is too far / too close / off to the left, skip an out-of-height fruit, and sweep
to discover a fruit out of view.
"""

from __future__ import annotations

from dimos.robot.unitree.g1.harvest import announce
from dimos.robot.unitree.g1.harvest.announce import RecordingAnnouncer
from dimos.robot.unitree.g1.harvest.blackboard import HarvestConfig, initial_state
from dimos.robot.unitree.g1.harvest.graph import build_harvest_graph
from dimos.robot.unitree.g1.harvest.skills import FieldOkra, MockHarvestSkills

_RECURSION_LIMIT = 400
_CFG = HarvestConfig()  # reach x[0.10,0.50] y[0.30,0.60] z[0.40,1.10]; centre (0.30, 0.45)


def _run(skills: MockHarvestSkills, config: HarvestConfig | None = None) -> dict:
    cfg = config or _CFG
    app = build_harvest_graph(skills, cfg)
    return app.invoke(initial_state(), {"recursion_limit": _RECURSION_LIMIT})


def _mock(field: list[FieldOkra], **kw) -> MockHarvestSkills:
    return MockHarvestSkills(field, reach=_CFG.reach, fov=_CFG.fov, **kw)


def _no_reposition(skills: MockHarvestSkills) -> bool:
    """True iff every base move was a pure left-sweep (no approach/reposition).

    Harvest progresses right→left, so the discovery sweep steps -x. After the
    field is exhausted the robot sweeps ``max_empty_advances`` times to confirm
    "done" (§8), so move_calls is rarely empty — this checks the stronger
    property that nothing was *chased*.
    """
    sweep = (round(-_CFG.advance_step, 3), 0.0)
    return all(m == sweep for m in skills.move_calls)


def test_in_reach_okra_is_picked_immediately() -> None:
    """An okra already inside the reach box is grasped without repositioning."""
    field = [FieldOkra("a", x=0.30, y=0.45, z=0.80, ripeness=0.9)]
    skills = _mock(field)
    final = _run(skills)

    assert final["picks"] == 1
    assert _no_reposition(skills)  # picked in place; any moves are the §8 sweep


def test_too_far_okra_triggers_forward_then_pick() -> None:
    """An okra beyond the reach box (too far) → move FORWARD → pick."""
    field = [FieldOkra("far", x=0.30, y=0.85, z=0.80, ripeness=0.9)]
    skills = _mock(field)
    final = _run(skills)

    assert final["picks"] == 1
    assert len(skills.move_calls) >= 1
    # First move is forward (positive y), no lateral (already centred in x).
    lateral, forward = skills.move_calls[0]
    assert forward > 0 and abs(lateral) < 1e-9


def test_too_close_okra_triggers_backup_then_pick() -> None:
    """An okra closer than the reach box (ridge risk) → move BACK → pick."""
    field = [FieldOkra("near", x=0.30, y=0.20, z=0.80, ripeness=0.9)]
    skills = _mock(field)
    final = _run(skills)

    assert final["picks"] == 1
    lateral, forward = skills.move_calls[0]
    assert forward < 0  # backed off the ridge


def test_left_side_okra_triggers_left_strafe_then_pick() -> None:
    """An okra to the left → strafe LEFT to bring it into the right-side reach."""
    field = [FieldOkra("left", x=-0.40, y=0.45, z=0.80, ripeness=0.9)]
    skills = _mock(field)
    final = _run(skills)

    assert final["picks"] == 1
    lateral, _forward = skills.move_calls[0]
    assert lateral < 0  # moved left


def test_out_of_height_okra_is_skipped_not_chased() -> None:
    """A ripe okra above the arm's reach is skipped (G1 cannot squat)."""
    field = [FieldOkra("high", x=0.30, y=0.45, z=1.50, ripeness=0.95)]
    skills = _mock(field)
    final = _run(skills)

    assert final["picks"] == 0
    assert any(r["result"] == "skipped_height" for r in final["records"])
    assert _no_reposition(skills)  # never chased it; any moves are the §8 sweep


def test_sweep_discovers_okra_out_of_view() -> None:
    """No fruit in view here → advance_left until one is discovered, then pick."""
    # Out of FOV at the start (x below fov x-min -0.80); sweeping left reaches it.
    field = [FieldOkra("left_far", x=-1.05, y=0.45, z=0.80, ripeness=0.9)]
    skills = _mock(field)
    final = _run(skills)

    assert final["picks"] == 1
    assert final["iterations"] >= 2  # took at least one sweep + re-detect
    # The discovery move was a leftward sweep.
    assert skills.move_calls[0] == (round(-_CFG.advance_step, 3), 0.0)


def test_terminates_when_field_empty() -> None:
    """Empty field → sweep the cap then stop; loop must terminate."""
    skills = _mock([])
    final = _run(skills)

    assert final["picks"] == 0
    # Swept up to the cap, then done — no infinite loop.
    assert len(skills.move_calls) <= _CFG.max_empty_advances


def test_recovery_retries_then_succeeds() -> None:
    """A first failed verify triggers a bounded re-grasp, then succeeds."""
    field = [FieldOkra("flaky", x=0.30, y=0.45, z=0.80, ripeness=0.9)]
    skills = _mock(field, flaky_verifies={"flaky": 1})
    final = _run(skills)

    assert final["picks"] == 1
    assert len(skills.grasp_calls) == 2  # failed attempt + successful retry


def test_recovery_gives_up_after_max_retries() -> None:
    """If verify keeps failing past the retry cap, the okra is marked failed."""
    config = HarvestConfig(max_grasp_retries=3)
    field = [FieldOkra("stuck", x=0.30, y=0.45, z=0.80, ripeness=0.9)]
    skills = _mock(field, flaky_verifies={"stuck": 99})
    final = _run(skills, config)

    assert final["picks"] == 0
    assert len(skills.grasp_calls) == config.max_grasp_retries
    assert any("give_up" in line for line in final["log"])


def test_announces_japanese_at_key_points() -> None:
    """The robot speaks Japanese on start, on each pick, and on completion."""
    field = [FieldOkra("a", x=0.30, y=0.45, z=0.80, ripeness=0.9)]
    skills = _mock(field)
    voice = RecordingAnnouncer()
    app = build_harvest_graph(skills, _CFG, announcer=voice)
    app.invoke(initial_state(), {"recursion_limit": _RECURSION_LIMIT})

    assert announce.start() in voice.said
    assert announce.picked(1) in voice.said
    assert announce.done(1) in voice.said


def test_announces_approach_direction() -> None:
    """A too-far fruit is announced as a forward approach (depth) in Japanese."""
    field = [FieldOkra("far", x=0.30, y=0.85, z=0.80, ripeness=0.9)]
    skills = _mock(field)
    voice = RecordingAnnouncer()
    app = build_harvest_graph(skills, _CFG, announcer=voice)
    app.invoke(initial_state(), {"recursion_limit": _RECURSION_LIMIT})

    assert announce.approaching("forward") in voice.said


def test_silent_by_default() -> None:
    """With no announcer the graph runs silently (NullAnnouncer), no crash."""
    field = [FieldOkra("a", x=0.30, y=0.45, z=0.80, ripeness=0.9)]
    skills = _mock(field)
    final = _run(skills)  # no announcer passed

    assert final["picks"] == 1
