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

"""Tests for the okra-harvest LangGraph.

Covers the verify/retry recovery and the 3D §5 movement: approach a fruit that
is too far / too close / off to the left, skip an out-of-height fruit, and sweep
to discover a fruit out of view.
"""

from __future__ import annotations

from unittest.mock import patch

from dimos.robot.unitree.g1.harvest import announce
from dimos.robot.unitree.g1.harvest.announce import RecordingAnnouncer
from dimos.robot.unitree.g1.harvest.blackboard import HarvestConfig, initial_state
from dimos.robot.unitree.g1.harvest.graph import build_harvest_graph
from dimos.robot.unitree.g1.harvest.skills import FieldOkra, MockHarvestSkills

_RECURSION_LIMIT = 400
_CFG = HarvestConfig()  # reach x[-0.20,0.61] y[0.05,0.65] z[-0.35,0.85]; centre (0.205, 0.35)


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
    field = [FieldOkra("far", x=_CFG.reach.x_center, y=0.85, z=0.80, ripeness=0.9)]
    skills = _mock(field)
    final = _run(skills)

    assert final["picks"] == 1
    assert len(skills.move_calls) >= 1
    # First move is forward (positive y), no lateral (already centred in x).
    lateral, forward = skills.move_calls[0]
    assert forward > 0 and abs(lateral) < 1e-9


def test_too_close_okra_triggers_backup_then_pick() -> None:
    """An okra closer than the reach box (ridge risk) → move BACK → pick."""
    field = [FieldOkra("near", x=_CFG.reach.x_center, y=0.02, z=0.80, ripeness=0.9)]
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


def test_left_behind_okra_is_revisited() -> None:
    """An okra pushed out of view (to the right, where the left-sweep won't go)
    is remembered and revisited, so nothing is abandoned.

    okra 'a' (slightly left) is approached first (a left strafe), which pushes
    okra 'b' (far right) out of the FOV. The leftward discovery sweep moves away
    from b, so only the revisit (return to remembered position) recovers it.
    """
    field = [
        FieldOkra("a", x=-0.23, y=0.45, z=0.80, ripeness=0.90),  # nearest -> approached first
        FieldOkra("b", x=0.79, y=0.45, z=0.80, ripeness=0.88),  # seen, then pushed out right
    ]
    skills = _mock(field)
    voice = RecordingAnnouncer()
    app = build_harvest_graph(skills, _CFG, announcer=voice)
    final = app.invoke(initial_state(), {"recursion_limit": _RECURSION_LIMIT})

    assert final["picks"] == 2  # both picked, including the left-behind one
    assert announce.revisiting() in voice.said  # a revisit actually happened
    assert final["pending"] == {}  # nothing left behind at the end


def test_visits_multiple_stations() -> None:
    """When a station is exhausted, the robot drives to the next and continues."""
    s0 = [FieldOkra("a0", x=0.30, y=0.45, z=0.80, ripeness=0.9)]
    s1 = [FieldOkra("a1", x=0.30, y=0.45, z=0.80, ripeness=0.9)]
    skills = MockHarvestSkills(reach=_CFG.reach, fov=_CFG.fov, stations=[s0, s1])
    voice = RecordingAnnouncer()
    app = build_harvest_graph(skills, _CFG, announcer=voice)
    final = app.invoke(initial_state(), {"recursion_limit": _RECURSION_LIMIT})

    assert final["picks"] == 2  # one okra from each station
    assert skills.station_moves == [1]  # moved to station 1 exactly once
    assert final["station_id"] == 1
    # Two-part announcement (§6 HMI): "this station is done" is true either way
    # so it is said BEFORE the move; "moving to the next one" only once
    # go_to_next_station() confirms one exists, so it is said after.
    assert announce.station_done() in voice.said
    assert announce.next_station() in voice.said
    assert voice.said.index(announce.station_done()) < voice.said.index(announce.next_station())


def test_basket_full_swaps_then_continues() -> None:
    """Hitting basket capacity triggers a swap, then harvesting resumes."""
    config = HarvestConfig(basket_capacity=2)
    field = [
        FieldOkra("a", x=0.20, y=0.45, z=0.80, ripeness=0.95),
        FieldOkra("b", x=0.30, y=0.45, z=0.80, ripeness=0.90),
        FieldOkra("c", x=0.40, y=0.45, z=0.80, ripeness=0.85),
    ]
    skills = _mock(field)
    voice = RecordingAnnouncer()
    app = build_harvest_graph(skills, config, announcer=voice)
    final = app.invoke(initial_state(), {"recursion_limit": _RECURSION_LIMIT})

    assert final["picks"] == 3  # all picked despite the basket filling mid-run
    assert skills.basket_swaps == 1  # swapped once when it hit capacity (2)
    assert announce.basket_swap() in voice.said


def test_silent_by_default() -> None:
    """With no announcer the graph runs silently (NullAnnouncer), no crash."""
    field = [FieldOkra("a", x=0.30, y=0.45, z=0.80, ripeness=0.9)]
    skills = _mock(field)
    final = _run(skills)  # no announcer passed

    assert final["picks"] == 1


def test_voice_lead_s_waits_after_speaking_before_moving() -> None:
    """voice_lead_s>0: the robot pauses after speaking, before it physically
    moves — so an announcement is always heard before the action it describes
    (§6 HMI; the real G1 speaker queues audio and returns immediately)."""
    config = HarvestConfig(voice_lead_s=2.0)
    field = [
        FieldOkra("far", x=config.reach.x_center, y=0.85, z=0.80, ripeness=0.9)
    ]  # too far -> reposition
    skills = _mock(field)
    voice = RecordingAnnouncer()
    app = build_harvest_graph(skills, config, announcer=voice)
    with patch("dimos.robot.unitree.g1.harvest.graph.time.sleep") as mock_sleep:
        app.invoke(initial_state(), {"recursion_limit": _RECURSION_LIMIT})

    assert announce.approaching("forward") in voice.said
    assert mock_sleep.call_count >= 1  # reposition + grasp both wait
    assert all(call.args[0] == 2.0 for call in mock_sleep.call_args_list)


def test_voice_lead_s_defaults_to_no_wait() -> None:
    """voice_lead_s defaults to 0.0 — existing behaviour/tests are unaffected."""
    field = [FieldOkra("a", x=0.30, y=0.45, z=0.80, ripeness=0.9)]
    skills = _mock(field)
    voice = RecordingAnnouncer()
    app = build_harvest_graph(skills, _CFG, announcer=voice)  # _CFG: voice_lead_s=0.0
    with patch("dimos.robot.unitree.g1.harvest.graph.time.sleep") as mock_sleep:
        app.invoke(initial_state(), {"recursion_limit": _RECURSION_LIMIT})

    mock_sleep.assert_not_called()


def test_advance_left_waits_only_on_its_one_announcement() -> None:
    """searching() is announced once per dry spell (not every sweep step), so the
    voice_lead_s wait should likewise fire only on that first sweep, not once per
    subsequent sweep step."""
    config = HarvestConfig(voice_lead_s=2.0, max_empty_advances=3)
    skills = _mock([])  # empty field: sweeps up to the cap, then gives up on the station
    voice = RecordingAnnouncer()
    app = build_harvest_graph(skills, config, announcer=voice)
    with patch("dimos.robot.unitree.g1.harvest.graph.time.sleep") as mock_sleep:
        app.invoke(initial_state(), {"recursion_limit": _RECURSION_LIMIT})

    assert len(skills.move_calls) == 3  # three advance_left sweeps (the cap)
    assert voice.said.count(announce.searching()) == 1  # announced only on the first
    # One wait for searching() (first sweep only) + one for station_done()
    # (next_station, unconditional) — NOT one per sweep (that would be 3+).
    assert mock_sleep.call_count == 2


# --- 「オクラが無い」と「カメラ準備中」を音声で区別する（2026-09-14） --------------
# 実機では音声だけが判断材料になる場面があるため、count==0 の理由を読み分けられる
# ことをグラフ経由で担保する。


class _EmptyDetectSkills(MockHarvestSkills):
    """検出0件を返すが、3D化できず捨てた件数だけを差し替えられるスキル。"""

    def __init__(self, dropped: int) -> None:
        super().__init__(field=[])  # 空の畑
        self._dropped = dropped

    def detect_okra(self):  # type: ignore[no-untyped-def]
        return []

    def last_detect_dropped(self) -> int:
        return self._dropped


def test_voice_reports_detections_that_could_not_be_localized() -> None:
    voice = RecordingAnnouncer()
    app = build_harvest_graph(_EmptyDetectSkills(dropped=3), _CFG, announcer=voice)
    app.invoke(initial_state())
    assert announce.detect_result(0, 3) in voice.said
    assert announce.detect_result(0) not in voice.said


def test_voice_says_no_okra_when_nothing_was_dropped() -> None:
    voice = RecordingAnnouncer()
    app = build_harvest_graph(_EmptyDetectSkills(dropped=0), _CFG, announcer=voice)
    app.invoke(initial_state())
    assert announce.detect_result(0) in voice.said


def test_detect_result_wording_is_distinct() -> None:
    assert announce.detect_result(0, 0) == "オクラは見当たりません。"
    assert announce.detect_result(0, 3) == "オクラを3個見つけましたが、位置が測れません。"
    assert announce.detect_result(2, 3) == "オクラが2個見えます。"  # 有効があれば通常文言


def test_detect_log_records_dropped_count() -> None:
    app = build_harvest_graph(_EmptyDetectSkills(dropped=2), _CFG)
    out = app.invoke(initial_state())
    assert any("dropped 2" in line for line in out["log"])
