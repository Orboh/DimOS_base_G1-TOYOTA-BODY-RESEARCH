# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Offline tests for the base-motion / station-nav wiring (no robot)."""

from __future__ import annotations

from dimos.robot.unitree.g1.harvest.nav_skills import (
    make_navigate_stations,
    make_search_forward,
    make_twist_move_cmd,
)


def test_twist_move_cmd_drives_then_stops() -> None:
    published: list = []
    move_cmd = make_twist_move_cmd(published.append, settle_s=0.0)

    move_cmd(0.15, 0.0, 0.0, 0.01)  # forward velocity for ~10 ms, then stop

    assert len(published) == 2  # drive Twist, then stop Twist
    assert published[0].linear.x == 0.15  # forward velocity
    assert published[1].linear.x == 0.0 and published[1].angular.z == 0.0  # stop


def test_twist_move_cmd_lateral_and_turn() -> None:
    published: list = []
    make_twist_move_cmd(published.append, settle_s=0.0)(0.0, -0.2, 0.1, 0.01)
    assert published[0].linear.y == -0.2
    assert published[0].angular.z == 0.1


def test_navigate_stations_visits_then_ends() -> None:
    visited: list = []
    go = make_navigate_stations(visited.append, ["row-A", "row-B"])

    assert go() is True
    assert go() is True
    assert go() is False  # exhausted -> field done
    assert visited == ["row-A", "row-B"]


def test_navigate_stations_empty_is_false() -> None:
    assert make_navigate_stations(lambda t: None, [])() is False


def test_relative_move_uses_twist_move_cmd_end_to_end() -> None:
    """A DimosHarvestSkills built with the cmd_vel move_cmd drives the base."""
    from dimos.robot.unitree.g1.harvest.real_skills import build_dimos_harvest_skills

    published: list = []
    move_cmd = make_twist_move_cmd(published.append, settle_s=0.0)
    skills = build_dimos_harvest_skills(
        move_cmd=move_cmd,
        detect_fn=lambda: [],
        grasp_fn=lambda okra, force: None,
        verify_fn=lambda: True,
        next_station_fn=lambda: False,
        swap_fn=lambda: None,
        base_speed=100.0,  # fast so the timed move is ~instant in the test
    )
    skills.relative_move(lateral=0.0, forward=0.30)
    assert any(t.linear.x > 0 for t in published)  # drove forward
    assert published[-1].linear.x == 0.0  # ended stopped


def test_search_forward_steps_then_finishes() -> None:
    """go_to_next_station walks forward and returns True up to max, then False."""
    moves: list = []
    move_cmd = lambda vx, vy, vyaw, dur: moves.append((vx, vy, vyaw, dur))  # noqa: E731
    go = make_search_forward(move_cmd, step_m=0.30, speed=0.5, max_advances=2)

    assert go() is True  # advance 1
    assert go() is True  # advance 2
    assert go() is False  # exhausted -> field done (run terminates)
    assert len(moves) == 2  # only the two successful advances drove the base
    assert all(vx > 0 and vy == 0.0 for vx, vy, _, _ in moves)  # forward only
    assert moves[0][3] == 0.30 / 0.5  # dur = step / speed
