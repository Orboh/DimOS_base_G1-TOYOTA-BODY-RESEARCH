# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Offline checks for the real-robot wiring adapter (no robot needed).

Verifies the adapter shape: it satisfies the ``HarvestSkills`` protocol, turns a
relative displacement into the right velocity command, and actually drives the
harvest graph when backed by stand-in callables. The robot-touching paths
(``make_g1_speaker_announcer``, live VLM/ACT/nav) are NOT exercised here.
"""

from __future__ import annotations

from dimos.robot.unitree.g1.harvest.blackboard import HarvestConfig, Okra, initial_state
from dimos.robot.unitree.g1.harvest.graph import build_harvest_graph
from dimos.robot.unitree.g1.harvest.real_skills import (
    DimosHarvestSkills,
    build_dimos_harvest_skills,
    build_live_harvest_skills,
)
from dimos.robot.unitree.g1.harvest.skills import HarvestSkills

# Large base speed so the timed `move` sleeps are negligible in tests.
_FAST = 1000.0


def _adapter(**overrides):
    kw = dict(
        move_cmd=lambda vx, vy, vyaw, dur: None,
        detect_fn=lambda: [],
        grasp_fn=lambda okra, force: None,
        verify_fn=lambda: True,
        next_station_fn=lambda: False,
        swap_fn=lambda: None,
        base_speed=_FAST,
    )
    kw.update(overrides)
    return build_dimos_harvest_skills(**kw)


def test_adapter_satisfies_protocol() -> None:
    assert isinstance(_adapter(), HarvestSkills)
    assert isinstance(_adapter(), DimosHarvestSkills)


def test_relative_move_emits_velocity_commands() -> None:
    calls: list[tuple[float, float, float, float]] = []
    skills = _adapter(move_cmd=lambda vx, vy, vyaw, dur: calls.append((vx, vy, vyaw, dur)))

    skills.relative_move(lateral=0.0, forward=0.30)  # forward only
    assert len(calls) == 1
    vx, vy, _vyaw, dur = calls[0]
    assert vx > 0 and vy == 0.0 and dur > 0

    calls.clear()
    skills.relative_move(lateral=0.30, forward=0.0)  # right strafe -> vy (sign per robot)
    assert len(calls) == 1
    vx, vy, _vyaw, _dur = calls[0]
    assert vx == 0.0 and vy != 0.0


def test_graph_runs_with_real_adapter() -> None:
    """The real adapter plugs into the graph: one in-reach okra is picked."""
    cfg = HarvestConfig()
    seen = {"done": False}

    def detect():
        if seen["done"]:
            return []
        seen["done"] = True
        return [Okra(id="a", img_region="R", pos_3d={"x": 0.30, "y": 0.45, "z": 0.80},
                     ripeness=0.9, reachable=True)]

    grasps: list[tuple[str, float]] = []
    skills = _adapter(
        detect_fn=detect,
        grasp_fn=lambda okra, force: grasps.append((okra.id, force)),
        verify_fn=lambda: True,
    )
    app = build_harvest_graph(skills, cfg)
    final = app.invoke(initial_state(), {"recursion_limit": 400})

    assert final["picks"] == 1
    assert grasps == [("a", cfg.grasp_force)]


# --- LIVE assembly (real detect wiring) -------------------------------------

from dataclasses import dataclass  # noqa: E402

from dimos.robot.unitree.g1.harvest.graph import build_harvest_graph  # noqa: E402


@dataclass
class _FakeDet:
    name: str
    bbox: tuple
    confidence: float = 0.9
    track_id: int = 1


class _StubYolo:
    """Stands in for Yolo2DDetector: returns one in-reach 'okra' once, then none."""

    def __init__(self):
        self._seen = False

    def process_image(self, _frame):
        if self._seen:
            return []
        self._seen = True
        return [_FakeDet("okra", (300, 220, 340, 260), track_id=7)]


def test_build_live_harvest_skills_drives_graph() -> None:
    """LIVE assembly (real YOLO path, here with a stub detector) runs the graph."""
    cfg = HarvestConfig()
    skills, grasp = build_live_harvest_skills(
        frame_getter=lambda: object(),  # opaque frame; the stub detector ignores it
        target_classes={"okra"},
        detector=_StubYolo(),
        pixel_to_base=lambda u, v, det: {"x": 0.30, "y": 0.45, "z": 0.80},  # in reach box
    )
    assert isinstance(skills, DimosHarvestSkills)
    final = build_harvest_graph(skills, cfg).invoke(initial_state(), {"recursion_limit": 400})
    assert final["picks"] == 1  # detected (stub) + grasped (dummy grasp module)
    assert grasp is not None


def test_live_blueprint_imports_and_builds() -> None:
    from dimos.robot.unitree.g1.blueprints.manipulation.unitree_g1_okra_harvest_live import (
        unitree_g1_okra_harvest_live as bp,
    )

    assert bp is not None
