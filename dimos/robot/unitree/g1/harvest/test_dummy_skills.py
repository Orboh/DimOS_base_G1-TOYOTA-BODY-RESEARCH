# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the DUMMY skills: full pipeline runs, and a grasp can be cancelled
mid-reach by the SafetyMonitor (no robot)."""

from __future__ import annotations

import threading
import time

from dimos.robot.unitree.g1.harvest.blackboard import HarvestConfig, Okra, initial_state
from dimos.robot.unitree.g1.harvest.dummy_skills import (
    DummyGraspModule,
    DummyHarvestSkills,
    make_vlm_verify_harvest,
)
from dimos.robot.unitree.g1.harvest.graph import build_harvest_graph
from dimos.robot.unitree.g1.harvest.safety import SafetyCheck, SafetyMonitor
from dimos.robot.unitree.g1.harvest.skills import HarvestSkills

_OKRA = Okra(id="x", img_region="R", pos_3d={"x": 0.30, "y": 0.45, "z": 0.80}, ripeness=0.9)


def test_dummy_satisfies_protocol() -> None:
    assert isinstance(DummyHarvestSkills(), HarvestSkills)


def test_full_pipeline_runs_with_dummy_skills() -> None:
    skills = DummyHarvestSkills(num_okra=2, grasp_module=DummyGraspModule(steps=2, step_s=0.001))
    app = build_harvest_graph(skills, HarvestConfig())
    final = app.invoke(initial_state(), {"recursion_limit": 400})
    assert final["picks"] == 2


def test_grasp_module_stop_aborts_reach() -> None:
    grasp = DummyGraspModule(steps=200, step_s=0.01)  # ~2s if uninterrupted
    result: dict = {}
    t = threading.Thread(target=lambda: result.update(ok=grasp.run_episode(_OKRA, 0.3)))
    t.start()
    time.sleep(0.05)  # let the reach get going
    grasp.stop()
    t.join(timeout=2.0)

    assert result["ok"] is False  # cancelled, not completed
    okra_id, steps, cancelled = grasp.episodes[-1]
    assert cancelled is True and steps < 200


def test_safety_monitor_cancels_running_grasp() -> None:
    """A hazard during a reach -> monitor.on_pause -> grasp_module.stop()."""
    grasp = DummyGraspModule(steps=200, step_s=0.01)
    danger = {"on": False}
    mon = SafetyMonitor(
        [SafetyCheck("person", lambda: not danger["on"])],
        on_pause=lambda reason: grasp.stop(),
    )

    result: dict = {}
    t = threading.Thread(target=lambda: result.update(ok=grasp.run_episode(_OKRA, 0.3)))
    t.start()
    time.sleep(0.05)
    danger["on"] = True
    mon.step()  # trips -> on_pause -> grasp.stop()
    t.join(timeout=2.0)

    assert mon.gate.is_paused()
    assert result["ok"] is False  # the monitor actually stopped the reach
    assert grasp.episodes[-1][2] is True


def test_vlm_verify_routes_to_vlm() -> None:
    assert make_vlm_verify_harvest(lambda q: "Yes, it is holding an okra.")() is True
    assert make_vlm_verify_harvest(lambda q: "No, the gripper is empty.")() is False
