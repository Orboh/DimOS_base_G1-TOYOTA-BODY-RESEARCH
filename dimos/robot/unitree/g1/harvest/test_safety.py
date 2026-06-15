# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the SafetyMonitor + pause gate (no robot, no threads/sleeps).

Drives ``step()`` synchronously to verify trip/clear, the stop/resume hooks, the
expensive-check cadence, and that the harvest graph actually consults the gate.
"""

from __future__ import annotations

import threading

from dimos.robot.unitree.g1.harvest import announce
from dimos.robot.unitree.g1.harvest.announce import RecordingAnnouncer
from dimos.robot.unitree.g1.harvest.blackboard import HarvestConfig, Okra, initial_state
from dimos.robot.unitree.g1.harvest.graph import build_harvest_graph
from dimos.robot.unitree.g1.harvest.safety import (
    PauseGate,
    SafetyCheck,
    SafetyMonitor,
)


def test_trips_and_resumes_with_hooks_and_voice() -> None:
    danger = {"on": False}
    events: list[str] = []
    voice = RecordingAnnouncer()
    mon = SafetyMonitor(
        [SafetyCheck("person", lambda: not danger["on"])],
        on_pause=lambda reason: events.append(f"pause:{reason}"),
        on_resume=lambda: events.append("resume"),
        announcer=voice,
    )

    assert mon.step() == []  # safe initially
    assert not mon.gate.is_paused()

    danger["on"] = True
    assert mon.step() == ["person"]
    assert mon.gate.is_paused()
    assert events == ["pause:person"]
    assert announce.safety_stop("person") in voice.said

    # Still paused while danger persists; no duplicate pause hook.
    mon.step()
    assert events == ["pause:person"]

    danger["on"] = False
    mon.step()
    assert not mon.gate.is_paused()
    assert events == ["pause:person", "resume"]
    assert announce.safety_resume() in voice.said


def test_expensive_check_only_on_full_pass() -> None:
    calls = {"n": 0}

    def vlm_ok() -> bool:
        calls["n"] += 1
        return True

    mon = SafetyMonitor([SafetyCheck("vlm", vlm_ok, expensive=True)])
    mon.step(include_expensive=False)  # cheap tick: VLM not run
    assert calls["n"] == 0
    mon.step(include_expensive=True)  # full pass: VLM run
    assert calls["n"] == 1


def test_failing_check_is_treated_as_unsafe() -> None:
    def boom() -> bool:
        raise RuntimeError("sensor offline")

    mon = SafetyMonitor([SafetyCheck("imu", boom)])
    assert mon.step() == ["imu"]  # exception -> unsafe (fail safe)
    assert mon.gate.is_paused()


def test_gate_checkpoint_blocks_until_cleared() -> None:
    gate = PauseGate()
    gate.trip("test")
    assert gate.is_paused()
    # Clear from another thread; checkpoint should unblock.
    threading.Timer(0.05, gate.clear).start()
    assert gate.checkpoint(timeout=2.0) is True
    assert not gate.is_paused()


def test_graph_consults_the_gate() -> None:
    """The harvest graph calls gate.checkpoint at its motion nodes."""

    class CountingGate:
        def __init__(self) -> None:
            self.checkpoints = 0

        def checkpoint(self, timeout=None) -> bool:
            self.checkpoints += 1
            return True

        def is_paused(self) -> bool:
            return False

    cfg = HarvestConfig()
    field = [Okra(id="a", img_region="R", pos_3d={"x": 0.30, "y": 0.45, "z": 0.80},
                  ripeness=0.9, reachable=True)]
    seen = {"done": False}

    class _Skills:
        def detect_okra(self):
            if seen["done"]:
                return []
            seen["done"] = True
            return list(field)

        def relative_move(self, lateral, forward=0.0, yaw=0.0):
            pass

        def go_to_next_station(self):
            return False

        def swap_basket(self):
            pass

        def grasp_okra(self, okra, force):
            pass

        def verify_harvest(self):
            return True

        def record_harvest(self, record):
            pass

    gate = CountingGate()
    app = build_harvest_graph(_Skills(), cfg, safety=gate)
    final = app.invoke(initial_state(), {"recursion_limit": 400})

    assert final["picks"] == 1
    assert gate.checkpoints > 0  # the workflow consulted the safety gate
