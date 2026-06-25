# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Offline tests for GraspSequence (IK->ACT->cut-gate->cut), fully injected.

No robot, no pinocchio: ik_solve / act / cut_ok / publishers are fakes. These pin
the sequence ordering, the cut-gate, the blade-safety clamp, and stop-cancellation.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

# JointState pulls dimos_lcm (LCM-generated msgs, only present in the live/Orin
# env). GraspSequence only constructs JointState(name=, position=, ...) and reads
# .position back in tests, so stub a lightweight stand-in before importing it.
if "dimos_lcm" not in sys.modules:
    class _FakeJointState:
        def __init__(self, name=None, position=None, velocity=None, effort=None):
            self.name = name
            self.position = position
            self.velocity = velocity
            self.effort = effort

    _mod = types.ModuleType("dimos.msgs.sensor_msgs.JointState")
    _mod.JointState = _FakeJointState  # type: ignore[attr-defined]
    sys.modules["dimos.msgs.sensor_msgs.JointState"] = _mod

from dimos.robot.unitree.g1.harvest.blackboard import Okra
from dimos.robot.unitree.g1.harvest.grasp_sequence import GraspSequence


@dataclass
class _Sol:
    """Stand-in IkApproachResult (only the fields GraspSequence reads)."""

    arm14: list[float]
    joint_names: list[str]
    wait_s: float


def _ok_sol() -> _Sol:
    return _Sol(arm14=[0.0] * 14, joint_names=[f"j{i}" for i in range(14)], wait_s=0.0)


class _FakeAct:
    def __init__(self) -> None:
        self.ran = 0
        self.stopped = False

    def run_episode(self, okra=None, force=None) -> bool:
        self.ran += 1
        return True

    def stop(self) -> None:
        self.stopped = True


def _grippers() -> tuple[list, list]:
    arm_pub, grip_pub = [], []
    return arm_pub, grip_pub


def test_full_sequence_success_order() -> None:
    """IK -> ACT -> cut-gate(True) -> cut: returns True, cut publishes 4.4 rad."""
    events: list[str] = []
    act = _FakeAct()
    arm_pub: list = []
    grip_pub: list = []
    seq = GraspSequence(
        ik_solve=lambda o: (events.append("ik"), _ok_sol())[1],
        publish_arm=lambda js: arm_pub.append(js),
        act_module=act,
        cut_ok_fn=lambda: (events.append("gate"), True)[1],
        publish_gripper=lambda js: grip_pub.append(js),
    )
    ok = seq.run_episode(Okra(id="okra_1"), force=0.3)
    assert ok is True
    assert events == ["ik", "gate"]       # IK before gate
    assert act.ran == 1                    # ACT ran once
    assert len(arm_pub) == 1               # IK published the 14-joint target
    assert len(grip_pub) == 1              # cut published one gripper target
    assert grip_pub[0].position[0] == 4.4  # closed to the cut position
    assert seq.episodes[-1] == ("okra_1", "cut", True)


def test_ik_unreachable_fails_before_act() -> None:
    """IK returns None -> episode fails immediately, ACT and cut never run."""
    act = _FakeAct()
    grip_pub: list = []
    seq = GraspSequence(
        ik_solve=lambda o: None,
        act_module=act,
        publish_gripper=lambda js: grip_pub.append(js),
    )
    assert seq.run_episode(Okra(id="okra_x")) is False
    assert act.ran == 0
    assert grip_pub == []
    assert seq.episodes[-1] == ("okra_x", "ik", False)


def test_cut_gate_blocks_cut() -> None:
    """VLM cut-gate False -> no cut published, episode fails (after ACT)."""
    act = _FakeAct()
    grip_pub: list = []
    seq = GraspSequence(
        ik_solve=lambda o: _ok_sol(),
        act_module=act,
        cut_ok_fn=lambda: False,
        publish_gripper=lambda js: grip_pub.append(js),
    )
    assert seq.run_episode(Okra(id="okra_2")) is False
    assert act.ran == 1                    # ACT still ran (approach happened)
    assert grip_pub == []                  # but the blade never closed
    assert seq.episodes[-1] == ("okra_2", "cut_gate", False)


def test_blade_safety_clamp() -> None:
    """A cut command above the blade limit is clamped to 5.2 rad."""
    grip_pub: list = []
    seq = GraspSequence(
        ik_solve=lambda o: _ok_sol(),
        publish_gripper=lambda js: grip_pub.append(js),
        q_close=6.0,  # over the 5.2 blade limit on purpose
    )
    assert seq.run_episode(Okra(id="okra_3")) is True
    assert grip_pub[0].position[0] == 5.2  # clamped, blade protected


def test_stop_cancels_and_stops_act() -> None:
    """stop() before the episode -> returns False early and propagates to ACT."""
    act = _FakeAct()
    seq = GraspSequence(ik_solve=lambda o: _ok_sol(), act_module=act)
    seq.stop()
    assert seq.run_episode(Okra(id="okra_4")) is False
    assert act.stopped is True


def test_no_cut_gate_allows_cut() -> None:
    """cut_ok_fn=None (VLM unwired) -> cut proceeds (fallback allow)."""
    grip_pub: list = []
    seq = GraspSequence(
        ik_solve=lambda o: _ok_sol(),
        publish_gripper=lambda js: grip_pub.append(js),
    )
    assert seq.run_episode(Okra(id="okra_5")) is True
    assert grip_pub[0].position[0] == 4.4
