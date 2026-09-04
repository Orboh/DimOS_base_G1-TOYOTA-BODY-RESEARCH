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

"""Offline tests for ActGraspModule (fake ACT service + stub I/O, no robot).

Verifies the episode loop, the stop/cancel path (what the SafetyMonitor drives),
reset-on-first-step, and that arm + gripper targets are published — all without
ZMQ / act_service / a real arm.
"""

from __future__ import annotations

import sys
import threading
import time
import types

# _publish lazily imports JointState (pulls dimos_lcm, only in the live/Orin env).
# Stub a lightweight stand-in so the publish path is testable off-robot.
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

from dimos.robot.unitree.g1.harvest.act_grasp import ActGraspModule


class _Okra:
    id = "okra_1"


class _State:
    """Stand-in 29-DOF JointState (position only; name check is skipped when empty)."""

    def __init__(self) -> None:
        self.position = [0.0] * 29
        self.name: list[str] = []


def _module(act_call, **kw):
    published = {"arm": [], "grip": []}
    mod = ActGraspModule(
        image_getter=lambda: object(),  # opaque; encode is stubbed below
        state_getter=lambda: _State(),
        gripper_getter=lambda: 0.0,
        publish_arm=lambda js: published["arm"].append(js),
        publish_gripper=lambda js: published["grip"].append(js),
        act_call=act_call,
        rate_hz=kw.pop("rate_hz", 1000.0),  # fast loop for tests (overridable)
        **kw,
    )
    mod._encode = lambda image: b"JPEG"  # skip real cv2 jpeg encoding
    return mod, published


def test_two_camera_sends_both_images() -> None:
    """With a wrist_getter, the request carries cam_high + cam_right_wrist."""
    sent: list = []
    mod, _pub = _module(
        lambda obs: sent.append(obs) or [0.0] * 16,
        max_steps=1,
        wrist_getter=lambda: object(),
    )
    mod.run_episode(_Okra(), 0.3)
    images = sent[0]["images"]
    assert set(images) == {"cam_high", "cam_right_wrist"}


def test_7d_wrist_only_state_and_wire() -> None:
    """7-DOF wrist-only: state is 7 right-arm joints, sole image = wrist in image_jpeg."""
    sent: list = []

    def _state():
        s = _State()
        # right arm (motor 22-28) = recognizable values 1..7
        for i, v in enumerate(range(1, 8)):
            s.position[22 + i] = float(v)
        return s

    published = {"arm": [], "grip": []}
    mod = ActGraspModule(
        image_getter=lambda: None,  # head not needed in 7d mode
        state_getter=_state,
        gripper_getter=lambda: 0.0,
        publish_arm=lambda js: published["arm"].append(js),
        publish_gripper=lambda js: published["grip"].append(js),
        act_call=lambda obs: sent.append(obs) or [0.1] * 7,
        rate_hz=1000.0,
        max_steps=1,
        wrist_getter=lambda: object(),
        right_arm_only_7d=True,
    )
    mod._encode = lambda image: b"WRISTJPEG"
    mod.run_episode(_Okra(), 0.3)

    assert len(sent) == 1
    # state = exactly the 7 right-arm joints, no gripper dim
    assert sent[0]["state"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    # sole image is the wrist, and it is also sent in the head image_jpeg slot
    assert set(sent[0]["images"]) == {"cam_right_wrist"}
    assert sent[0]["image_jpeg"] == b"WRISTJPEG"


def test_7d_publishes_arm_not_gripper() -> None:
    """7-DOF mode publishes the right arm into arm_target and NEVER the gripper (cut is out of ACT)."""
    published = {"arm": [], "grip": []}
    mod = ActGraspModule(
        image_getter=lambda: None,
        state_getter=lambda: _State(),
        gripper_getter=lambda: 0.0,
        publish_arm=lambda js: published["arm"].append(js),
        publish_gripper=lambda js: published["grip"].append(js),
        act_call=lambda obs: [0.5] * 7,
        rate_hz=1000.0,
        max_steps=1,
        wrist_getter=lambda: object(),
        right_arm_only_7d=True,
    )
    mod._encode = lambda image: b"JPEG"
    mod.run_episode(_Okra(), 0.3)

    assert len(published["arm"]) == 1
    arm = published["arm"][0]
    assert len(arm.position) == 14  # full 14-joint arm_target
    assert arm.position[7:] == [0.5] * 7  # right half = ACT action
    assert published["grip"] == []  # gripper untouched by ACT


def test_two_camera_waits_for_wrist_frame() -> None:
    """If the wrist frame isn't there yet, no inference is sent (no startup error)."""
    sent: list = []
    mod, _pub = _module(
        lambda obs: sent.append(obs) or [0.0] * 16,
        max_steps=1,
        rate_hz=1000.0,
        wrist_getter=lambda: None,  # wrist never arrives
    )
    mod.run_episode(_Okra(), 0.3)
    assert sent == []  # waited for the wrist; never inferred
    assert mod.episodes[-1][1] == 0  # 0 steps


def test_episode_runs_and_publishes() -> None:
    obs_seen: list = []

    def act_call(obs):
        obs_seen.append(obs)
        return [0.0] * 16

    mod, pub = _module(act_call, max_steps=5)
    done = mod.run_episode(_Okra(), 0.3)

    assert done is True
    assert len(pub["arm"]) == 5 and len(pub["grip"]) == 5  # one per step
    assert obs_seen[0]["reset"] is True and obs_seen[1]["reset"] is False  # reset only first
    assert mod.episodes[-1] == ("okra_1", 5, False)


def test_stop_cancels_mid_reach() -> None:
    def act_call(obs):
        return [0.0] * 16

    mod, _pub = _module(act_call, max_steps=10_000, rate_hz=200.0)
    result: dict = {}
    t = threading.Thread(target=lambda: result.update(ok=mod.run_episode(_Okra(), 0.3)))
    t.start()
    time.sleep(0.05)
    mod.stop()
    t.join(timeout=2.0)

    assert result["ok"] is False  # cancelled
    okra_id, steps, cancelled = mod.episodes[-1]
    assert cancelled is True and steps < 10_000


def test_reached_fn_ends_episode_early() -> None:
    mod, pub = _module(lambda obs: [0.0] * 16, max_steps=100, reached_fn=lambda a, step: step >= 3)
    assert mod.run_episode(_Okra(), 0.3) is True
    assert len(pub["arm"]) == 3  # stopped as soon as reached_fn fired


def test_publishes_right_gripper_from_action() -> None:
    # action[15] is the right Dex1 target.
    action = [0.0] * 16
    action[15] = 0.42
    mod, pub = _module(lambda obs: action, max_steps=1)
    mod.run_episode(_Okra(), 0.3)
    assert pub["grip"][0].position[0] == 0.42
    assert len(pub["arm"][0].position) == 14  # 14 arm targets
