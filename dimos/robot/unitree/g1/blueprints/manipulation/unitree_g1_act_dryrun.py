#!/usr/bin/env python3
# Copyright 2025-2026 Dimensional Inc.
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

"""Live DRY-RUN (B0) for the okra ACT: real camera + real joint/gripper state, NO motion.

Exercises the FULL Stage B pipeline with ZERO DDS writes, so the observation
assembly, ACT inference and camera format can be checked against the real robot
without moving anything:

    TeleimagerCamera ─color_image─▶ ActBridge ─(ZMQ)─▶ act_service ─▶ action (LOGGED only)
    G1ArmSdkConnection(publish_cmd=False) ─motor_states─▶ ActBridge   (reads rt/lowstate; no rt/arm_sdk write)
    G1GripperConnection(publish_cmd=False) ─right_gripper_state─▶ ActBridge (reads rt/dex1/right/state; no cmd write)

SAFETY:
- Both DDS modules run with ``publish_cmd=False`` => they only SUBSCRIBE
  (rt/lowstate, rt/dex1/right/state) and republish state; nothing is ever
  written to rt/arm_sdk or rt/dex1/right/cmd. Sport mode is not touched.
- ``ActBridge(dry_run=True)`` => the predicted action is logged only.

Run (in order):
    # NX:      teleimager-server --rs              (head frames on :55555)
    # laptop:  ~/act-okura/.venv_act/bin/python scripts/act_service.py --serve
    # laptop:  ROBOT_INTERFACE=<nic> .venv/bin/dimos run unitree-g1-act-dryrun
"""

from __future__ import annotations

import os
from typing import Any

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.unitree.g1.act.act_bridge import ActBridge
from dimos.robot.unitree.g1.act.g1_arm_sdk_connection import G1ArmSdkConnection
from dimos.robot.unitree.g1.act.g1_gripper_connection import G1GripperConnection
from dimos.robot.unitree.g1.camera.teleimager_camera_module import TeleimagerCamera
from dimos.robot.unitree.g1.camera.zmq_camera_module import ZmqCamera


def _camera_blueprint() -> Any:
    """Head-camera source. Defaults to teleimager (okra's training format)."""
    source = os.getenv("DIMOS_CAMERA_SOURCE", "teleimager").strip().lower()
    if source == "zmq":
        return ZmqCamera.blueprint()
    return TeleimagerCamera.blueprint()


_NIC = os.getenv("ROBOT_INTERFACE", "")

unitree_g1_act_dryrun = autoconnect(
    _camera_blueprint(),
    G1ArmSdkConnection.blueprint(network_interface=_NIC, publish_cmd=False),
    G1GripperConnection.blueprint(network_interface=_NIC, publish_cmd=False),
    ActBridge.blueprint(dry_run=True),
).transports(
    {
        ("color_image", Image): LCMTransport("/color_image", Image),
        ("motor_states", JointState): LCMTransport("/g1/motor_states", JointState),
        ("arm_target", JointState): LCMTransport("/g1/arm_target", JointState),
        ("right_gripper_state", JointState): LCMTransport("/g1/right_gripper_state", JointState),
        ("gripper_target", JointState): LCMTransport("/g1/gripper_target", JointState),
    }
)

__all__ = ["unitree_g1_act_dryrun"]
