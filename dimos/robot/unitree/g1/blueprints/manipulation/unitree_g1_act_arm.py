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

"""LIVE okra-ACT upper-body control: arms MOVE via rt/arm_sdk + right Dex1.

⚠️ This MOVES the real robot's arms and right gripper. Legs stay on the onboard
balance controller (sport mode NOT released — "motion control mode"). The arm
safety is built into ``G1ArmSdkConnection`` (faithful port of the verified
unitree_lerobot ``G1_29_ArmController``): the command is clipped toward the
target relative to the MEASURED arm pose at ≤20 rad/s every 250 Hz cycle, the
waist is held stiff at its startup pose, weight ramps 0→1, motion starts from
the current pose. Keep an e-stop in hand and clear space around the arms.

Wiring (autoconnect by stream name):
    TeleimagerCamera.color_image ───────▶ ActBridge.color_image
    G1ArmSdkConnection.motor_states ────▶ ActBridge.motor_states
    G1GripperConnection.right_gripper_state ▶ ActBridge.right_gripper_state
    ActBridge.arm_target ───────────────▶ G1ArmSdkConnection.arm_target ─▶ rt/arm_sdk
    ActBridge.gripper_target ───────────▶ G1GripperConnection.gripper_target ▶ rt/dex1/right/cmd

The camera defaults to teleimager (the exact format the okra policy was trained
on); set ``DIMOS_CAMERA_SOURCE=zmq`` to fall back to the GEAR-SONIC publisher.

Run (in order):
    # NX:      teleimager-server --rs              (head frames on :55555)
    # laptop:  ~/act-okura/.venv_act/bin/python scripts/act_service.py --serve
    # laptop:  ROBOT_INTERFACE=<nic> .venv/bin/dimos run unitree-g1-act-arm
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

# Dataset (Orboh/okura-sub-lerobot) recorded first-frame arm pose [rad], left 7 +
# right 7. The arms slew here on startup so the policy begins in-distribution,
# mirroring eval_g1.py. ActBridge then waits _START_DELAY_S before inferring.
_INIT_ARM_POSE = [
    -0.110, -0.047, 0.112, 0.131, 0.012, -0.411, 0.157,   # left arm
    -0.294, 0.077, 0.174, 0.768, -0.340, -0.809, -0.476,  # right arm
]
_START_DELAY_S = 2.5

unitree_g1_act_arm = autoconnect(
    _camera_blueprint(),
    G1ArmSdkConnection.blueprint(network_interface=_NIC, initial_arm_pose=_INIT_ARM_POSE),
    G1GripperConnection.blueprint(network_interface=_NIC),
    ActBridge.blueprint(dry_run=False, startup_delay_s=_START_DELAY_S),
).transports(
    {
        ("color_image", Image): LCMTransport("/color_image", Image),
        ("motor_states", JointState): LCMTransport("/g1/motor_states", JointState),
        ("arm_target", JointState): LCMTransport("/g1/arm_target", JointState),
        ("right_gripper_state", JointState): LCMTransport("/g1/right_gripper_state", JointState),
        ("gripper_target", JointState): LCMTransport("/g1/gripper_target", JointState),
    }
)

__all__ = ["unitree_g1_act_arm"]
