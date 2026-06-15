#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Blueprint: okra harvest with the REAL arm (okra-ACT reach).

    dimos run unitree-g1-okra-harvest-live-arm

The harvest orchestrator drives the full live loop: head-camera YOLO detect →
select → **okra-ACT reach (the right arm MOVES, stoppable)** → Ollama-vision
verify → Japanese G1-speaker announce. Legs stay on the onboard balance
controller (motion-control mode; the arm runs on rt/arm_sdk).

⚠️ THE ARM MOVES. Keep an e-stop in hand and space around the arm. The
SafetyMonitor can cancel a reach mid-motion (currently a placeholder always-safe
check — wire REAL checks before relying on it). Base walking is OFF
(use_base_move=False); grasp-and-pull only (no cutter). Not robot-verified here.

Prereqs (same as unitree-g1-act-arm, plus this pipeline's deps):
  # NX:      teleimager-server --rs
  # laptop:  ~/act-okura/.venv_act/bin/python scripts/act_service.py --serve
  # laptop:  ollama serve  &&  ollama pull moondream
  # laptop:  ROBOT_INTERFACE=<nic> dimos run unitree-g1-okra-harvest-live-arm
"""

from __future__ import annotations

import os
from typing import Any

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.unitree.g1.act.g1_arm_sdk_connection import G1ArmSdkConnection
from dimos.robot.unitree.g1.act.g1_gripper_connection import G1GripperConnection
from dimos.robot.unitree.g1.camera.teleimager_camera_module import TeleimagerCamera
from dimos.robot.unitree.g1.camera.zmq_camera_module import ZmqCamera
from dimos.robot.unitree.g1.harvest.harvest_module import HarvestModule

_NIC = os.getenv("ROBOT_INTERFACE", "")

# Dataset first-frame arm pose [rad] (left 7 + right 7) — arms slew here on start
# so the policy begins in-distribution (mirrors unitree-g1-act-arm).
_INIT_ARM_POSE = [
    -0.110, -0.047, 0.112, 0.131, 0.012, -0.411, 0.157,
    -0.294, 0.077, 0.174, 0.768, -0.340, -0.809, -0.476,
]


def _camera_blueprint() -> Any:
    source = os.getenv("DIMOS_CAMERA_SOURCE", "teleimager").strip().lower()
    return ZmqCamera.blueprint() if source == "zmq" else TeleimagerCamera.blueprint()


unitree_g1_okra_harvest_live_arm = autoconnect(
    _camera_blueprint(),
    G1ArmSdkConnection.blueprint(network_interface=_NIC, initial_arm_pose=_INIT_ARM_POSE),
    G1GripperConnection.blueprint(network_interface=_NIC),
    HarvestModule.blueprint(
        use_dummy=False,
        use_act_grasp=True,      # ⚠️ real arm reach
        use_g1_speaker=True,     # Japanese G1 speaker
        vlm_model="moondream",   # local Ollama vision verify
        use_base_move=False,     # base walking off (safety)
    ),
).transports(
    {
        ("color_image", Image): LCMTransport("/color_image", Image),
        ("motor_states", JointState): LCMTransport("/g1/motor_states", JointState),
        ("arm_target", JointState): LCMTransport("/g1/arm_target", JointState),
        ("right_gripper_state", JointState): LCMTransport("/g1/right_gripper_state", JointState),
        ("gripper_target", JointState): LCMTransport("/g1/gripper_target", JointState),
    }
)

__all__ = ["unitree_g1_okra_harvest_live_arm"]
