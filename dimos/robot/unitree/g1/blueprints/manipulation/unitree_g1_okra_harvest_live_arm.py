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
from dimos.robot.unitree.g1.camera.teleimager_camera_module import (
    RightWristTeleimagerCamera,
    TeleimagerCamera,
)
from dimos.robot.unitree.g1.harvest.harvest_module import HarvestModule

_NIC = os.getenv("ROBOT_INTERFACE", "")

# Tree-model dataset (sotata/okura-pick-tree-20260615) first-frame arm pose [rad]
# (left 7 + right 7) — arms slew here on start so the policy begins in-distribution.
_INIT_ARM_POSE = [
    0.269, 0.196, -0.018, 0.986, 0.122, 0.028, 0.003,   # left arm
    -0.114, 0.029, 0.185, 0.538, 0.209, -0.755, 0.370,  # right arm
]


unitree_g1_okra_harvest_live_arm = (
    autoconnect(
        TeleimagerCamera.blueprint(camera="head"),
        RightWristTeleimagerCamera.blueprint(camera="right_wrist"),
        G1ArmSdkConnection.blueprint(network_interface=_NIC, initial_arm_pose=_INIT_ARM_POSE),
        G1GripperConnection.blueprint(network_interface=_NIC),
        HarvestModule.blueprint(
            use_dummy=False,
            use_act_grasp=True,      # ⚠️ real arm reach (2-camera tree model)
            use_g1_speaker=True,     # Japanese G1 speaker
            vlm_model="moondream",   # local Ollama vision verify (~1s caption+keyword)
            use_base_move=False,     # base walking off (safety)
        ),
    )
    .remappings(
        [
            # Wrist instance also publishes color_image; rename so it feeds the
            # HarvestModule/ActGraspModule wrist input, not the head.
            (RightWristTeleimagerCamera, "color_image", "cam_right_wrist"),
        ]
    )
    .transports(
        {
            ("color_image", Image): LCMTransport("/color_image", Image),
            ("cam_right_wrist", Image): LCMTransport("/cam_right_wrist", Image),
            ("motor_states", JointState): LCMTransport("/g1/motor_states", JointState),
            ("arm_target", JointState): LCMTransport("/g1/arm_target", JointState),
            ("right_gripper_state", JointState): LCMTransport("/g1/right_gripper_state", JointState),
            ("gripper_target", JointState): LCMTransport("/g1/gripper_target", JointState),
        }
    )
)

__all__ = ["unitree_g1_okra_harvest_live_arm"]
