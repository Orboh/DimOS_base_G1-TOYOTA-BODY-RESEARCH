#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Blueprint: FULL okra harvest — REAL arm (okra-ACT) + REAL base walking + voice.

    dimos run unitree-g1-okra-harvest-full

The "everything on" integration (Step 3): head + right-wrist cameras → YOLO
detect → select → **okra-ACT reach (arm MOVES, stoppable)** → Ollama-vision
verify → Japanese G1 speaker, with **base reposition/sweep walking (cmd_vel)**
for the §5 approach/advance/revisit moves. Legs run on the onboard balance
controller (LocoClient velocity in 'ai' mode) while the arm runs on
``rt/arm_sdk`` — the two coexist (motion-control mode), so walking and the arm
reach happen concurrently.

This combines ``-live-arm`` (arm + speaker + ACT) and ``-walk`` (cmd_vel base)
in ONE process. That is only safe because every DDS module now initialises the
channel factory through the idempotent ``ensure_channel_factory`` helper — the
first ``start()`` does the real ``ChannelFactoryInitialize``, the rest are
no-ops (no double-init crash). See ``dimos/robot/unitree/g1/act/dds_init.py``.

⚠️⚠️ THE ARM MOVES **and** THE ROBOT WALKS. Keep an e-stop in hand and clear
space around the whole robot. The SafetyMonitor file e-stop pauses the arm reach
mid-motion: ``touch /tmp/okra_estop`` to pause, ``rm`` to resume. (Person/balance
/VLM safety checks are still follow-ups — wire real checks before unattended use.)

Prereqs (union of -live-arm and -walk):
  # NX:      teleimager-server --rs
  # laptop:  ~/act-okura/.venv_act/bin/python scripts/act_service.py --serve
  # Jetson:  ollama serve   (moondream already pulled)
  # laptop:  ROBOT_INTERFACE=<nic> dimos run unitree-g1-okra-harvest-full
"""

from __future__ import annotations

import os

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.unitree.g1.act.g1_arm_sdk_connection import G1ArmSdkConnection
from dimos.robot.unitree.g1.act.g1_gripper_connection import G1GripperConnection
from dimos.robot.unitree.g1.camera.teleimager_camera_module import (
    RightWristTeleimagerCamera,
    TeleimagerCamera,
)
from dimos.robot.unitree.g1.effectors.high_level.dds_sdk import G1HighLevelDdsSdk
from dimos.robot.unitree.g1.harvest.harvest_module import HarvestModule

_NIC = os.getenv("ROBOT_INTERFACE", "")

# Tree-model dataset (sotata/okura-pick-tree-20260615) first-frame arm pose [rad]
# (left 7 + right 7) — arms slew here on start so the policy begins in-distribution.
_INIT_ARM_POSE = [
    0.269, 0.196, -0.018, 0.986, 0.122, 0.028, 0.003,   # left arm
    -0.114, 0.029, 0.185, 0.538, 0.209, -0.755, 0.370,  # right arm
]


unitree_g1_okra_harvest_full = (
    autoconnect(
        TeleimagerCamera.blueprint(camera="head"),
        RightWristTeleimagerCamera.blueprint(camera="right_wrist"),
        G1ArmSdkConnection.blueprint(network_interface=_NIC, initial_arm_pose=_INIT_ARM_POSE),
        G1GripperConnection.blueprint(network_interface=_NIC),
        G1HighLevelDdsSdk.blueprint(network_interface=_NIC),  # base walking (LocoClient)
        HarvestModule.blueprint(
            use_dummy=False,
            use_act_grasp=True,      # ⚠️ real arm reach (2-camera tree model)
            use_base_move=True,      # ⚠️ real base walking via cmd_vel -> LocoClient
            use_g1_speaker=True,     # Japanese G1 speaker
            vlm_model="moondream",   # local Ollama vision verify (~1s caption+keyword)
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
            ("cmd_vel", Twist): LCMTransport("/cmd_vel", Twist),
        }
    )
)

__all__ = ["unitree_g1_okra_harvest_full"]
