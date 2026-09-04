#!/usr/bin/env python3
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

"""LIVE okra-ACT (tree model) standalone grasp — TWO cameras, the arm MOVES.

Like ``unitree-g1-act-arm`` but for the two-camera "tree" pick model
(``sotata/act-okura-pick-tree-06152026``): head ``cam_high`` + right-wrist
``cam_right_wrist``. ActBridge sends both images to ``act_service.py``; the
right arm reaches the cut point (closing/cutting is separate). This is the ACT
policy ALONE (no harvest LangGraph) — for testing whether the model grasps.

⚠️ The arm MOVES. Legs stay on the onboard balance controller (rt/arm_sdk).
Keep an e-stop in hand and space around the arm.

Run (in order):
    # NX:      teleimager-server --rs            (head :55555 + right_wrist :55557)
    # laptop:  ~/act-okura/.venv_act/bin/python scripts/act_service.py --serve
    # laptop:  ROBOT_INTERFACE=<nic> dimos run unitree-g1-act-arm-tree
"""

from __future__ import annotations

import os

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.unitree.g1.act.act_bridge import ActBridge
from dimos.robot.unitree.g1.act.g1_arm_sdk_connection import G1ArmSdkConnection
from dimos.robot.unitree.g1.act.g1_gripper_connection import G1GripperConnection
from dimos.robot.unitree.g1.camera.teleimager_camera_module import (
    RightWristTeleimagerCamera,
    TeleimagerCamera,
)

_NIC = os.getenv("ROBOT_INTERFACE", "")

# Dataset (sotata/okura-pick-tree-20260615) recorded first-frame arm pose [rad],
# left 7 + right 7. The arms slew here on startup so the policy begins
# in-distribution (mirrors eval_g1.py); ActBridge then waits _START_DELAY_S.
_INIT_ARM_POSE = [
    0.269,
    0.196,
    -0.018,
    0.986,
    0.122,
    0.028,
    0.003,  # left arm
    -0.114,
    0.029,
    0.185,
    0.538,
    0.209,
    -0.755,
    0.370,  # right arm
]
_START_DELAY_S = 2.5

unitree_g1_act_arm_tree = (
    autoconnect(
        TeleimagerCamera.blueprint(camera="head"),
        RightWristTeleimagerCamera.blueprint(camera="right_wrist"),
        G1ArmSdkConnection.blueprint(network_interface=_NIC, initial_arm_pose=_INIT_ARM_POSE),
        G1GripperConnection.blueprint(network_interface=_NIC),
        ActBridge.blueprint(dry_run=False, startup_delay_s=_START_DELAY_S),
    )
    .remappings(
        [
            # The wrist instance also publishes color_image; rename it so it
            # feeds ActBridge.cam_right_wrist and not the head input.
            (RightWristTeleimagerCamera, "color_image", "cam_right_wrist"),
        ]
    )
    .transports(
        {
            ("color_image", Image): LCMTransport("/color_image", Image),
            ("cam_right_wrist", Image): LCMTransport("/cam_right_wrist", Image),
            ("motor_states", JointState): LCMTransport("/g1/motor_states", JointState),
            ("arm_target", JointState): LCMTransport("/g1/arm_target", JointState),
            ("right_gripper_state", JointState): LCMTransport(
                "/g1/right_gripper_state", JointState
            ),
            ("gripper_target", JointState): LCMTransport("/g1/gripper_target", JointState),
        }
    )
)

__all__ = ["unitree_g1_act_arm_tree"]
