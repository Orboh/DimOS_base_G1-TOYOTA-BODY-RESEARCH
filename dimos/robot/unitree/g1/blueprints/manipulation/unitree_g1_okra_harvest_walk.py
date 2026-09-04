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

"""Blueprint: okra-harvest with REAL base walking (Step 1 of walking integration).

    dimos run unitree-g1-okra-harvest-walk

Wires ``G1HighLevelDdsSdk`` (which calls ``MotionSwitcherClient.SelectMode('ai')``
before issuing LocoClient velocity commands) into the harvest flow.

REAL now: head-camera YOLO detect + base reposition/sweep walking (cmd_vel).
ARM is OFF to avoid the DDS double-init collision between G1HighLevelDdsSdk and
G1ArmSdkConnection. G1 speaker is also OFF for the same reason. Verify =
moondream Ollama (~1 s caption+keyword).

⚠️ THE ROBOT WALKS. Keep an e-stop in hand and space around the robot.
  The SafetyMonitor file e-stop: ``touch /tmp/okra_estop`` to pause, ``rm`` to resume.

Prereqs:
  # NX:     teleimager-server --rs
  # Jetson: ollama serve  (moondream already pulled)
  # laptop: ROBOT_INTERFACE=<nic> dimos run unitree-g1-okra-harvest-walk
"""

from __future__ import annotations

import os

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.Image import Image
from dimos.robot.unitree.g1.camera.teleimager_camera_module import TeleimagerCamera
from dimos.robot.unitree.g1.effectors.high_level.dds_sdk import G1HighLevelDdsSdk
from dimos.robot.unitree.g1.harvest.harvest_module import HarvestModule

_NIC = os.getenv("ROBOT_INTERFACE", "")

unitree_g1_okra_harvest_walk = autoconnect(
    TeleimagerCamera.blueprint(camera="head"),
    G1HighLevelDdsSdk.blueprint(network_interface=_NIC),
    HarvestModule.blueprint(
        use_dummy=False,
        use_base_move=True,  # ⚠️ real walking via cmd_vel -> G1HighLevelDdsSdk
        use_act_grasp=False,  # arm OFF — avoids DDS double-init
        use_g1_speaker=False,  # speaker OFF — avoids DDS double-init
        vlm_model="moondream",  # Jetson Ollama verify (~1 s)
    ),
).transports(
    {
        ("color_image", Image): LCMTransport("/color_image", Image),
        ("cmd_vel", Twist): LCMTransport("/cmd_vel", Twist),
    }
)

__all__ = ["unitree_g1_okra_harvest_walk"]
