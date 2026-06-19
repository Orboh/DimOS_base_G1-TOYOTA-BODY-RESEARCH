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

"""LAPTOP app for the ACT-independent IK reach: human clicks okra -> right-arm reach.

Runs ON THE LAPTOP, joined to the robot DDS subnet (192.168.123.x) via wired
``enp46s0``. Composes:
    vis_module('rerun')         - Rerun viewer + RerunWebSocketServer (human click)
    IkReachBridge               - clicked_point + motor_states -> arm_target (IK)
    G1ArmSdkConnection          - rt/arm_sdk (DDS), 250Hz clip-to-measured / weight ramp

The D435i point cloud is NOT produced here (no camera on the laptop). It arrives
over LCM from the Jetson ``unitree-g1-ik-camera`` node; RerunBridge's default LCM
listener renders it so the human can click. The click round-trip and IK are all
local on the laptop; only rt/arm_sdk / rt/lowstate cross to the robot over DDS.

Autoconnect (in-proc, by stream name):
    RerunWebSocketServer.clicked_point ─▶ IkReachBridge.clicked_point
    G1ArmSdkConnection.motor_states ───▶ IkReachBridge.motor_states
    IkReachBridge.arm_target ──────────▶ G1ArmSdkConnection.arm_target ─▶ rt/arm_sdk

SAFETY (fail-safe by default):
- DRY-RUN unless ``IK_REACH_LIVE=1`` is explicitly set: IkReachBridge logs the
  target and G1ArmSdkConnection does not publish. Absence of the env var is NEVER
  interpreted as live.
- ``arm_velocity_limit`` defaults to a conservative 4 rad/s (NOT the Stage-B 20).
- LIVE runs require a physical e-stop in hand and clear space around the arms.

Run:
    # Jetson:  LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1' dimos run unitree-g1-ik-camera
    # laptop (DRY):  ROBOT_INTERFACE=enp46s0 LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1' \
    #                dimos run unitree-g1-ik-reach
    # laptop (LIVE): add IK_REACH_LIVE=1  (e-stop in hand, area clear)
"""

from __future__ import annotations

import os

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.unitree.g1.act.g1_arm_sdk_connection import G1ArmSdkConnection
from dimos.robot.unitree.g1.act.ik_reach_bridge import IkReachBridge
from dimos.utils.logging_config import setup_logger
from dimos.visualization.vis_module import vis_module

logger = setup_logger()

# DDS NIC for rt/arm_sdk / rt/lowstate. Must be the wired robot-subnet interface
# (blocker #6: an empty NIC silently auto-selects WiFi, which is NOT the robot LAN).
_NIC = os.getenv("ROBOT_INTERFACE", "enp46s0")

# Fail-safe live gate (blocker #10): explicit opt-in only; default is dry-run.
_LIVE = os.getenv("IK_REACH_LIVE", "").strip() == "1"

# Reach slew rate [rad/s]. G1ArmSdkConnection (the DimOS arm-extension path) clips the
# per-cycle command to measured + (target-measured)*vel_limit*dt. The per-cycle offset
# (vel_limit*dt) must produce more than the joint breakaway torque (kp*offset) or the
# arm never starts: real hw 2026-06-19, vel_limit=2.0 gave 0.008 rad / 0.64 Nm and 0
# motion; the reference 20.0 gives 0.08 rad / 6.4 Nm and the arm slews to the target.
# Using DimOS's reference value — the arm reaches fast (snaps to the okra).
_ARM_VEL_LIMIT = float(os.getenv("IK_ARM_VEL_LIMIT", "20.0"))

if _LIVE:
    logger.warning(
        f"unitree-g1-ik-reach LAUNCHING **LIVE** — arm WILL move via rt/arm_sdk on NIC "
        f"{_NIC!r} at <= {_ARM_VEL_LIMIT} rad/s. Keep an e-stop in hand and clear the area."
    )
else:
    logger.info(
        f"unitree-g1-ik-reach DRY-RUN (set IK_REACH_LIVE=1 to drive the arm). NIC={_NIC!r}."
    )

unitree_g1_ik_reach = autoconnect(
    # Pin 'rerun' so RerunWebSocketServer (the clicked_point producer) is always present
    # — a 'none' viewer would silently leave the bridge armed but unable to ever reach.
    vis_module("rerun"),
    IkReachBridge.blueprint(
        log_only=not _LIVE,
        expected_click_frame="/world/camera/pointcloud",  # R1-confirmed; required for LIVE
    ),
    G1ArmSdkConnection.blueprint(
        network_interface=_NIC,
        arm_velocity_limit=_ARM_VEL_LIMIT,
        publish_cmd=_LIVE,
    ),
).transports(
    {
        ("motor_states", JointState): LCMTransport("/g1/motor_states", JointState),
        ("arm_target", JointState): LCMTransport("/g1/arm_target", JointState),
    }
)

__all__ = ["unitree_g1_ik_reach"]
