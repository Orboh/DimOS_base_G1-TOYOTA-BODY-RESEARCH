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

"""Left-shoulder gravity-feedforward probe, dry by default.

No target producer exists: the arm target is captured from ``rt/lowstate`` and
therefore this gate requests no joint displacement.  It evaluates the reduced
left-arm URDF gravity estimate and logs the bounded torque that *would* be sent.

LIVE is intentionally a separate, future physical gate.  It still affects only
LeftShoulderPitch, starts at 10% of the model estimate, ramps over five seconds,
and clips the feedforward contribution to +/-0.5 Nm.
"""

from __future__ import annotations

import os

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.unitree.g1.act.g1_arm_sdk_connection import G1ArmSdkConnection
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_NIC = os.getenv("ROBOT_INTERFACE", "").strip()
_LIVE_REQUESTED = os.getenv("G1_LEFT_SHOULDER_GRAVITY_PROBE_LIVE", "").strip() == "1"

if _LIVE_REQUESTED and not _NIC:
    raise RuntimeError(
        "G1_LEFT_SHOULDER_GRAVITY_PROBE_LIVE=1 requires ROBOT_INTERFACE to name the wired robot NIC"
    )

if _LIVE_REQUESTED:
    logger.warning(
        "unitree-g1-left-shoulder-gravity-probe LIVE: measured-pose hold only; "
        "LeftShoulderPitch gravity FF is 10%, clipped to +/-0.5 Nm. Physical STOP must be in hand."
    )
else:
    logger.info(
        "unitree-g1-left-shoulder-gravity-probe DRY-RUN: calculating bounded gravity FF "
        "but never writing rt/arm_sdk."
    )

unitree_g1_left_shoulder_gravity_probe = autoconnect(
    G1ArmSdkConnection.blueprint(
        network_interface=_NIC,
        arm_velocity_limit=20.0,
        kp_arm=160.0,
        kd_arm=6.0,
        initial_arm_pose=[],
        weight_ramp_s=5.0,
        publish_cmd=_LIVE_REQUESTED,
        enable_disconnect=False,
        collection_mode=False,
        stiff_gravity_compensation_left=True,
        stiff_gravity_left_joint_indices=[0],  # LeftShoulderPitch only
        stiff_gravity_tau_scale=0.10,
        stiff_gravity_tau_limit_nm=0.50,
        stiff_gravity_ramp_s=5.0,
    ),
).transports(
    {
        # This isolated topic has no target producer.  A generic arm-target
        # publisher therefore cannot turn the gravity-hold probe into a motion
        # command accidentally.
        ("arm_target", JointState): LCMTransport("/g1/gravity_probe/arm_target", JointState),
        ("motor_states", JointState): LCMTransport("/g1/gravity_probe/motor_states", JointState),
    }
)

__all__ = ["unitree_g1_left_shoulder_gravity_probe"]
