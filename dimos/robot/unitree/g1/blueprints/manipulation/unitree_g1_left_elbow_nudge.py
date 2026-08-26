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

"""G1 left-elbow physical commissioning gate, dry by default.

This is a purpose-specific successor to the completed wrist-roll gate.  It
waits 6.5 seconds while the measured-pose hold settles, then offsets only the
left elbow by +0.020 rad (about 1.15 degrees) at 0.010 rad/s.  It makes no
target change to the other arm joints, waist, legs, or grippers.

LIVE output requires both explicit opt-ins::

    ROBOT_INTERFACE=<wired-NIC> G1_LEFT_ELBOW_NUDGE_LIVE=1 \
        .venv/bin/dimos run unitree-g1-left-elbow-nudge
"""

from __future__ import annotations

import os

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.unitree.g1.act.g1_arm_sdk_connection import G1ArmSdkConnection
from dimos.robot.unitree.g1.act.g1_single_joint_nudge import G1SingleJointNudge
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_NIC = os.getenv("ROBOT_INTERFACE", "").strip()
_LIVE_REQUESTED = os.getenv("G1_LEFT_ELBOW_NUDGE_LIVE", "").strip() == "1"

if _LIVE_REQUESTED and not _NIC:
    raise RuntimeError(
        "G1_LEFT_ELBOW_NUDGE_LIVE=1 requires ROBOT_INTERFACE to name the wired robot NIC"
    )

if _LIVE_REQUESTED:
    logger.warning(
        "unitree-g1-left-elbow-nudge LIVE: left elbow only (+0.020 rad at "
        "0.010 rad/s). Physical STOP must be in hand and the area clear."
    )
else:
    logger.info(
        "unitree-g1-left-elbow-nudge DRY-RUN: no rt/arm_sdk write. Set both "
        "ROBOT_INTERFACE=<wired-NIC> and G1_LEFT_ELBOW_NUDGE_LIVE=1 for LIVE."
    )

unitree_g1_left_elbow_nudge = autoconnect(
    G1ArmSdkConnection.blueprint(
        network_interface=_NIC,
        arm_velocity_limit=20.0,
        initial_arm_pose=[],
        weight_ramp_s=5.0,
        publish_cmd=_LIVE_REQUESTED,
        enable_disconnect=False,
        collection_mode=False,
    ),
    G1SingleJointNudge.blueprint(joint_index=3, joint_name="LeftElbow"),
).transports(
    {
        ("motor_states", JointState): LCMTransport("/g1/motor_states", JointState),
        ("arm_target", JointState): LCMTransport("/g1/arm_target", JointState),
    }
)

__all__ = ["unitree_g1_left_elbow_nudge"]
