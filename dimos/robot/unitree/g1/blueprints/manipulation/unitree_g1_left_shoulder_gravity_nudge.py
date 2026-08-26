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

"""Left-shoulder gravity-feedforward tracking gate, dry by default.

After a 6.5-second measured-pose hold, only LeftShoulderPitch is offset by
+0.010 rad at 0.005 rad/s.  The existing 10%-scale, +/-0.5 Nm bounded gravity
feedforward remains applied only to that same joint.  All other arm targets,
waist, legs, and grippers are unchanged.

LIVE requires an independent physical-safety confirmation::

    ROBOT_INTERFACE=<wired-NIC> G1_LEFT_SHOULDER_GRAVITY_NUDGE_LIVE=1 \\
        .venv/bin/dimos run unitree-g1-left-shoulder-gravity-nudge
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
_LIVE_REQUESTED = os.getenv("G1_LEFT_SHOULDER_GRAVITY_NUDGE_LIVE", "").strip() == "1"

if _LIVE_REQUESTED and not _NIC:
    raise RuntimeError(
        "G1_LEFT_SHOULDER_GRAVITY_NUDGE_LIVE=1 requires ROBOT_INTERFACE to name the wired robot NIC"
    )

if _LIVE_REQUESTED:
    logger.warning(
        "unitree-g1-left-shoulder-gravity-nudge LIVE: LeftShoulderPitch only (+0.010 rad "
        "at 0.005 rad/s) with 10%-scale gravity FF clipped to +/-0.5 Nm. Physical STOP required."
    )
else:
    logger.info(
        "unitree-g1-left-shoulder-gravity-nudge DRY-RUN: no rt/arm_sdk write. It calculates "
        "the bounded gravity FF and the individual tracking-error trace."
    )

unitree_g1_left_shoulder_gravity_nudge = autoconnect(
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
        stiff_gravity_left_joint_indices=[0],
        stiff_gravity_tau_scale=0.10,
        stiff_gravity_tau_limit_nm=0.50,
        stiff_gravity_ramp_s=5.0,
    ),
    G1SingleJointNudge.blueprint(
        joint_index=0,
        joint_name="LeftShoulderPitch",
        delta_rad=0.010,
        rate_rad_s=0.005,
    ),
).transports(
    {
        ("motor_states", JointState): LCMTransport("/g1/motor_states", JointState),
        ("arm_target", JointState): LCMTransport("/g1/arm_target", JointState),
    }
)

__all__ = ["unitree_g1_left_shoulder_gravity_nudge"]
