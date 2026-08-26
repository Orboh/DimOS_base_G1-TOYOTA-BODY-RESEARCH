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

"""G1 upper-body authority handover / measured-pose hold.

This is the only live entry point intended for the first physical deployment
gate.  It contains *no* target producer: after receiving ``rt/lowstate``,
``G1ArmSdkConnection`` holds the measured startup pose while its arm-SDK
authority weight ramps from zero to one.  Consequently it cannot command a
home pose, a learned-policy action, a click/IK target, or a gripper command.

By default this is dry-run and never writes ``rt/arm_sdk``.  Live output is an
explicit, two-part opt-in:

    ROBOT_INTERFACE=<wired-NIC> G1_ARM_HOLD_LIVE=1 \
        .venv/bin/dimos run unitree-g1-arm-hold

Before live use: robot in motion-control/self-balancing mode, area clear, and
a separate operator holding the physical e-stop.  Ctrl-C once and wait for the
weight ramp-down to complete.  Do not run this together with any other
``rt/arm_sdk`` publisher.
"""

from __future__ import annotations

import os

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.unitree.g1.act.g1_arm_sdk_connection import G1ArmSdkConnection
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_NIC = os.getenv("ROBOT_INTERFACE", "").strip()
_LIVE_REQUESTED = os.getenv("G1_ARM_HOLD_LIVE", "").strip() == "1"

# Never let an omitted NIC fall back to the Wi-Fi/default route.  Failing before
# the module opens its DDS publisher is safer than attempting discovery on an
# arbitrary interface.
if _LIVE_REQUESTED and not _NIC:
    raise RuntimeError(
        "G1_ARM_HOLD_LIVE=1 requires ROBOT_INTERFACE to name the wired robot NIC"
    )

if _LIVE_REQUESTED:
    logger.warning(
        "unitree-g1-arm-hold LIVE: measured-pose hold only; no arm target or "
        "gripper target exists. E-stop must be in hand and the area clear."
    )
else:
    logger.info(
        "unitree-g1-arm-hold DRY-RUN: no rt/arm_sdk write. Set both "
        "ROBOT_INTERFACE=<wired-NIC> and G1_ARM_HOLD_LIVE=1 for the explicit "
        "measured-pose hold gate."
    )

unitree_g1_arm_hold = autoconnect(
    G1ArmSdkConnection.blueprint(
        network_interface=_NIC,
        # No arm_target producer is present, so this is belt-and-suspenders:
        # even a future accidental target connection cannot move the arm through
        # this commissioning blueprint.
        arm_velocity_limit=0.0,
        initial_arm_pose=[],
        weight_ramp_s=5.0,
        publish_cmd=_LIVE_REQUESTED,
        enable_disconnect=False,
        collection_mode=False,
    ),
)

__all__ = ["unitree_g1_arm_hold"]
