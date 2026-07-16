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

"""LAPTOP app: human-clicked okra -> IK pre-grasp -> scripted gripper close. NO ACT.

Registered in ``dimos/robot/all_blueprints.py`` (auto-generated -- see
``dimos/robot/test_all_blueprints_generation.py``) as
``unitree-g1-okra-ik-only-grasp``. Run it with ``dimos run
unitree-g1-okra-ik-only-grasp`` (set ``IK_REACH_LIVE=1`` for LIVE).
``oda/start_okra_ik_only_grasp.sh`` still handles the Jetson camera kick +
laptop network prep + viewer, ending in ``dimos run`` instead of a standalone
launcher.

This is the ACT-free sibling of ``dimos/robot/unitree/g1/blueprints/manipulation/
unitree_g1_okra_harvest.py``: same click -> IK pre-grasp reach, but instead of
handing ``reach_done`` to an ACT policy (``ActBridge``), it hands it to
``GripperGraspOnReach`` (``gripper_grasp_on_reach.py``, same directory), which
just closes the gripper to a fixed, scripted ``close_q`` and holds -- no
learned policy, no camera-fed inference. Everything else (IkReachBridge,
G1ArmSdkConnection, G1GripperConnection, the click UI) is reused unmodified.

Autoconnect (by stream name; reach_done/arm_target/etc. cross worker processes
via explicit LCMTransport, same topic names as unitree_g1_okra_harvest.py so
this is plug-compatible with the existing Jetson camera publisher / robot DDS
setup with zero changes on that side):
    RerunWebSocketServer.clicked_point ─▶ IkReachBridge.clicked_point
    G1ArmSdkConnection.motor_states ───▶ IkReachBridge.motor_states
    IkReachBridge.arm_target ──────────▶ G1ArmSdkConnection.arm_target ─▶ rt/arm_sdk
    IkReachBridge.reach_done ──────────▶ GripperGraspOnReach.reach_done (IK settled -> close)
    GripperGraspOnReach.gripper_target ▶ G1GripperConnection.gripper_target ▶ rt/dex1/right/cmd

SAFETY (fail-safe by default, same convention as the rest of dimos/):
- DRY-RUN unless ``IK_REACH_LIVE=1``: IkReachBridge logs the target,
  GripperGraspOnReach logs the would-be close, G1ArmSdkConnection does not
  write rt/arm_sdk. Absence of the env var is NEVER interpreted as live.
- ``close_q`` (OKRA_NOACT_CLOSE_Q) has NO known-good default -- it is a raw,
  untuned Dex1 motor position. Tune on hardware before a LIVE grasp attempt.
- LIVE runs require a physical e-stop in hand and clear space around the arms.

Run (recommended): bash oda/start_okra_ik_only_grasp.sh [--live]
Run (manual): dimos run unitree-g1-okra-ik-only-grasp   (env vars below)
"""

from __future__ import annotations

import os

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.Bool import Bool
from dimos.robot.unitree.g1.act.g1_arm_sdk_connection import G1ArmSdkConnection
from dimos.robot.unitree.g1.act.g1_gripper_connection import G1GripperConnection
from dimos.robot.unitree.g1.act.ik_reach_bridge import IkReachBridge
from dimos.robot.unitree.g1.blueprints.manipulation.gripper_grasp_on_reach import (
    GripperGraspOnReach,
)
from dimos.utils.logging_config import setup_logger
from dimos.visualization.vis_module import vis_module

logger = setup_logger()

# DDS NIC for rt/arm_sdk / rt/lowstate / rt/dex1. Must be the wired robot-subnet
# interface (an empty NIC silently auto-selects WiFi, which is NOT the robot LAN).
_NIC = os.getenv("ROBOT_INTERFACE", "enp46s0")

# Fail-safe live gate: explicit opt-in only; default is dry-run.
_LIVE = os.getenv("IK_REACH_LIVE", "").strip() == "1"

# Reach slew rate [rad/s]. Unchanged from the proven value (unitree_g1_ik_reach.py /
# unitree_g1_okra_harvest.py): 2.0 gave 0 motion on real hw 2026-06-19, 20.0 works.
_ARM_VEL_LIMIT = float(os.getenv("IK_ARM_VEL_LIMIT", "20.0"))

# Shoulder/elbow position-tracking gains (G1ArmSdkConnectionConfig.kp_arm/kd_arm,
# g1_arm_sdk_connection.py:92-93). Defaults (80.0/3.0) are the "proven" value shared
# with unitree_g1_okra_harvest.py -- but that pipeline's IK only ever reached to a
# pre-grasp standoff_m=0.05 short of the okra; THIS demo's standoff=0.0 asks IK to
# hold a more EXTENDED pose (the actual centroid). LIVE test 2026-07-14: reach errors
# decayed from ~1 rad toward a nonzero residual (0.36-0.55 rad) and then fully
# plateaued for 15+ seconds -- closure rate (~0.02-0.09 rad/s) was two orders of
# magnitude below arm_velocity_limit, i.e. NOT a velocity-limit problem; looks like
# the arm settling into a gravity/torque equilibrium it can't push through at this
# gain. Test knob: raise both to see if the residual shrinks (confirms the
# stiffness/torque hypothesis). Defaults left UNCHANGED (80.0/3.0) so this override
# is opt-in only -- set both together to keep the damping ratio from drifting.
_KP_ARM = float(os.getenv("OKRA_NOACT_KP_ARM", "80.0"))
_KD_ARM = float(os.getenv("OKRA_NOACT_KD_ARM", "3.0"))

# Vertical (torso Z) target compensation [m] -- same empirical correction used by
# unitree_g1_okra_harvest.py (arm-side droop / FK bias, not a camera error).
_TARGET_Z_OFFSET = float(os.getenv("OKRA_TARGET_Z_OFFSET", "0.05"))

# Click = the okra's CENTROID, but the cut point (stem) sits below it. This is a
# separate, semantic offset (okra geometry) from _TARGET_Z_OFFSET above (hardware
# droop calibration) -- kept as two named knobs so tuning one never fights the
# other. Positive value = aim this far BELOW the clicked centroid; 0.0 = target
# the centroid itself (current default -- no known-good cut-point distance yet).
_CUT_BELOW_CENTROID_M = float(os.getenv("OKRA_CUT_BELOW_CENTROID_M", "0.0"))

# Pre-grasp standoff [m] (IkReachBridgeConfig.standoff_m, ik_reach_bridge.py:124).
# In the ACT harvest pipeline this defaults to 0.05 so IK stops 5cm short and ACT
# advances the rest. THERE IS NO ACT HERE, so that default must NOT be inherited --
# it would leave the scripted gripper close 5cm away from the okra. 0.0 = drive the
# tip exactly onto the clicked centroid (config's own definition of 0.0).
_STANDOFF_M = float(os.getenv("OKRA_NOACT_STANDOFF_M", "0.0"))

# Scripted (no-ACT) gripper close target -- a raw Dex1 q, NOT meters. See
# GripperGraspOnReachConfig.close_q docstring: NO known-good value, must be
# tuned on hardware. Larger q = more open (empirical fit in okra_harvest.py:
# q~2.5->~2.5cm, q~3.0->~3cm, q~4.0->~4.5cm), so closing means going smaller.
_CLOSE_Q = float(os.getenv("OKRA_NOACT_CLOSE_Q", "0.0"))

# Ignore a second reach_done within this many seconds of the last one, then
# re-arm so the operator can retry with a fresh click without restarting.
_DEBOUNCE_S = float(os.getenv("OKRA_NOACT_DEBOUNCE_S", "3.0"))

# Separate live-gate for the GRIPPER, independent of IK_REACH_LIVE (arm). close_q has
# NO known-good value (see above) -- default OFF so the first LIVE runs can verify the
# arm reaches the okra correctly with the gripper staying log-only, before trusting an
# untuned close on a real grasp. Requires BOTH IK_REACH_LIVE=1 AND this =1 to actually
# close the gripper; either alone leaves it dry-run (fail-safe, explicit opt-in only).
_GRIP_LIVE = os.getenv("OKRA_NOACT_GRIP_LIVE", "").strip() == "1"


def _camera_info_overlay(ci):  # type: ignore[no-untyped-def]
    """RerunBridge visual_override: log the camera Pinhole at the COLOR IMAGE
    entity so the RGB frame renders co-located with the cloud (one Spatial3D
    view). Module-level (not a lambda) so the blueprint config stays picklable
    for the forkserver worker. Duplicated from unitree_g1_okra_harvest.py (a
    private, non-exported helper in that sibling file) rather than shared."""
    return ci.to_rerun(image_topic="world/camera/color_image")


def _pointcloud_rgb_overlay(pc):  # type: ignore[no-untyped-def]
    """RerunBridge visual_override: render the cloud with its TRUE per-point
    camera RGB (so the okra shows in real color, easy to click) instead of the
    default height colormap. Falls back to default if a cloud arrives without
    colors. Duplicated from unitree_g1_okra_harvest.py for the same reason as
    _camera_info_overlay above."""
    import os as _os

    import numpy as np
    import rerun as rr

    points, colors = pc.as_numpy()
    if colors is None or len(points) == 0:
        return pc.to_rerun()
    rgb = (np.asarray(colors) * 255.0).clip(0, 255).astype(np.uint8)
    radius = float(_os.getenv("IK_PC_POINT_RADIUS", "0.0015"))
    return rr.Points3D(positions=np.asarray(points), colors=rgb, radii=radius)


_HANDOFF_MSG = (
    f"NO ACT -- scripted gripper close_q={_CLOSE_Q:.3f} (OKRA_NOACT_CLOSE_Q, UNTUNED) "
    f"on reach_done, debounce={_DEBOUNCE_S}s | target Z offset = {_TARGET_Z_OFFSET:+.3f} m "
    f"| standoff = {_STANDOFF_M:.3f} m (0.0 = reach the clicked centroid itself, no ACT to close the gap) "
    f"| cut point = {_CUT_BELOW_CENTROID_M:.3f} m below the clicked centroid (OKRA_CUT_BELOW_CENTROID_M) "
    f"| kp_arm={_KP_ARM:.1f} kd_arm={_KD_ARM:.1f} (proven default 80.0/3.0, OKRA_NOACT_KP_ARM/KD_ARM)"
)
if _LIVE and _GRIP_LIVE:
    logger.warning(
        f"unitree_g1_okra_ik_only_grasp LAUNCHING **LIVE** -- arm WILL move via rt/arm_sdk "
        f"and the right Dex1 WILL close, on NIC {_NIC!r} at <= {_ARM_VEL_LIMIT} rad/s. "
        f"{_HANDOFF_MSG}. Keep an e-stop in hand."
    )
elif _LIVE:
    logger.warning(
        f"unitree_g1_okra_ik_only_grasp LAUNCHING **LIVE (arm only)** -- arm WILL move via "
        f"rt/arm_sdk on NIC {_NIC!r} at <= {_ARM_VEL_LIMIT} rad/s, but the GRIPPER stays "
        f"DRY-RUN (set OKRA_NOACT_GRIP_LIVE=1 to also close it). {_HANDOFF_MSG}. "
        f"Keep an e-stop in hand."
    )
else:
    logger.info(
        f"unitree_g1_okra_ik_only_grasp DRY-RUN (set IK_REACH_LIVE=1 to drive arm+gripper). "
        f"NIC={_NIC!r}. {_HANDOFF_MSG}."
    )

unitree_g1_okra_ik_only_grasp = autoconnect(
    # Pin 'rerun' so RerunWebSocketServer (the clicked_point producer) is present.
    vis_module(
        "rerun",
        rerun_config={
            "visual_override": {
                "world/camera/camera_info": _camera_info_overlay,
                "world/camera/pointcloud": _pointcloud_rgb_overlay,
            },
        },
    ),
    IkReachBridge.blueprint(
        log_only=not _LIVE,
        expected_click_frame="/world/camera/pointcloud",  # R1-confirmed; required for LIVE
        fire_reach_done=True,  # the whole point of this app: always hand off (to us, not ACT)
        # +Z is up (approach_offset_xyz docstring): droop compensation UP, cut-point offset DOWN.
        approach_offset_xyz=[0.0, 0.0, _TARGET_Z_OFFSET - _CUT_BELOW_CENTROID_M],
        standoff_m=_STANDOFF_M,  # no ACT to close the default 5cm gap -- reach the centroid itself
    ),
    GripperGraspOnReach.blueprint(
        close_q=_CLOSE_Q,
        debounce_s=_DEBOUNCE_S,
        # Independent of arm liveness: needs IK_REACH_LIVE=1 AND OKRA_NOACT_GRIP_LIVE=1
        # to actually close (see _GRIP_LIVE above) -- lets the first LIVE runs verify the
        # reach alone before trusting the untuned close_q on a real grasp.
        dry_run=not (_LIVE and _GRIP_LIVE),
    ),
    G1ArmSdkConnection.blueprint(
        network_interface=_NIC,
        arm_velocity_limit=_ARM_VEL_LIMIT,
        publish_cmd=_LIVE,
        kp_arm=_KP_ARM,  # test knob for the standoff=0 stall -- see _KP_ARM comment above
        kd_arm=_KD_ARM,
        # 2-stage stop: 'd' in the key helper publishes /g1/arm_sdk_disconnect ->
        # ramp weight->0 (hand the arm back to the onboard controller), then 'q' quits.
        enable_disconnect=True,
    ),
    G1GripperConnection.blueprint(
        network_interface=_NIC,
        # No hold_target_q override: the gripper must obey our gripper_target.
    ),
).transports(
    {
        ("motor_states", JointState): LCMTransport("/g1/motor_states", JointState),
        ("arm_target", JointState): LCMTransport("/g1/arm_target", JointState),
        ("gripper_target", JointState): LCMTransport("/g1/gripper_target", JointState),
        ("reach_done", Bool): LCMTransport("/g1/reach_done", Bool),
        # accepted okra position (torso frame) -- logged for calibration/analysis.
        ("okra_target", PointStamped): LCMTransport("/g1/okra_target", PointStamped),
        # operator 2-stage stop: the key helper publishes here to cut G1 transmission.
        ("disconnect", Bool): LCMTransport("/g1/arm_sdk_disconnect", Bool),
    }
)

__all__ = ["unitree_g1_okra_ik_only_grasp"]
