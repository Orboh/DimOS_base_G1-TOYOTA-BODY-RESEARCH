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

"""LAPTOP app: human-clicked okra -> IK pre-grasp -> (auto) ACT grasp. NO MCP.

First-stage IK->ACT pipeline (no DimOS skills / MCP / agent). A human clicks the
okra in the Rerun point cloud; ``IkReachBridge`` solves a one-shot right-arm reach
to a pre-grasp pose and, once the arm has SETTLED there, fires an internal
``reach_done`` (std_msgs/Bool). ``ActBridge`` (``trigger_mode=True``) idles until
that trigger, then runs the okra ACT grasp policy for ``grasp_duration_s`` and
stops. Both publish ``arm_target`` into the SAME ``G1ArmSdkConnection`` (250 Hz
clip-to-measured / weight ramp); they are strictly sequential so last-writer-wins
is safe. The left arm is HELD (basket hand): ActBridge drives only right arm +
right Dex1.

Runs ON THE LAPTOP, joined to the robot DDS subnet via wired ``enp46s0``. The
D435i point cloud + color image arrive over LCM from the Jetson
``unitree-g1-ik-camera`` standalone publisher (SSH-launched out-of-band; the
single head D435i cannot also serve teleimager). ACT therefore consumes the
ik-camera ``/camera/color_image`` stream -- the okra policy must be (re)trained on
that stream for in-distribution grasping (see plan "condition D").

Autoconnect (by stream name; reach_done/arm_target/etc. cross worker processes via
explicit LCMTransport):
    RerunWebSocketServer.clicked_point ─▶ IkReachBridge.clicked_point
    G1ArmSdkConnection.motor_states ───▶ IkReachBridge.motor_states / ActBridge.motor_states
    IkReachBridge.arm_target ──────────▶ G1ArmSdkConnection.arm_target ─▶ rt/arm_sdk
    IkReachBridge.reach_done ──────────▶ ActBridge.reach_done   (IK settled -> grasp)
    /camera/color_image (ik-cam, LCM) ─▶ ActBridge.color_image
    G1GripperConnection.right_gripper_state ▶ ActBridge.right_gripper_state
    ActBridge.arm_target ──────────────▶ G1ArmSdkConnection.arm_target ─▶ rt/arm_sdk
    ActBridge.gripper_target ──────────▶ G1GripperConnection.gripper_target ▶ rt/dex1/right/cmd

SAFETY (fail-safe by default):
- DRY-RUN unless ``IK_REACH_LIVE=1``: IkReachBridge logs the target (fires
  reach_done immediately, no settle wait, since the arm is not driven), ActBridge
  logs actions but publishes nothing, G1ArmSdkConnection does not write rt/arm_sdk.
- LIVE runs require a physical e-stop in hand and clear space around the arms.
- The ACT inference service (scripts/act_service.py --serve, ZMQ :5701) must be
  running; the launcher start_okra_harvest.sh starts it.

Run (recommended): bash dimos/robot/unitree/g1/examples/start_okra_harvest.sh [--live]
"""

from __future__ import annotations

import os

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.Bool import Bool
from dimos.robot.unitree.g1.act.act_bridge import ActBridge
from dimos.robot.unitree.g1.act.g1_arm_sdk_connection import G1ArmSdkConnection
from dimos.robot.unitree.g1.act.g1_gripper_connection import G1GripperConnection
from dimos.robot.unitree.g1.act.ik_reach_bridge import IkReachBridge
from dimos.utils.logging_config import setup_logger
from dimos.visualization.vis_module import vis_module

logger = setup_logger()

# DDS NIC for rt/arm_sdk / rt/lowstate / rt/dex1. Must be the wired robot-subnet
# interface (an empty NIC silently auto-selects WiFi, which is NOT the robot LAN).
_NIC = os.getenv("ROBOT_INTERFACE", "enp46s0")

# Fail-safe live gate: explicit opt-in only; default is dry-run.
_LIVE = os.getenv("IK_REACH_LIVE", "").strip() == "1"

# Reach slew rate [rad/s]. G1ArmSdkConnection clips per-cycle to measured +
# (target-measured)*vel_limit*dt; the reference 20.0 is required on real hw
# (vel_limit=2.0 gave 0 motion, 2026-06-19). Do NOT lower for "safety".
_ARM_VEL_LIMIT = float(os.getenv("IK_ARM_VEL_LIMIT", "20.0"))

# How long ACT drives the grasp after reach_done [s] (fixed-duration stop).
_GRASP_DURATION_S = float(os.getenv("OKRA_GRASP_DURATION_S", "8.0"))

# IK->ACT handoff. Default ON. Set OKRA_ACT_HANDOFF=0 to do the IK reach and HOLD the
# pre-grasp without ACT (no reach_done) — to inspect the reach/standoff in isolation.
_ACT_HANDOFF = os.getenv("OKRA_ACT_HANDOFF", "1").strip() != "0"

# Vertical (torso Z) target compensation [m] — EMPIRICAL correction for a constant
# DOWNWARD Z bias of the executed tip vs the commanded pose. Hand-eye calibration
# (scripts/handeye_calib.py, 2026-06-23) showed the camera Z is accurate (Δz≈-0.2cm,
# fit RMS≈nominal), so the ~5cm vertical miss is on the ARM side (commanded-vs-executed:
# arm_sdk droop AND/OR an FK / encoder-zero bias — indistinguishable in that experiment,
# not separately diagnosed). Raising the torso target Z by ~0.05 makes the real tip land
# on the okra (5cm -> ~1cm on real hw). Pose-dependent, so not exact. Override live via
# OKRA_TARGET_Z_OFFSET.
_TARGET_Z_OFFSET = float(os.getenv("OKRA_TARGET_Z_OFFSET", "0.05"))

# Hand-eye calibration: log the MEASURED gripper tip in torso every N motor_states
# (0 = off). Set e.g. OKRA_TIP_LOG=25 (~2 Hz at 50 Hz state) to read P_arm for the
# camera->torso extrinsic check (compare to the clicked tip's CALIB torso = P_cam).
_TIP_LOG_EVERY_N = int(os.getenv("OKRA_TIP_LOG", "0"))


def _camera_info_overlay(ci):  # type: ignore[no-untyped-def]
    """RerunBridge visual_override: log the camera Pinhole at the COLOR IMAGE
    entity so the RGB frame renders co-located with the cloud (one Spatial3D
    view). Module-level (not a lambda) so the blueprint config stays picklable
    for the forkserver worker. See unitree_g1_ik_reach for the full rationale."""
    return ci.to_rerun(image_topic="world/camera/color_image")


def _pointcloud_rgb_overlay(pc):  # type: ignore[no-untyped-def]
    """RerunBridge visual_override: render the cloud with its TRUE per-point
    camera RGB (so the okra shows in real color, easy to click) instead of the
    default height colormap. Falls back to default if a cloud arrives without
    colors. Module-level (not a lambda) for picklability."""
    import os

    import numpy as np
    import rerun as rr

    points, colors = pc.as_numpy()
    if colors is None or len(points) == 0:
        return pc.to_rerun()
    rgb = (np.asarray(colors) * 255.0).clip(0, 255).astype(np.uint8)
    radius = float(os.getenv("IK_PC_POINT_RADIUS", "0.0015"))
    return rr.Points3D(positions=np.asarray(points), colors=rgb, radii=radius)


_HANDOFF_MSG = (
    f"ACT handoff ON (grasp {_GRASP_DURATION_S}s)" if _ACT_HANDOFF
    else "ACT handoff OFF (OKRA_ACT_HANDOFF=0): IK reaches pre-grasp and HOLDS, ACT does not start"
)
_HANDOFF_MSG += f" | target Z offset = {_TARGET_Z_OFFSET:+.3f} m (OKRA_TARGET_Z_OFFSET)"
if _LIVE:
    logger.warning(
        f"unitree-g1-okra-harvest LAUNCHING **LIVE** — arm WILL move via rt/arm_sdk"
        f"{' and the right Dex1 WILL close' if _ACT_HANDOFF else ''}, on NIC {_NIC!r} at "
        f"<= {_ARM_VEL_LIMIT} rad/s. {_HANDOFF_MSG}. Keep an e-stop in hand."
    )
else:
    logger.info(
        f"unitree-g1-okra-harvest DRY-RUN (set IK_REACH_LIVE=1 to drive arm+gripper). "
        f"NIC={_NIC!r}. {_HANDOFF_MSG}."
    )

unitree_g1_okra_harvest = autoconnect(
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
        fire_reach_done=_ACT_HANDOFF,  # OKRA_ACT_HANDOFF=0 -> hold pre-grasp, no ACT
        approach_offset_xyz=[0.0, 0.0, _TARGET_Z_OFFSET],  # raise torso target Z (OKRA_TARGET_Z_OFFSET)
        tip_log_every_n=_TIP_LOG_EVERY_N,  # OKRA_TIP_LOG>0 -> log measured tip(torso) for hand-eye
    ),
    ActBridge.blueprint(
        dry_run=not _LIVE,
        trigger_mode=True,        # idle until reach_done; IK owns the pre-grasp
        startup_delay_s=0.0,      # arm is already at the IK pre-grasp (no slew)
        grasp_duration_s=_GRASP_DURATION_S,
        right_only=True,          # 8-DoF tree-right model: cam_high + cam_right_wrist, right arm+grip
    ),
    G1ArmSdkConnection.blueprint(
        network_interface=_NIC,
        arm_velocity_limit=_ARM_VEL_LIMIT,
        publish_cmd=_LIVE,
        # NOTE: no initial_arm_pose — ACT starts from the IK pre-grasp, not the
        # dataset first-frame pose (the IK reach provides the in-distribution start).
    ),
    G1GripperConnection.blueprint(network_interface=_NIC),
).transports(
    {
        # ik-camera color stream (NOT teleimager /color_image): the single head
        # D435i is owned by the standalone cloud publisher. This is ACT's cam_high.
        ("color_image", Image): LCMTransport("/camera/color_image", Image),
        # wrist UVC color (standalone V4L2 publisher on the NX) -> ACT cam_right_wrist.
        ("right_wrist_image", Image): LCMTransport("/camera/right_wrist_color", Image),
        ("motor_states", JointState): LCMTransport("/g1/motor_states", JointState),
        ("arm_target", JointState): LCMTransport("/g1/arm_target", JointState),
        ("gripper_target", JointState): LCMTransport("/g1/gripper_target", JointState),
        ("right_gripper_state", JointState): LCMTransport("/g1/right_gripper_state", JointState),
        # explicit transport so the IK->ACT trigger crosses forkserver workers.
        ("reach_done", Bool): LCMTransport("/g1/reach_done", Bool),
    }
)

__all__ = ["unitree_g1_okra_harvest"]
