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

"""LAPTOP app: CHEST ZED Mini -> human click -> IK pre-grasp -> scripted close. NO ACT.

The chest-ZED sibling of ``unitree_g1_okra_ik_only_grasp.py`` (same directory).
Two deltas from that file, everything else identical:

1. Camera source: the chest-mounted ZED Mini plugged into THIS laptop runs
   IN-PROCESS (``ZEDCamera``), replacing the head D435i relayed from the Jetson
   (``unitree-g1-ik-camera`` / ``ik_camera_standalone.py``). Same LCM topic
   contract (``/camera/pointcloud`` etc.), so the click UI is unchanged.
2. Click transform: ``IkReachBridge.camera_mount_xyzrpy`` overrides the head-
   D435i URDF constant with the chest-ZED mount measured 2026-07-16:
       xyz = [0.109, 0.030, 0.248]  (torso_link -> ZED LEFT lens [m];
             tape-measured from the shoulder-pitch axis, +-1cm)
       rpy = [0.0, -0.0209, 0.0]    (IMU-measured pitch -1.2deg = slightly
             nose-UP; the mount is ~level, NOT the sim's assumed 19.3deg down)
   This is a PLACEHOLDER pending hand-eye calibration (design doc SS-04:
   ``T_base_camera`` — same status as the AGX Orin port spec's blocker). Good
   enough for DRY-RUN and rough reaches; expect a few cm of error LIVE.

SAFETY: identical fail-safe gating to the D435i version — DRY-RUN unless
``IK_REACH_LIVE=1``; the gripper additionally needs ``OKRA_NOACT_GRIP_LIVE=1``.

Run: dimos run unitree-g1-okra-ik-only-grasp-zed   (env vars below)
     # viewer: dimos-viewer --connect rerun+http://127.0.0.1:9877/proxy \
     #                      --ws-url ws://127.0.0.1:3030/ws
No Jetson step needed (ZED is laptop-local); lo multicast + rmem prep still
required, same as every LCM app (see oda/start_okra_ik_only_grasp.sh [2/4]).
"""

from __future__ import annotations

import os

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.hardware.sensors.camera.zed.camera import ZEDCamera
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
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

# ---- robot side (identical to unitree_g1_okra_ik_only_grasp.py) -------------
_NIC = os.getenv("ROBOT_INTERFACE", "enp46s0")
_LIVE = os.getenv("IK_REACH_LIVE", "").strip() == "1"
_ARM_VEL_LIMIT = float(os.getenv("IK_ARM_VEL_LIMIT", "20.0"))
_KP_ARM = float(os.getenv("OKRA_NOACT_KP_ARM", "80.0"))
_KD_ARM = float(os.getenv("OKRA_NOACT_KD_ARM", "3.0"))
_TARGET_Z_OFFSET = float(os.getenv("OKRA_TARGET_Z_OFFSET", "0.05"))
_CUT_BELOW_CENTROID_M = float(os.getenv("OKRA_CUT_BELOW_CENTROID_M", "0.0"))
_STANDOFF_M = float(os.getenv("OKRA_NOACT_STANDOFF_M", "0.0"))
_CLOSE_Q = float(os.getenv("OKRA_NOACT_CLOSE_Q", "0.0"))
_DEBOUNCE_S = float(os.getenv("OKRA_NOACT_DEBOUNCE_S", "3.0"))
_GRIP_LIVE = os.getenv("OKRA_NOACT_GRIP_LIVE", "").strip() == "1"

# No physical Dex1 attached (e.g. hand removed for a reach-only test): skip the
# gripper modules entirely. Without this, G1GripperConnection fail-safe-refuses
# to start (no rt/dex1/right/state) and tears the whole app down. Reach-only:
# IkReachBridge still fires reach_done, it just has no consumer.
_NO_GRIPPER = os.getenv("OKRA_NO_GRIPPER", "").strip() == "1"

# Dex1 DDS topic prefix. This rig's PHYSICALLY-RIGHT-mounted Dex1 enumerates
# under the LEFT service (data cable in the left-hand port; only
# rt/dex1/left/state publishes — confirmed 2026-07-16), so the default here is
# the LEFT prefix. Set OKRA_DEX1_PREFIX=rt/dex1/right if the cabling is ever
# moved to the proper right port.
_DEX1_PREFIX = os.getenv("OKRA_DEX1_PREFIX", "rt/dex1/left").strip()

# Gripper position-servo gains. Default 5.0/0.05 = the proven soft grip
# (compliant: jaws stop where the okra resists, gentle hold). Grip FORCE at
# stall ≈ kp × position error, so raise kp for a firmer squeeze (e.g. the
# 2026-07-16 "はさみ切りたい" request — close harder onto the okra); the future
# cutter attachment will need this same knob. Raise gradually and watch the
# jaws/motor — there is no force sensor in this loop.
_GRIP_KP = float(os.getenv("OKRA_GRIP_KP", "5.0"))
_GRIP_KD = float(os.getenv("OKRA_GRIP_KD", "0.05"))

# ---- chest-ZED camera (same knobs as unitree_g1_zed_ik_view.py) -------------
_ZED_SERIAL = os.getenv("ZED_SERIAL", "").strip() or None
_PC_FPS = float(os.getenv("ZED_PC_FPS", "3.0"))
_DEPTH_MODE = os.getenv("ZED_DEPTH_MODE", "NEURAL")
_CAM_FPS = int(os.getenv("ZED_FPS", "15"))
# Cloud density: same tuning the D435i pipeline already converged on
# (ik_camera_standalone.py IK_CAMERA_VOXEL/IK_CAMERA_DEPTH_TRUNC): 2mm voxel
# ("dense enough to read the okra") + 0.8m depth truncation (cut the far
# background so the point budget stays on the workspace).
_PC_VOXEL = float(os.getenv("ZED_PC_VOXEL", "0.002"))
_DEPTH_TRUNC = float(os.getenv("ZED_DEPTH_TRUNC", "0.8"))

# Chest-ZED mount, torso_link <- ZED LEFT lens (the point-cloud origin), as
# "x,y,z,roll,pitch,yaw" [m]/[rad]. Default = 2026-07-16 measurement (see module
# docstring). PLACEHOLDER until hand-eye calibration; override via env to tune.
_ZED_MOUNT = [
    float(v)
    for v in os.getenv("ZED_MOUNT_XYZRPY", "0.109,0.030,0.248,0.0,-0.0209,0.0").split(",")
]


def _camera_info_overlay(ci):  # type: ignore[no-untyped-def]
    """RerunBridge visual_override: log the camera Pinhole at the COLOR IMAGE
    entity so the RGB frame renders co-located with the cloud (one Spatial3D
    view). Module-level (not a lambda) so the blueprint config stays picklable
    for the forkserver worker. Duplicated from unitree_g1_okra_ik_only_grasp.py
    (a private, non-exported helper in that sibling file) rather than shared."""
    return ci.to_rerun(image_topic="world/camera/color_image")


def _pointcloud_rgb_overlay(pc):  # type: ignore[no-untyped-def]
    """RerunBridge visual_override: render the cloud with its TRUE per-point
    camera RGB (so the okra shows in real color, easy to click) instead of the
    default height colormap. Falls back to default if a cloud arrives without
    colors. Duplicated from unitree_g1_okra_ik_only_grasp.py for the same
    reason as _camera_info_overlay above."""
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
    (
        "REACH-ONLY (OKRA_NO_GRIPPER=1): no Dex1 attached, gripper modules skipped "
        if _NO_GRIPPER
        else f"NO ACT -- scripted gripper close_q={_CLOSE_Q:.3f} (OKRA_NOACT_CLOSE_Q, UNTUNED) "
        f"on reach_done, debounce={_DEBOUNCE_S}s "
    )
    + f"| target Z offset = {_TARGET_Z_OFFSET:+.3f} m "
    f"| standoff = {_STANDOFF_M:.3f} m | cut point = {_CUT_BELOW_CENTROID_M:.3f} m below centroid "
    f"| kp_arm={_KP_ARM:.1f} kd_arm={_KD_ARM:.1f} "
    f"| CHEST ZED mount xyzrpy={_ZED_MOUNT} (ZED_MOUNT_XYZRPY, UNCALIBRATED tape+IMU value)"
)
if _LIVE and _GRIP_LIVE:
    logger.warning(
        f"unitree_g1_okra_ik_only_grasp_zed LAUNCHING **LIVE** -- arm WILL move via rt/arm_sdk "
        f"and the right Dex1 WILL close, on NIC {_NIC!r} at <= {_ARM_VEL_LIMIT} rad/s. "
        f"{_HANDOFF_MSG}. Keep an e-stop in hand."
    )
elif _LIVE:
    logger.warning(
        f"unitree_g1_okra_ik_only_grasp_zed LAUNCHING **LIVE (arm only)** -- arm WILL move via "
        f"rt/arm_sdk on NIC {_NIC!r} at <= {_ARM_VEL_LIMIT} rad/s, but the GRIPPER stays "
        f"DRY-RUN (set OKRA_NOACT_GRIP_LIVE=1 to also close it). {_HANDOFF_MSG}. "
        f"Keep an e-stop in hand."
    )
else:
    logger.info(
        f"unitree_g1_okra_ik_only_grasp_zed DRY-RUN (set IK_REACH_LIVE=1 to drive arm+gripper). "
        f"NIC={_NIC!r}. {_HANDOFF_MSG}."
    )

_MODULES = [
    vis_module(
        "rerun",
        rerun_config={
            "visual_override": {
                "world/camera/camera_info": _camera_info_overlay,
                "world/camera/pointcloud": _pointcloud_rgb_overlay,
            },
        },
    ),
    # Chest ZED runs IN-PROCESS on the laptop (no Jetson relay). Same config as
    # the stage-1 viewer blueprint (unitree-g1-zed-ik-view), which verified the
    # cloud renders + clicks on real hw 2026-07-16.
    ZEDCamera.blueprint(
        serial_number=_ZED_SERIAL,
        fps=_CAM_FPS,
        enable_depth=True,
        enable_pointcloud=True,
        align_depth_to_color=True,
        pointcloud_fps=_PC_FPS,
        pointcloud_voxel=_PC_VOXEL,
        pointcloud_depth_trunc=_DEPTH_TRUNC,
        camera_info_fps=1.0,
        depth_mode=_DEPTH_MODE,
        enable_tracking=False,  # rigid chest mount: no positional tracking needed
    ),
    IkReachBridge.blueprint(
        log_only=not _LIVE,
        expected_click_frame="/world/camera/pointcloud",  # same entity path as D435i app
        fire_reach_done=True,
        approach_offset_xyz=[0.0, 0.0, _TARGET_Z_OFFSET - _CUT_BELOW_CENTROID_M],
        standoff_m=_STANDOFF_M,
        # Chest-ZED mount override (torso <- camera body).
        camera_mount_xyzrpy=_ZED_MOUNT,
        # In-process ZEDCamera publishes TF, so the viewer resolves clicks into the
        # camera BODY frame (optical rotation already applied) — unlike the D435i
        # Jetson publisher (no TF, raw optical clicks). Verified 2026-07-16: a point
        # 40cm in front of the ZED clicked as [0.374,-0.006,0.002] (X-fwd, not Z-fwd).
        click_in_camera_body_frame=True,
    ),
    G1ArmSdkConnection.blueprint(
        network_interface=_NIC,
        arm_velocity_limit=_ARM_VEL_LIMIT,
        publish_cmd=_LIVE,
        kp_arm=_KP_ARM,
        kd_arm=_KD_ARM,
        enable_disconnect=True,
    ),
]
if not _NO_GRIPPER:
    _MODULES += [
        GripperGraspOnReach.blueprint(
            close_q=_CLOSE_Q,
            debounce_s=_DEBOUNCE_S,
            dry_run=not (_LIVE and _GRIP_LIVE),
        ),
        G1GripperConnection.blueprint(
            network_interface=_NIC,
            dex1_topic_prefix=_DEX1_PREFIX,
            kp=_GRIP_KP,
            kd=_GRIP_KD,
        ),
    ]

unitree_g1_okra_ik_only_grasp_zed = autoconnect(*_MODULES).transports(
    {
        # camera -> viewer (same contract as unitree-g1-ik-camera / -zed-ik-view)
        ("pointcloud", PointCloud2): LCMTransport("/camera/pointcloud", PointCloud2),
        ("color_image", Image): LCMTransport("/camera/color_image", Image),
        ("camera_info", CameraInfo): LCMTransport("/camera/camera_info", CameraInfo),
        # robot-side topics (identical to unitree_g1_okra_ik_only_grasp.py)
        ("motor_states", JointState): LCMTransport("/g1/motor_states", JointState),
        ("arm_target", JointState): LCMTransport("/g1/arm_target", JointState),
        ("gripper_target", JointState): LCMTransport("/g1/gripper_target", JointState),
        ("reach_done", Bool): LCMTransport("/g1/reach_done", Bool),
        ("okra_target", PointStamped): LCMTransport("/g1/okra_target", PointStamped),
        ("disconnect", Bool): LCMTransport("/g1/arm_sdk_disconnect", Bool),
    }
)

__all__ = ["unitree_g1_okra_ik_only_grasp_zed"]
