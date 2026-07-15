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

"""AGX Orin app: human-clicked okra -> IK pre-grasp -> scripted gripper close. NO ACT.

Chest-ZED-Mini sibling of ``unitree_g1_okra_ik_only_grasp.py`` (the laptop +
head-D435i original, ported per ``AGX_ORIN_PORT_SPEC.md``). Runs standalone on
the AGX Orin backpack: ``ZEDCamera`` talks to the ZED Mini over direct USB3 (no
Jetson-NX relay, no laptop network prep) and feeds the same Rerun click-UI +
``IkReachBridge`` + ``GripperGraspOnReach`` pipeline as the original.

Registered in ``dimos/robot/all_blueprints.py`` as
``unitree-g1-okra-ik-only-grasp-zed``. Run with ``dimos run
unitree-g1-okra-ik-only-grasp-zed`` (set ``IK_REACH_LIVE=1`` for LIVE).

Same handoff as the original (see ``gripper_grasp_on_reach.py``, same
directory): IK reach -> scripted gripper close, no learned policy.
    RerunWebSocketServer.clicked_point ─▶ IkReachBridge.clicked_point
    G1ArmSdkConnection.motor_states ───▶ IkReachBridge.motor_states
    IkReachBridge.arm_target ──────────▶ G1ArmSdkConnection.arm_target ─▶ rt/arm_sdk
    IkReachBridge.reach_done ──────────▶ GripperGraspOnReach.reach_done (IK settled -> close)
    GripperGraspOnReach.gripper_target ▶ G1GripperConnection.gripper_target ▶ rt/dex1/right/cmd
    ZEDCamera.{color_image,depth_image,pointcloud,camera_info} ─▶ Rerun (world/camera/*)

⚠️⚠️ TWO UNRESOLVED BLOCKERS before a real LIVE grasp attempt (see
``AGX_ORIN_PORT_SPEC.md`` §5):

1. **Chest-ZED hand-eye calibration is NOT done.** ``IkReachBridge`` computes
   click->torso using a STATIC extrinsic; the head-D435i value baked into
   ``ik_reach_bridge.py`` (``_D435_XYZ``/``_D435_RPY``) is WRONG for a
   chest-mounted camera. This blueprint exposes ``IK_ZED_CAM_XYZ`` /
   ``IK_ZED_CAM_RPY`` (torso_link -> zed_link, URDF fixed-joint convention:
   "x,y,z" [m] / "roll,pitch,yaw" [rad]) to override it -- leave them unset and
   the reach silently uses the (wrong, D435i) default, so LIVE clicks will
   reach to the WRONG point in space until these are measured and set. This is
   a DIFFERENT calibration, in a DIFFERENT convention, from
   ``OKRA_CAM_TO_TORSO``/``scripts/compute_cam_to_torso.py`` (that one is for
   the YOLO-detect harvest pipeline's own pinhole convention, not this
   pinocchio/URDF one) -- do not reuse one value for the other.
2. **``enable_disconnect`` (the operator's keyboard 2-stage soft-stop on
   ``G1ArmSdkConnection``) does not exist on this branch.** The laptop
   original wires a ``disconnect`` stream + ``enable_disconnect=True`` for a
   soft ramp-to-zero before quitting; neither is present on this branch's
   ``G1ArmSdkConnection`` (checked 2026-07-15), so this port omits both. Keep
   a physical e-stop in hand for any LIVE run regardless.

SAFETY (fail-safe by default, same convention as the rest of dimos/):
- DRY-RUN unless ``IK_REACH_LIVE=1``: IkReachBridge logs the target,
  GripperGraspOnReach logs the would-be close, G1ArmSdkConnection does not
  write rt/arm_sdk. Absence of the env var is NEVER interpreted as live.
- ``close_q`` (OKRA_NOACT_CLOSE_Q) has NO known-good value -- untuned Dex1
  motor position. Tune on hardware before a LIVE grasp attempt.
- LIVE runs require a physical e-stop in hand and clear space around the arm.

Run (manual): dimos run unitree-g1-okra-ik-only-grasp-zed   (env vars below)
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

# DDS NIC for rt/arm_sdk / rt/lowstate / rt/dex1 -- the G1's internal Ethernet from
# the backpack, NOT a laptop NIC. Empty = SDK auto-select (usually wrong); set
# ROBOT_INTERFACE explicitly (same convention as unitree_g1_okra_harvest_ik.py).
_NIC = os.getenv("ROBOT_INTERFACE", "")

# Fail-safe live gate: explicit opt-in only; default is dry-run.
_LIVE = os.getenv("IK_REACH_LIVE", "").strip() == "1"

# Reach slew rate [rad/s]. Unchanged proven value (see the laptop/D435i original).
_ARM_VEL_LIMIT = float(os.getenv("IK_ARM_VEL_LIMIT", "20.0"))

# Shoulder/elbow gains -- see the laptop/D435i original's _KP_ARM/_KD_ARM comment
# for the standoff=0 stall investigation. Defaults left UNCHANGED (80.0/3.0).
_KP_ARM = float(os.getenv("OKRA_NOACT_KP_ARM", "80.0"))
_KD_ARM = float(os.getenv("OKRA_NOACT_KD_ARM", "3.0"))

# Vertical (torso Z) target compensation [m] -- arm-side droop / FK bias.
_TARGET_Z_OFFSET = float(os.getenv("OKRA_TARGET_Z_OFFSET", "0.05"))

# Click = the okra's CENTROID; cut point sits below it (separate knob from the
# hardware-droop offset above). 0.0 = target the centroid itself.
_CUT_BELOW_CENTROID_M = float(os.getenv("OKRA_CUT_BELOW_CENTROID_M", "0.0"))

# Pre-grasp standoff [m]. NO ACT here, so 0.0 = drive the tip onto the centroid
# itself (unlike the ACT harvest pipeline's 0.05 standoff).
_STANDOFF_M = float(os.getenv("OKRA_NOACT_STANDOFF_M", "0.0"))

# Scripted (no-ACT) gripper close target -- raw Dex1 q, NOT meters. NO known-good
# value; must be tuned on hardware (see gripper_grasp_on_reach.py).
_CLOSE_Q = float(os.getenv("OKRA_NOACT_CLOSE_Q", "0.0"))

_DEBOUNCE_S = float(os.getenv("OKRA_NOACT_DEBOUNCE_S", "3.0"))

# Separate live-gate for the GRIPPER, independent of IK_REACH_LIVE (arm).
_GRIP_LIVE = os.getenv("OKRA_NOACT_GRIP_LIVE", "").strip() == "1"

# ZED depth mode: NEURAL (fewer holes, GPU-heavier) or PERFORMANCE (lighter).
_ZED_DEPTH_MODE = os.getenv("ZED_DEPTH_MODE", "NEURAL")


def _parse_xyz_or_empty(spec: str) -> list[float]:
    """``"x,y,z"`` -> ``[x,y,z]``; ``""`` -> ``[]`` (caller then uses the D435i
    default). Raises via logger.warning + returns [] on a malformed value rather
    than crashing the blueprint at import time."""
    spec = (spec or "").strip()
    if not spec:
        return []
    vals = [float(v) for v in spec.replace(" ", "").split(",")]
    if len(vals) != 3:
        logger.warning(f"expected 3 comma-separated values, got {len(vals)} in {spec!r}; ignoring")
        return []
    return vals


# Chest-ZED hand-eye override (torso_link -> zed_link; URDF fixed-joint convention:
# meters / radians). UNSET by default -- see the blocker-1 warning in the module
# docstring above. Measure on hardware before LIVE.
_CAM_XYZ = _parse_xyz_or_empty(os.getenv("IK_ZED_CAM_XYZ", ""))
_CAM_RPY = _parse_xyz_or_empty(os.getenv("IK_ZED_CAM_RPY", ""))
if _LIVE and not (_CAM_XYZ and _CAM_RPY):
    logger.error(
        "unitree_g1_okra_ik_only_grasp_zed: IK_ZED_CAM_XYZ/IK_ZED_CAM_RPY are UNSET -- "
        "IkReachBridge will fall back to the head-D435i extrinsic, which is WRONG for "
        "the chest ZED. LIVE clicks will reach to the wrong point. Measure and set both "
        "before trusting a LIVE reach (see AGX_ORIN_PORT_SPEC.md blocker 1)."
    )


def _camera_info_overlay(ci):  # type: ignore[no-untyped-def]
    """RerunBridge visual_override: log the camera Pinhole at the COLOR IMAGE
    entity so the RGB frame renders co-located with the cloud (one Spatial3D
    view). Duplicated from the laptop/D435i original (private, non-exported
    helper there) rather than shared."""
    return ci.to_rerun(image_topic="world/camera/color_image")


def _pointcloud_rgb_overlay(pc):  # type: ignore[no-untyped-def]
    """RerunBridge visual_override: render the cloud with its TRUE per-point
    camera RGB (so the okra shows in real color, easy to click) instead of the
    default height colormap. Falls back to default if a cloud arrives without
    colors. Duplicated from the laptop/D435i original for the same reason as
    _camera_info_overlay above."""
    import numpy as np
    import rerun as rr

    points, colors = pc.as_numpy()
    if colors is None or len(points) == 0:
        return pc.to_rerun()
    rgb = (np.asarray(colors) * 255.0).clip(0, 255).astype(np.uint8)
    radius = float(os.getenv("IK_PC_POINT_RADIUS", "0.0015"))
    return rr.Points3D(positions=np.asarray(points), colors=rgb, radii=radius)


_HANDOFF_MSG = (
    f"NO ACT -- scripted gripper close_q={_CLOSE_Q:.3f} (OKRA_NOACT_CLOSE_Q, UNTUNED) "
    f"on reach_done, debounce={_DEBOUNCE_S}s | target Z offset = {_TARGET_Z_OFFSET:+.3f} m "
    f"| standoff = {_STANDOFF_M:.3f} m | cut point = {_CUT_BELOW_CENTROID_M:.3f} m below centroid "
    f"| kp_arm={_KP_ARM:.1f} kd_arm={_KD_ARM:.1f} | cam_xyz={_CAM_XYZ or '(D435i default, WRONG for ZED)'} "
    f"cam_rpy={_CAM_RPY or '(D435i default, WRONG for ZED)'}"
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

unitree_g1_okra_ik_only_grasp_zed = autoconnect(
    # Chest ZED Mini, direct USB3 (no NX relay). Depth + pointcloud on for the
    # click UI and IkReachBridge's target; NEURAL by default (fewer holes).
    ZEDCamera.blueprint(
        enable_depth=True,
        enable_pointcloud=True,
        depth_mode=_ZED_DEPTH_MODE,
    ),
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
        # Same entity_prefix ("world") + topic ("/camera/pointcloud") as the
        # D435i original -> same Rerun entity path, so this stays valid
        # unchanged (it does not depend on which physical camera publishes it).
        expected_click_frame="/world/camera/pointcloud",
        fire_reach_done=True,
        approach_offset_xyz=[0.0, 0.0, _TARGET_Z_OFFSET - _CUT_BELOW_CENTROID_M],
        standoff_m=_STANDOFF_M,
        camera_xyz=_CAM_XYZ,  # [] until measured -> falls back to the (wrong) D435i default
        camera_rpy=_CAM_RPY,
    ),
    GripperGraspOnReach.blueprint(
        close_q=_CLOSE_Q,
        debounce_s=_DEBOUNCE_S,
        dry_run=not (_LIVE and _GRIP_LIVE),
    ),
    G1ArmSdkConnection.blueprint(
        network_interface=_NIC,
        arm_velocity_limit=_ARM_VEL_LIMIT,
        publish_cmd=_LIVE,
        kp_arm=_KP_ARM,
        kd_arm=_KD_ARM,
        # NOTE: enable_disconnect does not exist on this branch -- see blocker 2
        # in the module docstring above.
    ),
    G1GripperConnection.blueprint(
        network_interface=_NIC,
    ),
).transports(
    {
        ("motor_states", JointState): LCMTransport("/g1/motor_states", JointState),
        ("arm_target", JointState): LCMTransport("/g1/arm_target", JointState),
        ("gripper_target", JointState): LCMTransport("/g1/gripper_target", JointState),
        ("reach_done", Bool): LCMTransport("/g1/reach_done", Bool),
        ("okra_target", PointStamped): LCMTransport("/g1/okra_target", PointStamped),
        # ZED streams, on the same topic names the D435i standalone publisher used,
        # so the Rerun entity paths (world/camera/*) match expected_click_frame above.
        ("color_image", Image): LCMTransport("/camera/color_image", Image),
        ("depth_image", Image): LCMTransport("/camera/depth_image", Image),
        ("pointcloud", PointCloud2): LCMTransport("/camera/pointcloud", PointCloud2),
        ("camera_info", CameraInfo): LCMTransport("/camera/camera_info", CameraInfo),
    }
)

__all__ = ["unitree_g1_okra_ik_only_grasp_zed"]
