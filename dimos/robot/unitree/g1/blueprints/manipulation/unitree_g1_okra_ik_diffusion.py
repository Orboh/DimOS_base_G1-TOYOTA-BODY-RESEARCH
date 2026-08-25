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

"""LAPTOP app: CHEST ZED -> click -> IK pre-grasp -> UMI DIFFUSION EE micro-adjust.

The diffusion sibling of ``unitree_g1_okra_ik_only_grasp_zed.py``. Same front half
(chest ZED Mini in-process -> human/YOLO click -> ``IkReachBridge`` coarse reach), but
where the ZED blueprint hands ``reach_done`` to ``GripperGraspOnReach`` (scripted close),
this hands it to ``UmiDiffusionBridge``: a closed-loop EE fine-adjustment driven by the
trained UMI diffusion policy (via the co-located ``umi_policy_server`` in the ``umi``
conda env — start it FIRST, see oda/umi_diffusion/RUN.md).

Deltas from the ZED blueprint:
1. ``IkReachBridge`` stops SHORT at a pre-grasp standoff (default 0.05 m, approach
   legs OFF) so the policy has room to fine-adjust; still fires ``reach_done``.
2. ``GripperGraspOnReach`` + ``G1GripperConnection`` are REMOVED. The gripper open/close
   is the USER's SEPARATE program (out of scope here). ``UmiDiffusionBridge`` drives only
   ``arm_target`` and fires ``adjust_done`` (``/g1/adjust_done``) when the adjustment
   converges — the user's gripper program subscribes that to close.

SAFETY: DRY-RUN unless ``IK_REACH_LIVE=1`` (arm stays put; everything logged). Keep the
remote e-stop in hand (L2+B damping). Ctrl-C clean-stop on this rig is unreliable.

Run: (1) conda run -n umi python oda/umi_diffusion/umi_policy_server.py --cam-device /dev/videoN
     (2) dimos run unitree-g1-okra-ik-diffusion
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
from dimos.robot.unitree.g1.act.ik_reach_bridge import IkReachBridge
from dimos.robot.unitree.g1.act.umi_diffusion_bridge import UmiDiffusionBridge
from dimos.utils.logging_config import setup_logger
from dimos.visualization.vis_module import vis_module

logger = setup_logger()

# ---- robot side (same knobs as the ZED blueprint) ---------------------------
_NIC = os.getenv("ROBOT_INTERFACE", "enp46s0")
_LIVE = os.getenv("IK_REACH_LIVE", "").strip() == "1"
_ARM_VEL_LIMIT = float(os.getenv("IK_ARM_VEL_LIMIT", "12.0"))
_KP_ARM = float(os.getenv("OKRA_NOACT_KP_ARM", "80.0"))
_KD_ARM = float(os.getenv("OKRA_NOACT_KD_ARM", "3.0"))
# Gravity feedforward on the RIGHT arm during position tracking (default OFF = unchanged).
# Without it the arm holds a pose only by carrying a permanent position error
# (e = tau_gravity / kp; measured 3-7 deg = 45-90 mm at the tip on hw with kp=80 plus the
# wrist-mounted payload). That droop is what stalls the diffusion loop: the policy asks for
# "measured + delta", the arm settles at "commanded - droop", so the tip never advances.
# OKRA_GRAVITY_TAU_SCALE trims for what the URDF does NOT model (GoPro, Media Mod, mount,
# cabling): start LOW (0.6-0.8) and raise while watching `arm track` shrink.
_GRAVITY_FF = os.getenv("OKRA_GRAVITY_FF", "").strip() == "1"
_GRAVITY_TAU_SCALE = float(os.getenv("OKRA_GRAVITY_TAU_SCALE", "0.7"))
# End-effector payload the URDF does NOT model. It ends in a 0.170 kg `right_rubber_hand`
# (G1's display hand), so this is the mass to ADD: (real payload) - 0.170.
# 2026-08-25 hardware: Dex1-1 550 g + GoPro HERO9 158 g + attachments 100 g = 808 g
#   -> 0.808 - 0.170 = 0.638 kg.
_TIP_EXTRA_MASS_KG = float(os.getenv("OKRA_TIP_EXTRA_MASS_KG", "0.638"))
# Payload CoM in the wrist-yaw frame [m] (0.0415 = hand mount face, 0.1845 = fingertip).
_TIP_EXTRA_COM = [
    float(v) for v in os.getenv("OKRA_TIP_EXTRA_COM_XYZ", "0.113,-0.003,0.0").split(",")
]
# Pre-grasp standoff: stop the IK reach SHORT so the diffusion policy fine-adjusts the
# last leg. Default 0.05 m (was 0.0 in the scripted-close blueprint). approach legs OFF.
_STANDOFF_M = float(os.getenv("OKRA_STANDOFF_M", "0.05"))
_TARGET_Z_OFFSET = float(os.getenv("OKRA_TARGET_Z_OFFSET", "0.0"))
_CUT_BELOW_CENTROID_M = float(os.getenv("OKRA_CUT_BELOW_CENTROID_M", "0.0"))
_CONFIRM_CLICK = os.getenv("OKRA_CONFIRM_CLICK", "").strip() == "1"
_CONFIRM_MIN_GAP_S = float(os.getenv("OKRA_CONFIRM_MIN_GAP_S", "0.35"))
_CONFIRM_WINDOW_S = float(os.getenv("OKRA_CONFIRM_WINDOW_S", "3.5"))
_FIXED_ORI_RAW = os.getenv("OKRA_FIXED_ORI_XYZW", "").strip()
_FIXED_ORI = [float(v) for v in _FIXED_ORI_RAW.split(",")] if _FIXED_ORI_RAW else []

# Tool-tip offset from the wrist [m], WRIST frame. IkReach drives this onto the click;
# UmiDiffusionBridge uses it as the FK/IK EE frame. Step 6 aligns the diffusion one to
# the UMI TCP point (~/umi/okra_20260723_ishimaru dataset_plan grippers[0].tcp_pose).
_TIP_OFFSET = [float(v) for v in os.getenv("OKRA_TIP_OFFSET_XYZ", "0.1845,-0.003,0.0").split(",")]
_UMI_TIP_OFFSET = [
    float(v) for v in os.getenv("OKRA_UMI_TIP_OFFSET_XYZ", ",".join(map(str, _TIP_OFFSET))).split(",")
]

# ---- UMI diffusion bridge ----
_UMI_SERVER = os.getenv("UMI_SERVER_ADDR", "tcp://127.0.0.1:5599")
_UMI_CONTROL_HZ = float(os.getenv("UMI_CONTROL_HZ", "10.0"))
# n_exec waypoints per inference. Server inference ~88ms on this PC, so >=2 keeps the
# 10 Hz loop off the latency edge (raise if the loop overruns; lower for freshest obs).
_UMI_N_EXEC = int(os.getenv("UMI_N_EXEC_PER_INFER", "2"))
# Per-request budget. 88 ms measured on this PC on AC power; measured 440-500 ms with the
# laptop on BATTERY (RTX 3070 clocks down to ~0.9 GHz under load), which blows straight
# through a 300 ms budget on every single request -> 10 consecutive misses -> the episode
# aborts with the arm HELD and no adjust_done. Raise this if the GPU cannot be un-throttled.
_UMI_PREDICT_TIMEOUT_MS = int(os.getenv("UMI_PREDICT_TIMEOUT_MS", "300"))
# v1 fallback: position-only IK (orientation held) = safest first bring-up. Set
# UMI_POSITION_ONLY=0 for full 6-DOF (follow the policy's commanded orientation).
_UMI_POSITION_ONLY = os.getenv("UMI_POSITION_ONLY", "1").strip() == "1"
# Convergence -> adjust_done. Observed commanded steps were 3.2-4.3 mm on hw 2026-07-30,
# i.e. right at the 4 mm default, so keep these reachable without an edit-rebuild cycle.
_UMI_CONVERGE_EPS_M = float(os.getenv("UMI_CONVERGE_EPS_M", "0.004"))
_UMI_CONVERGE_HOLD_TICKS = int(os.getenv("UMI_CONVERGE_HOLD_TICKS", "8"))

# ---- UMI observability (see UmiDiffusionBridge's module docstring) ----
# Default = log every tick: the failure mode being chased is "the policy is not visibly
# adjusting", and the old 0.5 s throttle made the log thinnest exactly then.
_UMI_LOG_EVERY_N = int(os.getenv("UMI_LOG_EVERY_N", "1"))
_UMI_LOG_JOINTS = os.getenv("UMI_LOG_JOINTS", "1").strip() == "1"
_UMI_LOG_CHUNK_MAX = int(os.getenv("UMI_LOG_CHUNK_MAX", "4"))
# "auto" = <per-run log dir>/umi_diffusion_trace.jsonl. Empty string disables the trace.
_UMI_TRACE_PATH = os.getenv("UMI_TRACE_PATH", "auto")
# Wrist-camera preflight: refuse to start the adjustment on a stale/black frame. A dead
# camera is indistinguishable from an idle policy once running (2026-08-25: a whole LIVE
# run was driven from a 16-hour-old frame). Set 0 only to debug without the GoPro.
_UMI_REQUIRE_CAMERA = os.getenv("UMI_REQUIRE_CAMERA_OK", "1").strip() != "0"

# ---- spoken phase announcements (Japanese, G1 speaker) ----------------------
# From outside the robot, an IK coarse reach and a diffusion fine-adjustment look the
# same, and an adjustment that refused to start looks like nothing at all. The LangGraph
# harvest app announces every phase; this gives the bridge pipeline the same cue.
# Needs pyopenjtalk + scipy in the venv; without them it degrades to [VOICE] log lines.
_ANCHOR_DRIFT = float(os.getenv("UMI_MAX_ANCHOR_DRIFT_M", "0.05"))
_VOICE = os.getenv("OKRA_VOICE", "").strip() == "1"
_VOICE_VOLUME = int(os.getenv("OKRA_VOICE_VOLUME", "100"))

# ---- chest-ZED camera (same knobs as the ZED blueprint) ---------------------
_ZED_SERIAL = os.getenv("ZED_SERIAL", "").strip() or None
_PC_FPS = float(os.getenv("ZED_PC_FPS", "3.0"))
_DEPTH_MODE = os.getenv("ZED_DEPTH_MODE", "NEURAL")
_CAM_FPS = int(os.getenv("ZED_FPS", "15"))
_PC_VOXEL = float(os.getenv("ZED_PC_VOXEL", "0.002"))
_DEPTH_TRUNC = float(os.getenv("ZED_DEPTH_TRUNC", "0.8"))
_ZED_MOUNT = [
    float(v) for v in os.getenv("ZED_MOUNT_XYZRPY", "0.109,0.030,0.248,0.0,-0.0209,0.0").split(",")
]


def _camera_info_overlay(ci):  # type: ignore[no-untyped-def]
    return ci.to_rerun(image_topic="world/camera/color_image")


def _pointcloud_rgb_overlay(pc):  # type: ignore[no-untyped-def]
    import os as _os

    import numpy as np
    import rerun as rr

    points, colors = pc.as_numpy()
    if colors is None or len(points) == 0:
        return pc.to_rerun()
    rgb = (np.asarray(colors) * 255.0).clip(0, 255).astype(np.uint8)
    radius = float(_os.getenv("IK_PC_POINT_RADIUS", "0.0015"))
    return rr.Points3D(positions=np.asarray(points), colors=rgb, radii=radius)


if _LIVE:
    logger.warning(
        "unitree_g1_okra_ik_diffusion LAUNCHING **LIVE (arm only)** -- arm WILL move via "
        f"rt/arm_sdk on NIC {_NIC!r} at <= {_ARM_VEL_LIMIT} rad/s. IK reach -> standoff "
        f"{_STANDOFF_M} m -> UMI diffusion EE adjust ({'pos-only' if _UMI_POSITION_ONLY else '6-DOF'}, "
        f"server {_UMI_SERVER}). Gripper is your SEPARATE program (subscribe /g1/adjust_done). "
        "Keep an e-stop in hand."
    )
else:
    logger.info(
        f"unitree_g1_okra_ik_diffusion DRY-RUN (set IK_REACH_LIVE=1 to drive the arm). NIC={_NIC!r}. "
        f"Start the policy server first (umi env). standoff={_STANDOFF_M} m, "
        f"UMI {'pos-only' if _UMI_POSITION_ONLY else '6-DOF'} @ {_UMI_CONTROL_HZ} Hz."
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
        enable_tracking=False,
    ),
    IkReachBridge.blueprint(
        log_only=not _LIVE,
        expected_click_frame="/world/camera/pointcloud",
        fire_reach_done=True,
        approach_offset_xyz=[0.0, 0.0, _TARGET_Z_OFFSET - _CUT_BELOW_CENTROID_M],
        standoff_m=_STANDOFF_M,
        approach_above_m=0.0,   # diffusion owns the final approach
        approach_front_m=0.0,
        confirm_click=_CONFIRM_CLICK,
        confirm_min_gap_s=_CONFIRM_MIN_GAP_S,
        confirm_window_s=_CONFIRM_WINDOW_S,
        fixed_orientation_xyzw=_FIXED_ORI,
        gripper_offset_xyz=_TIP_OFFSET,
        camera_mount_xyzrpy=_ZED_MOUNT,
        click_in_camera_body_frame=True,
        voice=_VOICE,
        voice_nic=_NIC,
        voice_volume=_VOICE_VOLUME,
    ),
    G1ArmSdkConnection.blueprint(
        network_interface=_NIC,
        arm_velocity_limit=_ARM_VEL_LIMIT,
        voice=_VOICE,
        voice_nic=_NIC,
        voice_volume=_VOICE_VOLUME,
        publish_cmd=_LIVE,
        kp_arm=_KP_ARM,
        kd_arm=_KD_ARM,
        enable_disconnect=True,
        gravity_ff=_GRAVITY_FF,
        gravity_tau_scale=_GRAVITY_TAU_SCALE,
        tip_extra_mass_kg=_TIP_EXTRA_MASS_KG,
        tip_extra_com_xyz=_TIP_EXTRA_COM,
    ),
    UmiDiffusionBridge.blueprint(
        max_anchor_drift_m=_ANCHOR_DRIFT,
        server_addr=_UMI_SERVER,
        control_hz=_UMI_CONTROL_HZ,
        n_exec_per_infer=_UMI_N_EXEC,
        predict_timeout_ms=_UMI_PREDICT_TIMEOUT_MS,
        position_only=_UMI_POSITION_ONLY,
        gripper_offset_xyz=_UMI_TIP_OFFSET,
        converge_pos_eps_m=_UMI_CONVERGE_EPS_M,
        converge_hold_ticks=_UMI_CONVERGE_HOLD_TICKS,
        log_only=not _LIVE,
        require_camera_ok=_UMI_REQUIRE_CAMERA,
        voice=_VOICE,
        voice_nic=_NIC,
        voice_volume=_VOICE_VOLUME,
        log_every_n=_UMI_LOG_EVERY_N,
        log_joints=_UMI_LOG_JOINTS,
        log_chunk_max=_UMI_LOG_CHUNK_MAX,
        trace_path=_UMI_TRACE_PATH,
    ),
]

unitree_g1_okra_ik_diffusion = autoconnect(*_MODULES).transports(
    {
        ("pointcloud", PointCloud2): LCMTransport("/camera/pointcloud", PointCloud2),
        ("color_image", Image): LCMTransport("/camera/color_image", Image),
        ("camera_info", CameraInfo): LCMTransport("/camera/camera_info", CameraInfo),
        ("motor_states", JointState): LCMTransport("/g1/motor_states", JointState),
        ("arm_target", JointState): LCMTransport("/g1/arm_target", JointState),
        ("reach_done", Bool): LCMTransport("/g1/reach_done", Bool),
        # diffusion adjustment converged -> the user's separate gripper program closes.
        ("adjust_done", Bool): LCMTransport("/g1/adjust_done", Bool),
        ("okra_target", PointStamped): LCMTransport("/g1/okra_target", PointStamped),
        ("disconnect", Bool): LCMTransport("/g1/arm_sdk_disconnect", Bool),
    }
)

__all__ = ["unitree_g1_okra_ik_diffusion"]
