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

"""LAPTOP app: kinesthetic ACT data collection — click okra -> IK pre-grasp -> hand-guide.

For collecting the okra-grasp ACT retraining data (design-plan §10). A human clicks
the okra; IkReachBridge solves the right-arm IK and slews (stiff) to the pre-grasp;
on settle it fires reach_done; G1ArmSdkConnection (collection_mode) then makes the
RIGHT arm COMPLIANT (kp->0 + feedforward gravity tau) so the operator hand-guides the
final grasp, while the LEFT arm + waist stay stiff. The NEXT click re-stiffens the
right arm for the next reach.

This blueprint OWNS rt/arm_sdk (single publisher). It does NOT run ACT or the gripper.
The data RECORDER is a SEPARATE read-only process (scripts/okra_kinesthetic_capture.py)
that subscribes /g1/motor_states (right arm q) + /camera/right_wrist_color over LCM and
dumps raw episodes — no arm_sdk contention. Offline, scripts/okra_lerobot_writer.py
converts the raw dump to a wrist-only 7-DoF LeRobot dataset.

SAFETY:
- DRY-RUN unless IK_REACH_LIVE=1 (publish_cmd False; no rt/arm_sdk writes).
- LIVE: the arm moves AND the right arm goes limp-but-gravity-held on reach_done —
  SUPPORT the right arm by hand, keep an e-stop, clear the area.

Run: bash dimos/robot/unitree/g1/examples/start_okra_collect.sh [--live]
  (then run the recorder in another terminal; see that script's output)
"""

from __future__ import annotations

import os

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.Bool import Bool
from dimos.robot.unitree.g1.act.g1_arm_sdk_connection import G1ArmSdkConnection
from dimos.robot.unitree.g1.act.g1_gripper_connection import G1GripperConnection
from dimos.robot.unitree.g1.act.ik_reach_bridge import IkReachBridge
from dimos.utils.logging_config import setup_logger
from dimos.visualization.vis_module import vis_module

logger = setup_logger()

_NIC = os.getenv("ROBOT_INTERFACE", "enp46s0")
_LIVE = os.getenv("IK_REACH_LIVE", "").strip() == "1"
# Reach slew speed. clip-to-measured caps the per-cycle command step to vel_limit*dt;
# the arm is torque-limited (kp*step). 20 crawled (~20s); 80 was too fast; 53 (= 2/3 of
# 80) tuned on hw 2026-06-24. Override via IK_ARM_VEL_LIMIT. Collection only (no ACT).
_ARM_VEL_LIMIT = float(os.getenv("IK_ARM_VEL_LIMIT", "53.0"))
# Vertical droop compensation (same knob as harvest); empirical torso-Z target raise.
_TARGET_Z_OFFSET = float(os.getenv("OKRA_TARGET_Z_OFFSET", "0.05"))
# Hold the right Dex1 gripper OPEN at this q during teaching (collection). q=3.0 ≈ 3cm
# opening, from a 2-point hw fit 2026-06-24 (q4.0→4.5cm, q2.67→2.5cm => 1.504 cm/q,
# intercept -1.515cm). Override via OKRA_GRIP_OPEN_Q; set to empty to hold current.
_grip_env = os.getenv("OKRA_GRIP_OPEN_Q", "3.0").strip()
_GRIP_OPEN_Q = float(_grip_env) if _grip_env else None
# 重力モデルURDF(collection_modeのg(q)計算に使う右腕縮約モデルの元URDF)。既定は
# g1.urdf(手先=170gダミーラバーハンド)。実機にDex1-1を装着している場合は、Dex1-1
# 実装込みの公式URDF(g1_dex1_1_official.urdf, Dex1-1合計365g)に切り替えて重力補償の
# 効き方を比較する。既知の乖離: 実測Dex1-1=546g vs 本URDF365g(過小評価、2026-09-04)。
# 詳細は g1_dex1_1_official.urdf のヘッダーコメント参照。
_GRAVITY_URDF = os.getenv("OKRA_GRAVITY_URDF", "").strip()


def _camera_info_overlay(ci):  # type: ignore[no-untyped-def]
    """RerunBridge overlay: RGB image-plane co-located with the cloud (module-level for pickle)."""
    return ci.to_rerun(image_topic="world/camera/color_image")


def _pointcloud_rgb_overlay(pc):  # type: ignore[no-untyped-def]
    """RerunBridge overlay: render the cloud in true camera RGB (module-level for pickle)."""
    import os

    import numpy as np
    import rerun as rr

    points, colors = pc.as_numpy()
    if colors is None or len(points) == 0:
        return pc.to_rerun()
    rgb = (np.asarray(colors) * 255.0).clip(0, 255).astype(np.uint8)
    radius = float(os.getenv("IK_PC_POINT_RADIUS", "0.0015"))
    return rr.Points3D(positions=np.asarray(points), colors=rgb, radii=radius)


_grav_urdf_label = _GRAVITY_URDF or "g1.urdf (default, dummy rubber hand 170g)"
if _LIVE:
    logger.warning(
        f"unitree-g1-okra-collect LAUNCHING **LIVE** — click okra -> IK reaches pre-grasp (stiff) -> "
        f"on settle the RIGHT arm goes COMPLIANT (hand-guide it). Left arm + waist stay stiff. "
        f"NIC {_NIC!r}, <= {_ARM_VEL_LIMIT} rad/s. gravity model urdf={_grav_urdf_label!r}. "
        f"SUPPORT THE RIGHT ARM. E-stop in hand."
    )
else:
    logger.info(
        f"unitree-g1-okra-collect DRY-RUN (set IK_REACH_LIVE=1 to drive the arm). NIC={_NIC!r}. "
        f"gravity model urdf={_grav_urdf_label!r}. "
        f"Run scripts/okra_kinesthetic_capture.py in another terminal to record."
    )

unitree_g1_okra_collect = autoconnect(
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
        expected_click_frame="/world/camera/pointcloud",  # required for LIVE
        # Do NOT auto-fire reach_done: a click slews the arm to the pre-grasp and HOLDS
        # it STIFF. The OPERATOR presses 'c' in the recorder to make the right arm
        # compliant (recorder publishes /g1/reach_done). This guarantees "click -> the
        # hand actually goes to the IK spot" before any hand-guiding.
        fire_reach_done=False,
        approach_offset_xyz=[0.0, 0.0, _TARGET_Z_OFFSET],
    ),
    G1ArmSdkConnection.blueprint(
        network_interface=_NIC,
        arm_velocity_limit=_ARM_VEL_LIMIT,
        publish_cmd=_LIVE,
        collection_mode=True,  # reach_done -> right arm compliant (kp->0 + gravity tau)
        urdf_path=_GRAVITY_URDF,  # "" = g1.urdf既定(ダミーハンド170g)。OKRA_GRAVITY_URDFで上書き。
    ),
    # Hold the right Dex1 open (~3cm, tune OKRA_GRIP_OPEN_Q) during teaching. Gripper
    # stays OUT of the ACT model — this is just the physical open state while recording.
    G1GripperConnection.blueprint(
        network_interface=_NIC,
        publish_cmd=_LIVE,
        hold_target_q=_GRIP_OPEN_Q,
    ),
).transports(
    {
        ("motor_states", JointState): LCMTransport("/g1/motor_states", JointState),
        ("arm_target", JointState): LCMTransport("/g1/arm_target", JointState),
        ("reach_done", Bool): LCMTransport("/g1/reach_done", Bool),
    }
)

__all__ = ["unitree_g1_okra_collect"]
