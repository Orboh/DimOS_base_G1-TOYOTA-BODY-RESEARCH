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

"""Blueprint: オクラ収穫「本番」— IK粗アプローチを front（align→push）に切り替えた版。

    dimos run unitree-g1-okra-honban

``unitree_g1_okra_harvest_zed.py``（以下 zed 版）のフォーク。両者の唯一の違いは
IK粗アプローチの経路形状:

  - zed 版（既定・フォールバック用に温存）: above = LIFT（真上）→TRANSIT（対象の
    真上へ水平移動）→DESCEND（垂直降下）の3段階。
  - honban 版（本ファイル）: front = ① align（今の奥行き(X)はそのまま、対象の
    高さ(Z)と左右位置(Y)を同時に合わせる）→② push（その位置から奥行き(X)方向へ
    まっすぐ押し込む）の2段階。``IkApproachSkill.solve_legs``/``stream_legs`` の
    ``front_m`` 参照。

採用の経緯: front はもともとクリック駆動版 ``IkReachBridge.approach_front_m`` に
しか無かった方式（2026-07-23, Oda「まっすぐIKでもっていければ最高」）。
2026-09-11に自動収穫用の ``IkApproachSkill`` へ移植し、2026-09-12に Isaac Sim
上で above/front 両方式の手先軌道を実写比較（肩正面・10cm右寄せ・腰高さの3条件）
した結果、frontの方が「グリッパーが対象に正対したまま直進で挿入される」ことを
push レグ中の手先 Y/Z が完全一定であることで数値・映像の両方で確認した上で本番
採用とした（会話ログ参照）。

⚠️ 既知のリスク（未解消・実機未検証）: above は元々「対象へ一発の関節空間リーチ
だと手先が読めない弧を描き株を払う」という実機事故（2026-09-08, zed 版 docstring
参照）を受けて追加された安全策。front は水平方向から対象へ直進するため、above が
避けていた「横から株を払う」リスクが再発しないかは Isaac Sim・実機いずれでも
まだ評価できていない（現状の sim シーンには株/茎のジオメトリが無い）。本番投入は
このリスクを認識した上での判断。何かあれば ``OKRA_APPROACH_FRONT_M=0`` +
``OKRA_APPROACH_ABOVE_M`` を設定するか、単純に ``unitree-g1-okra-harvest-zed``
（above版）へ戻すこと。

それ以外の設定（VLMなし・ACTなし・音声・移動・籠投入・重力補償・3段階LIVEゲート
等）は zed 版と全く同じ。詳細な経緯コメントは重複を避けるため
``unitree_g1_okra_harvest_zed.py`` を参照。

前提条件（zed 版と同じ）:
  - ZED SDK + pyzed が実行ホスト（AGX Orin）にインストール済み
  - ZED-M カメラが USB3 で接続済み
  - Ollama + qwen3-vl:2b（任意。VLM を使いたい場合のみ OKRA_VLM_MODEL を設定。既定は未使用）
  - OKRA_YOLO_MODEL（既定 "okra11n-seg.pt"、data/models_yolo/ 配下）
  - OKRA_TARGET（既定 "okra"）
"""

from __future__ import annotations

import os

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.unitree.g1.act.g1_arm_sdk_connection import G1ArmSdkConnection
from dimos.robot.unitree.g1.act.g1_gripper_connection import G1GripperConnection
from dimos.robot.unitree.g1.effectors.high_level.dds_sdk import G1HighLevelDdsSdk
from dimos.robot.unitree.g1.harvest.harvest_module import HarvestModule
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# robot side（zed 版と同じ命名・同じ既定値）
_NIC = os.getenv("ROBOT_INTERFACE", "enp46s0")
_LIVE = os.getenv("IK_REACH_LIVE", "").strip() == "1"
_ARM_VEL_LIMIT = float(os.getenv("IK_ARM_VEL_LIMIT", "20.0"))
_KP_ARM = float(os.getenv("OKRA_NOACT_KP_ARM", "80.0"))
_KD_ARM = float(os.getenv("OKRA_NOACT_KD_ARM", "3.0"))

_GRAVITY_FF = os.getenv("OKRA_GRAVITY_FF", "").strip() == "1"
_GRAVITY_TAU_SCALE = float(os.getenv("OKRA_GRAVITY_TAU_SCALE", "1.0"))
_GRAVITY_JOINTS = [int(v) for v in os.getenv("OKRA_GRAVITY_JOINTS", "0,1,2,3,4,5,6").split(",")]
_GRAVITY_TAU_LIMIT_NM = float(os.getenv("OKRA_GRAVITY_TAU_LIMIT_NM", "12.0"))
_GRAVITY_URDF = os.getenv("OKRA_GRAVITY_URDF", "").strip() or (
    "dimos/robot/unitree/g1/g1_dex1_1_calibrated_550g.urdf"
)
_GRAVITY_RAMP_S = float(os.getenv("OKRA_GRAVITY_RAMP_S", "5.0"))

# ⭐ zed 版との唯一の本質的差分: above_m ではなく front_m を使う。
# front-approach 距離（対象手前で「高さ・左右」を合わせ終える位置までの奥行き
# マージン）[m]。sim比較検証で収束・軌道とも安定していた 0.20m を既定にする
# （IkApproachSkill.solve_legs/stream_legs の front_m 参照）。
_IK_APPROACH_FRONT_M = float(os.getenv("OKRA_APPROACH_FRONT_M", "0.20"))
# above は明示的に無効（0.0）— 両方>0だと ik_approach.py の規約で above が優先
# されてしまうため、front を使うには above_m=0 が必須。
_IK_APPROACH_ABOVE_M = 0.0
# standoff（切断点手前でIKを止める量）[m]。IkApproachSkill既定の0.05は「最後の
# standoff分はACTが詰める」前提の値だが、honbanはuse_act_grasp=False（no-ACT、
# 本ファイル上部docstring参照）で標準の詰めステップが無いため、既定のままだと
# 刃が莢まで届かない。honbanでは0.0にしてIK自体を重心（切断点）まで到達させる
# （2026-09-14 ユーザー指摘。IK自体の到達精度は既にmax_reach_pos_err_m=3mmへ
# 厳格化済み、ik_approach.py参照）。
_STANDOFF_M = float(os.getenv("OKRA_STANDOFF_M", "0.0"))
_IK_STREAM_LEGS = os.getenv("OKRA_IK_STREAM_LEGS", "1").strip() == "1"
_IK_STREAM_STEP_M = float(os.getenv("OKRA_IK_STREAM_STEP_M", "0.035"))
_IK_STREAM_CADENCE_S = float(os.getenv("OKRA_IK_STREAM_CADENCE_S", "0.18"))
_CUT_SETTLE_S = float(os.getenv("OKRA_CUT_SETTLE_S", "2.5"))
_VOICE_LEAD_S = float(os.getenv("OKRA_VOICE_LEAD_S", "2.0"))
# §5 sweep（HarvestConfig パススルー、既定は未指定=HarvestConfigのデフォルトのまま
# 後方互換）。sim検証で広い探索範囲(例: 10m先まで)を試す時だけ上書きする
# （2026-09-12 要望）。未指定なら advance_step=0.30m, max_empty_advances=2 のまま。
_ADVANCE_STEP_S = os.getenv("OKRA_ADVANCE_STEP", "").strip()
_ADVANCE_STEP = float(_ADVANCE_STEP_S) if _ADVANCE_STEP_S else None
_MAX_EMPTY_ADVANCES_S = os.getenv("OKRA_MAX_EMPTY_ADVANCES", "").strip()
_MAX_EMPTY_ADVANCES = int(_MAX_EMPTY_ADVANCES_S) if _MAX_EMPTY_ADVANCES_S else None

_GRIP_KP = float(os.getenv("OKRA_GRIP_KP", "5.0"))
_GRIP_KD = float(os.getenv("OKRA_GRIP_KD", "0.05"))
#  実機は右手のみDex1搭載（g1_gripper_connection.py の既定・SETUP.md/STAGE_B_PLAN.md
#  で確認済み: rt/dex1/left/state は実機・sim双方とも publish されない）。sim側
#  (sim_dds_bridge.py) も rt/dex1/right/state しかecho-backしていないため、既定を
#  "left" のままにすると G1GripperConnection の起動待ちが恒久的にタイムアウトする
#  （2026-09-12 sim検証で実際に "No rt/dex1/left/state received" で起動失敗を確認）。
#  unitree_g1_okra_ik_only_grasp_zed.py の "left" 既定は当該機体固有の配線都合
#  （2026-07-16確認）であり、honban（本番）はそれを継承すべきではない。
_DEX1_PREFIX = os.getenv("OKRA_DEX1_PREFIX", "rt/dex1/right").strip()
_GRIP_LIVE = os.getenv("OKRA_NOACT_GRIP_LIVE", "").strip() == "1"

_MOVE_LIVE = os.getenv("OKRA_MOVE_LIVE", "").strip() == "1"
_USE_BASE_MOVE = _LIVE and _MOVE_LIVE
_BASE_SPEED = float(os.getenv("OKRA_BASE_SPEED", "0.8"))
# ベース移動の送り先切替: 既定"real"(実機・G1HighLevelDdsSdk経由 LocoClient)/"sim"
# (docs/sim-setup/sim_dds_bridge.py の SIM_BASE_MOVE ファイル経由)。
# G1HighLevelDdsSdk が呼ぶ LocoClient.Move() は Unitree 独自の JSON-RPC DDS
# トピック(rt/api/loco/request)を使うため sim では解釈できず、sim実行時のみ
# SimCmdVelBridge に差し替える（cam_source と同じ切替パターン）。実機投入時は
# 本env未設定のままでよく、コード変更は不要（2026-09-12 GUIデモでの要望）。
_MOVE_SOURCE = os.getenv("OKRA_MOVE_SOURCE", "real").strip().lower()

_USE_BASKET_DEPOSIT = os.getenv("OKRA_BASKET_DEPOSIT", "1").strip() == "1"

_CAM_TO_TORSO = os.getenv(
    "OKRA_CAM_TO_TORSO",
    "0.1090,0.0300,0.2480,-0.49475,0.49475,-0.50520,0.50520",
)

# カメラ入力の切替: 既定 "zed"（実機・pyzed必須）/ "sim"（Isaac Sim sim_dds_bridge.py の
# ego_view ZMQ配信、pyzed不要）。ZEDCamera は pyzed をトップレベル import するため、sim
# 実行ホスト（pyzed未インストール）でも honban.py を import できるよう、選ばれた方だけを
# ここで遅延 import する（sim側では zed/camera.py を一切 import しない）。
# sim側は SIM_CAM_MODE=torso で起動した sim_dds_bridge.py の cam_to_torso が実測値
# （OKRA_CAM_TO_TORSO 既定値）と一致することを 2026-09-12 実地検証済み（[[cam_to_torso検証]]）。
_CAMERA_SOURCE = os.getenv("OKRA_CAMERA_SOURCE", "zed").strip().lower()
if _CAMERA_SOURCE == "sim":
    from dimos.simulation.engines.isaac_zmq_camera import IsaacZmqDepthCamera

    _camera_module = IsaacZmqDepthCamera.blueprint(
        zmq_host=os.getenv("OKRA_SIM_CAM_HOST", "127.0.0.1"),
        zmq_port=int(os.getenv("OKRA_SIM_CAM_PORT", "5555")),
        zmq_topic=os.getenv("OKRA_SIM_CAM_TOPIC", "ego_view"),
    )
else:
    from dimos.hardware.sensors.camera.zed.camera import ZEDCamera

    _camera_module = ZEDCamera.blueprint(depth_mode=os.getenv("ZED_DEPTH_MODE", "NEURAL"))

_MODULES = [
    _camera_module,
    HarvestModule.blueprint(
        use_dummy=False,
        use_zed_depth=True,
        use_g1_speaker=True,
        network_interface=_NIC,
        vlm_model=os.getenv("OKRA_VLM_MODEL", ""),
        yolo_model=os.getenv("OKRA_YOLO_MODEL", "okra11n-seg.pt"),
        target_classes=os.getenv("OKRA_TARGET", "okra"),
        cam_to_torso_xyzquat=_CAM_TO_TORSO,
        use_ik_grasp_sequence=True,
        use_act_grasp=False,
        cut_close_q=float(os.getenv("OKRA_CUT_CLOSE_Q", "4.4")),
        blade_max_q=float(os.getenv("OKRA_BLADE_MAX_Q", "5.2")),
        cut_settle_s=_CUT_SETTLE_S,
        use_basket_deposit=_USE_BASKET_DEPOSIT,
        gripper_live=(_LIVE and _GRIP_LIVE),
        pregrasp_settle_s=(_GRAVITY_RAMP_S if _GRAVITY_FF else 0.0),
        # ⭐ zed 版と違う2行: above を0で無効化し、front を渡す。
        ik_approach_above_m=_IK_APPROACH_ABOVE_M,
        ik_approach_front_m=_IK_APPROACH_FRONT_M,
        ik_approach_standoff_m=_STANDOFF_M,
        ik_stream_legs=_IK_STREAM_LEGS,
        ik_stream_step_m=_IK_STREAM_STEP_M,
        ik_stream_cadence_s=_IK_STREAM_CADENCE_S,
        use_base_move=_USE_BASE_MOVE,
        base_speed=_BASE_SPEED,
        voice_lead_s=_VOICE_LEAD_S,
        advance_step=_ADVANCE_STEP,
        max_empty_advances=_MAX_EMPTY_ADVANCES,
    ),
    G1ArmSdkConnection.blueprint(
        network_interface=_NIC,
        arm_velocity_limit=_ARM_VEL_LIMIT,
        publish_cmd=_LIVE,
        kp_arm=_KP_ARM,
        kd_arm=_KD_ARM,
        enable_disconnect=True,
        stiff_gravity_compensation_right=_GRAVITY_FF,
        stiff_gravity_right_joint_indices=(_GRAVITY_JOINTS if _GRAVITY_FF else []),
        stiff_gravity_tau_scale=_GRAVITY_TAU_SCALE,
        stiff_gravity_tau_limit_nm=_GRAVITY_TAU_LIMIT_NM,
        stiff_gravity_ramp_s=_GRAVITY_RAMP_S,
        urdf_path=_GRAVITY_URDF,
        log_track_err_every_n=int(os.getenv("OKRA_ARM_LOG_EVERY_N", "2500")),
    ),
    G1GripperConnection.blueprint(
        network_interface=_NIC,
        dex1_topic_prefix=_DEX1_PREFIX,
        kp=_GRIP_KP,
        kd=_GRIP_KD,
    ),
]
if _USE_BASE_MOVE:
    if _MOVE_SOURCE == "sim":
        from dimos.simulation.engines.sim_cmd_vel_bridge import SimCmdVelBridge

        _MODULES.append(
            SimCmdVelBridge.blueprint(
                base_move_file=os.getenv("OKRA_SIM_BASE_MOVE_FILE", "/tmp/sim_base_move.txt"),
            )
        )
    else:
        _MODULES.append(G1HighLevelDdsSdk.blueprint(network_interface=_NIC))

_approach_note = (
    f"camera_source={_CAMERA_SOURCE} "
    f"approach_front_m={_IK_APPROACH_FRONT_M} standoff_m={_STANDOFF_M} "
    f"pregrasp_settle_s={(_GRAVITY_RAMP_S if _GRAVITY_FF else 0.0):.1f} "
    f"stream_legs={_IK_STREAM_LEGS} cut_settle_s={_CUT_SETTLE_S:.1f} "
    f"cam_to_torso={'set' if _CAM_TO_TORSO else 'UNSET(camera-frame passthrough)'}"
)
_move_note = (
    f"move={f'LIVE({_MOVE_SOURCE})' if _USE_BASE_MOVE else 'DRY-RUN(placeholder)'} "
    f"base_speed={_BASE_SPEED:.2f}m/s voice_lead_s={_VOICE_LEAD_S:.1f}"
)
if _LIVE and _GRIP_LIVE:
    logger.warning(
        "unitree_g1_okra_honban LAUNCHING **LIVE (arm+gripper)** -- YOLO detects "
        f"automatically (no click needed) and the arm+gripper WILL move via rt/arm_sdk / "
        f"rt/dex1 on NIC {_NIC!r} at <= {_ARM_VEL_LIMIT} rad/s. grasp=IK(front)->cut(no-ACT, no-VLM) "
        f"gravity_ff={_GRAVITY_FF} urdf={_GRAVITY_URDF!r} {_approach_note} "
        f"basket_deposit={'ON' if _USE_BASKET_DEPOSIT else 'OFF'} {_move_note}. "
        f"{'⚠️⚠️ BASE WILL WALK (LocoClient, OKRA_MOVE_LIVE=1). ' if _USE_BASE_MOVE else ''}"
        "Keep an e-stop in hand."
    )
elif _LIVE:
    logger.warning(
        "unitree_g1_okra_honban LAUNCHING **LIVE (arm only)** -- the arm WILL move via "
        f"rt/arm_sdk on NIC {_NIC!r}, but the GRIPPER stays DRY-RUN (set "
        "OKRA_NOACT_GRIP_LIVE=1 to also close/open it). grasp=IK(front)->cut(no-ACT, no-VLM) "
        f"gravity_ff={_GRAVITY_FF} {_approach_note} "
        f"basket_deposit={'ON' if _USE_BASKET_DEPOSIT else 'OFF'} {_move_note}."
    )
else:
    logger.info(
        f"unitree_g1_okra_honban DRY-RUN (set IK_REACH_LIVE=1 and OKRA_NOACT_GRIP_LIVE=1 "
        f"to drive arm+gripper, +OKRA_MOVE_LIVE=1 to also drive the base). NIC={_NIC!r}. "
        f"grasp=IK(front)->cut(no-ACT, no-VLM) "
        f"gravity_ff={_GRAVITY_FF} {_approach_note} "
        f"basket_deposit={'ON' if _USE_BASKET_DEPOSIT else 'OFF'} {_move_note}."
    )

unitree_g1_okra_honban = (
    autoconnect(*_MODULES)
    .remappings(
        [
            (HarvestModule, "color_image", "color_image"),
            (HarvestModule, "depth_image", "depth_image"),
            (HarvestModule, "camera_info", "camera_info"),
        ]
    )
    .transports(
        {
            ("color_image", Image): LCMTransport("/color_image", Image),
            ("depth_image", Image): LCMTransport("/depth_image", Image),
            ("camera_info", CameraInfo): LCMTransport("/camera_info", CameraInfo),
            ("motor_states", JointState): LCMTransport("/g1/motor_states", JointState),
            ("arm_target", JointState): LCMTransport("/g1/arm_target", JointState),
            ("gripper_target", JointState): LCMTransport("/g1/gripper_target", JointState),
            (
                "right_gripper_state",
                JointState,
            ): LCMTransport("/g1/right_gripper_state", JointState),
            ("cmd_vel", Twist): LCMTransport("/cmd_vel", Twist),
        }
    )
)

__all__ = ["unitree_g1_okra_honban"]
