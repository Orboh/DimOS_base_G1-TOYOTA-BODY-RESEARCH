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

"""Blueprint: オクラ収穫「model_no_kensho」— ①IK粗アプローチ無しで、教示済みの
準備姿勢から直接 Diffusion/ACT/Flow matching モデルにオクラへの接近を委ねる実験版。

    dimos run unitree-g1-model-no-kensho

``unitree_g1_okra_honban.py``（本番版）のフォーク。両者の唯一の本質的差分は
①IK粗アプローチの扱い:

  - honban 版: ① IK（front: align→push）でオクラの重心へ寄せる → ② ACT無し
    → ③ 切断可否 → ④ 切断 → ⑤ 籠投入。
  - 本ファイル: ① を丸ごとスキップ（IKでは一切寄せない）。教示済みの準備姿勢
    （``_move_to_pregrasp_pose``、起動時に一度だけ実行）から直接 ② モデル推論
    （UmiDiffusionBridge 経由、Diffusion/ACT/Flow matching いずれもUMI形式の
    出力＝pos3+rot6d絶対EEウェイポイントを返すZMQサーバーという前提）に委ねる。
    ③④⑤は honban と共通。

なぜ①を無くすか: これまでは「IKで近づいてからモデルで微調整」という構成で
検証していたが、今回は「教示済みの固定姿勢からモデルだけでオクラまで到達
できるか」を単独で検証したい（2026-09-16 ユーザー要望）。IK到達確認や
reach-verify のような「うまくいっている前提」を挟まない検証、の意味で
"no_kensho"（検証なし）と名付けている。

モデルの切り替え: 3モデルとも UMI 形式（pos3+rot6d の絶対EEウェイポイント、
ROOTフレーム）で応答するZMQ REPサーバーという前提が成り立つため、DimOS側の
コード（``UmiDiffusionBridge``）は一切変更せず、接続先アドレス
（``OKRA_MODEL_SERVER_ADDR``）だけを切り替える。``OKRA_MODEL_NAME`` はログ・
音声アナウンス用のラベルにのみ使う。

⚠️ 座標変換に関する既知の失敗と対策（[[g1-umi-diffusion-ee-frame]]）:
UMI/roboharvestのTCPフレームはGoPro光学系（+z=前）、G1のgripper_tipフレーム
はtorso整列（+x=前）で、軸の対応が異なる。過去に変換なしでtip frameのまま
渡した際、モデルは「前進」を指示しているのに実機は「真上」に上がり続ける
という実機事故が起きている（2026-08-26）。``UmiDiffusionBridge`` の
``ee_frame="camera"``（既定・本ファイルでも維持）がこの変換を担っており、
外さないこと。

Enterトリガー（人間による手動終了）: モデル推論ループは収束(converge)を
自動検知した場合に加え、``oda/press_enter_to_cut.py``（別ターミナルで起動、
Enterを押すとLCM経由で ``/g1/model_cut_trigger`` へ1回publish）でも終了できる。
人間が「もう十分近い」と判断した瞬間の姿勢のまま③切断可否→④切断へ進む。

前提条件（honban 版と同じ）:
  - ZED SDK + pyzed が実行ホストにインストール済み／ZED-M カメラが USB3 で接続済み
  - OKRA_YOLO_MODEL・OKRA_YOLO_CONF・OKRA_TARGET は honban 版と同じ既定値
  - OKRA_MODEL_SERVER_ADDR（既定 "tcp://127.0.0.1:5599"）: UMI形式ZMQ推論サーバー
    のアドレス。モデルごとに別々に起動したサーバーへ向け先を変える。
  - OKRA_MODEL_NAME（既定 "diffusion"）: ログ・音声アナウンス用のラベルのみ
"""

from __future__ import annotations

import os

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.Bool import Bool
from dimos.robot.unitree.g1.act.g1_arm_sdk_connection import G1ArmSdkConnection
from dimos.robot.unitree.g1.act.g1_gripper_connection import G1GripperConnection
from dimos.robot.unitree.g1.act.umi_diffusion_bridge import UmiDiffusionBridge
from dimos.robot.unitree.g1.effectors.high_level.dds_sdk import G1HighLevelDdsSdk
from dimos.robot.unitree.g1.harvest.harvest_module import HarvestModule
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# robot side（honban 版と同じ命名・同じ既定値）
_NIC = os.getenv("ROBOT_INTERFACE", "enp46s0")
_LIVE = os.getenv("IK_REACH_LIVE", "").strip() == "1"
_ARM_VEL_LIMIT = float(os.getenv("IK_ARM_VEL_LIMIT", "20.0"))
_KP_ARM = float(os.getenv("OKRA_NOACT_KP_ARM", "80.0"))
_KD_ARM = float(os.getenv("OKRA_NOACT_KD_ARM", "3.0"))

# 重力補償（honban 版と同じ既定 ON。モデル推論の観測に使うFK/IK残差が重力droopで
# 汚染されないようにするため、model_no_kenshoではむしろ honban 以上に重要）。
_GRAVITY_FF = os.getenv("OKRA_GRAVITY_FF", "1").strip() == "1"
_GRAVITY_TAU_SCALE = float(os.getenv("OKRA_GRAVITY_TAU_SCALE", "1.0"))
_GRAVITY_JOINTS = [int(v) for v in os.getenv("OKRA_GRAVITY_JOINTS", "0,1,2,3,4,5,6").split(",")]
_GRAVITY_TAU_LIMIT_NM = float(os.getenv("OKRA_GRAVITY_TAU_LIMIT_NM", "12.0"))
_GRAVITY_URDF = os.getenv("OKRA_GRAVITY_URDF", "").strip() or (
    "dimos/robot/unitree/g1/g1_dex1_1_calibrated_550g.urdf"
)
_GRAVITY_RAMP_S = float(os.getenv("OKRA_GRAVITY_RAMP_S", "5.0"))

# 起動直後・把持ループ開始前に一度だけ腕を動かす「IKの事前ポジション」
# （honban 版と同じ機構）。model_no_kenshoではここが唯一の"IK/教示による位置決め"
# であり、以降②のモデル推論はここを起点に完全に自力でオクラへ寄せる。
_PREGRASP_POSE_TORSO_XYZ = os.getenv("OKRA_PREGRASP_POSE_TORSO", "0.25,0.0,0.15")
_PREGRASP_POSE_Q7 = os.getenv("OKRA_PREGRASP_POSE_Q7", "")

_CUT_SETTLE_S = float(os.getenv("OKRA_CUT_SETTLE_S", "2.5"))
_CUT_CLOSE_Q = float(os.getenv("OKRA_CUT_CLOSE_Q", "1.6"))
_BLADE_MAX_Q = float(os.getenv("OKRA_BLADE_MAX_Q", "5.2"))
_BASKET_OPEN_Q = float(os.getenv("OKRA_BASKET_OPEN_Q", "5.2"))
_YOLO_CONF = float(os.getenv("OKRA_YOLO_CONF", "0.25"))
_VOICE_LEAD_S = float(os.getenv("OKRA_VOICE_LEAD_S", "2.0"))
_ADVANCE_STEP_S = os.getenv("OKRA_ADVANCE_STEP", "").strip()
_ADVANCE_STEP = float(_ADVANCE_STEP_S) if _ADVANCE_STEP_S else None
_MAX_EMPTY_ADVANCES_S = os.getenv("OKRA_MAX_EMPTY_ADVANCES", "").strip()
_MAX_EMPTY_ADVANCES = int(_MAX_EMPTY_ADVANCES_S) if _MAX_EMPTY_ADVANCES_S else None

_GRIP_KP = float(os.getenv("OKRA_GRIP_KP", "5.0"))
_GRIP_KD = float(os.getenv("OKRA_GRIP_KD", "0.05"))
_DEX1_PREFIX = os.getenv("OKRA_DEX1_PREFIX", "rt/dex1/right").strip()
_GRIP_LIVE = os.getenv("OKRA_NOACT_GRIP_LIVE", "").strip() == "1"

_MOVE_LIVE = os.getenv("OKRA_MOVE_LIVE", "").strip() == "1"
_USE_BASE_MOVE = _LIVE and _MOVE_LIVE
_BASE_SPEED = float(os.getenv("OKRA_BASE_SPEED", "0.8"))
_MOVE_SOURCE = os.getenv("OKRA_MOVE_SOURCE", "real").strip().lower()

# 既定ON: YOLOのオクラ検出を経由せず、準備姿勢への移動完了直後に1回だけ②モデル
# 推論を自動開始する（Fiducial Cube等、okra重みでは検出できない対象を machine
# の前に置いて検証する用途、2026-09-16 ユーザー要望）。実際にYOLOでオクラを
# 検出させたい場合（本来のオクラ収穫用モデルに差し替えた場合等）は
# OKRA_MODEL_AUTO_START=0 にすること。
_MODEL_AUTO_START = os.getenv("OKRA_MODEL_AUTO_START", "1").strip() == "1"

_USE_BASKET_DEPOSIT = os.getenv("OKRA_BASKET_DEPOSIT", "1").strip() == "1"
_BASKET_ENTRY_Q7 = os.getenv("OKRA_BASKET_ENTRY_Q7", "")
_BASKET_DROP_Q7 = os.getenv("OKRA_BASKET_DROP_Q7", "")
_BASKET_RETREAT_Q7 = os.getenv("OKRA_BASKET_RETREAT_Q7", "")

# 胸ZEDのハンドアイ変換（honban 版と同じCAD実測値）。
_CAM_TO_TORSO = os.getenv(
    "OKRA_CAM_TO_TORSO",
    "0.1110,0.0250,0.2585,-0.49475,0.49475,-0.50520,0.50520",
)

# モデル推論（UmiDiffusionBridge経由、3モデル共通のUMI形式インターフェース）。
# 接続先だけをモデルごとに切り替える。既存の unitree_g1_okra_ik_diffusion.py と
# 同じ環境変数名（UMI_*）を踏襲する — 「全部UMI形式」という前提のもとでは、
# コード上も「UMIサーバー」という一つの抽象で扱って差し支えないため。
_MODEL_NAME = os.getenv("OKRA_MODEL_NAME", "diffusion").strip()
_MODEL_SERVER_ADDR = os.getenv("OKRA_MODEL_SERVER_ADDR", "tcp://127.0.0.1:5599")
_MODEL_CONTROL_HZ = float(os.getenv("UMI_CONTROL_HZ", "10.0"))
_MODEL_N_EXEC = int(os.getenv("UMI_N_EXEC_PER_INFER", "2"))
_MODEL_PREDICT_TIMEOUT_MS = int(os.getenv("UMI_PREDICT_TIMEOUT_MS", "300"))
_MODEL_MAX_DURATION_S = float(os.getenv("OKRA_MODEL_MAX_DURATION_S", "60.0"))
# v1 既定: position-only（向きは現状維持）。6-DOFにするなら0にする。
_MODEL_POSITION_ONLY = os.getenv("UMI_POSITION_ONLY", "1").strip() == "1"
# ⚠️ "camera" が正しい変換（module docstring参照）。"tip" にすると2026-08-26と
# 同じ「前進のはずが真上に上がり続ける」事故を再現する — デバッグ目的以外で
# 変更しないこと。
_MODEL_EE_FRAME = os.getenv("UMI_EE_FRAME", "camera").strip().lower()
_MODEL_TIP_OFFSET = [
    float(v) for v in os.getenv("OKRA_TIP_OFFSET_XYZ", "0.1845,-0.003,0.0").split(",")
]
_MODEL_TIP_TO_TCP = [float(v) for v in os.getenv("UMI_TIP_TO_TCP_XYZ", "0,0,0").split(",")]
_MODEL_CONVERGE_EPS_M = float(os.getenv("UMI_CONVERGE_EPS_M", "0.004"))
_MODEL_CONVERGE_HOLD_TICKS = int(os.getenv("UMI_CONVERGE_HOLD_TICKS", "8"))
_MODEL_REQUIRE_CAMERA_OK = os.getenv("UMI_REQUIRE_CAMERA_OK", "1").strip() != "0"
_MODEL_LOG_EVERY_N = int(os.getenv("UMI_LOG_EVERY_N", "1"))
_MODEL_LOG_JOINTS = os.getenv("UMI_LOG_JOINTS", "1").strip() == "1"
_MODEL_LOG_CHUNK_MAX = int(os.getenv("UMI_LOG_CHUNK_MAX", "4"))
_MODEL_TRACE_PATH = os.getenv("UMI_TRACE_PATH", "auto")
# GraspSequence(ModelGraspAdapter)がadjust_doneを待つ最大秒数。人間の判断待ち
# （Enterトリガー）を含むため、収束オンリーの構成よりはるかに長めに取る。
_MODEL_GRASP_WAIT_S = float(os.getenv("OKRA_MODEL_GRASP_WAIT_S", "300.0"))

# カメラ入力の切替（honban 版と同じ）。
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
        yolo_conf=_YOLO_CONF,
        target_classes=os.getenv("OKRA_TARGET", "okra"),
        cam_to_torso_xyzquat=_CAM_TO_TORSO,
        use_ik_grasp_sequence=True,
        use_act_grasp=False,
        # ⭐ honban版との唯一の本質的差分: ①IKをスキップし②をモデル推論に委ねる。
        use_model_grasp=True,
        model_grasp_wait_s=_MODEL_GRASP_WAIT_S,
        model_grasp_note=_MODEL_NAME,
        model_grasp_auto_start=_MODEL_AUTO_START,
        cut_close_q=_CUT_CLOSE_Q,
        blade_max_q=_BLADE_MAX_Q,
        cut_settle_s=_CUT_SETTLE_S,
        use_basket_deposit=_USE_BASKET_DEPOSIT,
        basket_open_q=_BASKET_OPEN_Q,
        basket_entry_q7=_BASKET_ENTRY_Q7,
        basket_drop_q7=_BASKET_DROP_Q7,
        basket_retreat_q7=_BASKET_RETREAT_Q7,
        gripper_live=(_LIVE and _GRIP_LIVE),
        pregrasp_settle_s=(_GRAVITY_RAMP_S if _GRAVITY_FF else 0.0),
        pregrasp_pose_torso_xyz=_PREGRASP_POSE_TORSO_XYZ,
        pregrasp_pose_q7=_PREGRASP_POSE_Q7,
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
    # ②のモデル推論本体。HarvestModule(ModelGraspAdapter)からreach_doneで開始を
    # 通知され、収束またはcut_trigger（人間のEnter）でadjust_doneを返す。
    UmiDiffusionBridge.blueprint(
        server_addr=_MODEL_SERVER_ADDR,
        control_hz=_MODEL_CONTROL_HZ,
        n_exec_per_infer=_MODEL_N_EXEC,
        predict_timeout_ms=_MODEL_PREDICT_TIMEOUT_MS,
        max_duration_s=_MODEL_MAX_DURATION_S,
        position_only=_MODEL_POSITION_ONLY,
        ee_frame=_MODEL_EE_FRAME,
        gripper_offset_xyz=_MODEL_TIP_OFFSET,
        tip_to_tcp_xyz=_MODEL_TIP_TO_TCP,
        converge_pos_eps_m=_MODEL_CONVERGE_EPS_M,
        converge_hold_ticks=_MODEL_CONVERGE_HOLD_TICKS,
        require_camera_ok=_MODEL_REQUIRE_CAMERA_OK,
        urdf_path=_GRAVITY_URDF,
        log_only=not _LIVE,
        log_every_n=_MODEL_LOG_EVERY_N,
        log_joints=_MODEL_LOG_JOINTS,
        log_chunk_max=_MODEL_LOG_CHUNK_MAX,
        trace_path=_MODEL_TRACE_PATH,
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
    f"camera_source={_CAMERA_SOURCE} yolo_conf={_YOLO_CONF} "
    f"model_auto_start={_MODEL_AUTO_START} "
    f"model={_MODEL_NAME}@{_MODEL_SERVER_ADDR} ee_frame={_MODEL_EE_FRAME} "
    f"control_hz={_MODEL_CONTROL_HZ} converge={_MODEL_CONVERGE_EPS_M}m "
    f"max_duration={_MODEL_MAX_DURATION_S:.0f}s grasp_wait={_MODEL_GRASP_WAIT_S:.0f}s "
    f"pregrasp_pose_q7={_PREGRASP_POSE_Q7 or '(unset, falls back to torso)'} "
    f"pregrasp_pose_torso={_PREGRASP_POSE_TORSO_XYZ or 'OFF'} "
    f"pregrasp_settle_s={(_GRAVITY_RAMP_S if _GRAVITY_FF else 0.0):.1f} "
    f"cut_settle_s={_CUT_SETTLE_S:.1f} "
    f"cut_close_q={_CUT_CLOSE_Q} blade_max_q={_BLADE_MAX_Q} basket_open_q={_BASKET_OPEN_Q} "
    f"cam_to_torso={'set' if _CAM_TO_TORSO else 'UNSET(camera-frame passthrough)'}"
)
_move_note = (
    f"move={f'LIVE({_MOVE_SOURCE})' if _USE_BASE_MOVE else 'DRY-RUN(placeholder)'} "
    f"base_speed={_BASE_SPEED:.2f}m/s voice_lead_s={_VOICE_LEAD_S:.1f}"
)
if _LIVE and _GRIP_LIVE:
    logger.warning(
        "unitree_g1_model_no_kensho LAUNCHING **LIVE (arm+gripper)** -- YOLO detects "
        f"automatically (no click needed), ①IK approach is SKIPPED, and the arm+gripper "
        f"WILL move via rt/arm_sdk / rt/dex1 on NIC {_NIC!r} at <= {_ARM_VEL_LIMIT} rad/s "
        f"once the model server answers. grasp=model->cut(no-IK, no-VLM) "
        f"gravity_ff={_GRAVITY_FF} urdf={_GRAVITY_URDF!r} {_approach_note} "
        f"basket_deposit={'ON(' + ('taught' if (_BASKET_ENTRY_Q7 and _BASKET_DROP_Q7 and _BASKET_RETREAT_Q7) else 'IK') + ')' if _USE_BASKET_DEPOSIT else 'OFF'} {_move_note}. "
        f"{'⚠️⚠️ BASE WILL WALK (LocoClient, OKRA_MOVE_LIVE=1). ' if _USE_BASE_MOVE else ''}"
        "Start the model server FIRST (see docstring). "
        "Press Enter in oda/press_enter_to_cut.py to hand off manually at any time. "
        "Keep an e-stop in hand."
    )
elif _LIVE:
    logger.warning(
        "unitree_g1_model_no_kensho LAUNCHING **LIVE (arm only)** -- the arm WILL move via "
        f"rt/arm_sdk on NIC {_NIC!r} once the model server answers, but the GRIPPER stays "
        "DRY-RUN (set OKRA_NOACT_GRIP_LIVE=1 to also close/open it). "
        f"grasp=model->cut(no-IK, no-VLM) gravity_ff={_GRAVITY_FF} {_approach_note} "
        f"basket_deposit={'ON(' + ('taught' if (_BASKET_ENTRY_Q7 and _BASKET_DROP_Q7 and _BASKET_RETREAT_Q7) else 'IK') + ')' if _USE_BASKET_DEPOSIT else 'OFF'} {_move_note}."
    )
else:
    logger.info(
        f"unitree_g1_model_no_kensho DRY-RUN (set IK_REACH_LIVE=1 and OKRA_NOACT_GRIP_LIVE=1 "
        f"to drive arm+gripper, +OKRA_MOVE_LIVE=1 to also drive the base). NIC={_NIC!r}. "
        f"grasp=model->cut(no-IK, no-VLM) "
        f"gravity_ff={_GRAVITY_FF} {_approach_note} "
        f"basket_deposit={'ON(' + ('taught' if (_BASKET_ENTRY_Q7 and _BASKET_DROP_Q7 and _BASKET_RETREAT_Q7) else 'IK') + ')' if _USE_BASKET_DEPOSIT else 'OFF'} {_move_note}."
    )

unitree_g1_model_no_kensho = (
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
            # ②モデル推論の開始/終了合図（HarvestModule.ModelGraspAdapter <->
            # UmiDiffusionBridge）。
            ("reach_done", Bool): LCMTransport("/g1/reach_done", Bool),
            ("adjust_done", Bool): LCMTransport("/g1/adjust_done", Bool),
            # 人間のEnterトリガー（oda/press_enter_to_cut.py が発行）。
            ("cut_trigger", Bool): LCMTransport("/g1/model_cut_trigger", Bool),
        }
    )
)

__all__ = ["unitree_g1_model_no_kensho"]
