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

# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ブループリント: オクラ収穫 統合構成（胸部ZED検出 → IK → ACT → 切断 + 歩行 + 音声）。

    dimos run unitree-g1-okra-harvest-ik

本書の確定設計（[[00-全体設計書]]）を1プロセスに合成した統合構成:
  胸部 ZED（color+depth）→ okra-seg 検出（マスク重心 + マスク内深度 median で3D）→
  select（reach box で最も右）→ **IK 粗アプローチ（重心へ）→ ACT 微調整（~4s, 閉じない）
  → moondream 切断可否（実を収穫でき・主茎を切らない位置か）→ 切断（グリッパ閉じ=切断+把持,
  刃保護 5.2 rad クランプ）** → 記録 → ループ。届かない時は cmd_vel 歩行で再配置/掃引。

把持は ``GraspSequence``（IK→ACT→cut を1エピソード, [[SS-04/05/06]]）で実行する
（``use_ik_grasp_sequence=True``）。脚は LocoClient（'ai' モード）、上半身は rt/arm_sdk で
同時可動。全 DDS モジュールは冪等な ``ensure_channel_factory`` を通すため共存できる。

⚠️⚠️ アームが動き、グリッパが切断し、ロボットが歩きます。E-stop を携行し全周にスペースを
確保すること。ファイル E-stop: ``touch /tmp/okra_estop`` で一時停止 / ``rm`` で再開。

前提条件 / 環境変数:
  # NX:      右手首 UVC ウェブカメラの配信（teleimager-server 等）
  # laptop or Orin: act_service.py --serve（ZMQ :5701, ACT 推論）
  #            ACT_REPO_ID=sotata/act-okura-kinesthetic-wrist-7d（手首単眼・右腕7次元）
  # Orin:    ollama serve（moondream を pull 済み）
  # Orin:    ROBOT_INTERFACE=<nic> dimos run unitree-g1-okra-harvest-ik
  #
  # OKRA_YOLO_MODEL : okra-seg 重み（既定は HuggingFace Kota0612/okra-seg-detector）
  # OKRA_TARGET     : 検出クラス（既定 "okra"; 重み未投入の検証時は "banana"）
  # OKRA_CAM_TO_TORSO : 胸部ZEDのハンドアイ校正 "x,y,z,qx,qy,qz,qw"（torso<-camera）。
  #                     未設定だと IK に camera 系座標が素通り → 必ず実測値を入れること。
  # ACT_ENDPOINT    : act_service の場所（既定 tcp://127.0.0.1:5701）
  # OLLAMA_HOST     : Ollama ベース URL（空 = Jetson 既定）
  # OKRA_ARM / OKRA_WALK : "0" でアーム / 歩行を個別に無効化（段階ブリングアップ用）
  # OKRA_ACT        : "0" で ACT 微調整を無効化（IK到達→スクリプト式グリッパ close のみ、
  #                   ACT不要・act_serviceも不要）。既定 "1"（従来どおり IK->ACT->cut）。
"""

from __future__ import annotations

import os

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.hardware.sensors.camera.zed.camera import ZEDCamera
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.unitree.g1.act.g1_arm_sdk_connection import G1ArmSdkConnection
from dimos.robot.unitree.g1.act.g1_gripper_connection import G1GripperConnection
from dimos.robot.unitree.g1.camera.teleimager_camera_module import RightWristTeleimagerCamera
from dimos.robot.unitree.g1.effectors.high_level.dds_sdk import G1HighLevelDdsSdk
from dimos.robot.unitree.g1.harvest.harvest_module import HarvestModule

_NIC = os.getenv("ROBOT_INTERFACE", "")

# 段階ブリングアップ: アーム / 歩行を個別に無効化できる（既定は両方 ON）。
_ARM = os.getenv("OKRA_ARM", "1").strip() != "0"
_WALK = os.getenv("OKRA_WALK", "1").strip() != "0"
# ACT 微調整の有無（既定 ON = 従来どおり IK->ACT->cut）。OKRA_ACT=0 で ACT を挟まず
# IK 到達後そのまま切断可否チェック→グリッパを閉じる（②相当、スクリプト式）。
# _ARM=0 のときは無関係（アーム自体が無効）。
_ACT = os.getenv("OKRA_ACT", "1").strip() != "0"
_USE_ACT_GRASP = _ARM and _ACT

_modules = [
    # 胸部 ZED: head の color + 実 depth（検出 + 重心3D）。深度は NEURAL 推奨（穴が少ない）。
    ZEDCamera.blueprint(depth_mode=os.getenv("ZED_DEPTH_MODE", "NEURAL")),
    G1ArmSdkConnection.blueprint(network_interface=_NIC),
    G1GripperConnection.blueprint(network_interface=_NIC),
]
if _WALK:
    # ベース歩行（LocoClient）。OKRA_WALK=0 なら省略 — G1HighLevelDdsSdk の
    # MotionSwitcherClient/LocoClient.Init() は dds_init.channel_lock を取らずに
    # DDS 初期化するため、G1ArmSdkConnection/G1GripperConnection の起動スレッドと
    # 並走すると cyclonedds の IDL 型登録が競合し、稀に
    # 'NoneType' object has no attribute 'SupportsBasic' で全体がクラッシュする
    # （dds_init.py の channel_lock コメント参照）。歩行不要な段階検証では外して回避。
    _modules.append(G1HighLevelDdsSdk.blueprint(network_interface=_NIC))
if _USE_ACT_GRASP:
    # 右手首カメラ: ACT 入力（据え置き UVC/teleimager）。ACT 無効時（OKRA_ACT=0 /
    # OKRA_ARM=0）は不要 — teleimager パッケージ未インストールの環境でも段階
    # ブリングアップ（IKのみ検証）できるよう、ACT 有効時のみ組み込む。
    _modules.append(RightWristTeleimagerCamera.blueprint(camera="right_wrist"))

unitree_g1_okra_harvest_ik = (
    autoconnect(
        *_modules,
        HarvestModule.blueprint(
            use_dummy=False,
            # 検出（胸部 ZED + okra-seg, [[SS-01]] / [[SS-04]]）
            use_zed_depth=True,
            yolo_model=os.getenv("OKRA_YOLO_MODEL", "Kota0612/okra-seg-detector"),
            target_classes=os.getenv("OKRA_TARGET", "okra"),
            # 把持（IK → (任意)ACT → 切断 シーケンス, [[SS-04/05/06]]）
            # OKRA_ACT=0 で ACT を無効化（IK到達→スクリプト式グリッパ close のみ）。
            use_act_grasp=_USE_ACT_GRASP,
            use_ik_grasp_sequence=_ARM,
            # ACT は手首単眼・右腕7次元（sotata/act-okura-kinesthetic-wrist-7d）。
            # act_service を ACT_REPO_ID=sotata/act-okura-kinesthetic-wrist-7d で起動すること。
            act_right_arm_only_7d=True,
            act_endpoint=os.getenv("ACT_ENDPOINT", "tcp://127.0.0.1:5701"),
            cut_close_q=4.4,  # 切断時のグリッパ閉じ位置 [rad]
            blade_max_q=5.2,  # 刃保護の上限 [rad]（BladeGuard）
            cam_to_torso_xyzquat=os.getenv(
                "OKRA_CAM_TO_TORSO", ""
            ),  # 要実測（未設定=camera素通り）
            # 切断可否 / 確認 VLM（moondream, [[SS-02]]）
            vlm_model="moondream",
            ollama_host=os.getenv("OLLAMA_HOST", ""),
            # 歩行（再配置/掃引, [[SS-07]]）+ 音声
            use_base_move=_WALK,
            use_forward_search=_WALK,
            use_g1_speaker=True,
        ),
    )
    .remappings(
        # 右手首は color_image を出すため、ACT 手首入力（cam_right_wrist）にリネーム。
        # ZED の color_image / depth_image は head のまま検出へ。ACT 無効時はモジュール
        # 自体が組み込まれていないので、remap も不要（空リスト）。
        [(RightWristTeleimagerCamera, "color_image", "cam_right_wrist")] if _USE_ACT_GRASP else []
    )
    .transports(
        {
            ("color_image", Image): LCMTransport("/color_image", Image),
            ("depth_image", Image): LCMTransport("/depth_image", Image),
            ("camera_info", CameraInfo): LCMTransport("/camera_info", CameraInfo),
            **(
                {("cam_right_wrist", Image): LCMTransport("/cam_right_wrist", Image)}
                if _USE_ACT_GRASP
                else {}
            ),
            ("motor_states", JointState): LCMTransport("/g1/motor_states", JointState),
            ("arm_target", JointState): LCMTransport("/g1/arm_target", JointState),
            ("right_gripper_state", JointState): LCMTransport(
                "/g1/right_gripper_state", JointState
            ),
            ("gripper_target", JointState): LCMTransport("/g1/gripper_target", JointState),
            ("cmd_vel", Twist): LCMTransport("/cmd_vel", Twist),
        }
    )
)

__all__ = ["unitree_g1_okra_harvest_ik"]
