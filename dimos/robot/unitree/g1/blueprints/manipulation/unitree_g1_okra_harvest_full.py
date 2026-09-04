#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ブループリント: フル構成のオクラ収穫 — 実機アーム（okra-ACT）+ 実機歩行ベース + 音声。

    dimos run unitree-g1-okra-harvest-full

「全機能 ON」の統合構成（ステップ 3）: ヘッド + 右手首カメラ → YOLO 検出 →
選択 → **okra-ACT 到達（アームが動く、停止可能）** → Ollama ビジョン確認 →
日本語 G1 スピーカー。§5 の接近/前進/再訪動作には **ベース再配置/掃引歩行（cmd_vel）**
を使用。脚はオンボードバランスコントローラー（'ai' モードで LocoClient 速度）で動き、
アームは ``rt/arm_sdk`` で動く — 両者は共存可能（モーション制御モード）なので、
歩行とアーム到達が同時に実行される。

本ブループリントは ``-live-arm``（アーム + スピーカー + ACT）と ``-walk``（cmd_vel ベース）を
1プロセスに統合したもの。これが安全に動作するのは、すべての DDS モジュールが
冪等な ``ensure_channel_factory`` ヘルパーを通じてチャネルファクトリを初期化するためで —
最初の ``start()`` で実際の ``ChannelFactoryInitialize`` を行い、残りは
no-op（二重初期化クラッシュなし）。``dimos/robot/unitree/g1/act/dds_init.py`` 参照。

⚠️⚠️ アームが動き、**かつ**ロボットが歩きます。E-stop を手元に置き、
ロボット周囲のスペースを確保してください。SafetyMonitor ファイル E-stop はアーム到達を
動作中に一時停止します: ``touch /tmp/okra_estop`` で一時停止、``rm`` で再開。
（人物/バランス/VLM 安全チェックは今後の対応 — 無人運用前に実チェックを接続すること。）

前提条件（-live-arm と -walk の合算）:
  # NX:      teleimager-server --rs
  # laptop:  ~/act-okura/.venv_act/bin/python scripts/act_service.py --serve
  # Jetson:  ollama serve   (moondream は pull 済み)
  # laptop:  ROBOT_INTERFACE=<nic> dimos run unitree-g1-okra-harvest-full
"""

from __future__ import annotations

import os

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.unitree.g1.act.g1_arm_sdk_connection import G1ArmSdkConnection
from dimos.robot.unitree.g1.act.g1_gripper_connection import G1GripperConnection
from dimos.robot.unitree.g1.camera.teleimager_camera_module import (
    RightWristTeleimagerCamera,
    TeleimagerCamera,
)
from dimos.robot.unitree.g1.effectors.high_level.dds_sdk import G1HighLevelDdsSdk
from dimos.robot.unitree.g1.harvest.harvest_module import HarvestModule

_NIC = os.getenv("ROBOT_INTERFACE", "")

# ツリーモデルデータセット（sotata/okura-pick-tree-20260615）の初期フレームアーム姿勢 [rad]
# （左7関節 + 右7関節）— 起動時にアームがここへスルーし、ポリシーが分布内から開始される。
_INIT_ARM_POSE = [
    0.269, 0.196, -0.018, 0.986, 0.122, 0.028, 0.003,   # 左アーム
    -0.114, 0.029, 0.185, 0.538, 0.209, -0.755, 0.370,  # 右アーム
]


unitree_g1_okra_harvest_full = (
    autoconnect(
        TeleimagerCamera.blueprint(camera="head"),
        RightWristTeleimagerCamera.blueprint(camera="right_wrist"),
        G1ArmSdkConnection.blueprint(network_interface=_NIC, initial_arm_pose=_INIT_ARM_POSE),
        G1GripperConnection.blueprint(network_interface=_NIC),
        G1HighLevelDdsSdk.blueprint(network_interface=_NIC),  # ベース歩行（LocoClient）
        HarvestModule.blueprint(
            use_dummy=False,
            use_act_grasp=True,      # ⚠️ 実機アーム到達（2カメラツリーモデル）
            use_base_move=True,      # ⚠️ cmd_vel -> LocoClient による実機ベース歩行
            use_g1_speaker=True,     # 日本語 G1 スピーカー
            vlm_model="moondream",   # ローカル Ollama ビジョン確認（約1秒 キャプション+キーワード）
            # 検出+確認用 Ollama ベース URL。空 -> ollama_vlm DEFAULT_HOST
            # （Jetson）。Jetson がロボット LAN から外れている場合は実行ごとに上書き。
            # 例: OLLAMA_HOST=http://127.0.0.1:11434 でラップトップローカルの Ollama を使用。
            ollama_host=os.getenv("OLLAMA_HOST", ""),
            use_vlm_detect=True,     # moondream でオクラを検出（存在確認）→ オクラ学習済み
                                     # YOLO なしで検出後フローを動作確認
            use_forward_search=True, # オクラなし → 左掃引後に前進歩行し
                                     # 探索継続（1スポットで終了しない）
        ),
    )
    .remappings(
        [
            # 右手首インスタンスも color_image をパブリッシュするため、
            # ヘッドではなく HarvestModule/ActGraspModule の手首入力に届くようリネーム。
            (RightWristTeleimagerCamera, "color_image", "cam_right_wrist"),
        ]
    )
    .transports(
        {
            ("color_image", Image): LCMTransport("/color_image", Image),
            ("cam_right_wrist", Image): LCMTransport("/cam_right_wrist", Image),
            ("motor_states", JointState): LCMTransport("/g1/motor_states", JointState),
            ("arm_target", JointState): LCMTransport("/g1/arm_target", JointState),
            ("right_gripper_state", JointState): LCMTransport("/g1/right_gripper_state", JointState),
            ("gripper_target", JointState): LCMTransport("/g1/gripper_target", JointState),
            ("cmd_vel", Twist): LCMTransport("/cmd_vel", Twist),
        }
    )
)

__all__ = ["unitree_g1_okra_harvest_full"]
