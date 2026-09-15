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

"""Blueprint: オクラ収穫（YOLO自動検出ループ + IK到達 + グリッパ閉 + 腹部かご投入）。

    dimos run unitree-g1-okra-harvest-zed

``dimos run`` するだけで ``HarvestModule.start()`` が LangGraph の収穫フローを起動し、
YOLO検出→選択→把持→検証→…のループが**クリック待ちなしで自動的に**回り始める
（``unitree-g1-okra-ik-only-grasp-zed`` 系がクリック/YOLOブリッジ駆動なのと対照的）。

このブランチの要求どおり **VLM なし・Diffusion なし・IK のみ**: `use_ik_grasp_sequence=True`
+ `use_act_grasp=False` で、ACT を挟まず「IK 粗アプローチ→到達→グリッパを閉じる（切断）」の
みで把持する（``GraspSequence.run_episode`` 参照）。既定で `vlm_model=""` のため切断可否
判定・把持後検証はいずれも VLM なし（常に許可）。到達後は `use_basket_deposit=True`
（既定 ON）で右腕のみ腹部固定かご（basket_link, pelvis接合）へ IK 投入 → 開放まで進む
（F-07, ``harvest/basket_deposit.py``）。

ZEDCamera (AGX Orin の ZED-M) を color + depth + camera_info で使用する。depth_image は
HarvestModule に配線され、YoloOkraDetector がマスク内の ZED 実深度 median を使用する。
camera_info（ZED 実測の焦点距離・画像中心）も配線され、ピクセル→3D の左右・上下位置は
D435i 頭部カメラ用の画角当て推量ではなく、実 intrinsics による正しい逆投影で計算される
（``detect_yolo.make_zed_pixel_to_base``）。

SAFETY: 既定 DRY-RUN（``IK_REACH_LIVE`` unset）— 腕/グリッパへは何も publish しない
（HarvestModule のフロー自体はログ上進行するので配線確認に使える）。腕は
``IK_REACH_LIVE=1`` で動く。グリッパは**さらに** ``OKRA_NOACT_GRIP_LIVE=1`` が必要
（二段階ゲート、``unitree_g1_okra_ik_only_grasp_zed.py`` と同じ命名）—
``G1GripperConnection`` 自体には DRY-RUN 切替が無いため、``HarvestModule`` 側の
``gripper_live`` でガードしている（2026-09-08 実機DRY-RUNで、二段階ゲート未実装のまま
``G1GripperConnection`` を接続すると、Dex1 state 受信待ちがタイムアウトでクラッシュする
前にグリッパ閉コマンドが無条件で publish されていたことが判明。詳細は
``harvest_module.py`` の ``gripper_live`` docstring 参照）。腹部かご投入の3点直接経路は
MuJoCo でのみ自己衝突検証済み（``basket_deposit.py`` の SAFETY 注記参照）— 初回 LIVE
実行は必ずログで q_sol を確認し、e-stop を手元に。

2026-09-08 実機LIVE初回実行で、重力補償ON時に腕が中途半端な位置で止まる不具合が発生
した。原因は2つ: ① `HarvestModule` がカメラフレーム受信直後（起動から1秒未満）に即座に
IK到達を開始するため、`stiff_gravity_ramp_s`（既定5s）で0→100%に立ち上がる重力補償が
16-41%程度しか効いていない状態で腕を動かし、重力に負けて目標に届かなかった、
② IK粗アプローチが目標へ一発の関節空間リーチで、クリック駆動版
(`IkReachBridge.approach_above_m`)にあった「LIFT→TRANSIT→DESCEND」の段階的Cartesian
アプローチを持っていなかった。対策として `pregrasp_settle_s`（重力補償ランプ完了待ち、
`OKRA_GRAVITY_FF=1` 時のみ自動で `stiff_gravity_ramp_s` と同じ秒数を待つ）と
`ik_approach_above_m`（既定 0.08m、`OKRA_APPROACH_ABOVE_M` で調整）を追加した
（`harvest_module.py` / `ik_approach.py` 参照）。

同日、腕は前より正しく上がったがクリック駆動版と動きの質感が違う（手先軌道が
関節空間補間任せで直線的でない）との指摘を受け、`IkApproachSkill.stream_legs`
（`IkReachBridge._stream_leg` と同じ密度の Cartesian ストリーミング）を追加し、
既定 ON にした（`ik_stream_legs`/`OKRA_IK_STREAM_LEGS`）。

さらに同日、「グリッパの開閉やかご投入が起きていないように見える」との指摘で
`GraspSequence` のバグが判明: 切断（グリッパ閉）コマンドを送った**直後、待ち時間
ゼロで**籠投入（グリッパ開）へ進んでいたため、グリッパが実際に閉じきる前に開き
指令が飛んでいた。`cut_settle_s`（既定 1.5s、`OKRA_CUT_SETTLE_S`）で切断後の
整定待ちを追加した（`grasp_sequence.py` 参照）。`use_basket_deposit=True` と
併用する場合は必ず正の値にすること。

2026-09-08 実機LIVEでの追加要望により既定を 1.5s → 2.5s（+1s）に延長。

2026-09-08 移動を配線: LangGraph の ``reposition()``/``advance_left()``/``revisit()``
（[[SS-07-移動と足配置]]、「近すぎたら standoff_min までバックする」等のロジックは
既存）は今まで ``skills.relative_move`` が LIVE-TODO プレースホルダーで実体が
無かった。``G1HighLevelDdsSdk``（LocoClient, ``unitree_g1_okra_harvest_ik.py`` と
同じ統合方式）を追加し、``use_base_move`` を ``OKRA_MOVE_LIVE=1`` で有効化できる
ようにした（三段目の LIVE ゲート、既定 OFF — 腕/グリッパと同じ「明示 opt-in」
方針）。ローカル G1 解体新書（``05_サービスインターフェース/動作サービス.md``）
によれば `rt/arm_sdk` は Locked Stance / Movement Control 1 / Movement Control 2
の3 FSM でのみ使用可能で、歩行中の使用も仕様上サポートされている。cyclonedds の
型登録競合（`dds_init.py` の `channel_lock`）は `dds_sdk.py` 側で対処済み。

2026-09-08 実機LIVEで「音声より先に動作が実行されていて今何をしているか分からない」
との指摘を受けた。原因は2つ: ① G1SpeakerAnnouncer.say() はキュー投入後すぐ返り再生を
待たない、② graph.py の reposition/advance_left/revisit/next_station/swap_basket が
そもそも「動作 → 結果に応じて音声」の順でコードされていた。把持/移動/籠交換など
物理動作を伴うノードを「音声 → voice_lead_s 秒待機 → 動作」の順に並べ替え、待ち秒数を
`voice_lead_s`（既定 2.0s、``OKRA_VOICE_LEAD_S`` で調整）として追加した
（``HarvestConfig.voice_lead_s`` / ``graph.py`` 参照）。next_station だけは動作結果
（次拠点の有無）で言うべき内容が変わるため、まず常に真の「この場所は採り終わりました」
を言ってから待機・移動を実行し、次拠点が実在した場合のみ追加で「次の収穫場所に
移動します」を言う二段階アナウンスにした（``announce.station_done()`` 新設）。

前提条件:
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
from dimos.hardware.sensors.camera.zed.camera import ZEDCamera
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

# robot side (同じ命名を unitree_g1_okra_ik_only_grasp_zed.py / _ik_diffusion.py と揃える)
_NIC = os.getenv("ROBOT_INTERFACE", "enp46s0")
_LIVE = os.getenv("IK_REACH_LIVE", "").strip() == "1"
_ARM_VEL_LIMIT = float(os.getenv("IK_ARM_VEL_LIMIT", "20.0"))
_KP_ARM = float(os.getenv("OKRA_NOACT_KP_ARM", "80.0"))
_KD_ARM = float(os.getenv("OKRA_NOACT_KD_ARM", "3.0"))

# Right-arm gravity feedforward (unitree_g1_okra_ik_diffusion.py / _ik_only_grasp_zed.py
# と同じ意味の同じノブ)。既定 OFF。
_GRAVITY_FF = os.getenv("OKRA_GRAVITY_FF", "").strip() == "1"
_GRAVITY_TAU_SCALE = float(os.getenv("OKRA_GRAVITY_TAU_SCALE", "1.0"))
_GRAVITY_JOINTS = [int(v) for v in os.getenv("OKRA_GRAVITY_JOINTS", "0,1,2,3,4,5,6").split(",")]
_GRAVITY_TAU_LIMIT_NM = float(os.getenv("OKRA_GRAVITY_TAU_LIMIT_NM", "12.0"))
# 校正済み Dex1-1 単体(550g) URDF。f12c0dd37 参照。
_GRAVITY_URDF = os.getenv("OKRA_GRAVITY_URDF", "").strip() or (
    "dimos/robot/unitree/g1/g1_dex1_1_calibrated_550g.urdf"
)
# G1ArmSdkConnectionConfig.stiff_gravity_ramp_s の既定値(5.0s)と揃える。HarvestModule
# 側の pregrasp_settle_s にそのまま渡し、重力補償が0->100%に立ち上がるまで最初のIK到達
# を待たせる（2026-09-08 実機LIVEで判明した「重力補償が16-41%の状態で腕を動かして
# 目標に届かない」問題への対策。harvest_module.py の pregrasp_settle_s docstring参照）。
_GRAVITY_RAMP_S = float(os.getenv("OKRA_GRAVITY_RAMP_S", "5.0"))
# IK 粗アプローチを LIFT→TRANSIT→DESCEND の3段階Cartesian経路にする高さ[m]
# （IkApproachSkill.solve_legs/stream_legs 参照）。0 = 直接一発リーチ（レガシー）。
# クリック駆動版の OKRA_APPROACH_ABOVE_M と同じ考え方 — 既定 0.08m（runbook の
# 推奨値を踏襲）。
_IK_APPROACH_ABOVE_M = float(os.getenv("OKRA_APPROACH_ABOVE_M", "0.08"))
# 既定 ON: 各レグを IkReachBridge._stream_leg と同じ密度（3.5cm間隔・0.18s周期）で
# Cartesian ストリーミングし、手先が実際にほぼ直線を描くようにする（レグの端点だけを
# 関節空間補間でつなぐ簡易版=0 との違いは harvest_module.py の ik_stream_legs 参照）。
# クリック駆動版と同じ動きの質感にしたいというユーザー要望（2026-09-08）で追加。
_IK_STREAM_LEGS = os.getenv("OKRA_IK_STREAM_LEGS", "1").strip() == "1"
_IK_STREAM_STEP_M = float(os.getenv("OKRA_IK_STREAM_STEP_M", "0.035"))
_IK_STREAM_CADENCE_S = float(os.getenv("OKRA_IK_STREAM_CADENCE_S", "0.18"))
# 切断（グリッパ閉）指令の後、実際に閉じきるまで待つ秒数。0のまま use_basket_deposit=1
# だと、グリッパが閉じきる前に籠投入の開き指令が飛ぶ（2026-09-08 実機LIVEで確認 —
# gripper_grasp_on_reach.py の grasp_settle_s と同じ値をデフォルトにしている）。
_CUT_SETTLE_S = float(os.getenv("OKRA_CUT_SETTLE_S", "2.5"))
# §6 HMI: 把持/移動/籠交換など物理動作を伴うノードで、音声を発してから実際に動作を
# 送信するまで待つ秒数。G1SpeakerAnnouncer.say() はキュー投入後すぐ返るため、待たないと
# 動作が音声より先に（あるいは重なって）始まり、今何をしているか分からなくなる
# （2026-09-09 実機LIVEでの指摘。harvest_module.py / graph.py の voice_lead_s 参照）。
_VOICE_LEAD_S = float(os.getenv("OKRA_VOICE_LEAD_S", "2.0"))

_GRIP_KP = float(os.getenv("OKRA_GRIP_KP", "5.0"))
_GRIP_KD = float(os.getenv("OKRA_GRIP_KD", "0.05"))
_DEX1_PREFIX = os.getenv("OKRA_DEX1_PREFIX", "rt/dex1/left").strip()
# ⚠️ SAFETY 二段階ゲート（unitree_g1_okra_ik_only_grasp_zed.py と同じ命名）:
# G1GripperConnection には publish_cmd 相当の DRY-RUN 切替が無いため、HarvestModule
# 側の gripper_live をここでガードする。IK_REACH_LIVE だけでは腕しか保護されない —
# グリッパも実際に動かすには OKRA_NOACT_GRIP_LIVE=1 も必要（harvest_module.py 参照）。
_GRIP_LIVE = os.getenv("OKRA_NOACT_GRIP_LIVE", "").strip() == "1"

# ⚠️ SAFETY 三段目のゲート: 腕(_LIVE)・グリッパ(_GRIP_LIVE)と同じ「明示 opt-in」方針。
# G1HighLevelDdsSdk（LocoClient）はモジュールを組み込むだけで MotionSwitcher の
# SelectMode("ai") が start() 時に走る（実際の歩行コマンドは HarvestModule が
# use_base_move=True のときだけ reposition()/advance_left()/revisit() から
# cmd_vel 経由で送る）。IK_REACH_LIVE だけでは足回りは保護されない —
# 実際に歩かせるには OKRA_MOVE_LIVE=1 も必要（2026-09-08、[[SS-07-移動と足配置]]）。
_MOVE_LIVE = os.getenv("OKRA_MOVE_LIVE", "").strip() == "1"
_USE_BASE_MOVE = _LIVE and _MOVE_LIVE
# relative_move の変位[m]→タイムド速度指令の並進速度[m/s]。G1 LocoClient.Move
# (continuous_move=True) は ≥0.3 m/s でないと実際には歩き出さない
# （real_skills.py._BASE_SPEED 参照）。既定 0.8（ユーザー要望で 0.5→0.8、
# 2026-09-08、[[SS-07-移動と足配置]]の BASE_SPEED と同期させること）。
_BASE_SPEED = float(os.getenv("OKRA_BASE_SPEED", "0.8"))

# 切断（グリッパ閉）後、右腕のみで腹部固定かごへ IK 投入→開放まで進む(F-07)。
# 既定 ON — このブランチの要求（掴んだら次まで進む）に合わせる。0 で切断後は保持のまま止める。
_USE_BASKET_DEPOSIT = os.getenv("OKRA_BASKET_DEPOSIT", "1").strip() == "1"

# ZED→torso ハンドアイ外部パラメータ（HarvestModuleConfig.cam_to_torso_xyzquat、
# "x,y,z,qx,qy,qz,qw"）。既定値は unitree_g1_okra_ik_only_grasp_zed.py の
# ZED_MOUNT_XYZRPY（2026-07-16、肩ピッチ軸からのテープ+IMU実測、torso<-ZED body）を
# 同じ胸ZED光学フレーム変換になるよう xyzquat 形式へ変換したもの（OPTICAL_ROTATION合成
# 込み）。2026-09-08 実機LIVEで、この値を設定していない（空文字=カメラ座標そのまま）
# 状態だと検出座標がワークスペース外と誤判定されIKが失敗し続けることを確認。
# 平行移動は 2026-09-15 に CAD 実測値へ更新（honban.py と同一値 — 校正値はブループリント
# ではなく**カメラの取り付けという物理**の属性なので、同じ機体では一致していなければ
# ならない）。回転は未測定のため据え置き。詳細は unitree_g1_okra_honban.py のコメント。
_CAM_TO_TORSO = os.getenv(
    "OKRA_CAM_TO_TORSO",
    "0.1110,0.0250,0.2585,-0.49475,0.49475,-0.50520,0.50520",
)

_MODULES = [
    ZEDCamera.blueprint(depth_mode=os.getenv("ZED_DEPTH_MODE", "NEURAL")),
    HarvestModule.blueprint(
        use_dummy=False,
        use_zed_depth=True,
        use_g1_speaker=True,
        network_interface=_NIC,
        # 既定 "" = VLM 不使用。切断可否ゲート・把持後検証とも常時許可になる
        # （このブランチの要求: VLM なし）。VLM を使いたい場合のみ設定する。
        vlm_model=os.getenv("OKRA_VLM_MODEL", ""),
        # オクラ専用 seg 重み（data/models_yolo/okra11n-seg.pt）。
        yolo_model=os.getenv("OKRA_YOLO_MODEL", "okra11n-seg.pt"),
        target_classes=os.getenv("OKRA_TARGET", "okra"),
        # ZED→torso ハンドアイ外部パラメータ（既定はクリック駆動版の実測値を移植した
        # 暫定値。空文字（OKRA_CAM_TO_TORSO=""）でカメラ座標系そのまま=未校正扱いに戻せる）。
        cam_to_torso_xyzquat=_CAM_TO_TORSO,
        # 把持: IK 粗アプローチ→(ACT なし)→切断可否(VLM なし=常許可)→切断（グリッパ閉）。
        use_ik_grasp_sequence=True,
        use_act_grasp=False,
        cut_close_q=float(os.getenv("OKRA_CUT_CLOSE_Q", "4.4")),
        blade_max_q=float(os.getenv("OKRA_BLADE_MAX_Q", "5.2")),
        cut_settle_s=_CUT_SETTLE_S,
        use_basket_deposit=_USE_BASKET_DEPOSIT,
        # 二段階ゲート: IK_REACH_LIVE と OKRA_NOACT_GRIP_LIVE の両方が立たないと
        # gripper_target は実際には publish されない（harvest_module.py 参照）。
        gripper_live=(_LIVE and _GRIP_LIVE),
        # 重力補償ON時のみ、そのランプ完了(既定5s)を待ってから最初のIK到達を始める。
        # 重力補償OFFならランプ待ちの意味が無いので0のまま(待たない)。
        pregrasp_settle_s=(_GRAVITY_RAMP_S if _GRAVITY_FF else 0.0),
        ik_approach_above_m=_IK_APPROACH_ABOVE_M,
        ik_stream_legs=_IK_STREAM_LEGS,
        ik_stream_step_m=_IK_STREAM_STEP_M,
        ik_stream_cadence_s=_IK_STREAM_CADENCE_S,
        # 移動（再配置/掃引, [[SS-07-移動と足配置]]）。OKRA_MOVE_LIVE=1 のときのみ
        # 実際に cmd_vel を publish する。False のままなら real_skills.py の
        # LIVE-TODO プレースホルダーのまま（今までどおり動かない）。
        use_base_move=_USE_BASE_MOVE,
        base_speed=_BASE_SPEED,
        # §6 HMI: 音声を発してから物理動作（把持/移動/籠交換）を送るまでの待ち秒数。
        voice_lead_s=_VOICE_LEAD_S,
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
        # 既定250(rate_hz=250Hzに対し約1秒に1回)は自動収穫ループでは冗長すぎる
        # （2026-09-08 ユーザー要望）。既定を10秒に1回程度まで落とす。デバッグ時は
        # OKRA_ARM_LOG_EVERY_N=250 等で元に戻せる。
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
    # OKRA_MOVE_LIVE=1（かつ IK_REACH_LIVE=1）のときだけ組み込む — モジュールを
    # 足すだけで MotionSwitcher.SelectMode("ai") が start() 時に走るため、DRY-RUN
    # では一切ロードしない（unitree_g1_okra_harvest_ik.py と同じ統合方式）。
    _MODULES.append(G1HighLevelDdsSdk.blueprint(network_interface=_NIC))

_approach_note = (
    f"approach_above_m={_IK_APPROACH_ABOVE_M} "
    f"pregrasp_settle_s={(_GRAVITY_RAMP_S if _GRAVITY_FF else 0.0):.1f} "
    f"stream_legs={_IK_STREAM_LEGS} cut_settle_s={_CUT_SETTLE_S:.1f} "
    f"cam_to_torso={'set' if _CAM_TO_TORSO else 'UNSET(camera-frame passthrough)'}"
)
_move_note = (
    f"move={'LIVE(LocoClient)' if _USE_BASE_MOVE else 'DRY-RUN(placeholder)'} "
    f"base_speed={_BASE_SPEED:.2f}m/s voice_lead_s={_VOICE_LEAD_S:.1f}"
)
if _LIVE and _GRIP_LIVE:
    logger.warning(
        "unitree_g1_okra_harvest_zed LAUNCHING **LIVE (arm+gripper)** -- YOLO detects "
        f"automatically (no click needed) and the arm+gripper WILL move via rt/arm_sdk / "
        f"rt/dex1 on NIC {_NIC!r} at <= {_ARM_VEL_LIMIT} rad/s. grasp=IK->cut(no-ACT, no-VLM) "
        f"gravity_ff={_GRAVITY_FF} urdf={_GRAVITY_URDF!r} {_approach_note} "
        f"basket_deposit={'ON' if _USE_BASKET_DEPOSIT else 'OFF'} {_move_note}. "
        f"{'⚠️⚠️ BASE WILL WALK (LocoClient, OKRA_MOVE_LIVE=1). ' if _USE_BASE_MOVE else ''}"
        "Keep an e-stop in hand."
    )
elif _LIVE:
    logger.warning(
        "unitree_g1_okra_harvest_zed LAUNCHING **LIVE (arm only)** -- the arm WILL move via "
        f"rt/arm_sdk on NIC {_NIC!r}, but the GRIPPER stays DRY-RUN (set "
        "OKRA_NOACT_GRIP_LIVE=1 to also close/open it). grasp=IK->cut(no-ACT, no-VLM) "
        f"gravity_ff={_GRAVITY_FF} {_approach_note} "
        f"basket_deposit={'ON' if _USE_BASKET_DEPOSIT else 'OFF'} {_move_note}."
    )
else:
    logger.info(
        f"unitree_g1_okra_harvest_zed DRY-RUN (set IK_REACH_LIVE=1 and OKRA_NOACT_GRIP_LIVE=1 "
        f"to drive arm+gripper, +OKRA_MOVE_LIVE=1 to also drive the base). NIC={_NIC!r}. "
        f"grasp=IK->cut(no-ACT, no-VLM) "
        f"gravity_ff={_GRAVITY_FF} {_approach_note} "
        f"basket_deposit={'ON' if _USE_BASKET_DEPOSIT else 'OFF'} {_move_note}."
    )

unitree_g1_okra_harvest_zed = (
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

__all__ = ["unitree_g1_okra_harvest_zed"]
