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

"""LAPTOP app: 重力補償のみでの姿勢保持テスト(IK・カメラ・グリッパー無し)。

G1ArmSdkConnection を単体で collection_mode 起動し、g(q) の重力フィードフォワード
トルクだけ(kp を 0 までランプダウン)で、人間が手で動かした右腕をその場に保持できる
かを確認するための最小構成。IkReachBridge によるリーチは行わない — 起動直後は
現在の実測姿勢を STIFF に保持するだけで、腕は勝手に動かない。

流れ:
  1. 起動直後: 右腕は現在の実測姿勢を STIFF 保持(他の arm_sdk ブループリントと同じ
     安全な初期化。arm_target は誰も publish しないので、放置しても動かない)。
  2. 別ターミナルで scripts/gravity_hold_toggle.py を実行し 'c' を押す
     -> /g1/reach_done を publish -> 右腕が COMPLIANT になる
        (kp を compliant_kp_ramp_s で 0 まで滑らかにランプダウン + 重力FFトルク
         tau = g(q) * gravity_tau_scale を投入。左腕・腰は STIFF のまま)。
  3. 人間が右腕を持って動かし、好きな位置で手を離す -> 位置PDには頼らず g(q) のみで
     その場に留まるかを目視確認する(kd_compliant はわずかな粘性減衰のみ)。
  4. 新しい arm_target が来れば再スティッフ化するが、このブループリントは arm_target
     を一切 publish しないので、Ctrl+C で安全に停止する(weight を 1->0 にランプダウン
     しつつ、停止直前に右腕を再スティッフ化してから手を離すハンドオフを行う)。

重力モデルURDFの切り替え(OKRA_GRAVITY_URDF):
  (未設定)                                                    -> g1.urdf(ダミーラバーハンド170g)
  dimos/robot/unitree/g1/g1_dex1_1_official.urdf               -> Unitree公式Dex1-1込みURDF(365g)
    ※ 2026-09-04時点の既知の乖離: 実測Dex1-1単体=546g、本URDFの計算値=365g
       (約1.5倍過小評価)。詳細は g1_dex1_1_official.urdf のヘッダーコメント参照。

SAFETY:
- DRY-RUN(既定): IK_REACH_LIVE 未設定 -> publish_cmd=False。rt/arm_sdk には一切書き込まない。
- LIVE: IK_REACH_LIVE=1 で起動。'c' を押した瞬間に右腕が脱力方向へ動くので、
  必ず押す前に右腕を手で支え、E-stop をすぐ押せる状態で行うこと。

実行: bash dimos/robot/unitree/g1/examples/start_gravity_hold_test.sh [--live]
      (その後、別ターミナルで scripts/gravity_hold_toggle.py を実行)
"""

from __future__ import annotations

import os

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.Bool import Bool
from dimos.robot.unitree.g1.act.g1_arm_sdk_connection import G1ArmSdkConnection
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_NIC = os.getenv("ROBOT_INTERFACE", "enp46s0")
_LIVE = os.getenv("IK_REACH_LIVE", "").strip() == "1"
# 重力モデルURDF。空 = g1_arm_sdk_connection.py 既定(g1.urdf, ダミーハンド170g)。
_GRAVITY_URDF = os.getenv("OKRA_GRAVITY_URDF", "").strip()
# 重力フィードフォワードのスケール。実測質量がURDFより重い場合は 1.0 より大きくして
# 効きを強める調整に使う(例: 実測546g/URDF365g ≈ 1.5 を試す、など)。
_GRAVITY_TAU_SCALE = float(os.getenv("OKRA_GRAVITY_TAU_SCALE", "1.0"))
# コンプライアント化にかける時間 [s]。急に脱力させず滑らかにkpを落とす。
_COMPLIANT_KP_RAMP_S = float(os.getenv("OKRA_COMPLIANT_KP_RAMP_S", "1.5"))

_grav_urdf_label = _GRAVITY_URDF or "g1.urdf (default, dummy rubber hand 170g)"
if _LIVE:
    logger.warning(
        f"unitree-g1-gravity-hold-test LAUNCHING **LIVE** — right arm holds its CURRENT "
        f"measured pose STIFFLY until scripts/gravity_hold_toggle.py sends 'c' "
        f"(/g1/reach_done), then it goes COMPLIANT (kp->0 over {_COMPLIANT_KP_RAMP_S}s + "
        f"gravity tau, scale={_GRAVITY_TAU_SCALE}). gravity model urdf={_grav_urdf_label!r}. "
        f"SUPPORT THE RIGHT ARM before pressing 'c'. E-stop in hand."
    )
else:
    logger.info(
        f"unitree-g1-gravity-hold-test DRY-RUN (set IK_REACH_LIVE=1 to drive the arm). "
        f"gravity model urdf={_grav_urdf_label!r}, tau_scale={_GRAVITY_TAU_SCALE}."
    )

unitree_g1_gravity_hold_test = autoconnect(
    G1ArmSdkConnection.blueprint(
        network_interface=_NIC,
        publish_cmd=_LIVE,
        collection_mode=True,
        urdf_path=_GRAVITY_URDF,
        gravity_tau_scale=_GRAVITY_TAU_SCALE,
        compliant_kp_ramp_s=_COMPLIANT_KP_RAMP_S,
    ),
).transports(
    {
        ("motor_states", JointState): LCMTransport("/g1/motor_states", JointState),
        ("arm_target", JointState): LCMTransport("/g1/arm_target", JointState),
        ("reach_done", Bool): LCMTransport("/g1/reach_done", Bool),
    }
)

__all__ = ["unitree_g1_gravity_hold_test"]
