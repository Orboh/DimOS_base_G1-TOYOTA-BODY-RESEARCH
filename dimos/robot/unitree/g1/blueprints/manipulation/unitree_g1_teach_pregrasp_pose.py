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

"""Blueprint: honban(unitree_g1_okra_honban.py)の準備姿勢(pregrasp pose)を教示するツール。

    dimos run unitree-g1-teach-pregrasp-pose

honban の front-approach（align）は「今の手先の奥行き(X)はそのまま」Y,Zを合わせる
設計だが、G1 の休憩姿勢（腕を下げた状態）の手先XはIKワークスペース下限(0.05m)を
わずかに下回っており、align 初手からworkspace外/関節可動域超過で失敗する
（2026-09-14 実機LIVEで確認）。IKで座標から自動計算した準備姿勢は解けはするが
「人間工学的に妥当か」までは保証しない。本ツールは実際に人の手で動かして決めた
姿勢の関節角度をそのまま使えるようにする（キネステティック教示）。

使い方（別ターミナルから ``touch`` する — Enterキー入力ではない。DimOSはモジュール
を別プロセスで実行するため、モジュール内の ``input()`` は起動元ターミナルの
キー入力を受け取れない。2026-09-14実機LIVEで確認済み）:
  1. ``IK_REACH_LIVE=1`` で起動する（付けないとDRY-RUNで何も送信されない）
  2. 別ターミナルで ``touch /tmp/teach_go``（``TEACH_GO_FILE``で変更可）
     → ``TEACH_COMPLIANT_DELAY_S``（既定2秒）後に右腕が自動的にコンプライアント
     （kp->0 + 重力補償トルクのみ）になる。**touchしてから脱力までの間に右腕を
     支える準備をしておくこと** — 左腕・腰は剛性のまま、右腕だけが脱力する。
  3. 右腕を手で持って好きな準備姿勢へ動かす。ログに ``TEACH_LOG_INTERVAL_S``
     （既定1秒）間隔で現在の右腕7関節角度(q7, 正準順)が表示され続ける。
  4. 良い姿勢になったら別ターミナルで ``touch /tmp/teach_save``
     （``TEACH_SAVE_FILE``で変更可）→ その瞬間の関節角度が ``TEACH_SAVE_PATH``
     （既定 ``/tmp/pregrasp_pose_candidates.log``）に1行追記保存される
     （何度でも touch して複数候補を比較できる）。
  5. Ctrl+C で終了。保存された行から選んだ値を
     ``unitree_g1_okra_honban.py`` 起動時の
     ``OKRA_PREGRASP_POSE_Q7="<7つの値をカンマ区切り>"`` に設定する。

``collection_mode`` 自体は 2026-06-24 に実機で検証済みの既存機能（キネステ
ティック教示用、``g1_arm_sdk_connection.py`` 参照）をそのまま使っている。本
ブループリントが新規に足すのは ``TeachPoseLogger``（touchトリガーのreach_done
+ 定期ログ + touchでの保存）のみ。
"""

from __future__ import annotations

import os

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.Bool import Bool
from dimos.robot.unitree.g1.act.g1_arm_sdk_connection import G1ArmSdkConnection
from dimos.robot.unitree.g1.harvest.teach_pose_logger import TeachPoseLogger
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_NIC = os.getenv("ROBOT_INTERFACE", "enp46s0")
_LIVE = os.getenv("IK_REACH_LIVE", "").strip() == "1"
# 重力モデルURDF: honban既定と同じ、校正済みDex1-1(550g)モデル。
_GRAVITY_URDF = os.getenv("OKRA_GRAVITY_URDF", "").strip() or (
    "dimos/robot/unitree/g1/g1_dex1_1_calibrated_550g.urdf"
)
_LOG_INTERVAL_S = float(os.getenv("TEACH_LOG_INTERVAL_S", "1.0"))
_COMPLIANT_DELAY_S = float(os.getenv("TEACH_COMPLIANT_DELAY_S", "2.0"))
_SAVE_PATH = os.getenv("TEACH_SAVE_PATH", "/tmp/pregrasp_pose_candidates.log")
_GO_TRIGGER_FILE = os.getenv("TEACH_GO_FILE", "/tmp/teach_go")
_SAVE_TRIGGER_FILE = os.getenv("TEACH_SAVE_FILE", "/tmp/teach_save")

_MODULES = [
    G1ArmSdkConnection.blueprint(
        network_interface=_NIC,
        publish_cmd=_LIVE,
        collection_mode=True,
        urdf_path=_GRAVITY_URDF,
    ),
    TeachPoseLogger.blueprint(
        log_interval_s=_LOG_INTERVAL_S,
        compliant_delay_s=_COMPLIANT_DELAY_S,
        save_path=_SAVE_PATH,
        go_trigger_file=_GO_TRIGGER_FILE,
        save_trigger_file=_SAVE_TRIGGER_FILE,
    ),
]

if _LIVE:
    logger.warning(
        f"unitree_g1_teach_pregrasp_pose LAUNCHING **LIVE** -- from another terminal, "
        f"`touch {_GO_TRIGGER_FILE}`, then after {_COMPLIANT_DELAY_S:.1f}s the RIGHT arm "
        f"goes COMPLIANT (kp->0 + gravity feedforward only, urdf={_GRAVITY_URDF!r}) so it "
        "can be hand-guided to a pregrasp pose. LEFT arm and waist stay stiff at their "
        f"current pose. SUPPORT the right arm between the touch and the "
        f"{_COMPLIANT_DELAY_S:.1f}s elapsing -- once compliant it holds on gravity "
        f"feedforward alone, not full position control. `touch {_SAVE_TRIGGER_FILE}` at "
        f"any good pose to save its q7 to {_SAVE_PATH!r}. Ctrl+C to stop."
    )
else:
    logger.info(
        "unitree_g1_teach_pregrasp_pose DRY-RUN (set IK_REACH_LIVE=1 to actually put "
        "the right arm in gravity-only compliant mode; without it nothing is "
        "written to rt/arm_sdk)."
    )

unitree_g1_teach_pregrasp_pose = autoconnect(*_MODULES).transports(
    {
        ("motor_states", JointState): LCMTransport("/g1/motor_states", JointState),
        ("reach_done", Bool): LCMTransport("/reach_done", Bool),
    }
)

__all__ = ["unitree_g1_teach_pregrasp_pose"]
