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

"""unitree-g1-model-no-kensho 用の人間トリガー。Enterを押すと、その瞬間の
ロボットの姿勢のまま切断シーケンスへ進む合図（cut_trigger）を送る。

背景: DimOS のモジュールは別プロセス（ワーカー）で実行されるため、モジュール
内で ``input()`` を使っても実際のターミナル入力には反応しない
（unitree_g1_teach_pregrasp_pose.py の教示ツールで2026-09-14に確認済み）。
本スクリプトはDimOSのモジュールではなく、独立したPythonプロセスとして動く
ため ``input()`` が正常に機能する。Enterを検知したら、DimOSの通常の
モジュール間通信（LCM）に乗せて ``/g1/model_cut_trigger`` を1回publishする
だけの薄いブリッジで、ファイルタッチ方式（teach_pose_logger と同じ手法）の
代わりにDimOSネイティブの経路を使う（2026-09-16 ユーザー要望）。

収穫プログラム側（unitree_g1_model_no_kensho.py、UmiDiffusionBridge）は
このトピックを購読しており、受け取ると収束を待たずに現在位置のまま
adjust_done を返し、③切断可否→④切断へ進む。

実行（収穫プログラムと同じマシン、別ターミナルで）:
  .venv/bin/python oda/press_enter_to_cut.py

操作: Enterを押すたびに1回送信する。'q'+Enterで終了。
"""

from __future__ import annotations

from dimos.core.transport import LCMTransport
from dimos.msgs.std_msgs.Bool import Bool
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_TOPIC = "/g1/model_cut_trigger"


def main() -> None:
    pub = LCMTransport(_TOPIC, Bool)
    pub.start()
    print(f"unitree-g1-model-no-kensho 用トリガー送信ツール（topic={_TOPIC}）")
    print("モデル推論中に Enter を押すと、今の位置のまま切断シーケンスへ進みます。")
    print("'q' + Enter で終了。")
    try:
        while True:
            s = input("\n[Enterで送信 / q で終了] > ").strip().lower()
            if s == "q":
                break
            pub.broadcast(None, Bool(data=True))
            print(f"  -> {_TOPIC} へ送信しました。")
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        pub.stop()


if __name__ == "__main__":
    main()
