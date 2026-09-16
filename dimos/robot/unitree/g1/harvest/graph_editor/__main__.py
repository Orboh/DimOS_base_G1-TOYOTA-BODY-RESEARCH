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

"""``python -m dimos.robot.unitree.g1.harvest.graph_editor`` のエントリポイント。

harvest グラフ・エディタの FastAPI アプリを uvicorn で起動する。
"""

from __future__ import annotations

import argparse

import uvicorn

# 既存サービスと衝突しなさそうな適当なポート番号。
_DEFAULT_PORT = 8420


def main() -> None:
    """コマンドライン引数を読み、``uvicorn`` でサーバーを起動する。"""
    parser = argparse.ArgumentParser(
        description="harvest グラフ・エディタ（GUIでLangGraph StateGraphの配線を可視化・編集する）"
    )
    parser.add_argument("--host", default="127.0.0.1", help="バインドするホスト（既定: 127.0.0.1）")
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT, help=f"待ち受けポート（既定: {_DEFAULT_PORT}）")
    parser.add_argument("--reload", action="store_true", help="コード変更を自動リロードする（開発用）")
    args = parser.parse_args()

    uvicorn.run(
        "dimos.robot.unitree.g1.harvest.graph_editor.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
