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

"""harvest グラフ・エディタの FastAPI アプリ。

やることは3つだけ:

* ``GET /api/graph``  — 現行の harvest グラフ構造を JSON で返す（:mod:`introspect`）
* ``POST /api/export`` — 編集後の nodes/edges JSON から ``StateGraph`` 構築コードを返す（:mod:`codegen`）
* ``GET /``           — ``static/index.html``（React Flow の単一ページGUI）を配信する

ビルドステップ無し・バンドラー無し。フロントは ``static/index.html`` 1枚。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dimos.robot.unitree.g1.harvest.graph_editor.codegen import (
    GraphCodegenError,
    generate_stategraph_code,
)
from dimos.robot.unitree.g1.harvest.graph_editor.introspect import build_graph_json
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_STATIC_DIR = Path(__file__).resolve().parent / "static"


class GraphJson(BaseModel):
    """React Flow のノード・エッジ JSON（GUIの編集結果）。

    形は意図的に緩め（各要素は自由な dict）にしてある —— GUI 側の
    React Flow ノード/エッジオブジェクトをほぼそのまま受け取れるようにする
    ため。厳密なバリデーションは :mod:`codegen` 側で行う。
    """

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]


class ExportResponse(BaseModel):
    """生成された Python コードを1つだけ持つレスポンス。"""

    code: str


def create_app() -> FastAPI:
    """FastAPI アプリを組み立てる（テストや ``__main__`` から呼べるようファクトリ化）。"""
    app = FastAPI(
        title="Harvest Graph Editor",
        description="LangGraph StateGraph（harvest）の構造をGUIで可視化・編集するツール",
    )

    @app.get("/api/graph")
    def get_graph() -> dict[str, Any]:
        """現行の harvest グラフ構造（12ノード・固定エッジ・条件分岐）を返す。"""
        try:
            return build_graph_json()
        except Exception as exc:
            logger.exception("harvest グラフの読み込みに失敗しました")
            raise HTTPException(
                status_code=500, detail=f"グラフの読み込みに失敗しました: {exc}"
            ) from exc

    @app.post("/api/export")
    def export_graph(payload: GraphJson) -> ExportResponse:
        """編集後の nodes/edges から ``StateGraph`` 構築コードを生成して返す。"""
        try:
            code = generate_stategraph_code(payload.nodes, payload.edges)
        except GraphCodegenError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("コード生成に失敗しました")
            raise HTTPException(
                status_code=400, detail=f"コード生成に失敗しました: {exc}"
            ) from exc
        return ExportResponse(code=code)

    # 静的ファイル（index.html 一枚）を "/" 配下に配信する。
    # 上の @app.get/@app.post を先に登録しているので、/api/* はそちらが
    # 優先的にマッチし、それ以外のパス（"/" 含む）だけがここに落ちる。
    if _STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
    else:
        logger.warning("static ディレクトリが見つかりません: %s", _STATIC_DIR)

    return app


app = create_app()

__all__ = ["ExportResponse", "GraphJson", "app", "create_app"]
