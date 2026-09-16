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

"""``build_harvest_graph()`` の構造を React Flow 用の JSON に変換する。

``build_harvest_graph()`` は :class:`DummyHarvestSkills` で組み立てて
``CompiledStateGraph.get_graph()``（LangGraph 標準の ``Graph`` = 旧
``DrawableGraph``）を取得する。これはノード一覧とエッジ一覧
（``source``/``target``/``conditional``）を持つが、条件分岐エッジが
「ルーターがどの文字列を返したらどのノードに飛ぶか」というラベルまでは
持っていない（``Edge.data`` は常に ``None``）。そのラベルは
``CompiledStateGraph.builder.branches`` （``StateGraph.add_conditional_edges``
の第3引数としてコンパイル前に保持されているマッピング表）にしか残っていない
ため、本モジュールは両方を突き合わせてラベル付きのエッジ情報を作る。

このモジュールが返す JSON は GUI（React Flow）がそのまま読み込める形。
ノードの中身（Python 実装）やルーターの判定ロジックはここでは一切扱わない
—— 扱うのは配線（どのノードがどう繋がっているか）だけ。
"""

from __future__ import annotations

from collections import deque
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from dimos.robot.unitree.g1.harvest.blackboard import HarvestConfig
from dimos.robot.unitree.g1.harvest.dummy_skills import DummyHarvestSkills
from dimos.robot.unitree.g1.harvest.graph import build_harvest_graph

# LangGraph が予約している開始/終了の擬似ノードID。
START_ID = "__start__"
END_ID = "__end__"

# レイアウトのグリッド間隔 [px]（GUI初期表示用の適当な値）。
_COL_SPACING = 220
_ROW_SPACING = 140


def _build_dummy_compiled_graph() -> CompiledStateGraph[Any, Any, Any, Any]:
    """検証・GUI表示専用に ``DummyHarvestSkills`` で harvest グラフを組み立てる。"""
    skills = DummyHarvestSkills(num_okra=3, stations=1)
    return build_harvest_graph(skills, HarvestConfig())


def _auto_layout(node_ids: list[str], adjacency: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    """BFS で段組みする簡易自動レイアウト（GUI初期表示用）。

    ``__start__`` を起点に BFS した到達段数を縦位置（行）に、同じ段の中の
    出現順を横位置（列）に使う。サイクルを含むグラフなので厳密なDAG的な
    層分けにはならないが、初期表示としては十分読める配置になる。
    """
    level: dict[str, int] = {}
    order_in_level: dict[int, list[str]] = {}

    start = START_ID if START_ID in node_ids else node_ids[0]
    level[start] = 0
    order_in_level[0] = [start]
    queue: deque[str] = deque([start])
    while queue:
        node = queue.popleft()
        for nxt in adjacency.get(node, []):
            if nxt in level:
                continue
            lvl = level[node] + 1
            level[nxt] = lvl
            order_in_level.setdefault(lvl, []).append(nxt)
            queue.append(nxt)

    # BFSで到達しなかったノード（孤立ノード）は末尾の行にまとめて置く。
    unreached = [n for n in node_ids if n not in level]
    if unreached:
        lvl = (max(level.values()) + 1) if level else 0
        for n in unreached:
            level[n] = lvl
            order_in_level.setdefault(lvl, []).append(n)

    positions: dict[str, dict[str, float]] = {}
    for lvl, names in order_in_level.items():
        for col, name in enumerate(names):
            positions[name] = {"x": float(col * _COL_SPACING), "y": float(lvl * _ROW_SPACING)}
    return positions


def _router_edge_labels(
    builder: Any,
) -> dict[tuple[str, str], list[str]]:
    """``builder.branches`` から (source, target) -> [ラベル,...] の対応表を作る。

    ``branches`` は ``{source: {router_fn_name: BranchSpec(ends={label: target})}}``
    という形をしている（``StateGraph.add_conditional_edges`` の第3引数がそのまま
    ``BranchSpec.ends`` として保持されている）。同じ (source, target) に複数の
    ラベルが対応することは今回のグラフには無いが、一般化のためリストで持つ。
    """
    labels: dict[tuple[str, str], list[str]] = {}
    branches = getattr(builder, "branches", {}) or {}
    for source, router_map in branches.items():
        for _router_name, branch_spec in router_map.items():
            ends = getattr(branch_spec, "ends", None) or {}
            for label, target in ends.items():
                labels.setdefault((source, target), []).append(str(label))
    return labels


def build_graph_json(app: CompiledStateGraph[Any, Any, Any, Any] | None = None) -> dict[str, Any]:
    """harvest グラフの構造を React Flow 用 JSON に変換する。

    Args:
        app: 変換対象のコンパイル済みグラフ。``None`` の場合は
            ``DummyHarvestSkills`` で組み立てた既定の harvest グラフを使う
            （``GET /api/graph`` から呼ばれる通常の使い方）。

    Returns:
        ``{"nodes": [...], "edges": [...]}`` 形式の dict。
        ``nodes`` の各要素は ``{"id","label","position"}``。
        ``edges`` の各要素は ``{"id","source","target","conditional","label"}``。
    """
    compiled = app if app is not None else _build_dummy_compiled_graph()
    graph = compiled.get_graph()

    node_ids = list(graph.nodes.keys())
    adjacency: dict[str, list[str]] = {}
    for edge in graph.edges:
        adjacency.setdefault(edge.source, []).append(edge.target)
    positions = _auto_layout(node_ids, adjacency)

    router_labels = _router_edge_labels(getattr(compiled, "builder", None))

    nodes: list[dict[str, Any]] = []
    for node_id, node in graph.nodes.items():
        if node_id == START_ID:
            label = "START"
        elif node_id == END_ID:
            label = "END"
        else:
            label = node_id
        nodes.append(
            {
                "id": node_id,
                "label": label,
                "position": positions.get(node_id, {"x": 0.0, "y": 0.0}),
            }
        )

    edges: list[dict[str, Any]] = []
    # 同じ source/target の組が複数回現れることは無い想定だが、念のため
    # 連番でユニークな id を振る。
    seen_pairs: dict[tuple[str, str], int] = {}
    for edge in graph.edges:
        pair = (edge.source, edge.target)
        seen_pairs[pair] = seen_pairs.get(pair, 0) + 1
        suffix = "" if seen_pairs[pair] == 1 else f"_{seen_pairs[pair]}"
        edge_id = f"e__{edge.source}__{edge.target}{suffix}"

        label: str | None = None
        if edge.conditional:
            candidates = router_labels.get(pair, [])
            label = candidates[0] if candidates else edge.target

        edges.append(
            {
                "id": edge_id,
                "source": edge.source,
                "target": edge.target,
                "conditional": bool(edge.conditional),
                "label": label,
            }
        )

    return {"nodes": nodes, "edges": edges}


__all__ = ["END_ID", "START_ID", "build_graph_json"]
