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

"""GUI で編集した nodes/edges から ``StateGraph`` 構築コードを生成する。

このモジュールが持つ責務は「配線（グラフ構造）」の再現だけである。ノードの
中身（Pythonロジック）やルーター関数の判定ロジック自体はGUIの編集対象外な
ので、ここでも一切生成しない —— 生成するのはプレースホルダーのコメント/
``NotImplementedError`` のみ。

ノード ``id`` が元の harvest グラフに実在する関数名（:data:`KNOWN_NODE_IDS`）
と一致する場合は「既存の関数への参照」として扱い、
``dimos.robot.unitree.g1.harvest.graph`` からインポートするコードを生成する。
一致しない場合（GUI上で新規追加した、またはリネームしたノード）は、その
関数がまだ存在しないとみなしてプレースホルダー関数定義を生成する。
"""

from __future__ import annotations

import re
from typing import Any

# introspect.py が読み込む元の harvest グラフに実在するノード名（= 関数名）。
# GUI 上のノード id がこの集合に含まれていれば「既存の関数への参照」として
# 扱い、含まれていなければ新規/リネームとみなしてプレースホルダーを生成する。
KNOWN_NODE_IDS: frozenset[str] = frozenset(
    {
        "detect",
        "select",
        "grasp",
        "verify",
        "record",
        "give_up",
        "reposition",
        "advance_left",
        "revisit",
        "next_station",
        "swap_basket",
        "finish",
    }
)

START_ID = "__start__"
END_ID = "__end__"

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class GraphCodegenError(ValueError):
    """入力の nodes/edges JSON が不正でコード生成できないときの例外。"""


def _safe_identifier(raw: str, fallback: str) -> str:
    """任意の文字列を Python の識別子として使える形に正規化する。"""
    candidate = raw.strip() if raw else ""
    candidate = re.sub(r"\W", "_", candidate) if candidate else fallback
    if not candidate:
        candidate = fallback
    if candidate[0].isdigit():
        candidate = f"n_{candidate}"
    return candidate if _IDENT_RE.match(candidate) else fallback


def _node_var_name(node_id: str, used: dict[str, str]) -> str:
    """ノード id から重複しない Python 識別子を作る（キャッシュ付き）。"""
    if node_id in used:
        return used[node_id]
    base = _safe_identifier(node_id, fallback="node")
    name = base
    i = 2
    existing = set(used.values())
    while name in existing:
        name = f"{base}_{i}"
        i += 1
    used[node_id] = name
    return name


def generate_stategraph_code(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    """編集後の nodes/edges から ``StateGraph`` 構築コードの文字列を生成する。

    Args:
        nodes: ``{"id","label","position"}`` 形式のノード一覧
            （``__start__``/``__end__`` を含んでいてもよい。含まれない場合は
            START→END 直結として扱う）。
        edges: ``{"id","source","target","conditional","label"}`` 形式の
            エッジ一覧。

    Returns:
        そのまま貼り付けて使える ``StateGraph`` 構築コードの文字列。

    Raises:
        GraphCodegenError: ノードが1つも無い、またはエッジが存在しないノード
            を参照しているなど、コード生成できない入力だったとき。
    """
    if not isinstance(nodes, list) or not nodes:
        raise GraphCodegenError("nodes が空です。最低1つのノードが必要です。")

    node_ids: list[str] = []
    seen_ids: set[str] = set()
    for n in nodes:
        node_id = str(n.get("id", "")).strip()
        if not node_id:
            raise GraphCodegenError(f"id が空のノードがあります: {n!r}")
        if node_id in seen_ids:
            raise GraphCodegenError(f"ノード id が重複しています: {node_id!r}")
        seen_ids.add(node_id)
        node_ids.append(node_id)

    real_node_ids = [n for n in node_ids if n not in (START_ID, END_ID)]
    valid_ids = set(node_ids) | {START_ID, END_ID}

    for e in edges:
        source, target = str(e.get("source", "")), str(e.get("target", ""))
        if source not in valid_ids:
            raise GraphCodegenError(f"エッジの source が未知のノードです: {source!r}")
        if target not in valid_ids:
            raise GraphCodegenError(f"エッジの target が未知のノードです: {target!r}")

    var_names: dict[str, str] = {}
    for node_id in real_node_ids:
        _node_var_name(node_id, var_names)

    known_ids = [n for n in real_node_ids if n in KNOWN_NODE_IDS]
    new_ids = [n for n in real_node_ids if n not in KNOWN_NODE_IDS]

    # 固定エッジ / START・END に絡むエッジ / 条件分岐エッジに分類する。
    fixed_edges: list[tuple[str, str]] = []
    start_targets: list[str] = []
    end_sources: list[str] = []
    conditional_by_source: dict[str, list[tuple[str | None, str]]] = {}

    for e in edges:
        source, target = str(e["source"]), str(e["target"])
        conditional = bool(e.get("conditional", False))
        label = e.get("label")
        label = str(label) if label not in (None, "") else None

        if source == START_ID:
            start_targets.append(target)
            continue
        if target == END_ID:
            end_sources.append(source)
            continue

        if conditional:
            conditional_by_source.setdefault(source, []).append((label, target))
        else:
            fixed_edges.append((source, target))

    lines: list[str] = []
    lines.append('"""GUI編集結果から生成された StateGraph 構築コード（自動生成・要レビュー）。')
    lines.append("")
    lines.append("このコードはグラフ・エディタGUIが生成した「配線」だけのスニペットであり、")
    lines.append("そのまま実行できる完成品ではない。既存ノード")
    if known_ids:
        lines.append(f"（{', '.join(known_ids)}）")
    lines.append(
        "の実体は build_harvest_graph() の中で定義されたクロージャであり、この"
    )
    lines.append(
        "ファイル単体では解決できない —— このスニペットは build_harvest_graph()"
    )
    lines.append("の本体（クロージャ定義より後ろ、`# Wire the graph` の位置）に貼り付ける")
    lines.append("か、同名の関数が既にスコープにある場所で使うことを想定している。")
    lines.append("以下は必ず人手で埋めること:")
    lines.append("  * プレースホルダー関数（`# TODO: implement`）の中身")
    lines.append("  * プレースホルダー・ルーター関数（`# TODO: implement routing logic`）の判定ロジック")
    lines.append('"""')
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from langgraph.graph import END, START, StateGraph")
    lines.append("from langgraph.graph.state import CompiledStateGraph")
    lines.append("")
    lines.append("from dimos.robot.unitree.g1.harvest.blackboard import HarvestState")
    lines.append("")

    if new_ids:
        lines.append("")
        lines.append("# --- 新規/リネームされたノード（プレースホルダー。実装はここに書くこと） ---")
        for node_id in new_ids:
            var = var_names[node_id]
            lines.append("")
            lines.append(f"def {var}(state: HarvestState) -> HarvestState:  # TODO: implement")
            lines.append(f'    """ノード "{node_id}" の処理（GUIでは判定ロジックを持たないため未実装）。"""')
            lines.append("    raise NotImplementedError")
        lines.append("")

    router_var_names: dict[str, str] = {}
    if conditional_by_source:
        lines.append("")
        lines.append("# --- ルーター（プレースホルダー。判定ロジックは元の route_after_* 関数を ---")
        lines.append("# --- 参照して実装すること。GUIが編集するのは戻り値→遷移先の対応表のみ） ---")
        for source in conditional_by_source:
            router_var = _safe_identifier(f"route_after_{source}", fallback=f"route_after_{source}")
            router_var_names[source] = router_var
            lines.append("")
            lines.append(f"def {router_var}(state: HarvestState) -> str:  # TODO: implement routing logic")
            lines.append(f'    """"{source}" の次の遷移先を決めるルーター（判定ロジックは未実装）。"""')
            lines.append("    raise NotImplementedError")
        lines.append("")

    lines.append("")
    lines.append("def build_graph() -> CompiledStateGraph:")
    lines.append('    """GUIで編集したノード配置・エッジからグラフを組み立てる。"""')
    lines.append("    g = StateGraph(HarvestState)")
    lines.append("")
    for node_id in real_node_ids:
        var = var_names[node_id]
        lines.append(f'    g.add_node("{node_id}", {var})')
    lines.append("")

    for target in start_targets:
        lines.append(f'    g.add_edge(START, "{target}")')
    for source, target in fixed_edges:
        lines.append(f'    g.add_edge("{source}", "{target}")')
    for source in end_sources:
        lines.append(f'    g.add_edge("{source}", END)')

    for source, pairs in conditional_by_source.items():
        router_var = router_var_names[source]
        lines.append("    g.add_conditional_edges(")
        lines.append(f'        "{source}",')
        lines.append(f"        {router_var},")
        lines.append("        {")
        used_labels: set[str] = set()
        for label, target in pairs:
            resolved_label = label if label is not None else target
            # ラベルが衝突したら target を足して一意にする（保険）。
            while resolved_label in used_labels:
                resolved_label = f"{resolved_label}__{target}"
            used_labels.add(resolved_label)
            lines.append(f'            "{resolved_label}": "{target}",')
        lines.append("        },")
        lines.append("    )")

    lines.append("")
    lines.append("    return g.compile()")
    lines.append("")
    lines.append("")
    lines.append("__all__ = [\"build_graph\"]")
    lines.append("")

    return "\n".join(lines)


__all__ = ["GraphCodegenError", "KNOWN_NODE_IDS", "generate_stategraph_code"]
