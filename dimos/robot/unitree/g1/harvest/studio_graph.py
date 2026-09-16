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

"""LangGraph Studio 用のエントリポイント。

``langgraph dev``（``langgraph.json`` の ``graphs.harvest``）から読み込まれる
グラフファクトリ。``build_harvest_graph`` はロボット/カメラ接続の ``skills``
引数を要求するため、Studio 上でロボットなしにグラフ構造・条件分岐・state 遷移
を確認できるよう、常に :class:`DummyHarvestSkills` で組み立てる。

実機の skills を使いたい場合は流用せず、``harvest_module.py`` の
``HarvestModule.start()`` を参照すること（本ファイルは可視化・デバッグ専用）。
"""

from __future__ import annotations

from langgraph.graph.state import CompiledStateGraph

from dimos.robot.unitree.g1.harvest.blackboard import HarvestConfig
from dimos.robot.unitree.g1.harvest.dummy_skills import DummyHarvestSkills
from dimos.robot.unitree.g1.harvest.graph import build_harvest_graph


def graph() -> CompiledStateGraph:
    """Studio 用グラフファクトリ（DUMMY skills、ロボット非接続）。"""
    skills = DummyHarvestSkills(num_okra=3, stations=1)
    return build_harvest_graph(skills, HarvestConfig())


__all__ = ["graph"]
