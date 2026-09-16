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

"""Offline tests for ModelGraspAdapter (reach_done/adjust_done 越しの同期ラッパー)。

UmiDiffusionBridge本体（別モジュール）は使わず、on_adjust_done を直接呼んで
シミュレートする。実時間待ちを避けるため wait_timeout_s は短く設定する。
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time

from dimos.robot.unitree.g1.harvest.model_grasp_adapter import ModelGraspAdapter


@dataclass
class _Okra:
    id: str


@dataclass
class _BoolMsg:
    data: bool


def test_run_episode_fires_reach_done_and_waits_for_adjust_done() -> None:
    fired: list[str] = []
    adapter = ModelGraspAdapter(fire_reach_done=lambda: fired.append("reach_done"))

    def _complete_soon() -> None:
        time.sleep(0.02)
        adapter.on_adjust_done(_BoolMsg(data=True))

    threading.Thread(target=_complete_soon, daemon=True).start()
    ok = adapter.run_episode(_Okra(id="okra_1"), force=0.3)
    assert ok is True
    assert fired == ["reach_done"]
    assert adapter.episodes[-1] == ("okra_1", True)


def test_adjust_done_false_data_is_ignored() -> None:
    """data=False の adjust_done（他エピソード由来の残留等）はトリガーにならない。"""
    adapter = ModelGraspAdapter(fire_reach_done=lambda: None, wait_timeout_s=0.05)
    adapter.on_adjust_done(_BoolMsg(data=False))
    ok = adapter.run_episode(_Okra(id="okra_2"))
    assert ok is False  # タイムアウト
    assert adapter.episodes[-1] == ("okra_2", False)


def test_timeout_returns_false() -> None:
    adapter = ModelGraspAdapter(fire_reach_done=lambda: None, wait_timeout_s=0.05)
    ok = adapter.run_episode(_Okra(id="okra_3"))
    assert ok is False
    assert adapter.episodes[-1] == ("okra_3", False)


def test_stop_unblocks_waiting_episode_immediately() -> None:
    adapter = ModelGraspAdapter(fire_reach_done=lambda: None, wait_timeout_s=10.0)

    def _stop_soon() -> None:
        time.sleep(0.02)
        adapter.stop()

    threading.Thread(target=_stop_soon, daemon=True).start()
    t0 = time.time()
    ok = adapter.run_episode(_Okra(id="okra_4"))
    elapsed = time.time() - t0
    assert ok is False
    assert elapsed < 5.0  # 10s のタイムアウトを待たずに即座に解放された


def test_stop_then_next_episode_is_refused_once_then_recovers() -> None:
    """stop()の直後は1回だけ拒否され、その後クリアされて次は使える
    （GraspSequence/ActGraspModuleと同じ作法）。"""
    adapter = ModelGraspAdapter(fire_reach_done=lambda: None, wait_timeout_s=0.05)
    adapter.stop()
    assert adapter.run_episode(_Okra(id="okra_5")) is False
    assert adapter.episodes[-1] == ("okra_5", False)

    # 次のエピソードでは stop_requested がクリアされ、通常通り動作する。
    def _complete_soon() -> None:
        time.sleep(0.02)
        adapter.on_adjust_done(_BoolMsg(data=True))

    adapter2 = ModelGraspAdapter(fire_reach_done=lambda: None, wait_timeout_s=5.0)
    adapter2.stop()
    assert adapter2.run_episode(_Okra(id="okra_6")) is False

    def _complete_soon2() -> None:
        time.sleep(0.02)
        adapter2.on_adjust_done(_BoolMsg(data=True))

    threading.Thread(target=_complete_soon2, daemon=True).start()
    assert adapter2.run_episode(_Okra(id="okra_7")) is True
