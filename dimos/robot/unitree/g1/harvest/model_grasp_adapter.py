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

"""GraspSequence の act_module スロットへ、別モジュール（UmiDiffusionBridge、
またはUMI形式の推論サーバーに繋がる同等のブリッジ）を差し込むための同期ラッパー。

model_no_kensho ブループリント用（2026-09-16）: IKで対象へ寄せず、教示済みの
準備姿勢から直接 Diffusion/ACT/Flow matching モデルにオクラへの接近を委ねる
構成では、モデル推論の実体（UmiDiffusionBridge）は HarvestModule とは別モジュール
（別プロセス）で動く。両者は直接のメソッド呼び出しができないため、DimOSの
ストリーム（reach_done / adjust_done）越しにやり取りする。

このクラスは ``ActGraspModule`` と同じ ``run_episode(okra, force) -> bool`` /
``stop()`` インターフェースを提供し、GraspSequence からは同期呼び出しに見える
ようにする:

  run_episode() -> fire_reach_done() で開始を通知 -> adjust_done（収束 or
  人間のcut_triggerによる手動終了）が来るまでブロッキング待機 -> True/False

``on_adjust_done`` は呼び出し側（HarvestModule）が adjust_done ストリームの
購読ハンドラとして登録する。
"""

from __future__ import annotations

from collections.abc import Callable
import threading

from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class ModelGraspAdapter:
    """UmiDiffusionBridge（非同期・別モジュール）を1エピソードとして同期的に呼ぶ。"""

    def __init__(
        self,
        fire_reach_done: Callable[[], None],
        *,
        wait_timeout_s: float = 300.0,
        model_name: str = "model",
    ) -> None:
        self._fire_reach_done = fire_reach_done
        self._wait_timeout_s = float(wait_timeout_s)
        self._model_name = model_name
        self._adjust_done_event = threading.Event()
        self._stop_requested = threading.Event()
        # (okra_id, ok) per episode — GraspSequence.episodes と同じ用途のトレース。
        self.episodes: list[tuple[str, bool]] = []

    def on_adjust_done(self, msg: object) -> None:
        """adjust_done ストリームの購読ハンドラ（呼び出し側が配線する）。"""
        if getattr(msg, "data", False):
            self._adjust_done_event.set()

    def run_episode(self, okra: object = None, force: float | None = None) -> bool:
        """reach_done を送って adjust_done（収束 or 手動終了）を待つ。

        SafetyMonitor が停止中に開始しないよう、開始前に stop_requested を確認する
        （ActGraspModule/GraspSequence 同様の作法）。
        """
        okra_id = getattr(okra, "id", "?")
        # 呼び出しの度に必ずクリアする（ActGraspModuleと同じ作法）— そうしないと
        # 一度 stop() されたインスタンスが以後ずっと拒否され続けてしまう。
        # was_stopped は「クリアする前に立っていたか」を見て、今回**だけ**拒否する。
        was_stopped = self._stop_requested.is_set()
        self._stop_requested.clear()
        if was_stopped:
            logger.info(f"[model-grasp:{self._model_name}] {okra_id}: 停止要求中のため開始しない")
            self.episodes.append((okra_id, False))
            return False
        self._adjust_done_event.clear()
        logger.info(
            f"[model-grasp:{self._model_name}] {okra_id}: reach_done送信 — "
            f"モデル推論を開始（Enterで手動終了 or 収束を待つ、最大{self._wait_timeout_s:.0f}s）"
        )
        self._fire_reach_done()
        got = self._adjust_done_event.wait(timeout=self._wait_timeout_s)
        if self._stop_requested.is_set():
            logger.warning(f"[model-grasp:{self._model_name}] {okra_id}: 停止要求により中断")
            self.episodes.append((okra_id, False))
            return False
        if not got:
            logger.warning(
                f"[model-grasp:{self._model_name}] {okra_id}: "
                f"adjust_done が {self._wait_timeout_s:.0f}s 以内に来なかった（タイムアウト）"
            )
            self.episodes.append((okra_id, False))
            return False
        logger.info(f"[model-grasp:{self._model_name}] {okra_id}: adjust_done受信 — ②完了")
        self.episodes.append((okra_id, True))
        return True

    def stop(self) -> None:
        """SafetyMonitor.on_pause が呼ぶ。待機中の run_episode を即座に失敗させる。"""
        self._stop_requested.set()
        self._adjust_done_event.set()  # wait() を即座に解放する


__all__ = ["ModelGraspAdapter"]
