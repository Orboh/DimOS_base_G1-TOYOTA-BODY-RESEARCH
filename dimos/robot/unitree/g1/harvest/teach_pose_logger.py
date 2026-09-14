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

"""準備姿勢(pregrasp pose)の教示ロガー。

``G1ArmSdkConnection(collection_mode=True)`` と組み合わせて使う
（``unitree_g1_teach_pregrasp_pose.py`` ブループリント）。操作フロー:

  1. 起動直後は通常の剛性姿勢のまま待機（何も動かない）。
  2. 別ターミナルから ``touch <go_trigger_file>`` すると ``compliant_delay_s``
     （既定2秒）後に ``reach_done`` を publish し、右腕がコンプライアント
     （kp->0 + 重力補償トルクのみ、左腕・腰は剛性のまま）になる。**touchして
     から脱力までの間に右腕を支えること**。
  3. コンプライアント化した後は、``log_interval_s`` 間隔で現在の右腕7関節
     角度を定期ログしつつ、``touch <save_trigger_file>`` するたびに **その
     瞬間の関節角度を ``save_path`` に追記保存**する（何度でも押せる —
     複数候補を試して比較できる）。

使い方: 起動 → touch(go) → （支える）→ 2秒後に脱力 → 手で好きな準備姿勢へ
動かす → 良い姿勢で touch(save) → 保存されたlog行を
``unitree_g1_okra_honban.py`` の ``OKRA_PREGRASP_POSE_Q7`` にコピーする。

⚠️ 標準入力の ``input()`` はここでは使わない: DimOS はモジュールを
``ModuleCoordinator`` 経由で別プロセス（forkserverワーカー）上で実行するため、
そのプロセス内の ``input()`` は起動元ターミナルのキー入力を受け取れない
（2026-09-14 実機LIVEで確認 — Enterを押しても無反応だった）。かわりに
``safety_checks.FileEStop`` と同じ「ファイルの存在をポーリングする」方式で
プロセスの壁を越える。

``collection_mode`` 自体は 2026-06-24 に実機で検証済みの既存機能（キネステ
ティック教示用）を流用しているだけで、本モジュールが新規に足すのは
「touchトリガーの reach_done」「定期ログ」「touchでの保存」のみ（腕の制御
ロジックには一切関与しない）。
"""

from __future__ import annotations

from pathlib import Path
import threading
from threading import Thread
import time
from typing import Any

from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.Bool import Bool
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# 29-DOF motor_states のうち右腕7関節（正準順、g1_arm_sdk_connection.py の
# _ARM_IDX[7:14] と同じ）。
_RIGHT_ARM_SLICE = slice(22, 29)


class TeachPoseLoggerConfig(ModuleConfig):
    log_interval_s: float = 1.0  # コンプライアント化後、関節角度を定期ログする間隔 [s]
    # go_trigger_file の touch から reach_done を publish するまでの待機 [s]。
    # この間に右腕を支える（脱力する瞬間に落下・急変させないため）。
    compliant_delay_s: float = 2.0
    # touch(save_trigger_file)で保存した関節角度の追記先（1行=1候補、カンマ区切り7値）。
    save_path: str = "/tmp/pregrasp_pose_candidates.log"
    # touchすると compliant_delay_s 後にコンプライアント化するトリガーファイル。
    go_trigger_file: str = "/tmp/teach_go"
    # touchすると現在の関節角度を即座に保存するトリガーファイル。
    save_trigger_file: str = "/tmp/teach_save"
    poll_interval_s: float = 0.1  # トリガーファイルのポーリング間隔 [s]


class TeachPoseLogger(Module):
    """touchファイルトリガーの右腕コンプライアント化 + 定期ログ + 姿勢保存。"""

    config: TeachPoseLoggerConfig
    motor_states: In[JointState]
    reach_done: Out[Bool]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._latest: JointState | None = None
        self._stop = threading.Event()
        self._run_thread: Thread | None = None
        self._log_thread: Thread | None = None

    def _on_state(self, msg: JointState) -> None:
        with self._lock:
            self._latest = msg

    def _current_q7(self) -> list[float] | None:
        with self._lock:
            state = self._latest
        if state is None:
            return None
        pos = list(state.position)
        if len(pos) < 29:
            return None
        return [round(float(x), 4) for x in pos[_RIGHT_ARM_SLICE]]

    def _wait_for_touch(self, path: Path) -> bool:
        """path が touch されるまでポーリングする。stop 済みなら False。"""
        while not self._stop.is_set():
            if path.exists():
                path.unlink(missing_ok=True)
                return True
            time.sleep(self.config.poll_interval_s)
        return False

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.motor_states.subscribe(self._on_state)))
        self._run_thread = Thread(target=self._run, daemon=True, name="teach-pose-run")
        self._run_thread.start()

    def _run(self) -> None:
        go_path = Path(self.config.go_trigger_file)
        save_trigger_path = Path(self.config.save_trigger_file)
        go_path.unlink(missing_ok=True)
        save_trigger_path.unlink(missing_ok=True)

        logger.info(
            f"TeachPoseLogger: `touch {go_path}` すると"
            f"{self.config.compliant_delay_s:.1f}s後に右腕がコンプライアント化します"
            "（その間に右腕を支える準備をしてください）"
        )
        if not self._wait_for_touch(go_path):
            return

        logger.info(
            f"TeachPoseLogger: {self.config.compliant_delay_s:.1f}s後に右腕を"
            "コンプライアント化します。支えてください。"
        )
        time.sleep(self.config.compliant_delay_s)
        if self._stop.is_set():
            return
        self.reach_done.publish(Bool(data=True))
        logger.info(
            "TeachPoseLogger: reach_done publish完了 — 右腕がコンプライアント "
            "(kp->0 + 重力補償トルクのみ)になったはずです。手で持って良い準備姿勢へ "
            f"動かしてください（左腕・腰は剛性のまま）。良い姿勢になったら "
            f"`touch {save_trigger_path}` するとその時の関節角度を保存します"
            "（何度でも押せます）。Ctrl+Cで終了。"
        )
        self._log_thread = Thread(
            target=self._periodic_log, daemon=True, name="teach-pose-periodic-log"
        )
        self._log_thread.start()
        while self._wait_for_touch(save_trigger_path):
            self._save_current_pose()

    def _periodic_log(self) -> None:
        while not self._stop.is_set():
            time.sleep(self.config.log_interval_s)
            q7 = self._current_q7()
            if q7 is not None:
                logger.info(f"[teach-pose] right arm q7 (rad, 正準順) = {q7}")

    def _save_current_pose(self) -> None:
        q7 = self._current_q7()
        if q7 is None:
            logger.warning("TeachPoseLogger: motor_states 未受信のため保存できません")
            return
        line = ",".join(f"{v:.4f}" for v in q7)
        logger.info(f"[teach-pose] SAVED q7 = {line}")
        try:
            with open(self.config.save_path, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")
        except OSError as exc:
            logger.warning(f"TeachPoseLogger: {self.config.save_path} への書き込み失敗: {exc}")
            return
        logger.info(f"TeachPoseLogger: {self.config.save_path} に追記しました")

    @rpc
    def stop(self) -> None:
        self._stop.set()
        if self._log_thread is not None:
            self._log_thread.join(timeout=2.0)
        if self._run_thread is not None:
            self._run_thread.join(timeout=2.0)
        super().stop()


__all__ = ["TeachPoseLogger", "TeachPoseLoggerConfig"]
