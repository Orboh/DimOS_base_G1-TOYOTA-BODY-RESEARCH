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

"""籠投入後、次のオクラ探索へ戻る前に「起動時の姿勢」へ腕を復帰させる — LangGraph の
grasp_sequence 用（``GraspSequence.return_to_rest_fn``）。

⚠️ 2026-09-08 実機LIVEで判明: 籠投入（``basket_deposit.py``）の retreat leg は腹部
かご開口部の手前（``RETREAT_TORSO``）止まりで、把持開始前の姿勢（ユーザーが手動で
「腕を垂らして」から起動した姿勢）には戻らない。この状態のまま次周回の IK 解を
``measured_position``（＝ retreat 姿勢）から計算すると、見た目上「固定モーションの
最後で止まっている」ように見え、次のオクラ探索の基準姿勢もばらつく。

``G1ArmSdkConnection.stop()``（切断時の handback）と同じ考え方 — 一発で目標角へ
飛ばすと ``arm_velocity_limit``（既定20 rad/s）任せの急な動きになるため、控えめな
速度（既定0.5 rad/s、handback と同じ値）で多段補間して publish する。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import time

from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# G1ArmSdkConnection.stop() の handback と同じ値（急な戻り動作を避けるための
# 控えめな速度）。速く/遅くしたい場合はここではなく呼び出し側の引数で調整する。
_DEFAULT_SPEED_RAD_S = 0.5
_DEFAULT_RATE_HZ = 50.0
_DEFAULT_MAX_TRAVEL_S = 6.0

# G1 の 29-DOF motor_states 内、左腕7 + 右腕7 のインデックス（g1_arm_sdk_connection.py
# の ``_ARM_IDX`` と同じ並び）。
_LEFT_ARM_SLICE = slice(15, 22)
_RIGHT_ARM_SLICE = slice(22, 29)


def make_return_to_rest_fn(
    *,
    send_arm: Callable[[list[float]], None],
    get_measured: Callable[[], Sequence[float]],
    rest_q_getter: Callable[[], Sequence[float] | None],
    speed_rad_s: float = _DEFAULT_SPEED_RAD_S,
    rate_hz: float = _DEFAULT_RATE_HZ,
    max_travel_s: float = _DEFAULT_MAX_TRAVEL_S,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Callable[[], bool]:
    """起動時姿勢（左7+右7, 正準順）へ多段補間で戻る ``() -> bool`` を組み立てて返す。

    Args:
        send_arm: 14関節目標（左7+右7, 正準順）を arm_target へ publish する。
        get_measured: ``() -> 29-DOF 現在角``（開始点として使う）。
        rest_q_getter: ``() -> 14要素の起動時姿勢 | None``。None なら復帰をスキップ
            （起動直後にまだキャプチャできていない等）して False を返す。
        speed_rad_s: 最も動く関節の目標速度 [rad/s]（既定 0.5、handback と同じ）。
        rate_hz: 補間ステップの更新レート [Hz]。
        max_travel_s: 補間にかける最大時間 [s]（暴走防止のキャップ）。

    Returns:
        ``go_home() -> bool``。rest_q_getter が None を返せば何もせず False。
        それ以外は目標へ滑らかに近づけて publish し続け、完了で True。
    """

    def go_home() -> bool:
        rest14 = rest_q_getter()
        if rest14 is None:
            logger.warning(
                "[return-to-rest] 起動時姿勢が未キャプチャのため復帰をスキップ"
            )
            return False
        meas = list(get_measured())
        start14 = list(meas[_LEFT_ARM_SLICE]) + list(meas[_RIGHT_ARM_SLICE])
        goal14 = list(rest14)
        delta = max(abs(g - s) for g, s in zip(goal14, start14, strict=True))
        travel_s = min(max(delta / speed_rad_s, 0.0), max_travel_s)
        steps = max(int(travel_s * rate_hz), 1)
        for k in range(1, steps + 1):
            frac = k / steps
            q = [s + (g - s) * frac for s, g in zip(start14, goal14, strict=True)]
            send_arm(q)
            sleep_fn(1.0 / rate_hz)
        logger.info(
            f"[return-to-rest] 起動時姿勢へ復帰完了 (delta={delta:.3f} rad, {travel_s:.2f}s)"
        )
        return True

    return go_home


__all__ = ["make_return_to_rest_fn"]
