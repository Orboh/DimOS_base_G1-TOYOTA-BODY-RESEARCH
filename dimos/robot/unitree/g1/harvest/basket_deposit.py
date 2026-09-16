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

"""F-07 腹部固定かごへの投入（同期版・右腕のみ・IKのみ）— LangGraph の grasp_sequence 用。

``place_basket.py``（左手に持つ籠、``LEFT_PRESENT_BASKET`` で左腕を提示する両腕版）
の腹部固定かご版。かごを G1 の腹部（pelvis）に固定するYokoteさんの設計
（Obsidian ``Free-space/yokote/20260825/腹部かご搭載_方針書.md`` + 8/25-8/27作業ログ）
に合わせ、**左腕は一切動かさず右腕のみ**で entry→drop→retreat の3点をIKで運び、
グリッパを開いてリリースする。

``act/basket_deposit_bridge.py``（LCM 経由の非同期 Module 版、クリック/YOLOブリッジ
直結の ``unitree-g1-okra-ik-only-grasp-zed`` 用）と同じ考え方・同じ投入座標
（``ENTRY_TORSO``/``DROP_TORSO``/``RETREAT_TORSO`` をそこから import）だが、
こちらは ``GraspSequence.place_basket_fn``（``() -> None`` の同期呼び出し）として
差し込めるよう、``IkApproachSkill`` を使った同期関数の形で提供する。

⚠️ SAFETY: ``basket_deposit_bridge.py`` と同じ注意 — このIKベース3点直接経路は
MuJoCoでのみ自己衝突検証済み。実機Phase 5デモ（Yokote, 2026-08-27）では右脚接触
を避けるため追加の退避ウェイポイントが使われた。IK座標(entry_torso等)は自己干渉
モデルを持たないため、実測状態からのwarm-startによってはお腹や籠の縁に干渉する
経路を解いてしまうリスクがある（2026-09-14 ユーザー指摘）。**推奨は
entry_q7/drop_q7/retreat_q7 での教示モード**（下記）— IKを使わず、人の手で実際に
確認した安全な軌道をそのまま再生する。IKモードは教示前の暫定/フォールバック用途。
実オクラでのLIVE実行前に必ずDRY-RUNでq_solを確認すること。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import time

from dimos.robot.unitree.g1.act.basket_deposit_bridge import (
    BASKET_OPEN_Q,
    DROP_TORSO,
    ENTRY_TORSO,
    RETREAT_TORSO,
)
from dimos.robot.unitree.g1.harvest.ik_approach import IkApproachSkill
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_LEFT_SLICE = slice(15, 22)
_RIGHT_SLICE = slice(22, 29)
# 教示済み姿勢間を移動する際の控えめ速度[rad/s]（return_to_rest.pyのhandback速度と同じ考え方）。
_TAUGHT_SPEED_RAD_S = 0.5
_TAUGHT_MIN_WAIT_S = 0.8
_TAUGHT_MAX_WAIT_S = 3.0


def make_basket_deposit_fn(
    *,
    send_arm: Callable[[list[float], float], None],
    open_gripper: Callable[[float, float], None],
    get_measured: Callable[[], Sequence[float]],
    entry_torso: Sequence[float] = ENTRY_TORSO,
    drop_torso: Sequence[float] = DROP_TORSO,
    retreat_torso: Sequence[float] = RETREAT_TORSO,
    entry_q7: Sequence[float] | None = None,
    drop_q7: Sequence[float] | None = None,
    retreat_q7: Sequence[float] | None = None,
    q_open: float = BASKET_OPEN_Q,
    settle_secs: float = 1.2,
    ik: IkApproachSkill | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Callable[[], bool]:
    """F-07（腹部かご版）投入動作を行う ``() -> bool``（成功=True）を組み立てて返す。

    Args:
        send_arm: ``(arm14, secs)`` — 14関節目標（左7+右7, 正準順）へ ``secs`` 秒で
            スルーして保持する。左腕側は現在角のまま（IkApproachSkill.solve が
            hold する）なので、呼び出し側は特別な左腕制御をしなくてよい。
        open_gripper: ``(q, secs)`` — グリッパ目標角 ``q`` を ``secs`` 秒送出（開き＝リリース）。
        get_measured: ``() -> 29-DOF 現在角``（warm-start / 教示モードの左腕holdに使う）。
        entry_torso / drop_torso / retreat_torso: torso_link フレームの3点 [m]（IKモード用）。
            既定は ``basket_deposit_bridge.py`` と共有（同じ物理かごを指す）。
        entry_q7 / drop_q7 / retreat_q7: 教示済みの右腕7関節角度[rad]（正準順、
            ``unitree-g1-teach-pregrasp-pose`` と同じキネステティック教示で取得）。
            **3つとも指定されていれば、IKを一切使わずこちらを優先する**（推奨）。
            entry_torso等のIK座標は自己干渉モデルを持たないため、実測状態からの
            warm-startによってはお腹や籠の縁に干渉する経路を解いてしまうリスクが
            ある（2026-09-14 ユーザー指摘）。人の手で実際に確認した軌道をそのまま
            再生する方が確実。一部のみ指定された場合は警告してIKモードへ
            フォールバックする。
        q_open: リリース時のグリッパ開き角 [rad]。
        settle_secs: 開き整定時間 [s]。
        ik: 投入用の IK スキル（IKモードのみ使用）。None なら standoff=0（真上を狙う）・
            投入向けのタイトな許容誤差（0.02m）で既定生成する。

    Returns:
        ``place() -> bool``。教示モード: 常に True（教示済み姿勢は既に安全性を
        確認済みという前提のため、失敗判定を持たない）。IKモード: 3点いずれかで
        IK が解けなければ False（呼び出し側でリトライ/スキップ）。
        いずれもentry→dropの順にスルーし、**drop到達直後にグリッパを開いて
        リリース**してからretreatへ引く（2026-09-14 実機LIVEで判明: 旧実装は
        entry→drop→retreat 全部を移動し終えてから最後に開いており、実際には
        「かごから離れた後に開く」動作になっていた）。
    """
    use_taught = entry_q7 is not None and drop_q7 is not None and retreat_q7 is not None
    if not use_taught and (entry_q7 is not None or drop_q7 is not None or retreat_q7 is not None):
        logger.warning(
            "[basket-deposit] entry_q7/drop_q7/retreat_q7 が一部のみ指定されています"
            "（教示は3点セットが必要）— IKモードにフォールバックします。"
        )

    if use_taught:
        taught_legs = (
            ("entry", [float(v) for v in entry_q7]),  # type: ignore[union-attr]
            ("drop", [float(v) for v in drop_q7]),  # type: ignore[union-attr]
            ("retreat", [float(v) for v in retreat_q7]),  # type: ignore[union-attr]
        )

        def place_taught() -> bool:
            q_right_cur = None
            for label, q7 in taught_legs:
                meas = list(get_measured())
                q_left = list(meas[_LEFT_SLICE])
                if q_right_cur is None:
                    q_right_cur = list(meas[_RIGHT_SLICE])
                delta = max(abs(g - s) for g, s in zip(q7, q_right_cur, strict=True))
                wait_s = min(
                    max(delta / _TAUGHT_SPEED_RAD_S, _TAUGHT_MIN_WAIT_S), _TAUGHT_MAX_WAIT_S
                )
                logger.info(
                    f"[basket-deposit] {label}(教示): q_right={[round(v, 3) for v in q7]} "
                    f"wait={wait_s:.2f}s"
                )
                send_arm(q_left + q7, wait_s)
                sleep_fn(wait_s)
                q_right_cur = q7
                if label == "drop":
                    # drop到達直後にリリース（retreatへ引く前）。
                    open_gripper(q_open, settle_secs)
                    sleep_fn(settle_secs)
                    logger.info(
                        f"[basket-deposit] drop(教示)到達 — グリッパ q={q_open:.3f} で開放"
                    )

            logger.info("[basket-deposit] 投入完了(教示)")
            return True

        return place_taught

    if ik is None:
        ik = IkApproachSkill(
            standoff_m=0.0,  # かご開口の真上を狙う（切断リーチのような手前止めは不要）
            max_reach_pos_err_m=0.02,  # 投入はかご開口部が狭いので把持リーチより厳しめ
        )

    legs = (("entry", entry_torso), ("drop", drop_torso), ("retreat", retreat_torso))

    def place() -> bool:
        for label, target in legs:
            meas = list(get_measured())
            res = ik.solve(target, meas)
            if res is None:
                logger.warning(
                    f"[basket-deposit] {label} 目標 {list(target)} へ IK 解けず"
                    "（投入中止・要リトライ/スキップ）"
                )
                return False
            logger.info(
                f"[basket-deposit] {label}: torso{list(target)} err={res.err:.4f} m "
                f"wait={res.wait_s:.2f}s"
            )
            send_arm(res.arm14, res.wait_s)
            sleep_fn(res.wait_s)
            if label == "drop":
                # drop到達直後にリリース（retreatへ引く前）。
                open_gripper(q_open, settle_secs)
                sleep_fn(settle_secs)
                logger.info(f"[basket-deposit] drop到達 — グリッパ q={q_open:.3f} で開放")

        logger.info("[basket-deposit] 投入完了")
        return True

    return place


__all__ = ["make_basket_deposit_fn"]
