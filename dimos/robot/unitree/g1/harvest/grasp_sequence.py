# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""把持シーケンス: IK 粗アプローチ → ACT 微調整 → 切断可否 VLM → 切断（Phase 1–3）。

LangGraph の grasp ノードが呼ぶ ``grasp_okra`` の実体。設計方針 (P)（1ノード内で
①〜⑤を同期実行）に従い、次を**1回のエピソード**として束ねる:

  ① ik_approach : オクラ実の重心(torso 3D)へ右腕 IK（[[SS-04-粗アプローチIK]], 同期）
                  解けない/届かない → エピソード失敗（verify が False を見て retry/give_up）
  ② act_final   : 手首単眼 ACT で ~4s、切断点までハンドを寄せる（閉じない, [[SS-05-精密把持ACT]]）
  ③ cut 可否    : moondream で「実を収穫でき、かつ主茎を切らない位置か」判定（[[SS-02-状態判定VLM]]）
                  NG → 切断せずエピソード失敗
  ④ cut         : グリッパを閉じ位置(4.4 rad)へ → 切断＋把持（[[SS-06-切断と籠収納]]）
                  目標角は刃保護上限(5.2 rad)でクランプ（BladeGuard）
  ⑤ place_basket: 保留（プレースホルダ。別途実機開発中）

``ActGraspModule`` と同じ ``run_episode(okra, force)`` / ``stop()`` を提供するので、
既存の ``grasp_fn`` 配線と SafetyMonitor の途中停止フックにそのまま入る。
各サブステップ（IK 解、ACT、cut 可否、cut）は注入可能なので、実機なしで単体テストできる。

⚠️ 座標系: IK は torso フレームの重心3Dを要求する。検出が出す ``Okra.pos_3d`` を torso へ
変換するのは ``centroid_torso_getter``（呼び出し側＝detect/ブループリント配線の責務）。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from dimos.robot.unitree.g1.harvest.blackboard import Okra
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_RIGHT_GRIPPER_JOINT = "g1/right_gripper"

# Dex1-1 切断グリッパ（[[SS-06-切断と籠収納]]）。
_Q_CLOSE_CUT = 4.4   # [rad] 閉じ位置＝切断＋把持
_Q_BLADE_MAX = 5.2   # [rad] 刃保護の上限（機械限界 5.4 の手前。過電流フォルト回避）


class GraspSequence:
    """IK→ACT→切断可否→切断 を1エピソードに束ねる（停止可能）。

    Args:
        ik_solve: ``() -> arm14_or_None``。torso 重心3Dを IK で解き、14関節目標と
            待機時間を返す同期スキル呼び出し（呼び出し側が現在の対象オクラに束ねて渡す）。
            None なら「届かない/解けない」→ エピソード失敗。
            返り値は ``(arm14: list[float], joint_names: list[str], wait_s: float)``。
        publish_arm: 14関節 JointState を arm_target へ publish。
        act_module: ``run_episode()`` を持つ ACT（``ActGraspModule`` 互換）。閉じずに
            切断点まで寄せる（``grasp_duration`` ~4s 相当）。None ならスキップ（IK のみ）。
        cut_ok_fn: ``() -> bool``。切断可否（実を収穫でき・主茎を切らない位置か）。
            None なら常に許可（VLM 未配線時のフォールバック）。
        publish_gripper: 1関節 JointState を gripper_target へ publish（切断）。None なら
            切断スキップ（IK/ACT のみの検証時）。
        place_basket_fn: ``() -> None``。籠投入（保留中はプレースホルダ）。
    """

    def __init__(
        self,
        *,
        ik_solve: Callable[[Okra], Any | None],
        publish_arm: Callable[[Any], None] | None = None,
        act_module: Any = None,
        cut_ok_fn: Callable[[], bool] | None = None,
        publish_gripper: Callable[[Any], None] | None = None,
        place_basket_fn: Callable[[], None] | None = None,
        q_close: float = _Q_CLOSE_CUT,
        q_blade_max: float = _Q_BLADE_MAX,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._ik_solve = ik_solve
        self._publish_arm = publish_arm
        self._act = act_module
        self._cut_ok_fn = cut_ok_fn
        self._publish_gripper = publish_gripper
        self._place_basket_fn = place_basket_fn
        self._q_close = float(q_close)
        self._q_blade_max = float(q_blade_max)
        self._sleep = sleep_fn
        self._stop = threading.Event()
        # (okra_id, reached_phase, ok) per episode — for trace / assertions.
        self.episodes: list[tuple[str, str, bool]] = []

    def stop(self) -> None:
        """進行中のエピソードを中断（SafetyMonitor.on_pause が呼ぶ）。"""
        self._stop.set()
        if self._act is not None and hasattr(self._act, "stop"):
            self._act.stop()

    def _cut(self, q: float) -> None:
        """グリッパ目標角 q[rad] を publish。刃保護上限でクランプ（BladeGuard）。"""
        from dimos.msgs.sensor_msgs.JointState import JointState

        q_safe = min(self._q_blade_max, float(q))  # 刃保護: 5.2 rad を超えない
        if q_safe != q:
            logger.warning(f"[grasp-seq] cut q {q:.3f} clamped to blade-safe {q_safe:.3f} rad")
        if self._publish_gripper is not None:
            self._publish_gripper(
                JointState(
                    name=[_RIGHT_GRIPPER_JOINT],
                    position=[q_safe],
                    velocity=[0.0],
                    effort=[0.0],
                )
            )

    def run_episode(self, okra: Okra | None = None, force: float | None = None) -> bool:
        """1エピソード = IK→ACT→切断可否→切断。成功で True、途中失敗/中断で False。"""
        # 開始前に停止要求が立っていたら尊重する（SafetyMonitor が一時停止中に
        # エピソードを始めない）。clear で取りこぼすと、停止中でも把持が走ってしまう。
        if self._stop.is_set():
            logger.info("[grasp-seq] stop requested before start; refusing episode")
            return False
        self._stop.clear()
        okra_id = getattr(okra, "id", "?")

        # ① IK 粗アプローチ ----------------------------------------------------
        if self._stop.is_set():
            return False
        sol = self._ik_solve(okra) if okra is not None else None
        if sol is None:
            logger.info(f"[grasp-seq] {okra_id}: IK unreachable/unsolved -> episode fail")
            self.episodes.append((okra_id, "ik", False))
            return False
        arm14, joint_names, wait_s = sol.arm14, sol.joint_names, sol.wait_s
        if self._publish_arm is not None:
            from dimos.msgs.sensor_msgs.JointState import JointState

            self._publish_arm(
                JointState(
                    name=list(joint_names),
                    position=[float(x) for x in arm14],
                    velocity=[0.0] * len(arm14),
                    effort=[0.0] * len(arm14),
                )
            )
        logger.info(f"[grasp-seq] {okra_id}: IK reach -> waiting {wait_s:.2f}s for arm to settle")
        # open-loop 整定待ち（中断可能）
        if self._stop.wait(wait_s):
            return False

        # ② ACT 微調整（切断点まで、閉じない） --------------------------------
        if self._act is not None:
            if self._stop.is_set():
                return False
            logger.info(f"[grasp-seq] {okra_id}: ACT final approach (no close)")
            self._act.run_episode(okra, force)
            if self._stop.is_set():
                self.episodes.append((okra_id, "act", False))
                return False

        # ③ 切断可否 VLM（実を収穫でき・主茎を切らない位置か） ----------------
        if self._cut_ok_fn is not None:
            if not self._cut_ok_fn():
                logger.info(f"[grasp-seq] {okra_id}: VLM says NOT safe to cut -> episode fail (no cut)")
                self.episodes.append((okra_id, "cut_gate", False))
                return False

        # ④ 切断（グリッパ閉じ＝切断＋把持） -----------------------------------
        if self._stop.is_set():
            return False
        logger.info(f"[grasp-seq] {okra_id}: CUT (gripper -> {self._q_close} rad, blade limit {self._q_blade_max})")
        self._cut(self._q_close)

        # ⑤ 籠投入（保留: プレースホルダ） ------------------------------------
        if self._place_basket_fn is not None:
            self._place_basket_fn()

        self.episodes.append((okra_id, "cut", True))
        return True


__all__ = ["GraspSequence"]
