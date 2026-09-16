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

from collections.abc import Callable
import threading
import time
from typing import Any

from dimos.robot.unitree.g1.harvest import announce
from dimos.robot.unitree.g1.harvest.announce import Announcer, NullAnnouncer
from dimos.robot.unitree.g1.harvest.blackboard import Okra
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_RIGHT_GRIPPER_JOINT = "g1/right_gripper"

# Dex1-1 切断グリッパ（[[SS-06-切断と籠収納]]）。
_Q_CLOSE_CUT = 4.4  # [rad] 閉じ位置＝切断＋把持（呼び出し元は通常 config.cut_close_q で上書き）
# [rad] 開き方向の安全上限（機械限界 5.4 の手前。過電流フォルト回避）。qが小さい
# ほど閉じる/大きいほど開く（2026-09-14 実機確認、harvest_module.py の
# cut_close_q コメント参照）ため、実質「開きすぎ防止の上限」。
_Q_BLADE_MAX = 5.2


class GraspSequence:
    """IK→ACT→切断可否→切断 を1エピソードに束ねる（停止可能）。

    Args:
        ik_solve: ``() -> IkApproachResult | list[IkApproachResult] | None``。torso
            重心3Dを IK で解き、14関節目標と待機時間を返す同期スキル呼び出し（呼び出し側が
            現在の対象オクラに束ねて渡す）。None なら「届かない/解けない」→ エピソード失敗。
            単一の結果と、複数waypoint（``IkApproachSkill.solve_legs`` の
            LIFT→TRANSIT→DESCEND 段階的アプローチ）のリストの両方を受け付ける — リストなら
            各レグを順に publish_arm→待機し、レグ間も ``stop()`` で中断できる。
        publish_arm: 14関節 JointState を arm_target へ publish。
        act_module: ``run_episode()`` を持つ ACT（``ActGraspModule`` 互換）。閉じずに
            切断点まで寄せる（``grasp_duration`` ~4s 相当）。None ならスキップ（IK のみ）。
        cut_ok_fn: ``() -> bool``。切断可否（実を収穫でき・主茎を切らない位置か）。
            None なら常に許可（VLM 未配線時のフォールバック）。
        publish_gripper: 1関節 JointState を gripper_target へ publish（切断）。None なら
            切断スキップ（IK/ACT のみの検証時）。
        place_basket_fn: ``() -> None``。籠投入（保留中はプレースホルダ）。
        cut_settle_s: 切断（グリッパ閉）指令の後、実際に閉じきるまで待つ秒数（力覚/
            位置フィードバックなしの固定 open-loop 待機）。0（既定）だと、⑤籠投入が
            グリッパの閉じ完了を待たずに開始してしまい、グリッパが閉じきる前に
            開き指令が飛ぶ（2026-09-08 実機LIVEで確認 — 目視では「開閉していない」
            ように見える）。``place_basket_fn`` を渡すなら必ず正の値を設定すること。
        return_to_rest_fn: ``() -> bool``。籠投入の後、起動時の姿勢へ腕を戻す
            （``harvest/return_to_rest.py`` 参照）。None（既定）ならスキップ。
            2026-09-08 実機LIVEで判明: 籠投入の retreat leg 止まりのまま次周回の
            IK を計算すると、目視では「固定モーションの最後で止まっている」ように
            見え、次のオクラ探索の基準姿勢もばらつく。
        announcer: G1スピーカー等への発話（``harvest/announce.py`` の ``Announcer``
            プロトコル）。None（既定）なら ``NullAnnouncer``（何も話さない）。切断・
            籠投入の開始時に発話し、グリッパ開閉/固定モーションが実際に作動している
            ことを操作者が音で確認できるようにする（2026-09-08 ユーザー要望）。
        post_reach_verify_fn: ``() -> None``。①IK区間（全legのopen-loop待機）完了
            直後、②ACTの前に呼ばれる任意フック。None（既定）なら何もしない
            （後方互換）。IK の ``err``（ソルバー内部の残差）は「計算上その角度で
            目標に届くはずか」でしかなく、実機のPD制御が実際にそこまで追従した
            か、その結果エンドエフェクタが本当に対象の重心座標に届いたかは別問題
            — 2026-09-14 ユーザー指摘（5cm程度ズレて把持した実機事例）を受け、
            実測角度からFKを計算し目標座標との残差をログする「到達確認」を
            呼び出し側（harvest_module.py）が注入できるようにした。
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
        return_to_rest_fn: Callable[[], bool] | None = None,
        post_reach_verify_fn: Callable[[], None] | None = None,
        q_close: float = _Q_CLOSE_CUT,
        q_blade_max: float = _Q_BLADE_MAX,
        cut_settle_s: float = 0.0,
        announcer: Announcer | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._ik_solve = ik_solve
        self._publish_arm = publish_arm
        self._act = act_module
        self._cut_ok_fn = cut_ok_fn
        self._publish_gripper = publish_gripper
        self._place_basket_fn = place_basket_fn
        self._return_to_rest_fn = return_to_rest_fn
        self._post_reach_verify_fn = post_reach_verify_fn
        self._q_close = float(q_close)
        self._q_blade_max = float(q_blade_max)
        self._cut_settle_s = float(cut_settle_s)
        self._announcer = announcer if announcer is not None else NullAnnouncer()
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
            # select() の grasping()（「オクラを収穫します」）で予告した後、実際の IK 解
            # が失敗しても従来ここは無音だった（2026-09-08 実機LIVEで、手が動かず何も
            # 発話されない現象として発覚）。何が起きたか操作者に必ず伝える。
            self._announcer.say(announce.reach_fail())
            self.episodes.append((okra_id, "ik", False))
            return False
        # 単一結果 / 複数waypoint（LIFT→TRANSIT→DESCEND）のリスト、両方を受け付ける。
        legs = sol if isinstance(sol, list) else [sol]
        for i, leg in enumerate(legs):
            if self._stop.is_set():
                return False
            arm14, joint_names, wait_s = leg.arm14, leg.joint_names, leg.wait_s
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
            logger.info(
                f"[grasp-seq] {okra_id}: IK leg {i + 1}/{len(legs)} -> "
                f"waiting {wait_s:.2f}s for arm to settle"
            )
            # open-loop 整定待ち（中断可能、レグ間でも中断チェックが効く）
            if self._stop.wait(wait_s):
                return False

        # 到達確認（任意）: 実測角度からFKを計算し目標座標との残差をログするなど。
        # IK到達そのものは検証しない（何もしなくてもエピソードは続行する） —
        # あくまで観測・原因分析用のフック。
        if self._post_reach_verify_fn is not None:
            self._post_reach_verify_fn()

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
                logger.info(
                    f"[grasp-seq] {okra_id}: VLM says NOT safe to cut -> episode fail (no cut)"
                )
                self.episodes.append((okra_id, "cut_gate", False))
                return False

        # ④ 切断（グリッパ閉じ＝切断＋把持） -----------------------------------
        if self._stop.is_set():
            return False
        logger.info(
            f"[grasp-seq] {okra_id}: CUT (gripper -> {self._q_close} rad, blade limit {self._q_blade_max})"
        )
        self._announcer.say(announce.cutting())
        self._cut(self._q_close)
        # グリッパが実際に閉じきるまで待つ（力覚/位置フィードバックなし、固定の
        # open-loop 待機 — gripper_grasp_on_reach.py の grasp_settle_s と同じ考え方）。
        # ⚠️ 2026-09-08 実機LIVEで、ここで待たずに⑤へ進んだ結果、グリッパが閉じきる
        # 前に籠投入の開き指令(q_open)が飛び、目視では「開閉していないように見える」
        # 現象が確認された。
        if self._cut_settle_s > 0.0:
            if self._stop.wait(self._cut_settle_s):
                return False

        # ⑤ 籠投入（保留: プレースホルダ） ------------------------------------
        if self._place_basket_fn is not None:
            self._announcer.say(announce.basket_depositing())
            self._place_basket_fn()
            self._announcer.say(announce.basket_deposited())
            # ⑥ 復帰: retreat 止まりのままだと次周回の基準姿勢がずれる／目視では
            # 「固定モーションの最後で止まっている」ように見える(2026-09-08)。
            if self._return_to_rest_fn is not None:
                self._return_to_rest_fn()

        self.episodes.append((okra_id, "cut", True))
        return True


__all__ = ["GraspSequence"]
