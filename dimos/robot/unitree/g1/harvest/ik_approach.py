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

"""同期版 IK 粗アプローチスキル（F-04 Phase 1）。

``IkReachBridge``（非同期 DimOS モジュール: クリック点を購読 → IK → ``arm_target``
を publish → ``reach_done`` 発火）の **数理だけを取り出した同期関数** 。LangGraph の
grasp ノードから「呼ぶ → 結果が返る → 次へ」で使える（DDS 購読・スレッド・クリック
フレーム処理は持たない）。設計方針 ii-a（把持を全て LangGraph 同期スキルに揃える）。

役割（[[SS-04-粗アプローチIK]]）:
  オクラの **実の重心3D** を狙い、右腕7自由度 IK で pre-grasp 姿勢の14関節目標を解く。
  最後の数 cm の切断点合わせは ACT（[[SS-05-精密把持ACT]]）に任せるため、IK は重心へ
  寄せれば十分。切断点の手前 ``standoff_m`` で止める。

⚠️ **入力座標系は torso_link フレーム**（X=前方, Y=左, Z=上 — pinocchio/右腕モデルの
   基準）。検出（F-01）が出すのはカメラ→base のハンドアイ校正後の3Dなので、**呼び出し側
   （detect/grasp ノード）が torso フレームの重心3Dに変換して渡す**こと。本スキルは
   座標変換（T_torso_camera）を持たない（校正はカメラ取付の属性で、検出側の責務）。

到達可能性の最終判定（[[SS-04-粗アプローチIK]] §4「IK 収束で確認」）:
  reach box（select の事前フィルタ）とは別に、**本スキルが None を返したら「届かない」**
  とみなして次の候補へフォールバックする。収束しない / 関節デルタ過大 / 関節リミット
  違反 / ワークスペース外 はいずれも None。

アプローチ経路（``solve_legs``/``stream_legs`` の ``above_m``/``front_m``）:
  ``above_m>0`` = 上から降りる（LIFT→TRANSIT→DESCEND、``IkReachBridge._reach`` から
  移植）。``front_m>0`` = 先に対象の高さ(Z)と左右位置(Y)を合わせてから、奥行き
  (X)方向へまっすぐ押し込む（align→push）。両方>0なら above を優先。
  2026-09-11、sim比較用に ``IkReachBridge`` にしか無かった front approach を
  移植したが、当初の移植は「対象の手前へ斜めに直線移動→+X押し込み」で高さと
  水平が同時に動いてしまい意図と異なっていた。2026-09-12 に「先に高さ(Z)だけ
  合わせる→その高さのまま水平(X,Y)に伸ばす」へ一度修正したが、対象のYが
  現在の手先Yと違うと②の水平移動がX,Y同時の斜め移動になり、グリッパーが
  対象に正対せず挿入できない問題が残っていた（ユーザー指摘）。同日中に
  「①高さ(Z)と左右(Y)を先に合わせる→②奥行き(X)だけをまっすぐ進める」の
  現行版へ再修正した（``docs/sim-setup/compare_ik_approach.py`` 参照）。
  2026-09-14、実機LIVEで①のXを完全固定する仕様が原因の失敗（体に近い浅い
  Xから始めるとY,Zを合わせる自由度が2軸しか無く関節可動域超過で頻繁に
  reject）が判明したため、①のXを「現在のXと『対象の手前
  front_align_margin_m(既定0.05m)』の大きい方」まで前進させてよいことにした
  （対象の手前 front_align_margin_m 分は必ず残すため、②は必ず何らかの直進を
  含む＝正対挿入の設計は変わらない）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# 正準 29-DOF G1 関節ベクトル: 腕は 15-28（左 15-21, 右 22-28）。
_ARM_START = 15
_NUM_ARM = 14
_LEFT_SLICE = slice(15, 22)
_RIGHT_SLICE = slice(22, 29)


@dataclass
class IkApproachResult:
    """IK 粗アプローチの解。``arm14`` を arm_target に流し、``wait_s`` だけ待つ。"""

    arm14: list[float]  # 14関節目標 [rad]（左7=現値hold + 右7=IK解, 正準順）
    joint_names: list[str]  # arm14 に対応する関節名（arm_target の name に使う）
    wait_s: float  # スルー完了見込みの open-loop 待機時間 [s]
    err: float  # IK 残差 [m]
    converged: bool  # ソルバ収束フラグ
    q_right: list[float]  # 右腕7関節の解（デバッグ/連続性確認用）


class IkApproachSkill:
    """torso フレームの目標点 → 右腕14関節目標を同期で解く（IkReachBridge._reach の抽出）。

    既定値は ``IkReachBridgeConfig`` の実機検証済み値に合わせてある（standoff 0.05 m、
    Dex1-1 指先オフセット、ワークスペース箱、関節デルタ上限、待機推定）。
    """

    def __init__(
        self,
        urdf_path: str | None = None,
        *,
        gripper_offset_xyz: tuple[float, float, float] = (0.1845, -0.003, 0.0),
        standoff_m: float = 0.05,  # 切断点手前で止める量（torso -X）[m]
        approach_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),  # torso 補正 [m]
        ws_x: tuple[float, float] = (0.05, 0.65),  # ワークスペース箱（torso, 前方）[m]
        # 右腕は -Y 側 [m]。下限(最も右)は旧-0.75から-0.61へ変更（2026-09-12
        # ユーザー指摘）: 3mm厳格化後の実測(9x9グリッドをY方向に0.025m刻みで
        # 走査)で、y=-0.675が到達率>0%になる最遠点（本当の最大到達距離）、
        # y=-0.70以遠は到達率0%だった。その最大到達距離の10%内側
        # (-0.675*0.9≈-0.6075、-0.61に丸め)を新しい下限にすることで、
        # 「箱には入っているのに実際はほぼ届かない」際どい候補（届く確率
        # 24.7%@-0.625、17.3%@-0.65、8.6%@-0.675）を事前に除外し、より
        # 確実に取れる候補だけに絞る。
        # 上限側(+0.20, 左＝体の正中線をまたぐ側)は変更していない。左側は
        # 右腕を伸ばすとG1自身のお腹と干渉するケースがある（ユーザー指摘）が、
        # これはIK単体（自己干渉モデルなし）では検出できない別問題であり、
        # 本変更のスコープ外（要フォロー）。
        ws_y: tuple[float, float] = (-0.61, 0.20),
        ws_z: tuple[float, float] = (-0.35, 0.85),  # [m]
        max_joint_delta_deg: float = 90.0,  # 一発リーチの関節デルタ上限 [deg]
        require_converged: bool = True,
        # 許容残差 [m]。旧既定0.05（5cm）は「莢の茎の間へ数mm精度で刃を入れる」
        # 収穫作業に対して粗すぎた（2026-09-12 ユーザー指摘）。内部ソルバー
        # 自体の収束判定は eps=1e-4（0.1mm、pinocchio_ik.py）と非常に厳しいため、
        # converged=True の解は元々0.1mm級の精度がある。この閾値が効くのは
        # 「100回反復しても収束しきらなかった（best-effort）」少数ケースのみ
        # だが、旧既定5cmだとそのケースを丸ごと許容してしまい、ACTなし
        # （no-ACT）構成では粗いIKの着地点がそのまま切断位置になっていた。
        # 3mm（要求精度と同じ）に厳格化し、それ以上ズレる解は素直に reject
        # （None＝「届かない」）して次の候補へフォールバックさせる。
        max_reach_pos_err_m: float = 0.003,
        # front approach（align→push）専用: align フェーズで許容する最大前進量
        # [m]（対象の手前この距離までは、Y,Zを合わせる際に奥行き(X)が動いて
        # よい）。旧仕様（align中はXを現在位置に完全固定）は、体に近い浅い
        # Xから始めるとY,Zを動かす自由度が2軸しか無く、手首が窮屈な姿勢になり
        # 関節可動域超過で頻繁に reject されていた（2026-09-14 実機LIVEで
        # 確認: align初手からworkspace外、または途中で関節限界超過）。
        # align 終端のXを「現在のXと『対象の手前 front_align_margin_m』の
        # 大きい方（=より対象に近い方）」にすることで、Y,Zを合わせる際にXも
        # 使える自由度が増え、解ける可能性が上がる。ただし対象の手前
        # front_align_margin_m 分は必ず残すため、pushフェーズが必ず何らかの
        # 直進を含む（グリッパーが正対したまま挿入される設計は変わらない）。
        front_align_margin_m: float = 0.05,
        fixed_orientation_xyzw: list[float] | None = None,  # 空=現在のEE姿勢を保持
        nominal_speed_rad_s: float = 1.0,  # 待機推定の有効スルー速度 [rad/s]
        margin_s: float = 0.5,  # 追加の整定マージン [s]
        min_wait_s: float = 0.8,
        max_wait_s: float = 3.0,
    ) -> None:
        from dimos.robot.unitree.g1.ik_reach.right_arm_model import (
            DEFAULT_URDF,
            load_g1_right_arm_ik,
        )

        self._arm = load_g1_right_arm_ik(
            urdf_path or str(DEFAULT_URDF),
            gripper_offset_xyz=list(gripper_offset_xyz),
        )
        # FAIL CLOSED: 全インデックス（warm-start pos[22:29], q_sol, 14-vec, デルタ）が
        # 正準右腕順を前提にしている。順序が違うと黙って肩→手首にマップされ検知不能なので拒否。
        if not self._arm.order_matches_canonical:
            raise RuntimeError(
                f"reduced right-arm order {self._arm.joint_names} != canonical; "
                "index mapping would be silently wrong."
            )
        self._standoff_m = float(standoff_m)
        self._approach_offset = np.asarray(approach_offset_xyz, dtype=float)
        self._ws_x, self._ws_y, self._ws_z = ws_x, ws_y, ws_z
        self._max_joint_delta_deg = float(max_joint_delta_deg)
        self._require_converged = bool(require_converged)
        self._max_reach_pos_err_m = float(max_reach_pos_err_m)
        self._front_align_margin_m = float(front_align_margin_m)
        self._fixed_orientation_xyzw = list(fixed_orientation_xyzw or [])
        self._nominal_speed = float(nominal_speed_rad_s)
        self._margin_s = float(margin_s)
        self._min_wait_s = float(min_wait_s)
        self._max_wait_s = float(max_wait_s)

    @property
    def joint_names(self) -> list[str]:
        """arm14 に対応する正準腕関節名（左7 + 右7）。"""
        from dimos.control.components import make_humanoid_joints

        return list(make_humanoid_joints("g1"))[_ARM_START : _ARM_START + _NUM_ARM]

    def tip_torso(self, q_right: Any) -> list[float]:
        """右腕7関節角 → 現在の手先(tip)の torso_link 座標 [m]。

        内部の FK 計算（``solve_legs``/``stream_legs`` が above/front の起点に使う
        のと同じもの）を公開したヘルパー。ログ/可視化/sim比較用途（例:
        ``docs/sim-setup/compare_ik_approach.py`` が ``stream_legs`` の
        ``on_waypoint`` コールバックで受け取った ``arm14`` から手先の実軌跡を
        再構成するのに使う）。制御ロジック自体はこれを使わない。
        """
        q = np.asarray(q_right, dtype=float).flatten()
        pos = np.asarray(self._arm.root_to_torso_pose(self._arm.fk_tip(q)).translation).flatten()
        return [float(x) for x in pos]

    def _clamp_violation_detail(self, q_sol: np.ndarray) -> str:
        """``clamp_ok`` 失敗時、どの関節がどれだけ限界を超えたかを1行で返す（デバッグ用）。"""
        q = np.asarray(q_sol, dtype=float).flatten()
        lower, upper = self._arm.lower, self._arm.upper
        parts = []
        for i, name in enumerate(self._arm.joint_names):
            if q[i] < lower[i] - 1e-6:
                parts.append(f"{name}={q[i]:.3f}<{lower[i]:.3f}(lower)")
            elif q[i] > upper[i] + 1e-6:
                parts.append(f"{name}={q[i]:.3f}>{upper[i]:.3f}(upper)")
        return "; ".join(parts) if parts else "(clamp_ok は False だが詳細特定できず)"

    def solve(self, target_torso: Any, measured_position: Any) -> IkApproachResult | None:
        """torso フレームの目標点 → 14関節目標。届かない/解けない場合は None。

        Args:
            target_torso: オクラ実の重心3D ``[X, Y, Z]``（torso_link フレーム, [m]）。
            measured_position: 計測関節角（29-DOF 以上のシーケンス。warm-start と
                左腕 hold、姿勢保持に使う）。

        Returns:
            :class:`IkApproachResult`、または到達不能（ワークスペース外/不収束/デルタ過大/
            リミット違反）なら ``None``。
        """
        import pinocchio  # lazy: pinocchio は実行ホスト（Orin）にのみ在る

        from dimos.manipulation.planning.kinematics.pinocchio_ik import (
            check_joint_delta,
            get_worst_joint_delta,
        )

        pos = list(measured_position)
        if len(pos) < _ARM_START + _NUM_ARM:
            logger.warning(f"[ik-approach] measured pose has {len(pos)} joints; expected >= 29")
            return None
        q_left = np.array([float(x) for x in pos[_LEFT_SLICE]])
        q_right = np.array([float(x) for x in pos[_RIGHT_SLICE]])
        if not (np.all(np.isfinite(q_left)) and np.all(np.isfinite(q_right))):
            logger.warning("[ik-approach] measured arm pose has non-finite values; rejecting.")
            return None

        # 目標点（torso）: 重心 + approach_offset、さらに切断点手前 standoff（torso -X）。
        p_torso = np.asarray(target_torso, dtype=float) + self._approach_offset
        p_torso = p_torso - np.array([self._standoff_m, 0.0, 0.0])

        # ワークスペース箱（torso フレーム）: 妥当でない目標は弾く。
        if not (
            self._ws_x[0] <= p_torso[0] <= self._ws_x[1]
            and self._ws_y[0] <= p_torso[1] <= self._ws_y[1]
            and self._ws_z[0] <= p_torso[2] <= self._ws_z[1]
        ):
            logger.info(
                f"[ik-approach] torso target {np.round(p_torso, 3)} outside workspace box; reject."
            )
            return None

        p_root = self._arm.torso_to_root(p_torso)

        # 姿勢: 固定（ROOT フレーム）or 現在の EE 姿勢を保持（位置のみリーチ）。
        if self._fixed_orientation_xyzw:
            qx, qy, qz, qw = self._fixed_orientation_xyzw
            rot = pinocchio.Quaternion(qw, qx, qy, qz).normalized().toRotationMatrix()
        else:
            rot = self._arm.fk_root(q_right).rotation
        target = pinocchio.SE3(rot, np.asarray(p_root, dtype=float))

        # 解 + 安全ゲート
        q_sol, converged, err = self._arm.ik.solve(target, q_right)
        q_sol = np.asarray(q_sol, dtype=float).flatten()

        if self._require_converged and not converged and err > self._max_reach_pos_err_m:
            logger.info(
                f"[ik-approach] IK err={err:.4f} m > tol {self._max_reach_pos_err_m} m; reject."
            )
            return None
        if not converged:
            logger.info(f"[ik-approach] best-effort reach (err={err:.4f} m ≤ tol).")
        if not check_joint_delta(q_sol, q_right, self._max_joint_delta_deg):
            wi, wd = get_worst_joint_delta(q_sol, q_right)
            logger.info(
                f"[ik-approach] joint {self._arm.joint_names[wi]} delta {wd:.1f}° "
                f"> {self._max_joint_delta_deg}°; reject."
            )
            return None
        if not self._arm.clamp_ok(q_sol):
            logger.info(
                f"[ik-approach] q_sol {np.round(q_sol, 3)} violates joint limits "
                f"({self._clamp_violation_detail(q_sol)}); reject."
            )
            return None

        arm14 = np.concatenate([q_left, q_sol])  # 左7 hold + 右7 IK（正準順）

        # open-loop 待機: 関節移動量から推定（計測で完了判定すると ACT がスルー途中に
        # 誤発火した経緯があるため、時間ベースにする）。
        delta = float(np.max(np.abs(q_sol - q_right)))
        wait_s = delta / max(self._nominal_speed, 1e-3) + self._margin_s
        wait_s = min(max(wait_s, self._min_wait_s), self._max_wait_s)

        return IkApproachResult(
            arm14=[float(x) for x in arm14],
            joint_names=self.joint_names,
            wait_s=wait_s,
            err=float(err),
            converged=bool(converged),
            q_right=[float(x) for x in q_sol],
        )

    def solve_legs(
        self,
        target_torso: Any,
        measured_position: Any,
        *,
        above_m: float = 0.0,
        front_m: float = 0.0,
    ) -> list[IkApproachResult] | None:
        """複数waypointの段階的アプローチ版 ``solve()``。

        ``above_m <= 0`` かつ ``front_m <= 0``（既定）: ``solve()`` 単体と同じ、
        1要素のリスト ``[result]`` を返す（後方互換 — 呼び出し側は常にリストとして
        扱ってよい）。

        ``above_m > 0``: ``IkReachBridge._reach`` の approach-from-above と同じ
        「① LIFT: 現在の手先 x,y のまま真上へ ② TRANSIT: 対象の真上（高度を保持）へ
        水平移動 ③ DESCEND: 対象へ垂直降下」の3レグで解く。低い休憩姿勢から目標へ
        一発で関節空間リーチすると、手先の実軌道が読めない弧を描き株を払ったり、
        重力補償の立ち上がり待ちと相まって中途半端な位置で止まったりする
        （2026-09-08 実機LIVEで確認）ことへの対策 — 各レグは前レグの解を warm-start
        にした個別の position-only IK 解であり、``IkReachBridge`` のような密な
        Cartesian ストリーミングではない（レグの端点だけを直線的につなぐ簡易版）。

        ``front_m > 0``（``above_m<=0`` の場合のみ有効 — 両方>0なら above を優先）:
        「① ALIGN: 現在の手先 x（奥行き）のまま対象の高さ(z)と左右位置(y)へ
        同時に合わせる ② PUSH: その位置から奥行き(+x)方向だけへまっすぐ押し込む」
        の2レグで解く。above の「一旦持ち上げてから水平移動→垂直降下」とは違い、
        「先に位置(y,z)を合わせてから奥行きに正対して押し込む」ため、上から
        株へ降りない・横から払わないだけでなく、②が純粋な直線移動になり
        グリッパーが対象へ正面から（斜めにならず）挿入される。
        （2026-09-12、ユーザー指摘により2段階で修正: 当初「対象の手前へ斜め移動
        →+X押し込み」→「①高さのみ合わせる→②水平(x,y)に伸ばす」に直したが、
        対象のyが現在の手先yと違うと②が斜め移動になりグリッパーが正対しない
        問題が残っていたため、①でy,zを両方合わせる現行版へ再修正）。
        front_m の具体的な値は経路形状には使わない（``>0`` で本モードを有効化するだけ）。

        いずれかのレグが届かない/解けない場合は全体を ``None``（「届かない」）として
        扱う — 一部のレグだけ実行して中断すると、手先が中途半端な高さで止まり危険。
        """
        if above_m <= 0.0 and front_m <= 0.0:
            res = self.solve(target_torso, measured_position)
            return [res] if res is not None else None

        import pinocchio

        from dimos.manipulation.planning.kinematics.pinocchio_ik import (
            check_joint_delta,
            get_worst_joint_delta,
        )

        legs = self._legs_torso(target_torso, measured_position, above_m, front_m)
        if legs is None:
            return None
        legs_torso, q_left, q_right, rot = legs

        results: list[IkApproachResult] = []
        q_seed = q_right
        for label, p_torso in legs_torso:
            p_root = self._arm.torso_to_root(p_torso)
            target = pinocchio.SE3(rot, np.asarray(p_root, dtype=float))
            q_sol, converged, err = self._arm.ik.solve(target, q_seed)
            q_sol = np.asarray(q_sol, dtype=float).flatten()
            if self._require_converged and not converged and err > self._max_reach_pos_err_m:
                logger.info(
                    f"[ik-approach] {label} leg IK err={err:.4f} m > tol "
                    f"{self._max_reach_pos_err_m} m; rejecting whole approach."
                )
                return None
            if not check_joint_delta(q_sol, q_seed, self._max_joint_delta_deg):
                wi, wd = get_worst_joint_delta(q_sol, q_seed)
                logger.info(
                    f"[ik-approach] {label} leg joint {self.joint_names[wi]} delta {wd:.1f}° > "
                    f"{self._max_joint_delta_deg}°; rejecting whole approach."
                )
                return None
            if not self._arm.clamp_ok(q_sol):
                logger.info(
                    f"[ik-approach] {label} leg q_sol {np.round(q_sol, 3)} violates joint "
                    f"limits ({self._clamp_violation_detail(q_sol)}); rejecting whole approach."
                )
                return None
            arm14 = np.concatenate([q_left, q_sol])
            delta = float(np.max(np.abs(q_sol - q_seed)))
            wait_s = delta / max(self._nominal_speed, 1e-3) + self._margin_s
            wait_s = min(max(wait_s, self._min_wait_s), self._max_wait_s)
            results.append(
                IkApproachResult(
                    arm14=[float(x) for x in arm14],
                    joint_names=self.joint_names,
                    wait_s=wait_s,
                    err=float(err),
                    converged=bool(converged),
                    q_right=[float(x) for x in q_sol],
                )
            )
            q_seed = q_sol
        return results

    def _legs_torso(
        self,
        target_torso: Any,
        measured_position: Any,
        above_m: float,
        front_m: float = 0.0,
    ) -> tuple[list[tuple[str, np.ndarray]], np.ndarray, np.ndarray, np.ndarray] | None:
        """``solve_legs``/``stream_legs`` 共通: torso座標系のレグ端点リストと
        ``(q_left, q_right, rot)`` を計算する。届かない/計測なしなら ``None``。

        ``above_m`` と ``front_m`` がともに ``<= 0`` でも「direct」1レグのリストを
        返す（``stream_legs`` がアプローチ高さ/前進距離無しでも同じ密ストリーミング
        経路を使えるように）。両方 ``> 0`` なら ``above`` を優先する
        （``IkReachBridge._reach`` と同じ規約）。
        """
        import pinocchio

        pos = list(measured_position)
        if len(pos) < _ARM_START + _NUM_ARM:
            logger.warning(f"[ik-approach] measured pose has {len(pos)} joints; expected >= 29")
            return None
        q_left = np.array([float(x) for x in pos[_LEFT_SLICE]])
        q_right = np.array([float(x) for x in pos[_RIGHT_SLICE]])
        if not (np.all(np.isfinite(q_left)) and np.all(np.isfinite(q_right))):
            logger.warning("[ik-approach] measured arm pose has non-finite values; rejecting.")
            return None

        p_final = np.asarray(target_torso, dtype=float) + self._approach_offset
        p_final = p_final - np.array([self._standoff_m, 0.0, 0.0])
        if not (
            self._ws_x[0] <= p_final[0] <= self._ws_x[1]
            and self._ws_y[0] <= p_final[1] <= self._ws_y[1]
            and self._ws_z[0] <= p_final[2] <= self._ws_z[1]
        ):
            logger.info(
                f"[ik-approach] final torso target {np.round(p_final, 3)} outside workspace box; reject."
            )
            return None

        if self._fixed_orientation_xyzw:
            qx, qy, qz, qw = self._fixed_orientation_xyzw
            rot = pinocchio.Quaternion(qw, qx, qy, qz).normalized().toRotationMatrix()
        else:
            rot = self._arm.fk_root(q_right).rotation

        legs_torso: list[tuple[str, np.ndarray]] = []
        if above_m > 0.0 or front_m > 0.0:
            tip_now = np.asarray(
                self._arm.root_to_torso_pose(self._arm.fk_tip(q_right)).translation
            ).flatten()
        if above_m > 0.0:
            z_transit = max(float(p_final[2]) + float(above_m), float(tip_now[2]))
            if float(tip_now[2]) < z_transit - 0.02:
                legs_torso.append(("lift", np.array([tip_now[0], tip_now[1], z_transit])))
            legs_torso.append(("transit", np.array([p_final[0], p_final[1], z_transit])))
            legs_torso.append(("descend", p_final))
        elif front_m > 0.0:
            # 正面アプローチ（2026-09-12 再修正、2026-09-14 align のX許容幅を追加）:
            # ① 対象の高さ(Z)と左右位置(Y)を合わせる(align)。奥行き(X)は現在位置
            # のまま、ただし「対象の手前 front_align_margin_m まで」なら前進して
            # よい ② そこから奥行き(+X)方向へ「まっすぐ」対象へ押し込む(push)。
            #
            # ⚠️ 最初の修正（2026-09-12午前）では①を「高さ(Z)のみ」合わせ、Y,Xは
            # ②でまとめて動かしていた。これだと対象のYが今の手先Yと違う場合、②が
            # X,Y同時の斜め移動になり、グリッパーが正面からまっすぐ刺さらず対象を
            # 挟めない（ユーザー指摘: 「対象に対してグリッパーが正面から刺さらないと
            # 収穫できない」）。①で高さとY（左右）を先に合わせきることで、②は
            # 純粋に奥行き(X)だけの直線移動になり、ジョーが対象に正対したまま
            # まっすぐ挿入される。above が「一旦持ち上げて→水平移動→垂直降下」
            # なのに対し、front は「先に位置(Y,Z)を合わせる→奥行きに正対して押し
            # 込む」。front_m の具体的な値は経路形状には使わない（>0 で本モードを
            # 有効化するだけ）。
            #
            # ⚠️ 2026-09-14 実機LIVEで判明: align中にXを完全固定すると、体に近い
            # 浅いX（休憩姿勢由来）から始めた場合、Y,Zを合わせる自由度が2軸しか
            # 無く手首が窮屈な姿勢になり、workspace外/関節可動域超過で頻繁に
            # rejectされていた（ユーザー所見: 「奥行きを変えずY,Zだけ合わせるのは
            # IKで解けないケースが多い」）。align終端のXを「現在のXと『対象の手前
            # front_align_margin_m』の大きい方」にすることで、Y,Zを合わせる際に
            # Xも自由に使えるようにしつつ、対象の手前 front_align_margin_m 分は
            # 必ず残す（=pushが必ず直進を含み、正対挿入の設計は変わらない）。
            align_x = max(float(tip_now[0]), float(p_final[0]) - self._front_align_margin_m)
            legs_torso.append(("align", np.array([align_x, p_final[1], p_final[2]])))
            legs_torso.append(("push", p_final))
        else:
            legs_torso.append(("direct", p_final))

        return legs_torso, q_left, q_right, rot

    def stream_legs(
        self,
        target_torso: Any,
        measured_position: Any,
        *,
        above_m: float = 0.0,
        front_m: float = 0.0,
        send_arm: Callable[[list[float]], None],
        on_waypoint: Callable[[str, list[float], list[float]], None] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        step_m: float = 0.035,
        cadence_s: float = 0.18,
    ) -> IkApproachResult | None:
        """段階的アプローチを ``IkReachBridge._stream_leg`` と同じ密度でストリーミング実行する。

        ``solve_legs()``（レグの端点だけを解いて関節空間補間任せにする簡易版）とは違い、
        各レグの区間を ``step_m``[m] 間隔の waypoint に分割し、waypoint ごとに個別へ
        IK を再計算しながら ``send_arm(arm14)`` で逐次 publish、``cadence_s``[s] ごとに
        ``sleep_fn`` で待つ（tip speed ≈ step_m/cadence_s）。手先が実際にほぼ直線を描く
        — クリック駆動版(``IkReachBridge``)の動きに近い。``above_m`` と ``front_m`` が
        ともに ``<=0`` でも「direct」1区間の密ストリーミングとして動く。両方 ``>0``
        なら ``above`` を優先する（``_legs_torso`` と同じ規約）。

        Args:
            on_waypoint: ``(leg_label, p_torso_xyz, arm14) -> None``。各waypointの
                送出直後に呼ばれる任意コールバック（ログ/可視化/比較用 — sim比較で
                「above」と「front」の手先軌跡を記録して重ねるのに使う想定。
                ``harvest_module.py`` 等の実運用呼び出しでは None のまま）。

        ⚠️ この関数は呼び出し中ブロックする（全waypoint分の ``sleep_fn`` を合計すると
        数秒かかる）。中断チェックは持たない — 呼び出し側で早期に止めたい場合は
        ``send_arm`` 側やハードウェアの e-stop に頼ること（SafetyMonitor のファイル
        e-stop はこの関数が返るまで効かない）。

        Returns:
            全waypoint完走なら最終姿勢の ``IkApproachResult``（``wait_s=0.0`` — 既に
            ストリーミング中に整定時間を消化済みなので、呼び出し側の追加待機は不要）。
            どこかの waypoint が届かない/解けない場合は ``None``（既に送出済みの
            waypointまでは動いてしまっている点に注意 — ``solve_legs`` と違い全か無か
            ではない。ワークスペース逸脱などレアケースのみで、通常運用では起きない
            前提の設計）。
        """
        legs = self._legs_torso(target_torso, measured_position, above_m, front_m)
        if legs is None:
            return None
        legs_torso, q_left, q_right, rot = legs

        import pinocchio

        from dimos.manipulation.planning.kinematics.pinocchio_ik import (
            check_joint_delta,
            get_worst_joint_delta,
        )

        step = max(float(step_m), 0.005)
        cadence = max(float(cadence_s), 0.02)
        q_cur = q_right
        p_cur = np.asarray(
            self._arm.root_to_torso_pose(self._arm.fk_tip(q_right)).translation
        ).flatten()
        last_err = 0.0
        last_converged = True
        for label, p_to in legs_torso:
            p_to = np.asarray(p_to, dtype=float)
            seg = p_to - p_cur
            n = max(1, int(np.ceil(float(np.linalg.norm(seg)) / step)))
            for i in range(1, n + 1):
                p_i = p_to if i == n else p_cur + seg * (i / n)
                if not (
                    self._ws_x[0] <= p_i[0] <= self._ws_x[1]
                    and self._ws_y[0] <= p_i[1] <= self._ws_y[1]
                    and self._ws_z[0] <= p_i[2] <= self._ws_z[1]
                ):
                    logger.info(
                        f"[ik-approach] {label} step {i}/{n} torso{np.round(p_i, 3)} "
                        f"outside workspace (ws_x={self._ws_x} ws_y={self._ws_y} "
                        f"ws_z={self._ws_z}); rejecting whole approach."
                    )
                    return None
                target = pinocchio.SE3(
                    rot, np.asarray(self._arm.torso_to_root(p_i), dtype=float)
                )
                q_sol, converged, err = self._arm.ik.solve(target, q_cur)
                q_sol = np.asarray(q_sol, dtype=float).flatten()
                if (
                    self._require_converged
                    and not converged
                    and err > self._max_reach_pos_err_m
                ):
                    logger.info(
                        f"[ik-approach] {label} step {i}/{n} torso{np.round(p_i, 3)} "
                        f"IK err={err:.4f} m > tol; rejecting whole approach."
                    )
                    return None
                if not check_joint_delta(q_sol, q_cur, self._max_joint_delta_deg):
                    wi, wd = get_worst_joint_delta(q_sol, q_cur)
                    logger.info(
                        f"[ik-approach] {label} step {i}/{n} torso{np.round(p_i, 3)} joint "
                        f"{self.joint_names[wi]} delta {wd:.1f}° > "
                        f"{self._max_joint_delta_deg}°; rejecting whole approach."
                    )
                    return None
                if not self._arm.clamp_ok(q_sol):
                    logger.info(
                        f"[ik-approach] {label} step {i}/{n} torso{np.round(p_i, 3)} "
                        f"violates joint limits ({self._clamp_violation_detail(q_sol)}); "
                        "rejecting whole approach."
                    )
                    return None
                arm14 = np.concatenate([q_left, q_sol])
                arm14_list = [float(x) for x in arm14]
                send_arm(arm14_list)
                if on_waypoint is not None:
                    on_waypoint(label, [float(x) for x in p_i], arm14_list)
                q_cur = q_sol
                last_err, last_converged = float(err), bool(converged)
                sleep_fn(cadence)
            p_cur = p_to
            logger.info(f"[ik-approach] {label} leg done ({n} steps -> torso{np.round(p_cur, 3)})")

        return IkApproachResult(
            arm14=[float(x) for x in np.concatenate([q_left, q_cur])],
            joint_names=self.joint_names,
            wait_s=0.0,
            err=last_err,
            converged=last_converged,
            q_right=[float(x) for x in q_cur],
        )


__all__ = ["IkApproachResult", "IkApproachSkill"]
