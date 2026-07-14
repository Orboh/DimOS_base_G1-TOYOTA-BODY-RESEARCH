# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""F-07 籠収納（本番用・同期スキル）— 掴んだオクラを左手の籠へ運び、開いてリリースする。

[[SS-06-切断と籠収納]] の F-07。外部リポ ``Orboh/dex1_1_service`` の ``drop_to_basket``
投入動作（左腕で籠を提示 → 右腕を籠上空へ → グリッパ開きでリリース）を dimos へ移植した
もの。ただし投入姿勢は**固定関節角の再生ではなく、籠位置（torso フレーム, GT/校正値）へ
``IkApproachSkill`` で取り直す**（モデル差・個体差に頑健。設計 F-07 DoD「カゴ位置GT→IK投入→
開く」と一致）。

DDS 送信（腕＝``rt/arm_sdk`` / グリッパ＝``rt/dex1/right/cmd``）は**注入**するので、sim
（bridge 経由）でも実機でも**同一コード**が動く＝本番用。``grasp_sequence`` の
``place_basket_fn`` にも、sim スキルの収納段にも差し込める。
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from dimos.robot.unitree.g1.harvest.ik_approach import IkApproachSkill
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# 横手が実機で録った左腕「籠提示」姿勢（L字）。正準左腕順
# [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw] [rad]。
# 籠は左手首に付くので、この姿勢が籠の3D位置を決める（＝右腕が上空へ届く位置に提示する）。
LEFT_PRESENT_BASKET: list[float] = [-0.1170, -0.0167, -0.3997, 1.1330, 0.0834, -1.0673, -0.2355]

Q_GRIPPER_OPEN = 5.2  # [rad] グリッパ開き（[[SS-06]] の刃保護上限 Q_MAX。これでリリース）


def make_place_basket_fn(
    *,
    basket_torso: Sequence[float],
    send_arm: Callable[[list[float], float], None],
    open_gripper: Callable[[float, float], None],
    get_measured: Callable[[], Sequence[float]],
    left_present: Sequence[float] = LEFT_PRESENT_BASKET,
    right_drop: Sequence[float] | None = None,
    clearance_z: float = 0.10,
    q_open: float = Q_GRIPPER_OPEN,
    present_secs: float = 2.0,
    move_secs: float = 2.5,
    settle_secs: float = 1.2,
    ik: IkApproachSkill | None = None,
) -> Callable[[], bool]:
    """F-07 投入動作を行う ``() -> bool``（成功=True）を組み立てて返す。

    Args:
        basket_torso: 籠中心の3D ``[X前, Y左, Z上]``（torso_link フレーム, [m]）。
            籠は左腕に付くので、**左腕を ``left_present`` にした状態で実測した GT/校正値**を渡す。
        send_arm: ``(arm14, secs)`` — 14関節目標（左7+右7, 正準順）へ ``secs`` 秒で補間送出
            して保持する。**グリッパは閉じ保持のまま**（オクラを落とさない）呼び出し側で担保。
        open_gripper: ``(q, secs)`` — グリッパ目標角 ``q`` を ``secs`` 秒送出（開き＝リリース）。
        get_measured: ``() -> 29-DOF 現在角``（warm-start と左腕 hold に使う）。
        left_present: 左腕の籠提示姿勢（正準左腕7関節 [rad]）。
        right_drop: 右腕の投入姿勢を**固定教示角**（正準右腕7関節 [rad]）で与える。指定時は
            籠上空 IK を使わずこの角度へ動かす（外部リポ dex1_1_service の教示 drop 角に一致
            させたい場合に使用）。None（既定）なら従来どおり ``basket_torso`` へ IK で取り直す。
        clearance_z: 籠中心からどれだけ上を狙うか [m]（ここで開くとオクラが籠へ落ちる）。
        q_open: リリース時のグリッパ開き角 [rad]。
        present_secs / move_secs / settle_secs: 左腕提示 / 右腕投入 / 開き整定の各所要時間 [s]。
        ik: 投入用の IK スキル。None なら籠上空向けの既定（standoff=0, 横に広い WS）を生成。

    Returns:
        ``place() -> bool``。IK が解ければ「左腕提示 → 右腕を籠上空へ → 開いてリリース」を
        実行し True。籠へ IK が解けなければ何もせず False（呼び出し側でリトライ/スキップ）。
    """
    if ik is None:
        # 投入用 IK: reach（オクラ把持）用とは別パラメータ。
        #  - standoff_m=0: 手前で止めず籠の真上を狙う（reach は切断点手前で止める）。
        #  - ws_y を広げる: 籠は体の中央〜やや左。右腕のクロスリーチ目標を弾かないため。
        ik = IkApproachSkill(standoff_m=0.0, ws_y=(-0.75, 0.45))

    target = np.asarray(basket_torso, dtype=float) + np.array([0.0, 0.0, float(clearance_z)])
    left7 = [float(x) for x in left_present]
    right7_fixed = [float(x) for x in right_drop] if right_drop is not None else None

    def place() -> bool:
        meas = list(get_measured())

        # 右腕の投入先: right_drop 指定時は固定教示角（IK 不使用）、なければ籠上空へ IK
        # （解けなければ提示もせず中止）。左腕はどちらも提示姿勢を保持。
        if right7_fixed is None:
            res = ik.solve(target, meas)
            if res is None:
                logger.warning(
                    f"[place-basket] 籠上空 {np.round(target, 3)} へ IK 解けず（投入中止・要リトライ/スキップ）"
                )
                return False
            right7 = [float(x) for x in res.arm14[7:14]]
            drop_secs = max(move_secs, res.wait_s)
            logger.info(
                f"[place-basket] 籠上空へ投入: IK err={res.err:.4f} m "
                f"(target torso={np.round(target, 3)})"
            )
        else:
            right7 = list(right7_fixed)
            drop_secs = move_secs
            logger.info(f"[place-basket] 籠へ投入（固定教示角）right7={np.round(right7, 3)}")

        # Phase A: 左腕を籠提示姿勢へ（右腕は現在角を保持＝オクラを大きく揺らさない）。
        meas_arm = [float(x) for x in meas[15:29]]  # 14（左7+右7）
        arm_present = list(meas_arm)
        arm_present[0:7] = left7
        send_arm(arm_present, present_secs)

        # Phase B: 右腕を投入位置へ（左腕は提示姿勢を保持）。
        arm_drop = list(arm_present)
        arm_drop[0:7] = left7
        arm_drop[7:14] = right7
        send_arm(arm_drop, drop_secs)

        # Phase C: グリッパ開き → オクラをリリース（重力ON+籠コライダーなら物理落下）。
        open_gripper(q_open, settle_secs)
        return True

    return place


__all__ = ["make_place_basket_fn", "LEFT_PRESENT_BASKET", "Q_GRIPPER_OPEN"]
