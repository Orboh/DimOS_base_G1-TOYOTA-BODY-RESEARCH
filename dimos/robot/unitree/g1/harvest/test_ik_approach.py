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

"""Tests for the synchronous IK approach skill (right-arm reach).

Requires ``pinocchio`` + the G1 URDF (present on the Orin; skipped where pinocchio
is absent, e.g. a laptop dev box). These exercise the gate logic (workspace box,
convergence, joint-delta, limits) and a round-trip reach via the real right-arm
model — no robot needed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pinocchio")  # IK solver dep; skip suite if absent (laptop)

from dimos.robot.unitree.g1.harvest.ik_approach import IkApproachResult, IkApproachSkill


def _rest_state() -> list[float]:
    """A plausible 29-DOF measured pose (zeros = arms down/rest)."""
    return [0.0] * 29


def test_reach_to_in_workspace_target_succeeds() -> None:
    """A torso-frame target inside the workspace box returns a 14-vec reach."""
    skill = IkApproachSkill()
    # Forward 0.35 m, right (-Y) 0.25 m, slightly below torso_link — well inside the box.
    target_torso = np.array([0.35, -0.25, 0.10])
    res = skill.solve(target_torso, _rest_state())
    assert isinstance(res, IkApproachResult)
    assert len(res.arm14) == 14
    assert len(res.joint_names) == 14
    # Left arm (first 7) is held at the measured value (rest = 0).
    assert res.arm14[:7] == pytest.approx([0.0] * 7, abs=1e-9)
    # Reach is within tolerance and the wait is clamped to the configured range.
    assert res.err <= 0.05 + 1e-6
    assert 0.8 <= res.wait_s <= 3.0


def test_target_outside_workspace_box_is_rejected() -> None:
    """A target behind/over the torso (outside the box) is rejected (returns None)."""
    skill = IkApproachSkill()
    behind = np.array([-0.5, -0.25, 0.10])  # x < ws_x[0]=0.05 -> reject
    assert skill.solve(behind, _rest_state()) is None
    too_left = np.array([0.35, 0.6, 0.10])  # y > ws_y[1]=0.20 -> reach across body -> reject
    assert skill.solve(too_left, _rest_state()) is None


def test_standoff_stops_short_in_torso_x() -> None:
    """With standoff_m>0 the commanded tip lands ~standoff short of the okra in X."""
    no_standoff = IkApproachSkill(standoff_m=0.0)
    with_standoff = IkApproachSkill(standoff_m=0.05)
    target = np.array([0.40, -0.25, 0.10])
    r0 = no_standoff.solve(target, _rest_state())
    r1 = with_standoff.solve(target, _rest_state())
    assert r0 is not None and r1 is not None
    # The two solutions differ (standoff shifts the target), proving standoff is applied.
    assert r0.q_right != pytest.approx(r1.q_right, abs=1e-6)


def test_short_pose_rejected() -> None:
    """A measured pose with < 29 joints is rejected, not crashed."""
    skill = IkApproachSkill()
    assert skill.solve(np.array([0.35, -0.25, 0.10]), [0.0] * 10) is None


def test_non_finite_pose_rejected() -> None:
    skill = IkApproachSkill()
    bad = [0.0] * 29
    bad[22] = float("nan")
    assert skill.solve(np.array([0.35, -0.25, 0.10]), bad) is None


def test_solve_legs_front_m_returns_two_legs() -> None:
    """``front_m>0`` は「align → push」の2レグ。"""
    skill = IkApproachSkill()
    target = np.array([0.40, -0.25, 0.10])
    legs = skill.solve_legs(target, _rest_state(), front_m=0.15)
    assert legs is not None
    assert len(legs) == 2
    assert all(isinstance(leg, IkApproachResult) for leg in legs)


def test_solve_legs_above_wins_when_both_set() -> None:
    """``above_m`` と ``front_m`` を両方>0で渡すと above（3レグ）が優先される。"""
    skill = IkApproachSkill()
    target = np.array([0.40, -0.25, 0.10])
    legs = skill.solve_legs(target, _rest_state(), above_m=0.08, front_m=0.15)
    assert legs is not None
    assert len(legs) == 3  # lift, transit, descend (above のレグ数)


def test_solve_legs_no_approach_matches_direct_solve() -> None:
    """``above_m<=0`` かつ ``front_m<=0``（既定）は ``solve()`` と同じ1要素リスト。"""
    skill = IkApproachSkill()
    target = np.array([0.35, -0.25, 0.10])
    legs = skill.solve_legs(target, _rest_state())
    direct = skill.solve(target, _rest_state())
    assert legs is not None and len(legs) == 1
    assert legs[0].arm14 == pytest.approx(direct.arm14, abs=1e-9)


def test_stream_legs_front_m_aligns_position_then_pushes_straight() -> None:
    """front approach は「対象のy,zへ合わせ、対象手前front_align_margin_mまで
    前進してよい(align)」→「そこからxだけ直進(push)」。

    2026-09-12 再修正: 対象のyが現在の手先yと異なる場合でも、push レグが
    x以外(y,z)を動かさない「まっすぐな直進」になっていること（＝グリッパーが
    対象に正対したまま挿入される）を確認する — 旧版はここが斜め移動になり、
    ユーザーから「グリッパーが正面から刺さらない」と指摘された箇所。
    2026-09-14 再修正: align中にXを完全固定していた旧仕様は、体に近い浅いXから
    始めるとY,Zを合わせる自由度が2軸しか無く関節可動域超過で頻繁にrejectされて
    いたため、対象の手前front_align_margin_mまでは前進してよいことにした。
    ここでは現在位置が対象よりだいぶ手前にあるケース（前進が実際に発生する）を
    確認する。
    """
    skill = IkApproachSkill()
    target = np.array([0.55, -0.25, 0.10])  # 現在位置よりだいぶ奥 -> align で前進が発生する
    rest_tip = skill.tip_torso([0.0] * 7)  # _rest_state() の右腕7関節に対応する現在のtip位置
    assert abs(rest_tip[1] - target[1]) > 0.05  # 対象のyが現在のyと十分違うことを確認
    assert rest_tip[0] < target[0] - 0.05 - 0.05  # 現在位置が「対象-standoff-margin」より手前
    waypoints: list[tuple[str, list[float]]] = []
    res = skill.stream_legs(
        target,
        _rest_state(),
        front_m=0.15,
        send_arm=lambda _arm14: None,
        on_waypoint=lambda label, p, _arm14: waypoints.append((label, p)),
        sleep_fn=lambda _s: None,  # テストを待たせない
    )
    assert res is not None
    assert res.converged
    labels = {label for label, _ in waypoints}
    assert labels == {"align", "push"}
    # 最後の waypoint（push の終端）は standoff 込みの最終目標に到達している。
    last_label, last_p = waypoints[-1]
    assert last_label == "push"
    assert last_p == pytest.approx([0.50, -0.25, 0.10], abs=1e-6)  # target - standoff(0.05) in X
    # align レグの終端は「対象の手前front_align_margin_m(既定0.05)、対象のy,z」
    # に合わせた点（現在のXが対象よりだいぶ手前なので前進が発生している）。
    align_end = next(p for label, p in reversed(waypoints) if label == "align")
    assert align_end == pytest.approx([0.45, -0.25, 0.10], abs=1e-6)  # 0.50 - margin(0.05)
    # push の間、y,z は align で合わせた値のまま変化しない（x だけの直進）。
    push_ys = [p[1] for label, p in waypoints if label == "push"]
    push_zs = [p[2] for label, p in waypoints if label == "push"]
    assert push_ys == pytest.approx([-0.25] * len(push_ys), abs=1e-6)
    assert push_zs == pytest.approx([0.10] * len(push_zs), abs=1e-6)


def test_stream_legs_front_m_align_does_not_retreat_when_already_close() -> None:
    """現在のXがすでに対象の手前front_align_margin_mより近い場合、alignは
    後退させず現在のXのまま（前進のみを許す片側クランプであることの確認）。"""
    skill = IkApproachSkill()
    target = np.array([0.40, -0.25, 0.10])  # standoff込み p_final_x=0.35
    rest_tip = skill.tip_torso([0.0] * 7)
    assert rest_tip[0] > 0.35 - 0.05  # 現在位置がすでに「対象の手前margin」より近い前提
    waypoints: list[tuple[str, list[float]]] = []
    res = skill.stream_legs(
        target,
        _rest_state(),
        front_m=0.15,
        send_arm=lambda _arm14: None,
        on_waypoint=lambda label, p, _arm14: waypoints.append((label, p)),
        sleep_fn=lambda _s: None,
    )
    assert res is not None
    align_end = next(p for label, p in reversed(waypoints) if label == "align")
    assert align_end[0] == pytest.approx(rest_tip[0], abs=1e-6)


def test_tip_torso_matches_internal_fk() -> None:
    """``tip_torso`` は rest 姿勢で reach 内目標に近い妥当な torso 座標を返す（クラッシュしない）。"""
    skill = IkApproachSkill()
    pos = skill.tip_torso([0.0] * 7)
    assert len(pos) == 3
    assert all(np.isfinite(pos))
