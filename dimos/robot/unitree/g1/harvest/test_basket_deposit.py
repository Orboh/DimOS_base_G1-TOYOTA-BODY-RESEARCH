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

"""Tests for make_basket_deposit_fn: 教示q7モード（推奨）とIKフォールバックモード。

教示モードはpinocchio/IKを一切使わないため、ここではpinocchioに依存しない
（IKフォールバックの経路確認だけフェイクのIKスキルを注入する）。
"""

from __future__ import annotations

from dimos.robot.unitree.g1.harvest.basket_deposit import make_basket_deposit_fn


def _measured(q_left: list[float] | None = None, q_right: list[float] | None = None) -> list[float]:
    """29-DOF 計測ポーズのスタブ（左右腕スライスだけが意味を持つ）。"""
    pos = [0.0] * 29
    if q_left is not None:
        pos[15:22] = q_left
    if q_right is not None:
        pos[22:29] = q_right
    return pos


def test_taught_mode_replays_three_poses_in_order() -> None:
    """entry/drop/retreatの3点とも教示q7が揃っていれば、IKを使わず直接再生する。"""
    sent: list[tuple[list[float], float]] = []
    opened: list[tuple[float, float]] = []
    entry_q7 = [0.1] * 7
    drop_q7 = [0.2] * 7
    retreat_q7 = [0.1] * 7  # == entry

    place = make_basket_deposit_fn(
        send_arm=lambda arm14, secs: sent.append((arm14, secs)),
        open_gripper=lambda q, secs: opened.append((q, secs)),
        get_measured=lambda: _measured(q_left=[9.0] * 7, q_right=[0.0] * 7),
        entry_q7=entry_q7,
        drop_q7=drop_q7,
        retreat_q7=retreat_q7,
        sleep_fn=lambda _s: None,
    )
    ok = place()

    assert ok is True
    assert len(sent) == 3
    for arm14, _secs in sent:
        assert arm14[:7] == [9.0] * 7  # 左腕は常に計測値のままhold
    assert sent[0][0][7:] == entry_q7
    assert sent[1][0][7:] == drop_q7
    assert sent[2][0][7:] == retreat_q7
    assert len(opened) == 1  # 一度だけグリッパを開放（タイミングは別テストで検証）


def test_taught_mode_opens_gripper_right_after_drop_not_after_retreat() -> None:
    """グリッパはdrop到達直後に開き、retreatより前でなければならない。

    2026-09-14 実機LIVEで判明したバグの再発防止: entry→drop→retreat 全部を
    移動し終えてから最後に開く実装になっており、実際には「かごから離れた後に
    開く」という誤った動作になっていた。
    """
    events: list[str] = []
    entry_q7 = [0.1] * 7
    drop_q7 = [0.2] * 7
    retreat_q7 = [0.3] * 7  # entryと区別するため異なる値にする

    place = make_basket_deposit_fn(
        send_arm=lambda arm14, secs: events.append(f"send:{arm14[7]}"),
        open_gripper=lambda q, secs: events.append("open"),
        get_measured=lambda: _measured(q_left=[9.0] * 7, q_right=[0.0] * 7),
        entry_q7=entry_q7,
        drop_q7=drop_q7,
        retreat_q7=retreat_q7,
        sleep_fn=lambda _s: None,
    )
    assert place() is True
    assert events == ["send:0.1", "send:0.2", "open", "send:0.3"]


def test_taught_mode_never_fails() -> None:
    """教示モードは常にTrueを返す（教示済み姿勢は安全確認済みという前提のため）。"""
    place = make_basket_deposit_fn(
        send_arm=lambda arm14, secs: None,
        open_gripper=lambda q, secs: None,
        get_measured=lambda: _measured(),
        entry_q7=[0.0] * 7,
        drop_q7=[0.0] * 7,
        retreat_q7=[0.0] * 7,
        sleep_fn=lambda _s: None,
    )
    assert place() is True


def test_partial_taught_q7_falls_back_to_ik_mode() -> None:
    """entry/drop/retreatの一部だけ指定 -> 教示は不成立、IKモードへフォールバックする。"""

    class _FakeIk:
        def __init__(self) -> None:
            self.calls = 0

        def solve(self, target: object, meas: object) -> None:
            self.calls += 1
            return None  # 届かない扱い -> place() は最初のレグでFalseを返すはず

    fake_ik = _FakeIk()
    place = make_basket_deposit_fn(
        send_arm=lambda arm14, secs: None,
        open_gripper=lambda q, secs: None,
        get_measured=lambda: _measured(),
        entry_q7=[0.0] * 7,  # drop_q7/retreat_q7 が無い -> 教示3点セットが不成立
        ik=fake_ik,
        sleep_fn=lambda _s: None,
    )

    assert place() is False
    assert fake_ik.calls == 1  # IKモードのフォールバック経路に入ったことの確認


def test_ik_mode_opens_gripper_right_after_drop_not_after_retreat() -> None:
    """IKモードでも教示モードと同様に、dropのIK到達直後にリリースし、
    retreatより前でなければならない（2026-09-14 実機LIVEで判明したバグの
    再発防止。教示モードと同じ順序保証をIKモードにも要求する）。
    """

    class _FakeResult:
        def __init__(self, tag: float) -> None:
            self.arm14 = [tag] * 14
            self.joint_names = [f"j{i}" for i in range(14)]
            self.wait_s = 0.0
            self.err = 0.0
            self.converged = True
            self.q_right = [tag] * 7

    class _FakeIk:
        def solve(self, target: object, meas: object) -> _FakeResult:
            return _FakeResult(float(target[0]))  # torso目標のXをそのままタグに使う

    events: list[str] = []
    place = make_basket_deposit_fn(
        send_arm=lambda arm14, secs: events.append(f"send:{arm14[0]}"),
        open_gripper=lambda q, secs: events.append("open"),
        get_measured=lambda: _measured(),
        entry_torso=[1.0, 0.0, 0.0],
        drop_torso=[2.0, 0.0, 0.0],
        retreat_torso=[3.0, 0.0, 0.0],
        ik=_FakeIk(),
        sleep_fn=lambda _s: None,
    )
    assert place() is True
    assert events == ["send:1.0", "send:2.0", "open", "send:3.0"]
