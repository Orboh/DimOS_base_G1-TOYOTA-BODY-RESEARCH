#!/usr/bin/env python3
"""カッター装着グリッパの開閉範囲キャリブレーション補助ツール。

目的: q(モーター指令値)と実際の刃の開き幅[mm]の対応表を作り、
  - 適切な開き位置(茎が入る+余裕、開きすぎない)
  - 切断位置(閉じ切り)
を決める。対話式: 指定した q へ動かす → 実測した刃の開きをメモ → 次へ。

前提:
  - アプリ(unitree-g1-okra-ik-only-grasp-zed 等、G1GripperConnection入り)が起動中
    であること(このツールは正規の /g1/gripper_target 経由で動かすため)。
  - G1電源投入は「刃を完全に閉じた状態で」行っておくこと(ゼロ点が全閉になる)。
  - 刃の間に指を入れない。測るときはノギス/定規を刃に軽く当てる。

Run:
    cd ~/Toyota-auto-body-PoC/DimOS_oda
    CYCLONEDDS_HOME=~/cyclonedds-noshm LD_LIBRARY_PATH=~/cyclonedds-noshm/lib \
    LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1' \
    .venv/bin/python oda/gripper_range_probe.py

操作: q値を入力してEnter(例 0.5 → その位置へ)。実測値を聞かれたらmmで入力
(スキップは空Enter)。'q' で終了し、対応表を表示+ファイル保存。
"""

from __future__ import annotations

import sys
import time

from dimos.core.transport import LCMTransport
from dimos.msgs.sensor_msgs.JointState import JointState
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_

NIC = "enp46s0"
DEX1_STATE_TOPIC = "rt/dex1/left/state"  # この機体の配線都合(右手だが左サービス)


def main() -> None:
    ChannelFactoryInitialize(0, NIC)
    latest: dict = {}

    def cb(m) -> None:  # type: ignore[no-untyped-def]
        s = m.states[0]
        latest.update(q=s.q, tau=s.tau_est, mode=s.mode)

    sub = ChannelSubscriber(DEX1_STATE_TOPIC, MotorStates_)
    sub.Init(cb, 10)
    time.sleep(1.0)
    if "q" not in latest:
        print(f"ERROR: {DEX1_STATE_TOPIC} を受信できない(G1電源/ハンド接続を確認)")
        sys.exit(1)
    if latest.get("mode") == 0:
        print("ERROR: グリッパのモーターが無効状態(mode=0)。")
        print("  → G1電源OFF → ハンドのコネクタ挿し直し → 刃を閉じて電源ON")
        sys.exit(1)

    pub = LCMTransport("/g1/gripper_target", JointState)
    rows: list[tuple[float, float, float, str]] = []
    print(f"開始。現在 q={latest['q']:.3f} tau={latest['tau']:.2f} mode={latest['mode']}")
    print("q値を入力してEnter(例 1.5)。'q'+Enterで終了。")

    while True:
        try:
            s = input("\n目標q > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if s.lower() == "q":
            break
        try:
            target = float(s)
        except ValueError:
            print("  数値か 'q' を入力")
            continue
        for _ in range(3):
            pub.publish(
                JointState(
                    name=["g1/right_gripper"],
                    position=[target],
                    velocity=[0.0],
                    effort=[0.0],
                )
            )
            time.sleep(0.2)
        time.sleep(1.5)  # 整定待ち
        q, tau = latest["q"], latest["tau"]
        print(f"  実測 q={q:.3f} tau={tau:.2f}"
              + ("  ← 突き当たり/噛み合い(目標に届いていない)" if abs(q - target) > 0.15 else ""))
        gap = input("  刃の開き実測[mm](空Enterでスキップ) > ").strip()
        rows.append((target, q, tau, gap or "-"))

    print("\n==== q↔開き幅 対応表 ====")
    print(f"{'目標q':>8} {'実測q':>8} {'tau':>7}  開き[mm]")
    for t, q, tau, gap in rows:
        print(f"{t:8.2f} {q:8.3f} {tau:7.2f}  {gap}")
    if rows:
        path = "oda/gripper_range_result.txt"
        with open(path, "a") as f:
            f.write(f"\n# {time.strftime('%Y-%m-%d %H:%M')} カッター開閉範囲\n")
            for t, q, tau, gap in rows:
                f.write(f"target={t:.2f} q={q:.3f} tau={tau:.2f} gap_mm={gap}\n")
        print(f"→ {path} に追記保存した。")
    print("決めるもの: 開き位置(茎径+余裕) / 切断位置(閉じ切り=tauが立つ点)")


if __name__ == "__main__":
    main()
