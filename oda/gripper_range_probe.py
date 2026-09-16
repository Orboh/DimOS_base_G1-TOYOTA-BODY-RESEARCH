#!/usr/bin/env python3
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

"""Dex1 グリッパの開閉範囲キャリブレーション補助ツール（DDS直叩き版）。

目的: q(モーター指令値)と実際の刃の開き幅[mm]の対応表を作り、
  - 適切な開き位置(茎が入る+余裕、開きすぎない)
  - 切断位置(閉じ切り、あるいは意図的に隙間を残す位置)
を決める。対話式: 指定した q へ動かす → 実測した刃の開きをメモ → 次へ。

2026-09-15 全面改修: 旧版は G1GripperConnection（アプリ経由の /g1/gripper_target）
前提だったため、キャリブレーションのためだけにアプリ全体を起動する必要があり
煩雑だった。oda/gripper_move_probe.py / gripper_close_probe.py と同じ、
unitree_sdk2py を直接使う単体スクリプトに書き換えた（他のアプリを起動せず
単独で安全に使える）。対話中も位置を保持できるよう、バックグラウンドスレッドで
継続的に目標qをpublishし続ける。

前提:
  - G1電源投入は「刃を完全に閉じた状態で」行っておくこと(ゼロ点が全閉になる。
    Orboh/dex1_1_service の README §3 Calibration 参照)。
  - Dex1-1 は q が小さいほど閉じる・大きいほど開く（2026-09-14 実機確認、
    oda/gripper_move_probe.py / gripper_close_probe.py 参照）。
  - 刃の間に指を入れない。測るときはノギス/定規を刃に軽く当てる。
  - honban等のアプリは起動していない状態で単独実行すること（同時起動すると
    DDS上でコマンドが競合する）。

実行(あなたのターミナルで。念のため e-stop を手元に):
  ROBOT_INTERFACE=<有線NIC名> OKRA_DEX1_PREFIX=rt/dex1/right \
  .venv/bin/python oda/gripper_range_probe.py

操作: q値を入力してEnter(例 0.5 → その位置へ)。実測値を聞かれたらmmで入力
(スキップは空Enter)。'q' で終了し、対応表を表示+ファイル保存。
"""

from __future__ import annotations

import os
import sys
import threading
import time

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

NIC = os.getenv("ROBOT_INTERFACE", "enp46s0")
PREFIX = os.getenv("OKRA_DEX1_PREFIX", "rt/dex1/right")
KP = float(os.getenv("GRIPPER_PROBE_KP", "20.0"))  # gripper_move_probe.py と同じ、農場実績値
SETTLE_S = 1.5  # 目標変更後、実測を読むまでの整定待ち [s]


def main() -> None:
    ChannelFactoryInitialize(0, NIC)
    latest: dict = {}

    def cb(m) -> None:  # type: ignore[no-untyped-def]
        s = m.states[0]
        latest.update(q=s.q, tau=s.tau_est, mode=s.mode)

    sub = ChannelSubscriber(f"{PREFIX}/state", MotorStates_)
    sub.Init(cb, 10)
    t0 = time.time()
    while "q" not in latest and time.time() - t0 < 5:
        time.sleep(0.05)
    if "q" not in latest:
        print(f"ERROR: {PREFIX}/state を受信できない(G1電源/ハンド接続を確認)")
        sys.exit(1)

    pub = ChannelPublisher(f"{PREFIX}/cmd", MotorCmds_)
    pub.Init()
    cmd = MotorCmds_()
    cmd.cmds = [unitree_go_msg_dds__MotorCmd_()]
    cmd.cmds[0].dq = 0.0
    cmd.cmds[0].tau = 0.0
    cmd.cmds[0].kp = KP
    cmd.cmds[0].kd = 0.05

    # 対話中も位置を保持するため、目標qを継続的にpublishし続けるスレッドを立てる
    # （DDS直叩きでは、publishを止めるとモーターが指令を見失う可能性があるため）。
    current_target = [float(latest["q"])]  # 初期値=現在位置（動かない）
    stop_event = threading.Event()

    def _publish_loop() -> None:
        while not stop_event.is_set():
            cmd.cmds[0].q = float(current_target[0])
            pub.Write(cmd)
            time.sleep(0.02)

    pub_thread = threading.Thread(target=_publish_loop, daemon=True)
    pub_thread.start()

    rows: list[tuple[float, float, float, str]] = []
    print(f"開始。現在 q={latest['q']:.3f} tau={latest['tau']:.2f} mode={latest['mode']}")
    print("q値を入力してEnter(例 1.5)。'q'+Enterで終了。")

    try:
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
            current_target[0] = target
            time.sleep(SETTLE_S)
            q, tau = latest["q"], latest["tau"]
            print(
                f"  実測 q={q:.3f} tau={tau:.2f}"
                + ("  ← 突き当たり/噛み合い(目標に届いていない)" if abs(q - target) > 0.15 else "")
            )
            gap = input("  刃の開き実測[mm](空Enterでスキップ) > ").strip()
            rows.append((target, q, tau, gap or "-"))
    finally:
        # 送信を止める前に、そっと元の位置へ戻す指令(急に離すと脱力するだけなので安全)
        current_target[0] = float(latest["q"]) if not rows else rows[0][1]
        time.sleep(0.5)
        stop_event.set()
        pub_thread.join(timeout=1.0)
        pub.Close()

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
