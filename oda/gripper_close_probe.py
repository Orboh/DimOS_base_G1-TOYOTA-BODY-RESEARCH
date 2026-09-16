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

"""Dex1 グリッパの閉じ方向動作確認（gripper_move_probe.py の対）。

gripper_move_probe.py が「開き方向(q増加)」を確認するのに対し、こちらは
「閉じ方向(q減少)」を確認する（2026-09-02 kobayashi実機ログで作成・確認された
ものの再現。当時のログでは q=2.9881 -> 目標1.4881 へ追従、実測1.4806、
tau最大6.15を確認済み）。

公式 dex1_1_service のキャリブレーション手順（README: 手で固く閉じた状態を
q=0 として記録する）に基づけば、q減少は「全閉」に近づく方向のはずである。
このスクリプトの目的はまさにそれ（q減少=閉じる、q=0付近=全閉）を実機で
再確認すること — 2026-09-14、honban.pyのcut_close_q/BASKET_OPEN_Qの大小関係
がこの理解と矛盾している疑いが生じたため。

安全策（closeはopenよりリスクが高いため必須）:
  - 全閉(q≈0)までは詰めず、既定では現在位置から-1.5rad分だけ動かす
  - 目標が0を下回るなら0でクランプ（万一q0が既に低い位置でも過閉じしない）
  - 動作中、|tau|が閾値を超えたら即座に中断し、元の位置へ戻す
    （想定外の抵抗＝何かを挟んでいる/機構の端に達した、を示すため）

⚠️ 実行前に必ずグリッパの爪の間に指や物を入れないこと。

実行(あなたのターミナルで。念のため e-stop を手元に。honban等のアプリは
起動していない状態で単独実行すること — 同時起動するとDDS上でコマンドが競合する):
  ROBOT_INTERFACE=<有線NIC名> OKRA_DEX1_PREFIX=rt/dex1/right \
  .venv/bin/python oda/gripper_close_probe.py

判定:
  q が目標へ向かって滑らかに動く → 閉じ方向として正常に追従している。
  |tau|が閾値を超えて自動中断    → 途中に想定外の抵抗（何か挟んでいる等）。
"""

import os
import time

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

NIC = os.getenv("ROBOT_INTERFACE", "enp2s0")
PREFIX = os.getenv("OKRA_DEX1_PREFIX", "rt/dex1/left")
KP = 20.0  # gripper_move_probe.py と同じ、農場実績値
CLOSE_DELTA = -1.5  # 現在位置から閉じ方向へ-1.5rad（全閉q≈0までは詰めない安全マージン）
TAU_ABORT = 10.0  # この|tau|を超えたら即中断（想定外の抵抗）

ChannelFactoryInitialize(0, NIC)
st: dict = {}
sub = ChannelSubscriber(f"{PREFIX}/state", MotorStates_)
sub.Init(lambda m: st.update(q=m.states[0].q, dq=m.states[0].dq, tau=m.states[0].tau_est), 10)
t0 = time.time()
while "q" not in st and time.time() - t0 < 5:
    time.sleep(0.05)
if "q" not in st:
    raise SystemExit("NO STATE — サービス/配線を確認")

q0 = st["q"]
target = max(0.0, q0 + CLOSE_DELTA)  # 0未満にはしない（万一q0が既に低くても過閉じしない）
print(f"start:  q={q0:.4f} tau={st['tau']:.4f}  ->  target q={target:.4f} (閉じ方向, kp={KP})")

pub = ChannelPublisher(f"{PREFIX}/cmd", MotorCmds_)
pub.Init()
cmd = MotorCmds_()
cmd.cmds = [unitree_go_msg_dds__MotorCmd_()]
cmd.cmds[0].q = float(target)
cmd.cmds[0].dq = 0.0
cmd.cmds[0].tau = 0.0
cmd.cmds[0].kp = KP
cmd.cmds[0].kd = 0.05

qmin = qmax = q0
taumax = 0.0
aborted = False
end = time.time() + 3.0
while time.time() < end:
    pub.Write(cmd)
    q = st["q"]
    tau = st["tau"]
    qmin = min(qmin, q)
    qmax = max(qmax, q)
    taumax = max(taumax, abs(tau))
    if abs(tau) > TAU_ABORT:
        aborted = True
        print(f"!!! |tau|={abs(tau):.2f} > {TAU_ABORT} — 想定外の抵抗を検知、中断します。")
        break
    time.sleep(0.005)
time.sleep(0.2)
moved = qmax - qmin
print(f"end:    q={st['q']:.4f} tau={st['tau']:.4f} aborted={aborted}")
print(f"observed: q移動量={moved:.4f} rad, |tau|最大={taumax:.4f}")

# 送信を止める前に、そっと元の位置へ戻す指令(急に離すと脱力するだけなので安全)
cmd.cmds[0].q = float(q0)
for _ in range(100):
    pub.Write(cmd)
    time.sleep(0.005)
pub.Close()

if aborted:
    print(">>> 中断: 何かを挟んでいる、または機構の端に達した可能性。目視確認すること。")
elif moved > 0.05 or taumax > 0.1:
    print(">>> 動いた/力が出た = 閉じ方向として正常に追従している。")
else:
    print(">>> 全く動かず力も出ない = モーターが指令を実行しない(故障濃厚)。")
