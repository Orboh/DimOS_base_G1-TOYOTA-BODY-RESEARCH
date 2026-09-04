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

"""Dex1 グリッパを開いて解放する(2026-07-20).

開き位置(q=OPEN_Q)へ動かし、HOLD_S 秒だけ開いたまま保持してから終了する。
その間に掴んでいる物(葉など)を抜き取る。腕/ZED不要、グリッパのみ。

実行:
  cd ~/workspace/DimOS_base_G1-TOYOTA-BODY-
  CYCLONEDDS_HOME=~/cyclonedds-noshm LD_LIBRARY_PATH=~/cyclonedds-noshm/lib \
  .venv/bin/python oda/gripper_open.py
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

# 機体側NIC: RTX5050機=enp2s0(既定) / RTX3070ラップトップ=DEX1_NIC=enp46s0 を付けて実行
NIC = os.getenv("DEX1_NIC", "enp2s0")
PREFIX = "rt/dex1/left"
OPEN_Q = float(os.getenv("DEX1_OPEN_Q", "2.5"))  # 開き位置(刃付き全開は3.7が実測安全値)
KP = 8.0
HOLD_S = float(os.getenv("DEX1_HOLD_S", "8.0"))  # 開いたまま保持する秒数(この間に葉を抜く)

ChannelFactoryInitialize(0, NIC)
st = {}
ChannelSubscriber(f"{PREFIX}/state", MotorStates_).Init(
    lambda m: st.update(q=m.states[0].q, tau=m.states[0].tau_est), 10
)
t0 = time.time()
while "q" not in st and time.time() - t0 < 5:
    time.sleep(0.05)
if "q" not in st:
    raise SystemExit("NO STATE — サービス/配線を確認")
print(f"start q={st['q']:.3f}  ->  開きます (q={OPEN_Q}). {HOLD_S:.0f}秒以内に葉を抜いてください")

pub = ChannelPublisher(f"{PREFIX}/cmd", MotorCmds_)
pub.Init()
cmd = MotorCmds_()
cmd.cmds = [unitree_go_msg_dds__MotorCmd_()]
cmd.cmds[0].q = float(OPEN_Q)
cmd.cmds[0].dq = 0.0
cmd.cmds[0].tau = 0.0
cmd.cmds[0].kp = KP
cmd.cmds[0].kd = 0.05

end = time.time() + HOLD_S
last = 0.0
while time.time() < end:
    pub.Write(cmd)
    now = time.time()
    if now - last >= 1.0:
        last = now
        print(f"  開いて保持中: q={st['q']:.3f} (残り{end - now:.0f}s)")
    time.sleep(0.005)
pub.Close()
print(f"done. 最終 q={st['q']:.3f}  (送信停止でグリッパは脱力/ブレーキ)")
