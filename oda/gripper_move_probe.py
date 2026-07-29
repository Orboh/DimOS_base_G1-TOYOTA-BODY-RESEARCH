#!/usr/bin/env python3
"""Dex1 グリッパのモーター生死判定(2026-07-20, 動作テスト版).

サービス(main.cpp)は指令受信時に mode を FOC へ強制上書きし、
状態は q/dq/tau_est しか publish しない(mode/temp は未設定=常に0)。
よって「現在位置保持」ではモーターの生死を判定できない。

このスクリプトは **開き方向(q増加)へ実際に動かす指令**を送り、q が
追従して動くかを見る。開き方向はアタッチメント限界から離れる向きなので安全。

実行(あなたのターミナルで。念のため e-stop を手元に):
  cd ~/workspace/DimOS_base_G1-TOYOTA-BODY-
  CYCLONEDDS_HOME=~/cyclonedds-noshm LD_LIBRARY_PATH=~/cyclonedds-noshm/lib \
  .venv/bin/python oda/gripper_move_probe.py

判定:
  q が目標へ向かって動く/tau が出る → モーターは生きている(復帰可能)。
  q が全く動かず tau≈0 のまま      → モーターが指令を実行しない(故障濃厚)。
"""
import time
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

NIC = "enp2s0"
PREFIX = "rt/dex1/left"
KP = 20.0          # 意味のある剛性(農場実績値と同じ)
OPEN_DELTA = 0.8   # 現在位置から開き方向へ+0.8rad(アタッチメント限界から離れる)

ChannelFactoryInitialize(0, NIC)
st = {}
sub = ChannelSubscriber(f"{PREFIX}/state", MotorStates_)
sub.Init(lambda m: st.update(q=m.states[0].q, dq=m.states[0].dq, tau=m.states[0].tau_est), 10)
t0 = time.time()
while "q" not in st and time.time() - t0 < 5:
    time.sleep(0.05)
if "q" not in st:
    raise SystemExit("NO STATE — サービス/配線を確認")

q0 = st["q"]
target = q0 + OPEN_DELTA
print(f"start:  q={q0:.4f} tau={st['tau']:.4f}  ->  target q={target:.4f} (開き方向, kp={KP})")

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
end = time.time() + 3.0
while time.time() < end:
    pub.Write(cmd)
    q = st["q"]; qmin = min(qmin, q); qmax = max(qmax, q)
    taumax = max(taumax, abs(st["tau"]))
    time.sleep(0.005)
time.sleep(0.2)
moved = qmax - qmin
print(f"end:    q={st['q']:.4f} tau={st['tau']:.4f}")
print(f"observed: q移動量={moved:.4f} rad, |tau|最大={taumax:.4f}")

# 送信を止める前に、そっと元の位置へ戻す指令(急に離すと脱力するだけなので安全)
cmd.cmds[0].q = float(q0)
for _ in range(100):
    pub.Write(cmd); time.sleep(0.005)
pub.Close()

if moved > 0.05 or taumax > 0.1:
    print(">>> 動いた/力が出た = モーターは生きている。把持復帰の見込みあり。")
else:
    print(">>> 全く動かず力も出ない = モーターが指令を実行しない(故障濃厚)。")
