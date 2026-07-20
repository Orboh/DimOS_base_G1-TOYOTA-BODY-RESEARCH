#!/usr/bin/env python3
"""Dex1 グリッパのモーター有効化テスト(2026-07-20, このPC用診断).

mode=0 で指令を無視するグリッパに対し、enable ビット(mode=1)を明示して
「現在位置を保持する」指令を送り、モーターが FOC を有効化するか確認する。
目標=現在位置なので爪は動かない(安全)。

実行:
  cd ~/workspace/DimOS_base_G1-TOYOTA-BODY-
  CYCLONEDDS_HOME=~/cyclonedds-noshm LD_LIBRARY_PATH=~/cyclonedds-noshm/lib \
  .venv/bin/python oda/gripper_enable_probe.py

判定:
  after で mode が 1 になり temperature が非0になれば「復帰可能」。
  mode=0/temp=0 のままなら「モーター基板の故障(ハード交換相当)」。
"""
import time
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

NIC = "enp2s0"
PREFIX = "rt/dex1/left"   # この機体は右手首Dex1が left トピックに出る

ChannelFactoryInitialize(0, NIC)
st = {}
def on_state(m):
    s = m.states[0]
    st.update(q=s.q, tau=s.tau_est, mode=s.mode, temp=s.temperature)
sub = ChannelSubscriber(f"{PREFIX}/state", MotorStates_)
sub.Init(on_state, 10)

t0 = time.time()
while "q" not in st and time.time() - t0 < 5:
    time.sleep(0.05)
if "q" not in st:
    raise SystemExit("NO STATE — サービス/配線を確認")
q_now = st["q"]
print(f"before: q={q_now:.4f} tau={st['tau']:.4f} mode={st['mode']} temp={st['temp']}")

pub = ChannelPublisher(f"{PREFIX}/cmd", MotorCmds_)
pub.Init()
cmd = MotorCmds_()
cmd.cmds = [unitree_go_msg_dds__MotorCmd_()]
cmd.cmds[0].mode = 1          # ★ enable ビットを明示
cmd.cmds[0].q = float(q_now)  # 現在位置を保持(動かない)
cmd.cmds[0].dq = 0.0
cmd.cmds[0].tau = 0.0
cmd.cmds[0].kp = 5.0          # 柔らかい既定値
cmd.cmds[0].kd = 0.05

print("mode=1 + 現在位置保持 を 200Hz で 3秒送信中...")
end = time.time() + 3.0
while time.time() < end:
    pub.Write(cmd)
    time.sleep(0.005)
time.sleep(0.2)
print(f"after:  q={st['q']:.4f} tau={st['tau']:.4f} mode={st['mode']} temp={st['temp']}")
pub.Close()

if st["mode"] == 1:
    print(">>> 復帰した! モーターは生きている。指令に mode=1 を持たせれば把持できる。")
else:
    print(">>> mode=0 のまま。enable ビットでも起きない = モーター基板側の故障が濃厚。")
