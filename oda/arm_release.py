#!/usr/bin/env python3
"""arm_sdk 権限をオンボード制御へ返す(クリーン停止と同じ weight 1->0 ランプ).

g1_arm_sdk_connection.py の start()/stop() の忠実な最小再現:
現在測定姿勢を保持指令しつつ weight(idx29) を 1.0->0.0 に2秒でランプ。
腕は跳ねずに、そのままオンボード制御へ滑らかに引き継がれる。
"""
import time

import numpy as np
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

NIC = "enp46s0"
WEIGHT_IDX = 29
WAIST_IDX = [12, 13, 14]
ARM_IDX = list(range(15, 29))
WRIST_IDX = {19, 20, 21, 26, 27, 28}

ChannelFactoryInitialize(0, NIC)
state = {}
sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(lambda m: state.update(mm=m.mode_machine,
                                q=[m.motor_state[i].q for i in range(29)]), 10)
t0 = time.time()
while "q" not in state and time.time() - t0 < 5:
    time.sleep(0.05)
if "q" not in state:
    raise SystemExit("NO LowState — G1電源/LAN確認")
print(f"LowState OK (mode_machine={state['mm']}); 現在姿勢を保持しつつ weight 1->0 ランプ開始")

pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
pub.Init()
crc = CRC()
cmd = unitree_hg_msg_dds__LowCmd_()
cmd.mode_pr = 0
cmd.mode_machine = state["mm"]
for i in ARM_IDX:
    cmd.motor_cmd[i].mode = 1
    cmd.motor_cmd[i].kp = 40.0 if i in WRIST_IDX else 80.0
    cmd.motor_cmd[i].kd = 1.5 if i in WRIST_IDX else 3.0
    cmd.motor_cmd[i].q = float(state["q"][i])
    cmd.motor_cmd[i].dq = 0.0
    cmd.motor_cmd[i].tau = 0.0
for i in WAIST_IDX:
    cmd.motor_cmd[i].mode = 1
    cmd.motor_cmd[i].kp = 300.0
    cmd.motor_cmd[i].kd = 3.0
    cmd.motor_cmd[i].q = float(state["q"][i])
    cmd.motor_cmd[i].dq = 0.0
    cmd.motor_cmd[i].tau = 0.0

for w in np.linspace(1.0, 0.0, 101):
    cmd.motor_cmd[WEIGHT_IDX].q = float(w)
    cmd.crc = crc.Crc(cmd)
    pub.Write(cmd)
    time.sleep(0.02)
pub.Close()
print("done: weight=0 — 腕はオンボード制御に返却済み(クリーン停止相当)")
