#!/usr/bin/env python3
"""Dex1 グリッパ 単独はさみテスト(2026-07-20, 刃なしアタッチメント付き).

腕/ZED/ビューア不要。グリッパのみ DDS で駆動する(フリーズの心配なし)。
シーケンス: 開く → OPEN_HOLD_S 秒キープ(この間に物を爪の間に置く)→
閉じて握る → GRIP_HOLD_S 秒保持(握力を表示)→ 離して終了。

実行(あなたのターミナルで。e-stop を手元に):
  cd ~/workspace/DimOS_base_G1-TOYOTA-BODY-
  CYCLONEDDS_HOME=~/cyclonedds-noshm LD_LIBRARY_PATH=~/cyclonedds-noshm/lib \
  .venv/bin/python oda/gripper_pinch.py

調整(必要なら上部の定数):
  GRIP_KP を上げると握力↑(まず8で試し、弱ければ15→20へ)
  CLOSE_Q を下げるとより強く閉じ込む(物が薄いとき)。閉じ側定位置≈1.29。
"""
import time
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

NIC = "enp2s0"
PREFIX = "rt/dex1/left"     # この機体は右手首Dex1が left トピック
OPEN_Q = 2.4               # 開き位置(検証済みの安全域。物を受け入れる)
OPEN_HOLD_S = 10.0         # 開いて待つ秒数(この間に物を置く)
CLOSE_Q = 0.9              # 握り目標(物 or 機械限界で止まり、kpで保持)
GRIP_KP = 8.0              # 握力(中程度。弱ければ上げる)
GRIP_KD = 0.05
GRIP_HOLD_S = 6.0          # 握って保持する秒数
RATE = 0.005               # 200Hz

ChannelFactoryInitialize(0, NIC)
st = {}
ChannelSubscriber(f"{PREFIX}/state", MotorStates_).Init(
    lambda m: st.update(q=m.states[0].q, tau=m.states[0].tau_est), 10)
t0 = time.time()
while "q" not in st and time.time() - t0 < 5:
    time.sleep(0.05)
if "q" not in st:
    raise SystemExit("NO STATE — サービス/配線を確認")

pub = ChannelPublisher(f"{PREFIX}/cmd", MotorCmds_); pub.Init()
cmd = MotorCmds_(); cmd.cmds = [unitree_go_msg_dds__MotorCmd_()]
cmd.cmds[0].dq = 0.0; cmd.cmds[0].tau = 0.0; cmd.cmds[0].kd = GRIP_KD

def drive(target_q, kp, seconds, label):
    cmd.cmds[0].q = float(target_q); cmd.cmds[0].kp = float(kp)
    end = time.time() + seconds
    last_print = 0.0
    while time.time() < end:
        pub.Write(cmd)
        now = time.time()
        if now - last_print >= 1.0:
            last_print = now
            print(f"  [{label}] q={st['q']:.3f} tau={st['tau']:.3f}  (残り{end-now:.0f}s)")
        time.sleep(RATE)

print(f"start q={st['q']:.3f}")
print(f">>> 開きます (q->{OPEN_Q}). {OPEN_HOLD_S:.0f}秒以内に爪の間に物を置いてください")
drive(OPEN_Q, 8.0, OPEN_HOLD_S, "OPEN")
print(f">>> 閉じて握ります (q->{CLOSE_Q}, kp={GRIP_KP})")
drive(CLOSE_Q, GRIP_KP, GRIP_HOLD_S, "GRIP")
grip_q, grip_tau = st["q"], st["tau"]
print(f">>> 握り込み結果: q={grip_q:.3f}  tau={grip_tau:.3f}")
print(f">>> 離します (q->{OPEN_Q})")
drive(OPEN_Q, 8.0, 2.0, "RELEASE")
pub.Close()
print("done. (送信停止でグリッパは脱力/ブレーキします)")

if abs(grip_tau) > 0.3:
    print(f"判定: 握力トルク {grip_tau:.2f} を検出 = 物を掴んでいた/機械限界で保持。OK")
else:
    print("判定: トルクほぼ0 = 空振り(物が無い/薄い)か、CLOSE_Qが物の位置より上。CLOSE_Qを下げるか物を確認。")
