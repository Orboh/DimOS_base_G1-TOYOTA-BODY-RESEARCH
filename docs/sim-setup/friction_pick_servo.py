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

"""立位（非walk）chinou セットアップでの「完璧配置」摩擦把持テスト（.venv）。

歩行ドリフトを排除するため base は動かさず、bridge が出す live `Δ(okra-gap)` を指令へ帰還する
Δサーボ（腕のみ）で指の隙間をオクラ中心へ合わせてから close→lift する。位置ずれを実測で
潰した上で「今の摩擦設定でオクラを保持できるか」だけを見る。

前提 bridge（立位・重力ON・摩擦モード）:
  SIM_LOAD_ROOM=1 SIM_TABLE=1 SIM_OKRA=10 SIM_GRASP_FRICTION=1 SIM_GRAVITY=1 SIM_SELF_COLLISION=0
  SIM_LOG_EVERY=0.5  → ログを BRIDGE_LOG で渡す。

実行（.venv）:
  BRIDGE_LOG=/tmp/bridge_probe.log SIM_DDS_IFACE=lo SIM_DDS_PEERS=127.0.0.1 PICK_IDX=0 \
    .venv/bin/python docs/sim-setup/friction_pick_servo.py
"""

import os
import sys
import time

REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/docs/sim-setup")
sys.path.insert(0, "/home/kota-ueda/Desktop/unitree_sdk2_python")

import numpy as np
from walk_approach_pick import X_CMD_MAX, X_TIP, Z_COMP, BridgeLog

_WEIGHT_IDX = 29
_ARM_START = 15
_Q_CLOSE = 4.4
_CONV = (float(os.getenv("CONV_XY", "0.015")), float(os.getenv("CONV_Z", "0.02")))  # 収束閾値[m]


def main() -> int:
    iface = os.getenv("SIM_DDS_IFACE", "lo")
    peers = [p.strip() for p in os.getenv("SIM_DDS_PEERS", "127.0.0.1").split(",") if p.strip()]
    idx = int(os.getenv("PICK_IDX", "0"))
    log = BridgeLog(os.environ["BRIDGE_LOG"])

    # bridge の gap/Δ 診断が対象オクラを追うよう target file を書く
    with open(os.getenv("SIM_GRASP_TARGET_FILE", "/tmp/sim_grasp_target.txt"), "w") as f:
        f.write(str(idx))

    from dimos.robot.unitree.g1.harvest.ik_approach import IkApproachSkill

    ik = IkApproachSkill()

    import unitree_sdk2py.core.channel as ch

    if peers:
        pj = "".join(f'<Peer address="{p}"/>' for p in peers)
        ch.ChannelConfigHasInterface = (
            '<?xml version="1.0" encoding="UTF-8" ?><CycloneDDS><Domain Id="any">'
            "<General><Interfaces>"
            '<NetworkInterface name="$__IF_NAME__$" priority="default" multicast="false"/>'
            "</Interfaces><AllowMulticast>false</AllowMulticast>"
            "<EnableMulticastLoopback>false</EnableMulticastLoopback></General>"
            "<Discovery><ParticipantIndex>auto</ParticipantIndex>"
            f"<MaxAutoParticipantIndex>32</MaxAutoParticipantIndex><Peers>{pj}</Peers></Discovery>"
            "</Domain></CycloneDDS>"
        )
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
    from unitree_sdk2py.idl.default import (
        unitree_go_msg_dds__MotorCmd_,
        unitree_hg_msg_dds__LowCmd_ as make_lowcmd,
    )
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

    ChannelFactoryInitialize(0, iface)
    arm_pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
    arm_pub.Init()
    grip_pub = ChannelPublisher("rt/dex1/right/cmd", MotorCmds_)
    grip_pub.Init()
    lc = make_lowcmd()
    gc = MotorCmds_()
    gc.cmds = [unitree_go_msg_dds__MotorCmd_()]
    gc.cmds[0].kp, gc.cmds[0].kd = 5.0, 0.05
    cur = [0.0] * 14

    def send(arm14, w, gq):
        lc.motor_cmd[_WEIGHT_IDX].q = w
        for j in range(14):
            lc.motor_cmd[_ARM_START + j].q = arm14[j]
        arm_pub.Write(lc)
        gc.cmds[0].q = gq
        grip_pub.Write(gc)
        cur[:] = list(arm14)

    def ramp(to14, secs, gq, w=1.0):
        frm = list(cur)
        n = max(1, int(secs * 50))
        for i in range(n):
            s = (i + 1) / n
            send([frm[j] + (to14[j] - frm[j]) * s for j in range(14)], w, gq)
            time.sleep(0.02)

    def hold(secs, gq):
        for _ in range(max(1, int(secs * 50))):
            send(cur, 1.0, gq)
            time.sleep(0.02)

    # 事前挙手（机より上・前方へ腕を出してから寄せる）
    r0 = ik.solve(np.array([0.40, -0.16, 0.20]), [0.0] * 29)
    if r0 is not None:
        ramp(list(r0.arm14), 2.0, gq=0.0)

    # 初期リーチ目標（実測式: x+X_TIP cap / z+Z_COMP）
    rest = log.okra_rest()
    if rest is None:
        rest = (0.424, -0.16, -0.066)
        print(f"[servo] okra_rest 読めず → calib 既定 {rest} を使用", flush=True)
    tgt = np.array([min(rest[0] + X_TIP, X_CMD_MAX), rest[1] + 0.013, rest[2] + Z_COMP])
    print(
        f"[servo] reach rest={tuple(round(v, 3) for v in rest)} → cmd={tuple(round(float(v), 3) for v in tgt)}",
        flush=True,
    )
    r_via = ik.solve(np.array([tgt[0], tgt[1], tgt[2] + 0.10]), [0.0] * 29)
    if r_via is not None:
        ramp(list(r_via.arm14), 2.0, gq=0.0)
    r = ik.solve(tgt, [0.0] * 29)
    if r is None:
        print("[servo] reach IK 解けず", flush=True)
        return 1
    ramp(list(r.arm14), 2.0, gq=0.0)
    log.mark()
    hold(1.8, gq=0.0)

    # Δサーボ（腕のみ・base は動かさない＝立位）。完璧配置まで詰める
    conv = False
    for it in range(int(os.getenv("SERVO_ITERS", "6"))):
        d = log.delta_for(idx)
        if d is None:
            print(f"[servo] Δ観測なし（it={it}）", flush=True)
            hold(0.6, gq=0.0)
            continue
        print(
            f"[servo] it{it}: Δ(okra-gap)={tuple(round(v, 3) for v in d)} cmd={tuple(round(float(v), 3) for v in tgt)}",
            flush=True,
        )
        if abs(d[0]) < _CONV[0] and abs(d[1]) < _CONV[0] and abs(d[2]) < _CONV[1]:
            conv = True
            print(f"[servo]   ✅収束（|Δxy|<{_CONV[0]} |Δz|<{_CONV[1]}）＝完璧配置", flush=True)
            break
        tgt = tgt + 0.7 * np.clip(np.array(d), -0.05, 0.05)
        tgt[0] = float(np.clip(tgt[0], 0.30, X_CMD_MAX + 0.001))
        tgt[1] = float(np.clip(tgt[1], -0.30, 0.05))
        # z 下限: オクラは torso z≈-0.066。ここを 0.06 で止めると「オクラの12cm上」で頭打ち＝
        # 隙間が絶対に莢へ届かない（walk harvest skills が使う実績値 -0.08 と同型に下げる）。
        tgt[2] = float(np.clip(tgt[2], float(os.getenv("SERVO_Z_MIN", "-0.10")), 0.18))
        r = ik.solve(tgt, [0.0] * 29)
        if r is None:
            print(f"[servo]   IK 解けず cmd={tgt}", flush=True)
            break
        ramp(list(r.arm14), 1.2, gq=0.0)
        log.mark()
        hold(1.8, gq=0.0)
    if not conv:
        dlast = log.delta_for(idx)
        print(f"[servo] 未収束のまま close へ（Δ={dlast}）＝立位の到達限界の可能性", flush=True)

    # close → lift → 検証
    print("[servo] close（摩擦把持）", flush=True)
    log.mark()
    ramp(cur, 2.0, gq=_Q_CLOSE)  # 緩やかに閉じ（急閉じで剛体オクラを弾かない）
    hold(1.0, gq=_Q_CLOSE)
    lift = np.array([tgt[0] - 0.05, tgt[1], tgt[2] + 0.18])
    r_l = ik.solve(lift, [0.0] * 29)
    if r_l is not None:
        ramp(list(r_l.arm14), 2.0, gq=_Q_CLOSE)
    hold(1.5, gq=_Q_CLOSE)
    okz = log.okra_z_max()
    grasped = okz > 0.82
    print(
        f"[servo] === lift 後 okra_z(max)={okz:.3f} → {'✅ 把持成立（摩擦で保持）' if grasped else '❌ 未把持（滑落）'} ===",
        flush=True,
    )
    # 保持を確認するため close のまま少し維持（GUI 観察用）
    hold(float(os.getenv("HOLD_END", "3.0")), gq=_Q_CLOSE)
    return 0 if grasped else 2


if __name__ == "__main__":
    raise SystemExit(main())
