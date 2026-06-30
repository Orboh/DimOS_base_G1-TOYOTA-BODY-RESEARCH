#!/usr/bin/env python3
"""M3 机上ピック demo（.venv）: reach(IK)→閉じ(dex1)→持ち上げ(IK) を DDS で送る。

`rt/arm_sdk`(LowCmd_) と `rt/dex1/right/cmd`(MotorCmds_) を同時に publish。bridge(SIM_TABLE=1) が
dex1 閉じで対象オクラを手リンクへ FixedJoint 固定（world アンカーを外す＝収穫）し、続く lift の
腕姿勢でオクラが手と一緒に持ち上がる。dimos と同一の DDS 発行経路（unitree_sdk2py）。

target は torso 相対。既定=手前右 r0c1（bridge の SIM_GRASP_OKRA=1 と一致）。

実行（loopback）:
  # bridge: SIM_TABLE=1 SIM_OKRA=10 SIM_DDS_IFACE=lo SIM_DDS_PEERS=127.0.0.1 ... sim_dds_bridge.py
  SIM_DDS_IFACE=lo SIM_DDS_PEERS=127.0.0.1 .venv/bin/python docs/sim-setup/sim_pick_demo.py
"""
import os
import sys
import time

REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
sys.path.insert(0, REPO)
sys.path.insert(0, "/home/kota-ueda/Desktop/unitree_sdk2_python")

import numpy as np

_WEIGHT_IDX = 29
_ARM_START = 15
_Q_CLOSE = 4.4   # Dex1 閉じ（=切断/把持）[rad]


def main() -> int:
    iface = os.getenv("SIM_DDS_IFACE", "lo")
    peers = [p.strip() for p in os.getenv("SIM_DDS_PEERS", "").split(",") if p.strip()]
    reach = [float(x) for x in os.getenv("OKRA_TORSO", "0.414,-0.150,-0.066").split(",")]
    # 持ち上げ目標: 手前に少し引き（-X 0.05）＋上へ（+Z 0.18）
    lift = [reach[0] - 0.05, reach[1], reach[2] + 0.18]

    # --- IK（pinocchio） ---
    from dimos.robot.unitree.g1.harvest.ik_approach import IkApproachSkill

    skill = IkApproachSkill()
    r_reach = skill.solve(np.array(reach), [0.0] * 29)
    if r_reach is None:
        print(f"[pick] reach IK 解けず target={reach} → 中止", flush=True)
        return 1
    a_reach = list(r_reach.arm14)
    r_lift = skill.solve(np.array(lift), [0.0] * 29)
    a_lift = list(r_lift.arm14) if r_lift else a_reach
    print(f"[pick] reach IK err={r_reach.err:.4f}; "
          f"lift {'IK err=' + format(r_lift.err, '.4f') if r_lift else 'IK失敗→reach保持'}", flush=True)

    # --- DDS publishers（arm_sdk + dex1） ---
    import unitree_sdk2py.core.channel as ch

    if peers:
        pj = "".join(f'<Peer address="{p}"/>' for p in peers)
        ch.ChannelConfigHasInterface = (
            '<?xml version="1.0" encoding="UTF-8" ?><CycloneDDS><Domain Id="any">'
            '<General><Interfaces>'
            '<NetworkInterface name="$__IF_NAME__$" priority="default" multicast="false"/>'
            '</Interfaces><AllowMulticast>false</AllowMulticast>'
            '<EnableMulticastLoopback>false</EnableMulticastLoopback></General>'
            '<Discovery><ParticipantIndex>auto</ParticipantIndex>'
            f'<MaxAutoParticipantIndex>32</MaxAutoParticipantIndex><Peers>{pj}</Peers></Discovery>'
            '</Domain></CycloneDDS>'
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
    print(f"[pick] rt/arm_sdk + rt/dex1/right/cmd up (iface={iface}, peers={peers or 'mcast'})", flush=True)

    lc = make_lowcmd()
    gc = MotorCmds_()
    gc.cmds = [unitree_go_msg_dds__MotorCmd_()]
    gc.cmds[0].kp = 5.0
    gc.cmds[0].kd = 0.05

    def lerp(a, b, s):
        return [a[i] + (b[i] - a[i]) * s for i in range(len(a))]

    # フェーズ [s]: reach → 閉じ → 持ち上げ → 保持。env で各相を延長可（摩擦検証で定常を読むため）。
    #   例: 接近のみ長保持 = T_CLOSE=0 T_LIFT=0 T_REACH=2.5 T_HOLD=8
    T_REACH = float(os.getenv("T_REACH", "2.5"))
    T_CLOSE = float(os.getenv("T_CLOSE", "1.5"))
    T_LIFT = float(os.getenv("T_LIFT", "3.0"))
    T_HOLD = float(os.getenv("T_HOLD", "3.0"))
    total = T_REACH + T_CLOSE + T_LIFT + T_HOLD
    n = int(total * 50)
    phase_prev = ""
    for i in range(n):
        t = i / 50.0
        if t < T_REACH:                         # 腕を reach へ（weight も立ち上げ）
            s = t / T_REACH
            arm = [a_reach[j] * s for j in range(14)]
            w, gq, ph = s, 0.0, "reach"
        elif t < T_REACH + T_CLOSE:             # グリッパ閉じ（=把持）
            arm = a_reach
            w, gq, ph = 1.0, _Q_CLOSE * ((t - T_REACH) / T_CLOSE), "close"
        elif t < T_REACH + T_CLOSE + T_LIFT:    # 持ち上げ（オクラ追従）
            s = (t - T_REACH - T_CLOSE) / T_LIFT
            arm = lerp(a_reach, a_lift, s)
            w, gq, ph = 1.0, _Q_CLOSE, "lift"
        else:                                   # 保持
            arm, w, gq, ph = a_lift, 1.0, _Q_CLOSE, "hold"

        lc.motor_cmd[_WEIGHT_IDX].q = w
        for j in range(14):
            lc.motor_cmd[_ARM_START + j].q = arm[j]
        arm_pub.Write(lc)
        gc.cmds[0].q = gq
        grip_pub.Write(gc)
        if ph != phase_prev:
            print(f"[pick] t={t:.1f}s phase={ph}", flush=True)
            phase_prev = ph
        time.sleep(0.02)
    print("[pick] done（lift 姿勢 + 閉じ保持）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
