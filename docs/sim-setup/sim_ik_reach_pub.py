#!/usr/bin/env python3
"""M2 可視化: オクラ torso 3D → IK → ``rt/arm_sdk`` publish（仮想G1の右腕がオクラへ伸びる）。

.venv で実行（pinocchio + unitree_sdk2py + cyclonedds が在る）。`sim_dds_bridge.py`(SIM_TABLE=1) が
受けて仮想G1の右腕を駆動する。dimos と同一の DDS 発行経路（unitree_sdk2py ChannelPublisher）。

target は **torso_link 相対 [X=前, Y=左, Z=上]**（IkApproachSkill の入力系）。既定=手前右 r0c1
（`verify_m2_reach_ik.py` で IK err≈0 を確認した確実に届く1本）。`_dump_torso_m2.py` の JSON から
別のオクラを選ぶ場合は OKRA_TORSO で渡す。

実行（ローカル loopback）:
  # 1) bridge（別ターミナル/別プロセス, isaac-sim env, GUI）
  #   SIM_TABLE=1 SIM_OKRA=10 SIM_DDS_IFACE=lo SIM_HEADLESS=0 ... sim_dds_bridge.py
  # 2) 送信（.venv）
  SIM_DDS_IFACE=lo .venv/bin/python docs/sim-setup/sim_ik_reach_pub.py
  # tailscale 越し（Jetson→手元PC）なら SIM_DDS_IFACE=tailscale0 SIM_DDS_PEERS=<相手100.x>
"""
import os
import sys
import time

REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
sys.path.insert(0, REPO)
sys.path.insert(0, "/home/kota-ueda/Desktop/unitree_sdk2_python")

import numpy as np

_WEIGHT_IDX = 29   # motor_cmd[kNotUsedJoint0].q = weight
_ARM_START = 15    # 正準 motor_cmd の腕先頭（左7=15-21, 右7=22-28）


def main() -> int:
    iface = os.getenv("SIM_DDS_IFACE", "lo")
    peers = [p.strip() for p in os.getenv("SIM_DDS_PEERS", "").split(",") if p.strip()]
    secs = float(os.getenv("PUB_SECS", "12"))
    ramp_s = float(os.getenv("RAMP_S", "2.5"))
    tor = [float(x) for x in os.getenv("OKRA_TORSO", "0.414,-0.150,-0.066").split(",")]

    # --- IK（pinocchio + g1.urdf）: torso 目標 → 右腕14関節 ---
    from dimos.robot.unitree.g1.harvest.ik_approach import IkApproachSkill

    skill = IkApproachSkill()
    res = skill.solve(np.array(tor, dtype=float), [0.0] * 29)
    if res is None:
        print(f"[ik-pub] IK 解けず target(torso)={tor} → 中止（reach 内のオクラを指定して）", flush=True)
        return 1
    arm14 = [float(x) for x in res.arm14]
    print(f"[ik-pub] IK OK: err={res.err:.4f}m converged={res.converged} target_torso={tor}", flush=True)
    print(f"[ik-pub] 右腕7関節目標={[round(x, 3) for x in arm14[7:]]}", flush=True)

    # --- DDS publish（dimos と同じ unitree_sdk2py 経路）---
    import unitree_sdk2py.core.channel as ch

    if peers:  # tailscale 等の非マルチキャスト経路は unicast Peers 注入
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
        print(f"[ik-pub] unicast peers={peers}", flush=True)

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as make_lowcmd
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

    ChannelFactoryInitialize(0, iface)
    pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
    pub.Init()
    print(f"[ik-pub] rt/arm_sdk up (iface={iface}); ramp {ramp_s}s → hold（total {secs}s）", flush=True)

    lc = make_lowcmd()
    n = int(secs * 50)
    rn = max(1, int(ramp_s * 50))
    for i in range(n):
        s = min(1.0, i / rn)            # 0→1 ランプ（weight と関節を同時に立ち上げ）
        lc.motor_cmd[_WEIGHT_IDX].q = s
        for ji in range(14):
            lc.motor_cmd[_ARM_START + ji].q = arm14[ji] * s  # rest(0) → IK 目標
        pub.Write(lc)
        if i % 50 == 0:
            print(f"[ik-pub] t={i/50:.1f}s weight={s:.2f}", flush=True)
        time.sleep(0.02)
    print("[ik-pub] done（最終 IK 姿勢を保持）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
