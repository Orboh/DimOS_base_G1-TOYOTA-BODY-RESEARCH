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

"""摩擦把持の接近診断（.venv）: オクラへ IK reach して **ジョー開のまま長く保持**するだけ。

弾かれ（接近時にハンドが押し返される）の有無を、定常状態で読むための最小ドライバ。
閉じ・持ち上げはしない（grip=0 開いたまま）。bridge の dense diag（SIM_LOG_EVERY=0.5）で
hand_z / nearest okra / dist / finger を見て、reach が okra に届くか（弾かれないか）を確認する。

実行（bridge 起動中・摩擦モード）:
  SIM_DDS_IFACE=lo SIM_DDS_PEERS=127.0.0.1 OKRA_TORSO=0.424,-0.16,-0.066 HOLD_S=10 \
  .venv/bin/python docs/sim-setup/friction_reach_hold.py
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


def main() -> int:
    iface = os.getenv("SIM_DDS_IFACE", "lo")
    peers = [p.strip() for p in os.getenv("SIM_DDS_PEERS", "").split(",") if p.strip()]
    target = [float(x) for x in os.getenv("OKRA_TORSO", "0.424,-0.16,-0.066").split(",")]
    hold_s = float(os.getenv("HOLD_S", "10"))
    ramp_s = float(os.getenv("RAMP_S", "3"))

    from dimos.robot.unitree.g1.harvest.ik_approach import IkApproachSkill

    r = IkApproachSkill().solve(np.array(target), [0.0] * 29)
    if r is None:
        print(f"[reach-hold] IK 解けず target={target}", flush=True)
        return 1
    a = list(r.arm14)
    print(f"[reach-hold] reach IK err={r.err:.4f} target={target}", flush=True)

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
    print(
        f"[reach-hold] reach を {ramp_s}s で寄せ、ジョー開のまま {hold_s}s 保持（閉じない）",
        flush=True,
    )

    n = int((ramp_s + hold_s) * 50)
    for i in range(n):
        t = i / 50.0
        s = min(1.0, t / ramp_s) if ramp_s > 0 else 1.0
        arm = [a[j] * s for j in range(14)]
        lc.motor_cmd[_WEIGHT_IDX].q = s
        for j in range(14):
            lc.motor_cmd[_ARM_START + j].q = arm[j]
        arm_pub.Write(lc)
        gc.cmds[0].q = 0.0  # ジョー開のまま
        grip_pub.Write(gc)
        time.sleep(0.02)
    print("[reach-hold] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
