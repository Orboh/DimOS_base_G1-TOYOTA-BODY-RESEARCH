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

"""above/front どちらかの粗アプローチで実際にオクラを掴んで持ち上げるデモ（.venv）。

``sim_pick_demo.py``（M3: reach(単発solve)→閉じ→持ち上げ）の reach フェーズを、
``compare_ik_approach.py`` と同じ ``IkApproachSkill.stream_legs``（above_m/front_m の
段階的アプローチ）に差し替えたもの。「front のやり方で実際にオクラを取るところが見たい」
に応えるためのスクリプト — reach は密ストリーミングで見た目通りに実行し、そのまま
閉じ(dex1)→持ち上げ(IK)→保持まで一続きで行う。

bridge 側は ``SIM_GRASP_NEAREST=1`` を推奨（グリッパが閉じた瞬間、手に一番近い
未把持オクラを自動選択して掴む — target のオクラ番号を事前に知らなくてよい）。

実行（bridge起動済み・SIM_TABLE=1 SIM_GRASP_NEAREST=1 前提）:
  MODE=front OKRA_TORSO=0.42,0.12,-0.066 SIM_DDS_IFACE=lo SIM_DDS_PEERS=127.0.0.1 \
    .venv/bin/python docs/sim-setup/pick_demo_approach.py
"""

from __future__ import annotations

import os
import sys
import time

REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
sys.path.insert(0, REPO)
sys.path.insert(0, "/home/kota-ueda/Desktop/unitree_sdk2_python")

import numpy as np

_WEIGHT_IDX = 29
_ARM_START = 15
_Q_CLOSE = 4.4  # Dex1 閉じ（=切断/把持）[rad]（SS-06 Q_CLOSE と同じ）


def main() -> int:
    mode = os.getenv("MODE", "front").strip().lower()
    if mode not in ("above", "front"):
        print(f"[pick] MODE must be 'above' or 'front', got {mode!r}", flush=True)
        return 2

    iface = os.getenv("SIM_DDS_IFACE", "lo")
    peers = [p.strip() for p in os.getenv("SIM_DDS_PEERS", "").split(",") if p.strip()]
    target_torso = [float(v) for v in os.getenv("OKRA_TORSO", "0.42,0.12,-0.066").split(",")]
    above_m = float(os.getenv("ABOVE_M", "0.08"))
    front_m = float(os.getenv("FRONT_M", "0.20"))
    step_m = float(os.getenv("STEP_M", "0.035"))
    cadence_s = float(os.getenv("CADENCE_S", "0.18"))
    ramp_s = float(os.getenv("RAMP_S", "2.0"))
    t_close = float(os.getenv("T_CLOSE", "1.5"))
    t_lift = float(os.getenv("T_LIFT", "3.0"))
    t_hold = float(os.getenv("T_HOLD", "4.0"))

    from dimos.robot.unitree.g1.harvest.ik_approach import IkApproachSkill

    skill = IkApproachSkill()
    q0 = [0.0] * 29

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
        print(f"[pick] unicast peers={peers}", flush=True)

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
    gc.cmds[0].kp = 5.0
    gc.cmds[0].kd = 0.05
    gc.cmds[0].q = 0.0
    print(
        f"[pick] mode={mode} target_torso={target_torso} "
        f"rt/arm_sdk + rt/dex1/right/cmd up (iface={iface!r})",
        flush=True,
    )

    def publish_arm(arm14: list[float], weight: float = 1.0) -> None:
        lc.motor_cmd[_WEIGHT_IDX].q = weight
        for j in range(14):
            lc.motor_cmd[_ARM_START + j].q = arm14[j]
        arm_pub.Write(lc)

    def publish_grip(q: float) -> None:
        gc.cmds[0].q = q
        grip_pub.Write(gc)

    # ① weight ランプ（rest保持のまま arm_sdk 権限を立ち上げる）
    rn = max(1, int(ramp_s * 50))
    for i in range(rn):
        s = (i + 1) / rn
        publish_arm([0.0] * 14, weight=s)
        publish_grip(0.0)
        time.sleep(0.02)
    print(f"[pick] weight ramp done ({ramp_s:.1f}s) -- reach ({mode})", flush=True)

    # ② reach: above/front の段階的アプローチをそのままストリーミング実行（見た目通り）。
    def send_arm(arm14: list[float]) -> None:
        publish_arm(arm14, weight=1.0)
        publish_grip(0.0)  # reach 中はグリッパ開いたまま

    approach_kwargs = {"above_m": above_m} if mode == "above" else {"front_m": front_m}
    res = skill.stream_legs(
        np.array(target_torso),
        q0,
        send_arm=send_arm,
        step_m=step_m,
        cadence_s=cadence_s,
        **approach_kwargs,
    )
    if res is None:
        print(f"[pick] reach failed for target_torso={target_torso} ({mode}); abort", flush=True)
        return 1
    a_reach = list(res.arm14)
    print(f"[pick] reach done: err={res.err:.4f}m converged={res.converged}", flush=True)

    # ③ 持ち上げ目標（sim_pick_demo.py と同じ規約: 手前に-0.05, 上へ+0.18）
    lift_xyz = [target_torso[0] - 0.05, target_torso[1], target_torso[2] + 0.18]
    r_lift = skill.solve(np.array(lift_xyz), q0)
    a_lift = list(r_lift.arm14) if r_lift is not None else a_reach
    print(
        f"[pick] lift target={lift_xyz} "
        f"{'IK err=' + format(r_lift.err, '.4f') if r_lift is not None else '解けず→reach保持'}",
        flush=True,
    )

    def lerp(a: list[float], b: list[float], s: float) -> list[float]:
        return [a[i] + (b[i] - a[i]) * s for i in range(len(a))]

    # ④ close → ⑤ lift → ⑥ hold
    n_close = int(t_close * 50)
    for i in range(n_close):
        gq = _Q_CLOSE * ((i + 1) / n_close)
        publish_arm(a_reach, weight=1.0)
        publish_grip(gq)
        time.sleep(0.02)
    print(f"[pick] CLOSE done ({t_close:.1f}s, q={_Q_CLOSE})", flush=True)

    n_lift = int(t_lift * 50)
    for i in range(n_lift):
        s = (i + 1) / n_lift
        publish_arm(lerp(a_reach, a_lift, s), weight=1.0)
        publish_grip(_Q_CLOSE)
        time.sleep(0.02)
    print(f"[pick] LIFT done ({t_lift:.1f}s)", flush=True)

    n_hold = int(t_hold * 50)
    for _ in range(n_hold):
        publish_arm(a_lift, weight=1.0)
        publish_grip(_Q_CLOSE)
        time.sleep(0.02)
    print(f"[pick] HOLD done ({t_hold:.1f}s) -- 完了（{mode} で reach→掴む→持ち上げ）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
