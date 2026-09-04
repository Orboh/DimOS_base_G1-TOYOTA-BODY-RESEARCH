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

"""摩擦把持 → カゴ投入の通し（.venv）。マグネット不使用＝指の摩擦だけで掴み、運び、籠へ落とす。

bridge は SIM_GRASP_FRICTION=1 / SIM_SELF_COLLISION=0 / SIM_GRAVITY=1 で起動しておく。
流れ: reach(IK, okra+Δ補正) → close(指を閉じ＝摩擦把持) → lift(IK) →
      body_center(記録固定角) → drop_pose(籠上空・記録固定角) → open(離す＝重力で籠へ落下)。
※ 摩擦保持のまま大きく動かすので、各相は**ゆっくり**（落下リスクを下げる）。

実行:
  SIM_DDS_IFACE=lo SIM_DDS_PEERS=127.0.0.1 OKRA_TORSO=0.534,-0.16,-0.096 \
  .venv/bin/python docs/sim-setup/friction_pick_place.py
"""

import json
import os
import sys
import time

REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
sys.path.insert(0, REPO)
sys.path.insert(0, "/home/kota-ueda/Desktop/unitree_sdk2_python")

import numpy as np

_WEIGHT_IDX = 29
_ARM_START = 15
_Q_CLOSE = 4.4
_DROP_POSES = os.path.join(REPO, "docs/sim-setup/drop_poses.json")
_CANON14 = [
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]


def _pose14(d_right: dict, d_left: dict) -> list[float]:
    merged = {**d_left, **d_right}
    return [float(merged.get(k, 0.0)) for k in _CANON14]


def main() -> int:
    iface = os.getenv("SIM_DDS_IFACE", "lo")
    peers = [p.strip() for p in os.getenv("SIM_DDS_PEERS", "").split(",") if p.strip()]
    target = [float(x) for x in os.getenv("OKRA_TORSO", "0.534,-0.16,-0.096").split(",")]
    lift = [target[0] - 0.05, target[1], target[2] + 0.18]

    from dimos.robot.unitree.g1.harvest.ik_approach import IkApproachSkill

    ik = IkApproachSkill()
    r_reach = ik.solve(np.array(target), [0.0] * 29)
    if r_reach is None:
        print(f"[pp] reach IK 解けず target={target}", flush=True)
        return 1
    a_reach = list(r_reach.arm14)
    r_lift = ik.solve(np.array(lift), [0.0] * 29)
    a_lift = list(r_lift.arm14) if r_lift else a_reach
    print(
        f"[pp] reach IK err={r_reach.err:.4f} lift {'err=' + format(r_lift.err, '.4f') if r_lift else 'NG'}",
        flush=True,
    )

    dp = json.load(open(_DROP_POSES))
    body14 = _pose14(dp["right_arm_body_center"], dp["left_arm_body_center"])
    drop14 = _pose14(dp["right_arm_drop_pose"], dp["left_basket_pose"])

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

    # PP_VIA_UP>0: 目標の真上（+z[m]）を経由してから降ろす2段リーチ。休め姿勢（手の高さ~0.66m）
    # から直線補間すると机の前板(天板0.72m)に手が衝突する。歩行モード（floating base）では
    # この接触で押されて転倒するため、上空経由で机を越えてから降ろす。0=従来の1段（base-fix互換）。
    via_up = float(os.getenv("PP_VIA_UP", "0"))
    if via_up > 0:
        r_via = ik.solve(np.array([target[0], target[1], target[2] + via_up]), [0.0] * 29)
        if r_via is not None:
            print(f"[pp] pre-reach（上空 +{via_up:.2f}m 経由, err={r_via.err:.4f}）", flush=True)
            ramp(list(r_via.arm14), 2.0, gq=0.0, w=1.0)
    print("[pp] reach（開）", flush=True)
    ramp(a_reach, 2.5, gq=0.0, w=1.0)
    print("[pp] close（指を閉じ＝摩擦把持）", flush=True)
    ramp(a_reach, 1.5, gq=_Q_CLOSE)
    hold(0.8, gq=_Q_CLOSE)
    print("[pp] lift", flush=True)
    ramp(a_lift, 2.0, gq=_Q_CLOSE)
    hold(0.6, gq=_Q_CLOSE)
    print("[pp] → body_center（ゆっくり集約）", flush=True)
    ramp(body14, float(os.getenv("T_BODY", "3.0")), gq=_Q_CLOSE)
    hold(0.4, gq=_Q_CLOSE)
    print("[pp] → drop_pose（籠上空・ゆっくり）", flush=True)
    ramp(drop14, float(os.getenv("T_DROP", "3.0")), gq=_Q_CLOSE)
    hold(0.6, gq=_Q_CLOSE)
    print("[pp] open（離す＝重力で籠へ落下）", flush=True)
    hold(1.5, gq=0.0)
    print("[pp] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
