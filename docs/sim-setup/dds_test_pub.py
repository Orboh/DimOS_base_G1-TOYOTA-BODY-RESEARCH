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

"""loopback/疎通検証用: rt/arm_sdk に LowCmd_ を発行して sim の右腕を動かすテスト送信機。

dimos の代わりに、右肩pitch(正準idx22) と 右肘(idx25) を 0→目標へスイープし weight=1 で送る。
ブリッジ(sim_dds_bridge.py)がこれを受けて Isaac の腕を動かせば疎通OK。

実行（ブリッジと同じ SIM_DDS_IFACE / SIM_DDS_PEERS を使う）:
  PYTHONPATH=/home/kota-ueda/Desktop/unitree_sdk2_python \
  SIM_DDS_IFACE=wlp0s20f3 python docs/sim-setup/dds_test_pub.py
"""

import os
import sys
import time

sys.path.insert(0, "/home/kota-ueda/Desktop/unitree_sdk2_python")

from cyclonedds.domain import Domain, DomainParticipant
from cyclonedds.pub import DataWriter
from cyclonedds.topic import Topic
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as make_lowcmd
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

WEIGHT_IDX = 29
R_SHOULDER_PITCH = 22  # 正準 G1_29 idx
R_ELBOW = 25


def build_cfg() -> str:
    iface = os.getenv("SIM_DDS_IFACE", "wlp0s20f3")
    peers = [p.strip() for p in os.getenv("SIM_DDS_PEERS", "").split(",") if p.strip()]
    trace = os.getenv("SIM_DDS_TRACE", "")
    tr = (
        f"<Tracing><Verbosity>config</Verbosity><OutputFile>{trace}</OutputFile></Tracing>"
        if trace
        else ""
    )
    if peers:
        peers_xml = "<Peers>" + "".join(f'<Peer address="{p}"/>' for p in peers) + "</Peers>"
        return (
            f'<?xml version="1.0" encoding="UTF-8"?><CycloneDDS xmlns="https://cdds.io/config">'
            f'<Domain id="any"><General>'
            f'<Interfaces><NetworkInterface name="{iface}" priority="default" multicast="false"/></Interfaces>'
            f"<AllowMulticast>false</AllowMulticast><EnableMulticastLoopback>false</EnableMulticastLoopback>"
            f"</General><Discovery>"
            f"<ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>32</MaxAutoParticipantIndex>"
            f"{peers_xml}</Discovery>{tr}</Domain></CycloneDDS>"
        )
    return (
        f'<?xml version="1.0" encoding="UTF-8"?><CycloneDDS xmlns="https://cdds.io/config">'
        f'<Domain id="any"><General><Interfaces>'
        f'<NetworkInterface name="{iface}" priority="default" multicast="true"/>'
        f"</Interfaces></General><Discovery></Discovery>{tr}</Domain></CycloneDDS>"
    )


def main() -> None:
    domain_id = int(os.getenv("SIM_DDS_DOMAIN", "0"))
    secs = float(os.getenv("PUB_SECS", "8"))
    cfg = build_cfg()
    dom = Domain(domain_id, cfg)  # ★参照保持（GC回避）
    dp = DomainParticipant(domain_id)
    assert dom is not None
    t = Topic(dp, "rt/arm_sdk", LowCmd_)
    w = DataWriter(dp, t)
    lc = make_lowcmd()
    lc.motor_cmd[WEIGHT_IDX].q = 1.0  # arm_sdk authority weight = 1

    print(
        f"[pub] publishing rt/arm_sdk @50Hz for {secs}s (iface={os.getenv('SIM_DDS_IFACE', 'wlp0s20f3')})",
        flush=True,
    )
    n = int(secs * 50)
    for i in range(n):
        ramp = min(1.0, i / 100.0)
        lc.motor_cmd[R_SHOULDER_PITCH].q = 0.6 * ramp
        lc.motor_cmd[R_ELBOW].q = -0.8 * ramp
        w.write(lc)
        if i % 50 == 0:
            print(
                f"[pub] t={i / 50:.1f}s r_shoulder_pitch={lc.motor_cmd[R_SHOULDER_PITCH].q:.3f} "
                f"r_elbow={lc.motor_cmd[R_ELBOW].q:.3f}",
                flush=True,
            )
        time.sleep(0.02)
    print("[pub] done", flush=True)


if __name__ == "__main__":
    main()
