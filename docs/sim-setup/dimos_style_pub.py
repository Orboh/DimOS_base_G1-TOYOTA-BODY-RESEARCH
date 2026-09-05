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

"""dimos と同じ DDS 機構（unitree_sdk2py ChannelFactory + ChannelPublisher）で rt/arm_sdk を発行。

dimos の G1ArmSdkConnection と同じ経路（ChannelFactoryInitialize → ChannelPublisher("rt/arm_sdk",
LowCmd_)）を再現し、**tailscale 越し**に sim ブリッジへ届くことを実証する。
unitree_sdk2py は cyclonedds の config を固定（CYCLONEDDS_URI 無視）するので、
``ChannelConfigHasInterface`` テンプレに <Peers>+AllowMulticast=false を注入して unicast 化する。

実行（Jetson, unitree_sdk2py のある venv で）:
  SIM_DDS_IFACE=tailscale0 SIM_DDS_PEERS=100.100.126.61 \
  /home/tbr/workspace_ssd/unitree_mujoco/.venv/bin/python /tmp/dimos_style_pub.py
"""

import os
import time

WEIGHT_IDX = 29
R_SHOULDER_PITCH = 22
R_ELBOW = 25


def main() -> None:
    iface = os.getenv("SIM_DDS_IFACE", "tailscale0")
    peers = [p.strip() for p in os.getenv("SIM_DDS_PEERS", "").split(",") if p.strip()]
    secs = float(os.getenv("PUB_SECS", "12"))

    # dimos と同様に unitree_sdk2py を使うが、config テンプレに Peers を注入
    import unitree_sdk2py.core.channel as ch

    if peers:
        peers_xml = "<Peers>" + "".join(f'<Peer address="{p}"/>' for p in peers) + "</Peers>"
        ch.ChannelConfigHasInterface = (
            '<?xml version="1.0" encoding="UTF-8" ?><CycloneDDS><Domain Id="any">'
            "<General><Interfaces>"
            '<NetworkInterface name="$__IF_NAME__$" priority="default" multicast="false"/>'
            "</Interfaces><AllowMulticast>false</AllowMulticast>"
            "<EnableMulticastLoopback>false</EnableMulticastLoopback></General>"
            "<Discovery><ParticipantIndex>auto</ParticipantIndex>"
            f"<MaxAutoParticipantIndex>32</MaxAutoParticipantIndex>{peers_xml}</Discovery>"
            "</Domain></CycloneDDS>"
        )
        print(f"[dimos-pub] patched ChannelConfigHasInterface with peers={peers}", flush=True)

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as make_lowcmd
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

    # dimos の ensure_channel_factory と同じ呼び出し（domain 0, NIC 指定）
    ChannelFactoryInitialize(0, iface)
    pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
    pub.Init()
    print(f"[dimos-pub] ChannelPublisher rt/arm_sdk up (iface={iface})", flush=True)

    import math

    osc = os.getenv("PUB_OSC", "0") == "1"  # 1=往復スイープ（GUI観察用）, 0=一度上げて保持
    lc = make_lowcmd()
    lc.motor_cmd[WEIGHT_IDX].q = 1.0
    n = int(secs * 50)
    for i in range(n):
        if osc:
            s = (1.0 - math.cos(i / 50.0 * 1.4)) / 2.0  # 0..1 を周期的に
        else:
            s = min(1.0, i / 100.0)
        lc.motor_cmd[R_SHOULDER_PITCH].q = 0.7 * s
        lc.motor_cmd[R_ELBOW].q = -0.9 * s
        pub.Write(lc)
        if i % 50 == 0:
            print(
                f"[dimos-pub] t={i / 50:.1f}s r_shoulder_pitch={lc.motor_cmd[R_SHOULDER_PITCH].q:.3f}",
                flush=True,
            )
        time.sleep(0.02)
    print("[dimos-pub] done", flush=True)


if __name__ == "__main__":
    main()
