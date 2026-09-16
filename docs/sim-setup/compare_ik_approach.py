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

"""above vs front アプローチ比較（.venv）: 同一オクラ目標へ ``IkApproachSkill.stream_legs``
を ``above_m`` / ``front_m`` それぞれで実行し、仮想G1を実際に動かしつつ手先(tip)の
実軌跡を torso 座標でログする。

比較対象の2方式（``dimos/robot/unitree/g1/harvest/ik_approach.py`` 参照）:
  - **above**  = LIFT（真上）→TRANSIT（対象の真上へ水平移動）→DESCEND（垂直降下）。
    現在の自動収穫パイプライン（``unitree-g1-okra-harvest-zed``）が使う方式。
  - **front**  = front-approach（対象の手前 front_m へ直線移動）→push（そこから
    +X 方向へ水平に押し込む）。クリック駆動版 ``IkReachBridge`` にのみあった方式
    （2026-07-23, Oda「まっすぐIKでもっていければ最高」）を、2026-09-11に
    ``IkApproachSkill`` へ移植して本スクリプトで比較できるようにした。

``sim_ik_reach_pub.py``（単発 solve+ランプ）や ``friction_reach_hold.py``（単発
reach+保持）と同じ sim-in-the-loop の考え方だが、本スクリプトは ``stream_legs``
の密ストリーミングをそのまま使い、各waypointを ``rt/arm_sdk`` へ逐次 publish しつつ
JSON へロギングする。dimos と同一の DDS 発行経路（unitree_sdk2py）。

.venv で実行（pinocchio + unitree_sdk2py + cyclonedds が在る）。`sim_dds_bridge.py`
(SIM_TABLE=1) が受けて仮想G1の右腕を駆動する。

実行（3段, above→front→比較）:
  # 1) bridge（別ターミナル/別プロセス, isaac-sim env, GUI 推奨）
  SIM_TABLE=1 SIM_OKRA=10 SIM_DDS_IFACE=lo SIM_HEADLESS=0 \
    ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/sim_dds_bridge.py
  # 2) above 実行 → 完了後、腕が rest に戻ってから front を実行（重ならないように）
  MODE=above SIM_DDS_IFACE=lo .venv/bin/python docs/sim-setup/compare_ik_approach.py
  MODE=front SIM_DDS_IFACE=lo .venv/bin/python docs/sim-setup/compare_ik_approach.py
  # 3) 比較・プロット（.venv, DDS/simは不要）
  .venv/bin/python docs/sim-setup/plot_ik_approach_compare.py

env:
  MODE            : "above"（既定）| "front"
  OKRA_TORSO      : 目標のtorso座標 "X,Y,Z"[m]
                    （既定=r0c1, ``verify_m2_reach_ik.py``/``sim_ik_reach_pub.py`` と同じ
                    確実に届く1本）
  ABOVE_M         : above モードの持ち上げ高さ[m]（既定 0.08 = 実運用既定と同じ）
  FRONT_M         : front モードの手前距離[m]（既定 0.20 = DEMO_PLAN実証値と同じ）
  STANDOFF_M      : IkApproachSkill の切断点手前オフセット[m]（既定 0.0 = 実際に
                    OKRA_TORSO の座標まで到達。本番既定は 0.05 = そこからACTが
                    詰める前提。0のままにしないと「オクラに手が届いていない
                    ように見える」— 2026-09-12 ユーザー実機LIVE観察で発覚）
  STEP_M/CADENCE_S: stream_legs と同じ密度パラメータ（既定 0.035m / 0.18s）
  RAMP_S          : 開始前の arm_sdk weight ランプ秒数（既定 2.0）
  HOLD_S          : 完走後、最終姿勢を保持する秒数（既定 3.0、GUIで目視確認用）
  LOG_PATH        : 出力JSON（既定 /tmp/ik_approach_{MODE}.json）
  SIM_DDS_IFACE / SIM_DDS_PEERS: sim_ik_reach_pub.py と同じ（既定 lo=loopback）

⚠️ 2026-09-12 判明: `lo`（ループバック）がマルチキャスト非対応の環境（本機含む、コンテナ/
サンドボックス等でよくある — `ip link show lo` に `MULTICAST` フラグが無いか確認）では、
`SIM_DDS_PEERS` を明示しない限り本スクリプトと `sim_dds_bridge.py` が互いを発見できず、
**エラーは出ないまま arm_sdk が届かない**（bridge 側の `cmds_rx` が 0 のまま）。
必ず本スクリプトと `sim_dds_bridge.py` の**両方**に `SIM_DDS_IFACE=lo SIM_DDS_PEERS=127.0.0.1`
を渡すこと（片方だけでは discovery が非対称になり届かない）。`ip link show lo` に
`MULTICAST` フラグがある環境（実機NIC等）では従来通り `SIM_DDS_PEERS` 無しでもよい。
"""

from __future__ import annotations

import json
import os
import sys
import time

REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
sys.path.insert(0, REPO)
sys.path.insert(0, "/home/kota-ueda/Desktop/unitree_sdk2_python")

import numpy as np

_WEIGHT_IDX = 29  # motor_cmd[kNotUsedJoint0].q = weight
_ARM_START = 15  # 正準 motor_cmd の腕先頭（左7=15-21, 右7=22-28）
# 既定ターゲット: 手前右 r0c1（verify_m2_reach_ik.py / sim_ik_reach_pub.py で
# IK err≈0 を確認済みの確実に届く1本）。
_DEFAULT_OKRA_TORSO = "0.414,-0.150,-0.066"


def main() -> int:
    mode = os.getenv("MODE", "above").strip().lower()
    if mode not in ("above", "front"):
        print(f"[compare-ik] MODE must be 'above' or 'front', got {mode!r}", flush=True)
        return 2

    iface = os.getenv("SIM_DDS_IFACE", "lo")
    peers = [p.strip() for p in os.getenv("SIM_DDS_PEERS", "").split(",") if p.strip()]
    target_torso = [float(v) for v in os.getenv("OKRA_TORSO", _DEFAULT_OKRA_TORSO).split(",")]
    above_m = float(os.getenv("ABOVE_M", "0.08"))
    front_m = float(os.getenv("FRONT_M", "0.20"))
    step_m = float(os.getenv("STEP_M", "0.035"))
    cadence_s = float(os.getenv("CADENCE_S", "0.18"))
    ramp_s = float(os.getenv("RAMP_S", "2.0"))
    hold_s = float(os.getenv("HOLD_S", "3.0"))
    log_path = os.getenv("LOG_PATH", f"/tmp/ik_approach_{mode}.json")
    # ⚠️ 2026-09-12 判明: IkApproachSkill の既定 standoff_m=0.05 は「切断点手前で
    # 止める」本番設計（IK到達後にACTが最後の詰めを行う前提）のための量で、これが
    # 入ったままだと commanded 先端は OKRA_TORSO で指定した座標より常に torso -X
    # 方向へ 0.05m 手前で止まる — ACTを挟まない本比較（軌道の見た目比較のみ）では
    # 「指定したオクラ座標に手が届いていないように見える」原因になる
    # （ユーザー実機LIVE観察で発覚）。本比較には ACT 受け渡しの意味がないため
    # 既定 0.0（実際にオクラの座標まで到達）にした。本番同様に手前で止めた見た目
    # を比較したい場合のみ STANDOFF_M=0.05 等を明示する。
    standoff_m = float(os.getenv("STANDOFF_M", "0.0"))

    # IK（pinocchio + g1.urdf）: 現行の自動収穫パイプラインと同一クラスをそのまま使う。
    from dimos.robot.unitree.g1.harvest.ik_approach import IkApproachSkill

    skill = IkApproachSkill(standoff_m=standoff_m)
    q0 = [0.0] * 29  # rest 姿勢（sim_ik_reach_pub.py / friction_reach_hold.py と同じ前提）

    # DDS publish（dimos と同じ unitree_sdk2py 経路。sim_ik_reach_pub.py と同一パターン）
    import unitree_sdk2py.core.channel as ch

    if peers:  # tailscale 等の非マルチキャスト経路は unicast Peers 注入
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
        print(f"[compare-ik] unicast peers={peers}", flush=True)

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as make_lowcmd
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

    ChannelFactoryInitialize(0, iface)
    pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
    pub.Init()
    lc = make_lowcmd()
    print(f"[compare-ik] mode={mode} target_torso={target_torso} rt/arm_sdk up (iface={iface!r})",
          flush=True)

    # ① weight ランプ: 現在(rest, q=0のまま)を保持しつつ arm_sdk 権限を 0→1 に立ち上げる
    #    （G1ArmSdkConnection の重力補償ランプと同じ考え方 — weight=0からいきなり
    #    目標角へジャンプさせないための儀式。実機/simとも同じ）。
    rn = max(1, int(ramp_s * 50))
    for i in range(rn):
        s = (i + 1) / rn
        lc.motor_cmd[_WEIGHT_IDX].q = s
        for ji in range(14):
            lc.motor_cmd[_ARM_START + ji].q = 0.0
        pub.Write(lc)
        time.sleep(0.02)
    print(f"[compare-ik] weight ramp done ({ramp_s:.1f}s) -- streaming {mode} approach", flush=True)

    # ② stream_legs 本番: waypoint ごとに DDS publish + ログ収集。
    log: list[dict] = []
    t0 = time.monotonic()

    def send_arm(arm14: list[float]) -> None:
        lc.motor_cmd[_WEIGHT_IDX].q = 1.0
        for ji in range(14):
            lc.motor_cmd[_ARM_START + ji].q = arm14[ji]
        pub.Write(lc)

    def on_waypoint(label: str, p_torso: list[float], arm14: list[float]) -> None:
        log.append(
            {"t": time.monotonic() - t0, "leg": label, "torso_xyz": p_torso, "q_right": arm14[7:]}
        )

    approach_kwargs = {"above_m": above_m} if mode == "above" else {"front_m": front_m}
    res = skill.stream_legs(
        np.array(target_torso),
        q0,
        send_arm=send_arm,
        on_waypoint=on_waypoint,
        step_m=step_m,
        cadence_s=cadence_s,
        **approach_kwargs,
    )
    if res is None:
        print(
            f"[compare-ik] {mode}: unreachable/infeasible for target_torso={target_torso} "
            "(ワークスペース外 or IK不収束 or 関節リミット — ログは書き出さず終了)",
            flush=True,
        )
        return 1
    print(
        f"[compare-ik] {mode}: done, {len(log)} waypoints, "
        f"final err={res.err:.4f}m converged={res.converged}",
        flush=True,
    )

    # ③ 完走後、最終姿勢を hold_s 秒保持（GUIで最終姿勢を目視できるように）。
    hn = int(hold_s * 50)
    for _ in range(hn):
        pub.Write(lc)
        time.sleep(0.02)

    with open(log_path, "w") as f:
        json.dump(
            {
                "mode": mode,
                "target_torso": target_torso,
                "above_m": above_m if mode == "above" else 0.0,
                "front_m": front_m if mode == "front" else 0.0,
                "step_m": step_m,
                "cadence_s": cadence_s,
                "waypoints": log,
            },
            f,
            indent=2,
        )
    print(f"[compare-ik] log written -> {log_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
