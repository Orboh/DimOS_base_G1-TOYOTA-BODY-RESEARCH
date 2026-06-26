#!/usr/bin/env python3
"""M4 本命: 実 LangGraph harvest graph を sim で駆動するランナ（.venv）。

`build_harvest_graph(SimHarvestSkills, cfg)` を `invoke` し、detect→select→grasp(IK+dex1→sim)
→verify→record→loop で **手前右4本のオクラを自動収穫**する。bridge(SIM_TABLE=1) が受けて
仮想G1を動かす。reach box は torso 系で全4本を含むよう設定（reposition 不要）。

実行（bridge を SIM_TABLE=1 SIM_DDS_IFACE=lo SIM_DDS_PEERS=127.0.0.1 で起動した上で）:
  SIM_DDS_IFACE=lo SIM_DDS_PEERS=127.0.0.1 .venv/bin/python docs/sim-setup/sim_harvest_run.py
"""
from __future__ import annotations

import os
import sys

REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "docs/sim-setup"))

from dimos.robot.unitree.g1.harvest.blackboard import Box3D, HarvestConfig, initial_state
from dimos.robot.unitree.g1.harvest.graph import build_harvest_graph
from sim_harvest_skills import SimHarvestSkills

# 手前右4本（IK 可, m2 検証より）。(okra prim index, torso 相対 X前/Y左/Z上)
OKRA_TORSO = [
    (0, (0.344, -0.150, -0.066)),
    (1, (0.414, -0.150, -0.066)),
    (2, (0.484, -0.150, -0.066)),
    (3, (0.554, -0.150, -0.066)),
]


def main() -> int:
    iface = os.getenv("SIM_DDS_IFACE", "lo")
    peers = [p.strip() for p in os.getenv("SIM_DDS_PEERS", "127.0.0.1").split(",") if p.strip()]
    skills = SimHarvestSkills(OKRA_TORSO, iface=iface, peers=peers)

    # reach box は torso 系で 4本を含む（X前 0.30-0.60 / Y左 -0.30..0.10 / Z上 -0.20..0.10）。
    cfg = HarvestConfig(
        reach=Box3D(0.30, 0.60, -0.30, 0.10, -0.20, 0.10),
        basket_capacity=99,           # この demo は4本なので交換は起こさない（S3で別途検証済）
        max_empty_advances=0,         # 視界外探索しない（全部見えている）
        ripeness_threshold=0.5,
    )
    app = build_harvest_graph(skills, cfg)
    print("[run] LangGraph harvest を sim で開始（手前右4本）", flush=True)
    final = app.invoke(initial_state(), {"recursion_limit": 200})
    print(f"\n[run] 完了: picks={final.get('picks')} records={len(final.get('records', []))}", flush=True)
    for line in final.get("log", []):
        print(f"   {line}", flush=True)
    return 0 if final.get("picks", 0) >= 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
