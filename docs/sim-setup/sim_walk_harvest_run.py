#!/usr/bin/env python3
"""歩行モードの LangGraph 収穫ランナ（.venv）— 歩く→掴む→カゴ→横移動→次のオクラ。

bridge（SIM_WALK_POLICY=1 + SIM_TABLE=1 + SIM_GRASP_FRICTION=1 + SIM_OKRA_BREAK_N=1.0 +
SIM_WALK_SPAWN_X=-0.30 + SIM_WALK_LOG_TICKS=25 + SIM_LOG_EVERY=1）を起動した上で:

  BRIDGE_LOG=<bridgeログ> SIM_DDS_IFACE=lo SIM_DDS_PEERS=127.0.0.1 \
  SIM_WALK_PICK_IDS=0,1,2 .venv/bin/python docs/sim-setup/sim_walk_harvest_run.py

実 graph.py（LangGraph）が detect→select→grasp→verify→record→loop を回し、
WalkHarvestSkills が「事前挙手→位置合わせ歩行（前後/横）→Δサーボ→摩擦把持→籠投入」を実行する。
オクラは机上の A 配置（横 y に並ぶ）＝横移動収穫のデモになる。
"""
from __future__ import annotations

import os
import sys

REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "docs/sim-setup"))

from dimos.robot.unitree.g1.harvest.blackboard import Box3D, HarvestConfig, initial_state
from dimos.robot.unitree.g1.harvest.graph import build_harvest_graph
from sim_walk_harvest_skills import WalkHarvestSkills


def main() -> int:
    iface = os.getenv("SIM_DDS_IFACE", "lo")
    peers = [p.strip() for p in os.getenv("SIM_DDS_PEERS", "127.0.0.1").split(",") if p.strip()]
    pick_ids = [int(x) for x in os.getenv("SIM_WALK_PICK_IDS", "0,1,2").split(",")]
    skills = WalkHarvestSkills(iface=iface, peers=peers, pick_ids=pick_ids)

    # reach box は広く取る（実際の到達可否は skills 内の位置合わせ歩行が担保する）。
    cfg = HarvestConfig(
        reach=Box3D(0.0, 1.5, -0.6, 0.6, -0.3, 0.3),
        basket_capacity=99,
        max_empty_advances=0,
        ripeness_threshold=0.5,
    )
    app = build_harvest_graph(skills, cfg)
    print(f"[walk-run] LangGraph 歩行収穫 開始（対象 {pick_ids}）", flush=True)
    final = app.invoke(initial_state(), {"recursion_limit": 300})
    print(f"\n[walk-run] 完了: picks={final.get('picks')} records={len(final.get('records', []))}", flush=True)
    for line in final.get("log", []):
        print(f"   {line}", flush=True)
    return 0 if final.get("picks", 0) >= 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
