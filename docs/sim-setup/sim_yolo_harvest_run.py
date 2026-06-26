#!/usr/bin/env python3
"""YOLO 一連プロセス検証: 実検出→LangGraph→IK→把持 を sim で通す（.venv・知覚も本物）。

GT 座標注入を使わず、bridge の ego_view を **実 YOLO-seg で検出** して LangGraph harvest を駆動する。
detect(YOLO)→select→grasp(IK→arm_sdk + dex1, bridge最近傍把持)→verify→record→loop。

前提 bridge（near クリップ修正済・カメラ配信・最近傍把持）:
  SIM_DDS_IFACE=lo SIM_DDS_PEERS=127.0.0.1 SIM_HEADLESS=0 SIM_LOAD_ROOM=1 SIM_TABLE=1 SIM_OKRA=10 \
  SIM_PUB_CAMERA=1 SIM_CAM_MODE=torso SIM_CAM_NEAR=0.03 SIM_CAM_LOOK_WORLD=0.40,0.0,0.78 \
  SIM_GRASP_NEAREST=1 SIM_VIEWPORT_CAM=1 ... sim_dds_bridge.py

実行:
  SIM_DDS_IFACE=lo SIM_DDS_PEERS=127.0.0.1 .venv/bin/python docs/sim-setup/sim_yolo_harvest_run.py
"""
from __future__ import annotations

import os
import sys

REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "docs/sim-setup"))

from dimos.robot.unitree.g1.harvest.blackboard import Box3D, HarvestConfig, initial_state
from dimos.robot.unitree.g1.harvest.graph import build_harvest_graph
from sim_yolo_harvest_skills import SimYoloHarvestSkills


def main() -> int:
    iface = os.getenv("SIM_DDS_IFACE", "lo")
    peers = [p.strip() for p in os.getenv("SIM_DDS_PEERS", "127.0.0.1").split(",") if p.strip()]
    conf = float(os.getenv("YOLO_CONF", "0.25"))
    out = os.getenv("SIM_YOLO_OUT")  # 注釈画像保存（任意）

    skills = SimYoloHarvestSkills(iface=iface, peers=peers, conf=conf, save_dir=out)
    # reach box は torso 系（YOLO の torso 出力と同系）で机の届く範囲を含む。
    cfg = HarvestConfig(
        reach=Box3D(0.28, 0.62, -0.30, 0.30, -0.25, 0.15),
        basket_capacity=99,        # この demo は交換を起こさない（S3 は Tier0 済）
        max_empty_advances=0,      # 全部見えている（探索移動しない）
        ripeness_threshold=0.5,
        max_grasp_retries=1,       # 検出由来の重複/誤りはリトライせず次へ
    )
    app = build_harvest_graph(skills, cfg)
    print("[run] YOLO 検出で LangGraph harvest を sim 駆動（実検出→IK→把持）", flush=True)
    final = app.invoke(initial_state(), {"recursion_limit": 300})
    print(f"\n[run] 完了: picks={final.get('picks')} records={len(final.get('records', []))}", flush=True)
    for line in final.get("log", [])[-12:]:
        print(f"   {line}", flush=True)
    return 0 if final.get("picks", 0) >= 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
