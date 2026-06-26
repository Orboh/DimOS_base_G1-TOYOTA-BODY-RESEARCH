#!/usr/bin/env python3
"""S3 検証: カゴ満杯＝個数判定（basket_capacity=10）で 10 個ごとにカゴ交換されるか（F-13）。

計画書 [[12-検証計画-sim]] §6 S3 / §5 F-13。オクラ3Dモデル・Isaac Sim・実機いずれも不要
（MockHarvestSkills の抽象フィールドで段取りロジックだけを確かめる Tier 0 検証）。

実行:
  .venv/bin/python docs/sim-setup/verify_s3_basket10.py
"""
from __future__ import annotations

import os
import sys

# repo ルートを import パスに通す（このスクリプトは docs/sim-setup/ にある）。
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from dimos.robot.unitree.g1.harvest import announce
from dimos.robot.unitree.g1.harvest.announce import RecordingAnnouncer
from dimos.robot.unitree.g1.harvest.blackboard import HarvestConfig, initial_state
from dimos.robot.unitree.g1.harvest.graph import build_harvest_graph
from dimos.robot.unitree.g1.harvest.skills import FieldOkra, MockHarvestSkills

BASKET_CAP = 10  # ★ユーザ確定: 10 個でカゴ交換
N_OKRA = 11      # 交換(10個目)後も収穫が継続することを見るため容量+1 個置く


def main() -> int:
    cfg = HarvestConfig(basket_capacity=BASKET_CAP)
    # reach box x[0.10,0.50] の内側に N 個を均等配置（全て届く位置。y=0.45,z=0.80）。
    xs = [cfg.reach.x_min + (cfg.reach.x_max - cfg.reach.x_min) * i / (N_OKRA - 1) for i in range(N_OKRA)]
    field = [FieldOkra(f"o{i}", x=round(x, 3), y=0.45, z=0.80, ripeness=0.95) for i, x in enumerate(xs)]
    skills = MockHarvestSkills(field, reach=cfg.reach, fov=cfg.fov)
    voice = RecordingAnnouncer()

    app = build_harvest_graph(skills, cfg, announcer=voice)
    final = app.invoke(initial_state(), {"recursion_limit": 800})

    picks = final.get("picks", 0)
    swaps = skills.basket_swaps
    swap_said = announce.basket_swap() in voice.said
    # 交換時に basket_count が 0 に戻り、その後も採れていること（=継続）も確認。
    final_basket = final.get("basket_count", -1)

    checks = {
        "全 11 個を収穫 (picks==11)": picks == N_OKRA,
        "カゴ交換はちょうど 1 回 (basket_swaps==1)": swaps == 1,
        "「カゴ交換」アナウンスが出た": swap_said,
        "交換後リセットして継続 (終了時 basket_count==1)": final_basket == N_OKRA - BASKET_CAP,
    }
    print(f"[S3] basket_capacity={BASKET_CAP}, 配置オクラ={N_OKRA}")
    print(f"[S3] picks={picks} / basket_swaps={swaps} / 終了時 basket_count={final_basket}")
    for label, ok in checks.items():
        print(f"  [{'OK ' if ok else 'NG '}] {label}")
    passed = all(checks.values())
    print(f"[S3] RESULT: {'PASS ✅' if passed else 'FAIL ❌'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
