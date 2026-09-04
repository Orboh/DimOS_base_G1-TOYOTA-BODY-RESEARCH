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

"""M2 入口検証 phase B（.venv）: A配置の机上オクラを右腕 IK で掴めるか（F-01→F-04）。

phase A（`_dump_torso_m2.py`, isaac-sim env）が出した各オクラの **torso_link 相対座標** JSON を
読み、`IkApproachSkill.solve`（pinocchio + g1.urdf）で収束するか判定する。「右腕で実際に届く＝
IK が non-None」が到達可否の正本（reach box は placeholder, [[12-検証計画-sim]] §4）。

実行（2段）:
  M2_OUT=/tmp/m2_okra_torso.json
  PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES M2_OUT=$M2_OUT \
    ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/_dump_torso_m2.py   # phase A
  M2_OUT=$M2_OUT .venv/bin/python docs/sim-setup/verify_m2_reach_ik.py        # phase B
"""

from __future__ import annotations

import json
import os
import sys

REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
sys.path.insert(0, REPO)

import numpy as np

from dimos.robot.unitree.g1.harvest.ik_approach import IkApproachSkill

INP = os.environ.get("M2_OUT", "/tmp/m2_okra_torso.json")


def main() -> int:
    if not os.path.exists(INP):
        print(f"[m2] phase A 出力が無い: {INP}（先に _dump_torso_m2.py を実行）")
        return 2
    data = json.load(open(INP))
    skill = IkApproachSkill()  # ws_x[0.05,0.65] ws_y[-0.75,0.20] ws_z[-0.35,0.85], standoff 0.05
    rest = [0.0] * 29

    print(f"[m2] torso_world={[round(v, 3) for v in data['torso_world']]} lift={data['lift']:.3f}")
    print("[m2] 各オクラ: torso相対(X前,Y左,Z上) → 右腕 IK 収束?")
    print(f"{'okra':>12} {'torso(X,Y,Z)':>26} {'IK':>5} {'err[m]':>8}")
    solv = 0
    rows: dict[int, list[int]] = {0: [], 1: []}
    for o in data["okra"]:
        t = np.array(o["torso"], dtype=float)
        res = skill.solve(t, rest)
        ok = res is not None
        tag = f"r{o['row']}c{o['col']}" + ("(右手前)" if o["row"] == 0 else "(左奥)")
        err = f"{res.err:.4f}" if ok else "-"
        print(f"{tag:>12} ({t[0]:6.3f},{t[1]:6.3f},{t[2]:6.3f}) {'OK' if ok else 'NG':>5} {err:>8}")
        if ok:
            solv += 1
            rows[o["row"]].append(o["col"])
    print(
        f"\n[m2] IK 解ける本数 = {solv}/10   手前右 {len(rows[0])}/5 cols={sorted(rows[0])} / 左奥 {len(rows[1])}/5 cols={sorted(rows[1])}"
    )
    print(
        "[m2]  （左奥で解けない分は base 左移動=reposition の対象。手前右が掴めれば M2 の detect→IK は成立）"
    )
    print(
        f"[m2] RESULT: {'PASS ✅（最低1本 IK 可）' if solv >= 1 else 'FAIL ❌（0本＝配置/フレーム要確認）'}"
    )
    return 0 if solv >= 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
