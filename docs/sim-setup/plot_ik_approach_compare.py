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

"""above/front 比較プロット（.venv, DDS/Isaac Sim 不要）。

``compare_ik_approach.py`` が書き出した2本のJSONログ（手先 torso 軌跡）を読み込み、
  1) 3D軌跡を重ねたプロット(PNG)
  2) 経路長・waypoint数・z範囲などの数値比較
を出力する。simを起動していなくても、既に書き出されたログさえあれば実行できる。

実行:
  .venv/bin/python docs/sim-setup/plot_ik_approach_compare.py \
      --above /tmp/ik_approach_above.json --front /tmp/ik_approach_front.json \
      --out /tmp/ik_approach_compare.png
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _path_length(xyz: list[list[float]]) -> float:
    arr = np.asarray(xyz, dtype=float)
    if len(arr) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--above", default="/tmp/ik_approach_above.json")
    ap.add_argument("--front", default="/tmp/ik_approach_front.json")
    ap.add_argument("--out", default="/tmp/ik_approach_compare.png")
    args = ap.parse_args()

    data = {"above": _load(args.above), "front": _load(args.front)}
    xyz = {k: [w["torso_xyz"] for w in d["waypoints"]] for k, d in data.items()}

    print("=== IK approach 比較 (torso frame, X=前 Y=左 Z=上) ===")
    for k, d in data.items():
        pts = xyz[k]
        length = _path_length(pts)
        zs = [p[2] for p in pts]
        legs = sorted({w["leg"] for w in d["waypoints"]})
        print(
            f"[{k:5s}] target={d['target_torso']} above_m={d['above_m']} front_m={d['front_m']} "
            f"waypoints={len(pts)} path_len={length:.3f}m z_range=[{min(zs):.3f},{max(zs):.3f}]m "
            f"legs={legs}"
        )
    # 直線距離（開始点→目標）に対する実経路長の比 — 値が大きいほど「回り道」した軌道。
    for k, d in data.items():
        pts = np.asarray(xyz[k], dtype=float)
        straight = float(np.linalg.norm(pts[-1] - pts[0])) if len(pts) >= 2 else 0.0
        length = _path_length(xyz[k])
        ratio = length / straight if straight > 1e-6 else float("nan")
        print(f"[{k:5s}] straight_dist={straight:.3f}m path/straight_ratio={ratio:.2f}")

    import matplotlib

    matplotlib.use("Agg")  # ヘッドレスでもPNG保存できるように
    import matplotlib.pyplot as plt

    # DejaVu Sans（既定）はCJKグリフを持たず日本語ラベルが文字化けするため、
    # 導入済みのNoto Sans CJKを明示指定する（無ければ既定にフォールバック）。
    for jp_font in ("Noto Sans CJK JP", "IPAGothic", "TakaoGothic"):
        if jp_font in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            matplotlib.rcParams["font.family"] = jp_font
            break

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = {"above": "tab:blue", "front": "tab:orange"}
    for k, pts in xyz.items():
        arr = np.asarray(pts, dtype=float)
        ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], "-o", ms=2, color=colors[k], label=k)
        ax.scatter(*arr[0], color=colors[k], marker="^", s=80)  # start
        ax.scatter(*arr[-1], color=colors[k], marker="*", s=120)  # end
    tgt = data["above"]["target_torso"]
    ax.scatter(*tgt, color="k", marker="x", s=100, label="target")
    ax.set_xlabel("torso X (前) [m]")
    ax.set_ylabel("torso Y (左) [m]")
    ax.set_zlabel("torso Z (上) [m]")
    ax.set_title("IK approach 軌跡比較: above (LIFT→TRANSIT→DESCEND) vs front (front-approach→push)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
