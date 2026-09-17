#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""段階1（IK解析）と段階2（Isaac Sim 実機モデル）の突き合わせを図にする。

入力: reach_map.csv（段階1） + isaac_verify.csv（段階2）
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE1 = os.path.dirname(HERE)   # 段階1の出力は1つ上

_ap = argparse.ArgumentParser()
_ap.add_argument("--stage1-csv", default=os.path.join(STAGE1, "reach_map.csv"))
_ap.add_argument("--verify-csv", default=os.path.join(HERE, "isaac_verify.csv"))
_ap.add_argument("--out", default=os.path.join(HERE, "stage1_vs_stage2.png"))
_a = _ap.parse_args()
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
C_GOOD, C_BAD, C_NA = "#008300", "#e34948", "#b9b7b0"
C_SURFACE, C_INK, C_INK2, C_GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dddbd4"
N, W, H = 10, 640, 360

stage1 = list(csv.DictReader(open(_a.stage1_csv)))
stage2 = list(csv.DictReader(open(_a.verify_csv)))
depths = sorted({float(r["depth"]) for r in stage1})
nd = len(depths)

s1 = np.zeros((N, N), int)
for r in stage1:
    if r["reach_ok"] == "1":
        s1[int(r["row"]), int(r["col"])] += 1

s2 = np.zeros((N, N), int)        # 自己干渉＋幾何まで
s3 = np.zeros((N, N), int)        # さらにポケットにも当たらない
sp = np.zeros((N, N), int)        # ポケットに当たった回数
dev, err, pen, col, pk = [], [], [], [], []
for r in stage2:
    d = float(r["isaac_vs_target_m"]) * 1000
    c = r["self_collision"] == "1"
    k = r.get("pocket_hit", "0") == "1"
    if d < 5 and not c:
        s2[int(r["row"]), int(r["col"])] += 1
        if not k:
            s3[int(r["row"]), int(r["col"])] += 1
    if k:
        sp[int(r["row"]), int(r["col"])] += 1
    dev.append(float(r["joint_deviation_rad"]))
    err.append(d)
    pen.append(float(r["penetration_m"]) * 1000)
    col.append(c)
    pk.append(k)
dev, err, pen, col, pk = map(np.array, (dev, err, pen, col, pk))

steps = [BLUE[int(round(i * (len(BLUE) - 1) / nd))] for i in range(nd + 1)]
cmap = ListedColormap(steps)
norm = BoundaryNorm(np.arange(-0.5, nd + 1.5, 1.0), cmap.N)

fig = plt.figure(figsize=(19.5, 5.9), facecolor=C_SURFACE)


def grid_panel(ax, M, title, sub):
    ax.set_facecolor(C_SURFACE)
    im = ax.imshow(M, cmap=cmap, norm=norm, origin="upper",
                   extent=[0, W, H, 0], interpolation="nearest", aspect="auto")
    for r in range(N):
        for c in range(N):
            v = int(M[r, c])
            ax.text((c + .5) * W / N, (r + .5) * H / N, str(v) if v else "x",
                    ha="center", va="center", fontsize=8,
                    color=("#ffffff" if v > nd * .55 else (C_BAD if v == 0 else C_INK2)),
                    fontweight=("bold" if v == 0 else "normal"))
    for c in range(N + 1):
        ax.axvline(c * W / N, color=C_SURFACE, lw=1.4)
    for r in range(N + 1):
        ax.axhline(r * H / N, color=C_SURFACE, lw=1.4)
    ax.set_title(title, fontsize=11, color=C_INK, loc="left", pad=20)
    ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=8.5, color=C_INK2)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(C_GRID)
    return im


ax1 = fig.add_subplot(1, 4, 1)
im = grid_panel(ax1, s1, "a) Stage 1 — IK says reachable",
                "pinocchio + URDF only, no simulator")
ax2 = fig.add_subplot(1, 4, 2)
grid_panel(ax2, s2, "b) + robot model & self-contact",
           "pose achieved (<5 mm) AND no new self-contact")
ax2b = fig.add_subplot(1, 4, 3)
grid_panel(ax2b, s3, "c) + the chest pocket",
           "also clear of the pocket volume")

ax3 = fig.add_subplot(1, 4, 4)
ax3.set_facecolor(C_SURFACE)
for m, c, mk, lab in [(~col & ~pk, C_GOOD, "o", "clean"),
                      (col & ~pk, C_BAD, "X", "self-contact"),
                      (pk, "#eda100", "s", "hits pocket")]:
    ax3.scatter(dev[m], err[m], c=c, marker=mk, s=26, linewidths=0, alpha=0.8, label=lab)
ax3.axhline(5, color=C_INK2, lw=0.9, ls=":")
ax3.text(ax3.get_xlim()[1], 5.6, "5 mm", ha="right", fontsize=8, color=C_INK2)
ax3.set_xlabel("how far the joints were pushed back from the command [rad]",
               fontsize=9, color=C_INK2)
ax3.set_ylabel("fingertip vs target [mm]", fontsize=9, color=C_INK2)
ax3.set_yscale("symlog", linthresh=1)
ax3.set_ylim(bottom=-0.05)   # 誤差は非負。symlog が負側に伸びるのを抑える
ax3.set_title("d) Joint hold vs fingertip error",
              fontsize=11, color=C_INK, loc="left", pad=20)
ax3.text(0, 1.03, f"r = {np.corrcoef(dev, err)[0,1]:.2f}   |   "
                  f"joints held (<0.01 rad): median {np.median(err[dev<0.01]):.1f} mm",
         transform=ax3.transAxes, fontsize=8.5, color=C_INK2)
ax3.grid(True, color=C_GRID, lw=0.6)
ax3.set_axisbelow(True)
ax3.tick_params(colors=C_INK2, labelsize=8.5)
for sp in ax3.spines.values():
    sp.set_color(C_GRID)
ax3.legend(frameon=False, fontsize=9, labelcolor=C_INK2, loc="lower right")

cb = fig.colorbar(im, ax=[ax1, ax2, ax2b], ticks=range(nd + 1), orientation="horizontal",
                  fraction=0.045, pad=0.04, shrink=0.45, aspect=34)
cb.set_label(f"distances passing (of {nd})", color=C_INK2, fontsize=8.5)
cb.ax.tick_params(colors=C_INK2, labelsize=8)
cb.outline.set_edgecolor(C_GRID)

fig.suptitle("Where can the right hand still reach, once the robot model and the chest pocket are taken into account?",
             fontsize=12.5, color=C_INK, x=0.008, ha="left")
fig.text(0.006, 0.015,
         f"IK alone called {int(s1.sum())} cell-distance pairs reachable. {int(s2.sum())} survive the robot model, "
         f"and {int(s3.sum())} are also clear of the pocket — the pocket itself costs only "
         f"{int(s2.sum()) - int(s3.sum())} of them, all in the middle columns where the arm swings across the chest. "
         f"Most detected self-contacts are shallower than 1 mm (convex-hull grazing at the shoulder), so these "
         f"panels are a conservative filter, not a verdict that the arm collides.",
         fontsize=8.5, color=C_INK2)
fig.subplots_adjust(top=0.82, bottom=0.20, left=0.010, right=0.988, wspace=0.18)
out = _a.out
fig.savefig(out, dpi=150, facecolor=C_SURFACE)
print("wrote", out)
print(f"stage1={int(s1.sum())}  +model={int(s2.sum())}  +pocket={int(s3.sum())} "
      f"(pocket costs {int(s2.sum())-int(s3.sum())})")
