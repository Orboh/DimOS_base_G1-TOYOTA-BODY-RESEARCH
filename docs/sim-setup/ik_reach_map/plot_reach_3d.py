#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""reach_map.csv を「到達できた場所 / できなかった場所」の空間分布として描く。

zed_fov_reach_map.py の出力（500点の torso 座標＋可否）を読むだけ。IK は解き直さない。
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
C_GOOD, C_BAD, C_NA = "#008300", "#e34948", "#b9b7b0"
C_SURFACE, C_INK, C_INK2, C_GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dddbd4"

rows = list(csv.DictReader(open(os.path.join(HERE, "reach_map.csv"))))
P = np.array([[float(r["tx"]), float(r["ty"]), float(r["tz"])] for r in rows])
ok = np.array([r["reach_ok"] == "1" for r in rows])
box = np.array([r["in_ws_box"] == "1" for r in rows])
grp = [("reachable", ok, C_GOOD, "o", 22),
       ("out of reach (IK fails)", ~ok & box, C_BAD, "X", 26),
       ("outside workspace box", ~box, C_NA, ".", 12)]

# 基準点は torso_link の原点（定義が一意で誤りようがない）。肩位置は URDF から取らない限り
# 推測値になるので描かない。


def _style(a, xl, yl):
    a.set_facecolor(C_SURFACE)
    a.set_xlabel(xl, fontsize=9.5, color=C_INK2)
    a.set_ylabel(yl, fontsize=9.5, color=C_INK2)
    a.tick_params(colors=C_INK2, labelsize=8.5)
    a.grid(True, color=C_GRID, lw=0.6, alpha=0.9)
    a.set_axisbelow(True)
    for sp in a.spines.values():
        sp.set_color(C_GRID)


fig = plt.figure(figsize=(15.2, 5.8), facecolor=C_SURFACE)

# (a) 上から見る: X=前方, Y=左
ax = fig.add_subplot(1, 3, 1)
_style(ax, "X  forward from torso [m]", "Y  left [m]")
ax.scatter(P[:, 0], P[:, 1], c=C_NA, marker=".", s=10, linewidths=0, alpha=0.55)
ax.scatter(P[ok, 0], P[ok, 1], c=C_GOOD, marker="o", s=20, linewidths=0, alpha=0.85)
ax.scatter(0, 0, marker="+", s=130, c=C_INK, linewidths=1.6, zorder=5)
ax.annotate("torso origin", (0, 0), textcoords="offset points", xytext=(7, -11),
            fontsize=8.5, color=C_INK2)
ax.axhline(0, color=C_INK2, lw=0.8, ls=":")
ax.set_title("a) Top view (looking down)", fontsize=11, color=C_INK, loc="left")

# (b) 横から見る: X=前方, Z=上
ax = fig.add_subplot(1, 3, 2)
_style(ax, "X  forward from torso [m]", "Z  up [m]")
ax.scatter(P[:, 0], P[:, 2], c=C_NA, marker=".", s=10, linewidths=0, alpha=0.55)
ax.scatter(P[ok, 0], P[ok, 2], c=C_GOOD, marker="o", s=20, linewidths=0, alpha=0.85)
ax.scatter(0, 0, marker="+", s=130, c=C_INK, linewidths=1.6, zorder=5)
ax.annotate("torso origin", (0, 0), textcoords="offset points", xytext=(7, -11),
            fontsize=8.5, color=C_INK2)
ax.set_title("b) Side view (from the robot's right)", fontsize=11, color=C_INK, loc="left")

# (c) 3D
ax = fig.add_subplot(1, 3, 3, projection="3d")
ax.set_facecolor(C_SURFACE)
for lab, m, c, mk, s in grp:
    ax.scatter(P[m, 0], P[m, 1], P[m, 2], c=c, marker=mk, s=s, linewidths=0, alpha=0.8, label=lab)
ax.set_xlabel("X forward [m]", fontsize=8.5, color=C_INK2, labelpad=-2)
ax.set_ylabel("Y left [m]", fontsize=8.5, color=C_INK2, labelpad=-2)
ax.set_zlabel("Z up [m]", fontsize=8.5, color=C_INK2, labelpad=-4)
ax.tick_params(colors=C_INK2, labelsize=7.5, pad=0)
ax.view_init(elev=20, azim=-62)
for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
    pane.pane.set_facecolor(C_SURFACE)
    pane.pane.set_edgecolor(C_GRID)
ax.set_title("c) 3D — all 500 sampled points", fontsize=11, color=C_INK, loc="left")

n_ok, n_ik, n_box = int(ok.sum()), int((~ok & box).sum()), int((~box).sum())
handles = [Line2D([], [], ls="", marker=mk, ms=8, color=c,
                  label=f"{lab}  ({n})")
           for (lab, _, c, mk, _), n in zip(grp, [n_ok, n_ik, n_box])]
handles = ([Line2D([], [], ls="", marker="o", ms=7, color=C_GOOD,
                   label=f"reachable  ({n_ok})"),
            Line2D([], [], ls="", marker=".", ms=9, color=C_NA,
                   label=f"all sampled points  ({len(rows)})"),
            Line2D([], [], ls="", marker="X", ms=7, color=C_BAD,
                   label=f"out of reach, IK fails — panel c only  ({n_ik})")])
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           fontsize=9.5, labelcolor=C_INK2, bbox_to_anchor=(0.5, -0.005))
fig.text(0.008, 0.075, "Panels a/b project 500 points onto a plane, so marks overlap along the "
         "hidden axis — they show WHERE reachable points exist, not how many. Panel c keeps all "
         "three states separate.", fontsize=8.5, color=C_INK2)
fig.suptitle("Where the right hand can and cannot reach — 10x10 ZED cells x 5 distances, "
             "plotted in torso coordinates",
             fontsize=12.5, color=C_INK, x=0.008, ha="left")
fig.tight_layout(rect=[0, 0.115, 1, 0.94])
out = os.path.join(HERE, "reach_space.png")
fig.savefig(out, dpi=150, facecolor=C_SURFACE)
print("wrote", out)
print(f"reachable={n_ok}  out-of-reach={n_ik}  outside-box={n_box}  (total {len(rows)})")
