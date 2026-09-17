#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ポケットのサイズ違い（従来 / 0.9倍）を、ランダム2000点×3シードで比較する図。"""
from __future__ import annotations

import csv
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
C_S, C_INK, C_INK2, C_GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dddbd4"
SER = {"base": "#2a78d6", "p09": "#eb6834"}          # カテゴリカル slot1 / slot2
LAB = {"base": "8.0 x 17.0 x 8.5 cm (as built)", "p09": "7.2 x 15.3 x 7.7 cm (x0.9)"}
MK = {"base": "o", "p09": "s"}
ST_COL = {"reachable": "#008300", "hits pocket": "#eda100",
          "model/contact": "#eb6834", "IK fails": "#e34948", "outside box": "#b9b7b0"}
ORDER = ["reachable", "hits pocket", "model/contact", "IK fails", "outside box"]
SEEDS = [0, 1, 2]


def classify(d):
    s1 = list(csv.DictReader(open(f"{d}/reach_map.csv")))
    s2 = {r["row"]: r for r in csv.DictReader(open(f"{d}/isaac_verify_with_pocket.csv"))}
    out = []
    for r in s1:
        if r["in_ws_box"] != "1":      st = "outside box"
        elif r["reach_ok"] != "1":     st = "IK fails"
        else:
            v = s2.get(r["row"])
            if v is None:                          st = "model/contact"
            elif v.get("pocket_hit", "0") == "1":  st = "hits pocket"
            elif v["self_collision"] == "1" or float(v["isaac_vs_target_m"]) >= 0.005:
                                                   st = "model/contact"
            else:                                  st = "reachable"
        out.append((r, st))
    return out


data = {c: [classify(os.path.join(HERE, f"random_{c}_seed{s}")) for s in SEEDS]
        for c in ["base", "p09"]}
rates = {c: np.array([100 * Counter(s for _, s in t)["reachable"] / len(t) for t in data[c]])
         for c in data}

fig = plt.figure(figsize=(14.6, 5.2), facecolor=C_S)

# a) 試行ごとの成功率
ax = fig.add_subplot(1, 3, 1)
ax.set_facecolor(C_S)
for i, c in enumerate(["base", "p09"]):
    x = np.full(len(SEEDS), i, float) + np.linspace(-0.09, 0.09, len(SEEDS))
    ax.scatter(x, rates[c], s=60, c=SER[c], marker=MK[c], linewidths=0, zorder=3)
    m = rates[c].mean()
    ax.plot([i - 0.22, i + 0.22], [m, m], color=SER[c], lw=2.4, zorder=2)
    ax.text(i + 0.26, m, f"{m:.2f}%", color=C_INK, fontsize=10, va="center")
ax.set_xticks([0, 1])
ax.set_xticklabels(["as built", "x0.9"], fontsize=10, color=C_INK2)
ax.set_xlim(-0.5, 1.6)
ax.set_ylabel("reachable [% of 2000 sampled points]", fontsize=9.5, color=C_INK2)
ax.set_title("a) Success rate per trial", fontsize=11.5, color=C_INK, loc="left", pad=18)
ax.text(0, 1.02, f"3 seeds x 2000 random points each   |   difference "
                 f"{rates['p09'].mean()-rates['base'].mean():+.2f} pt",
        transform=ax.transAxes, fontsize=8.5, color=C_INK2)
ax.grid(True, axis="y", color=C_GRID, lw=0.6); ax.set_axisbelow(True)
ax.tick_params(colors=C_INK2, labelsize=9)
for sp in ax.spines.values():
    sp.set_color(C_GRID)

# b) 状態の内訳（平均）
ax = fig.add_subplot(1, 3, 2)
ax.set_facecolor(C_S)
w = 0.36
for i, c in enumerate(["base", "p09"]):
    means = [np.mean([Counter(s for _, s in t)[k] for t in data[c]]) for k in ORDER]
    bottom = 0
    for k, v in zip(ORDER, means):
        ax.bar(i, v, w, bottom=bottom, color=ST_COL[k], edgecolor=C_S, linewidth=2)
        if v > 120:
            ax.text(i, bottom + v / 2, f"{v:.0f}", ha="center", va="center",
                    fontsize=8.5, color="#ffffff", fontweight="bold")
        bottom += v
ax.set_xticks([0, 1]); ax.set_xticklabels(["as built", "x0.9"], fontsize=10, color=C_INK2)
ax.set_ylabel("points (mean of 3 trials)", fontsize=9.5, color=C_INK2)
ax.set_title("b) Where the 2000 points go", fontsize=11.5, color=C_INK, loc="left", pad=18)
ax.text(0, 1.02, "fewer 'hits pocket', but most move to 'model/contact'",
        transform=ax.transAxes, fontsize=8.5, color=C_INK2)
ax.grid(True, axis="y", color=C_GRID, lw=0.6); ax.set_axisbelow(True)
ax.tick_params(colors=C_INK2, labelsize=9)
for sp in ax.spines.values():
    sp.set_color(C_GRID)

# c) 同じ点群での増減（対応比較）
ax = fig.add_subplot(1, 3, 3)
ax.set_facecolor(C_S)
d_reach = [Counter(s for _, s in data["p09"][i])["reachable"]
           - Counter(s for _, s in data["base"][i])["reachable"] for i in range(len(SEEDS))]
d_pock = [Counter(s for _, s in data["p09"][i])["hits pocket"]
          - Counter(s for _, s in data["base"][i])["hits pocket"] for i in range(len(SEEDS))]
x = np.arange(len(SEEDS))
ax.bar(x - 0.18, d_pock, 0.34, color=ST_COL["hits pocket"])
ax.bar(x + 0.18, d_reach, 0.34, color=ST_COL["reachable"])
_lo, _hi = min(d_pock) * 1.28, max(d_reach) * 1.45
ax.set_ylim(_lo, _hi)
for xi, v in zip(x - 0.18, d_pock):
    ax.text(xi, v - 3, f"{v:+d}", ha="center", va="top", fontsize=9.5, color=C_INK,
            fontweight="bold")
for xi, v in zip(x + 0.18, d_reach):
    ax.text(xi, v + 1.5, f"{v:+d}", ha="center", va="bottom", fontsize=9.5, color=C_INK,
            fontweight="bold")
ax.axhline(0, color=C_INK2, lw=1.0)
ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s in SEEDS], fontsize=10, color=C_INK2)
ax.set_ylabel("change in point count (x0.9 minus as built)", fontsize=9.5, color=C_INK2)
ax.set_title("c) Same points, both pockets", fontsize=11.5, color=C_INK, loc="left", pad=18)
ax.text(0, 1.02, "same 2000 points in both conditions (paired)",
        transform=ax.transAxes, fontsize=8.5, color=C_INK2)
ax.grid(True, axis="y", color=C_GRID, lw=0.6); ax.set_axisbelow(True)
ax.tick_params(colors=C_INK2, labelsize=9)
for sp in ax.spines.values():
    sp.set_color(C_GRID)


fig.legend(handles=[Line2D([], [], ls="", marker="s", ms=9, color=ST_COL[k], label=k)
                    for k in ORDER],
           loc="lower center", ncol=5, frameon=False, fontsize=9, labelcolor=C_INK2,
           bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Does a smaller chest pocket free up reachable space?",
             fontsize=12.5, color=C_INK, x=0.007, ha="left")
fig.subplots_adjust(top=0.80, bottom=0.17, left=0.055, right=0.985, wspace=0.34)
out = os.path.join(HERE, "pocket_size_trials.png")
fig.savefig(out, dpi=150, facecolor=C_S)
print("wrote", out)
for c in ["base", "p09"]:
    print(f"  {c}: {rates[c].mean():.2f}% ± {rates[c].std(ddof=1):.2f}  {np.round(rates[c],2)}")
