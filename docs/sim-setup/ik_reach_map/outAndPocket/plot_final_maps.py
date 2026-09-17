#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ポケットを付けた状態での最終的な到達性マップ（ポケット無し版と同じ3枚組）。

段階1（reach_map.csv）と段階2（isaac_verify_with_pocket.csv）を突き合わせ、各点を5状態に分類する:
  reachable      … 最後まで通った
  hits pocket    … 胸ポケットの空間に腕が入る
  model/contact  … 自己干渉、または指令姿勢が実機モデルで再現できない（指先誤差 >= 5mm）
  IK fails       … そもそも IK が解けない
  outside box    … ワークスペース箱の外で IK を評価していない
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
ap = argparse.ArgumentParser()
ap.add_argument("--stage1", default=os.path.join(os.path.dirname(HERE), "reach_map.csv"))
ap.add_argument("--stage2", default=os.path.join(HERE, "isaac_verify_with_pocket.csv"))
ap.add_argument("--out", default=HERE)
a = ap.parse_args()

BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
C_S, C_INK, C_INK2, C_GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dddbd4"
# 状態の色。記号も必ず併記して色だけに頼らない
ST = {
    "ok":     ("#008300", "o", "reachable"),
    "pocket": ("#eda100", "P", "hits pocket"),
    "model":  ("#eb6834", "X", "model / self-contact"),
    "ik":     ("#e34948", "x", "IK fails"),
    "box":    ("#b9b7b0", ".", "outside workspace box"),
    # 半径モードでは、その方向にその距離の点が存在しないセルがある（光線が球と交わらない）
    "none":   ("#efeeea", "_", "no point at this distance"),
}
ORDER = ["ok", "pocket", "model", "ik", "box", "none"]

s1 = list(csv.DictReader(open(a.stage1)))
s2 = {(r["depth"], r["row"], r["col"]): r for r in csv.DictReader(open(a.stage2))}
depths = sorted({float(r["depth"]) for r in s1})
LABEL = os.environ.get("DIST_LABEL", "from the camera, along the optical axis")
N = max(int(r["row"]) for r in s1) + 1
W, H, nd = 640, 360, len(depths)


def classify(r):
    if r["in_ws_box"] != "1":
        return "box"
    if r["reach_ok"] != "1":
        return "ik"
    v = s2.get((r["depth"], r["row"], r["col"]))
    if v is None:
        return "model"
    if v.get("pocket_hit", "0") == "1":
        return "pocket"
    if v["self_collision"] == "1" or float(v["isaac_vs_target_m"]) >= 0.005:
        return "model"
    return "ok"


state = {}
score = np.zeros((N, N), int)
for r in s1:
    st = classify(r)
    state[(float(r["depth"]), int(r["row"]), int(r["col"]))] = st
    if st == "ok":
        score[int(r["row"]), int(r["col"])] += 1

cnt = {k: sum(1 for v in state.values() if v == k) for k in ORDER}
cnt["none"] = nd * N * N - len(state)
print("状態の内訳（全%d点）" % (nd * N * N))
for k in ORDER:
    print(f"  {ST[k][2]:<24} {cnt[k]:4d}")


def cellgrid(ax):
    for c in range(N + 1):
        ax.axvline(c * W / N, color=C_S, lw=1.5)
    for r in range(N + 1):
        ax.axhline(r * H / N, color=C_S, lw=1.5)


def style(ax):
    ax.set_facecolor(C_S)
    ax.tick_params(colors=C_INK2, labelsize=8.5)
    for sp in ax.spines.values():
        sp.set_color(C_GRID)


# ---- 図1: 到達スコア ---------------------------------------------------------
steps = [BLUE[int(round(i * (len(BLUE) - 1) / nd))] for i in range(nd + 1)]
cmap = ListedColormap(steps)
norm = BoundaryNorm(np.arange(-0.5, nd + 1.5, 1.0), cmap.N)
fig, ax = plt.subplots(figsize=(9.2, 5.6), facecolor=C_S)
style(ax)
im = ax.imshow(score, cmap=cmap, norm=norm, origin="upper",
               extent=[0, W, H, 0], interpolation="nearest", aspect="auto")
for r in range(N):
    for c in range(N):
        v = int(score[r, c])
        ax.text((c + .5) * W / N, (r + .5) * H / N, str(v) if v else "x",
                ha="center", va="center", fontsize=9,
                color=("#ffffff" if v > nd * .55 else (ST["ik"][0] if v == 0 else C_INK2)),
                fontweight=("bold" if v == 0 else "normal"))
cellgrid(ax)
ax.set_title("Reachability with the chest pocket fitted", fontsize=13, color=C_INK, loc="left", pad=26)
ax.text(0, 1.035, f"cell = how many of the {nd} distances ({LABEL}) survive "
                  f"IK + robot model + pocket   |   x = none",
        transform=ax.transAxes, fontsize=9.5, color=C_INK2)
ax.set_xlabel("image x [px]   (left edge = robot's left / +Y)", fontsize=9.5, color=C_INK2)
ax.set_ylabel("image y [px]   (top = up)", fontsize=9.5, color=C_INK2)
cb = fig.colorbar(im, ax=ax, ticks=range(nd + 1), fraction=0.030, pad=0.02, shrink=0.82)
cb.set_label("distances passing", color=C_INK2, fontsize=9)
cb.ax.tick_params(colors=C_INK2, labelsize=8.5)
cb.outline.set_edgecolor(C_GRID)
fig.tight_layout()
f1 = os.path.join(a.out, "pocket_reach_score.png")
fig.savefig(f1, dpi=150, facecolor=C_S)
plt.close(fig)

# ---- 図2: 距離ごとの内訳 -----------------------------------------------------
NC = min(nd, 6)
NR = int(np.ceil(nd / NC))
fig, axes = plt.subplots(NR, NC, figsize=(2.85 * NC, 3.35 * NR), facecolor=C_S)
axes = np.atleast_1d(axes).ravel()
for ax in axes[nd:]:
    ax.axis("off")
for di, d in enumerate(depths):
    ax = axes[di]
    style(ax)
    img = np.zeros((N, N, 3))
    nok = 0
    for r in range(N):
        for c in range(N):
            st = state.get((d, r, c), "none")
            col = ST[st][0]
            img[r, c] = [int(col[i:i + 2], 16) / 255 for i in (1, 3, 5)]
            if st == "ok":
                nok += 1
    ax.imshow(img, origin="upper", extent=[0, W, H, 0], interpolation="nearest", aspect="auto")
    for r in range(N):
        for c in range(N):
            _st = state.get((d, r, c), "none")
            ax.text((c + .5) * W / N, (r + .5) * H / N, ST[_st][1],
                    ha="center", va="center", fontsize=7,
                    color=(C_INK2 if _st == "none" else "#ffffff"), fontweight="bold")
    cellgrid(ax)
    ax.set_title(f"{d:g} m   —   {nok}/{N*N} reachable", fontsize=10, color=C_INK, loc="left", pad=6)
    ax.set_xticks([]); ax.set_yticks([])
fig.legend(handles=[Line2D([], [], ls="", marker=ST[k][1], ms=8, color=ST[k][0],
                           label=f"{ST[k][2]}  ({cnt[k]})") for k in ORDER],
           loc="lower center", ncol=6, frameon=False, fontsize=9, labelcolor=C_INK2,
           bbox_to_anchor=(0.5, -0.005))
fig.suptitle(f"Why each cell fails, per distance ({LABEL})",
             fontsize=11.5, color=C_INK, x=0.008, ha="left")
fig.subplots_adjust(top=0.86 if NR > 1 else 0.80, bottom=0.10 if NR > 1 else 0.17,
                    left=0.010, right=0.990, hspace=0.28)
f2 = os.path.join(a.out, "pocket_reach_by_distance.png")
fig.savefig(f2, dpi=150, facecolor=C_S)
plt.close(fig)

# ---- 図3: 空間分布 -----------------------------------------------------------
P = np.array([[float(r["tx"]), float(r["ty"]), float(r["tz"])] for r in s1])
stl = np.array([state[(float(r["depth"]), int(r["row"]), int(r["col"]))] for r in s1])
fig = plt.figure(figsize=(15.2, 5.8), facecolor=C_S)


def scat(ax, ix, iy, keys, s=20):
    for k in keys:
        m = stl == k
        ax.scatter(P[m, ix], P[m, iy], c=ST[k][0], marker=ST[k][1], s=s,
                   linewidths=0, alpha=0.85)


ax = fig.add_subplot(1, 3, 1)
style(ax)
scat(ax, 0, 1, ["box", "ik", "model", "pocket", "ok"])
ax.scatter(0, 0, marker="+", s=130, c=C_INK, linewidths=1.6, zorder=5)
ax.annotate("torso origin", (0, 0), textcoords="offset points", xytext=(7, -11), fontsize=8.5, color=C_INK2)
ax.axhline(0, color=C_INK2, lw=0.8, ls=":")
ax.set_xlabel("X  forward from torso [m]", fontsize=9.5, color=C_INK2)
ax.set_ylabel("Y  left [m]", fontsize=9.5, color=C_INK2)
ax.set_title("a) Top view (looking down)", fontsize=11, color=C_INK, loc="left")
ax.grid(True, color=C_GRID, lw=0.6); ax.set_axisbelow(True)

ax = fig.add_subplot(1, 3, 2)
style(ax)
scat(ax, 0, 2, ["box", "ik", "model", "pocket", "ok"])
# ポケットの断面を重ねる（真横から見ると長方形）
ox, oz, pd_, prz = 0.0709, 0.0770, 0.080, 0.085
ax.add_patch(plt.Rectangle((ox, oz - prz), pd_, prz, facecolor=ST["pocket"][0],
                           alpha=0.22, edgecolor=ST["pocket"][0], lw=1.4))
ax.annotate("pocket", (ox + pd_ * 0.5, oz - prz * 0.5), fontsize=8.5, color=C_INK2, ha="center")
ax.scatter(0, 0, marker="+", s=130, c=C_INK, linewidths=1.6, zorder=5)
ax.set_xlabel("X  forward from torso [m]", fontsize=9.5, color=C_INK2)
ax.set_ylabel("Z  up [m]", fontsize=9.5, color=C_INK2)
ax.set_title("b) Side view (pocket outline shown)", fontsize=11, color=C_INK, loc="left")
ax.grid(True, color=C_GRID, lw=0.6); ax.set_axisbelow(True)

ax = fig.add_subplot(1, 3, 3, projection="3d")
ax.set_facecolor(C_S)
for k in ["box", "ik", "model", "pocket", "ok"]:
    m = stl == k
    ax.scatter(P[m, 0], P[m, 1], P[m, 2], c=ST[k][0], marker=ST[k][1], s=16,
               linewidths=0, alpha=0.8)
ax.set_xlabel("X forward [m]", fontsize=8.5, color=C_INK2, labelpad=-2)
ax.set_ylabel("Y left [m]", fontsize=8.5, color=C_INK2, labelpad=-2)
ax.set_zlabel("Z up [m]", fontsize=8.5, color=C_INK2, labelpad=-4)
ax.tick_params(colors=C_INK2, labelsize=7.5, pad=0)
ax.view_init(elev=20, azim=-62)
for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
    pane.pane.set_facecolor(C_S); pane.pane.set_edgecolor(C_GRID)
ax.set_title("c) 3D — all 500 sampled points", fontsize=11, color=C_INK, loc="left")

fig.legend(handles=[Line2D([], [], ls="", marker=ST[k][1], ms=8, color=ST[k][0],
                           label=f"{ST[k][2]}  ({cnt[k]})") for k in ORDER],
           loc="lower center", ncol=6, frameon=False, fontsize=9.5, labelcolor=C_INK2,
           bbox_to_anchor=(0.5, -0.005))
fig.suptitle("Where the right hand can and cannot reach, with the chest pocket fitted",
             fontsize=12.5, color=C_INK, x=0.008, ha="left")
fig.text(0.008, 0.075, "Panels a/b project 500 points onto a plane, so marks overlap along the hidden "
                       "axis. Panel c keeps every state separate.", fontsize=8.5, color=C_INK2)
fig.tight_layout(rect=[0, 0.115, 1, 0.94])
f3 = os.path.join(a.out, "pocket_reach_space.png")
fig.savefig(f3, dpi=150, facecolor=C_S)
plt.close(fig)

with open(os.path.join(a.out, "pocket_reach_map.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["depth", "row", "col", "tx", "ty", "tz", "state", "final_ok"])
    for r in s1:
        k = (float(r["depth"]), int(r["row"]), int(r["col"]))
        w.writerow([r["depth"], r["row"], r["col"], r["tx"], r["ty"], r["tz"],
                    ST[state[k]][2], int(state[k] == "ok")])
print(f"\n出力:\n  {f1}\n  {f2}\n  {f3}\n  {os.path.join(a.out, 'pocket_reach_map.csv')}")
