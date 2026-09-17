#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""正中断面（真横から見た輪郭）で、G1 の体とポケットの位置関係を描く。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "out_halfcyl")
body = np.load(os.path.join(D, "profile_body.npy"))      # (x, z)
pock = np.load(os.path.join(D, "profile_pocket.npy"))
ox, oz, gap, z_lo, z_hi = np.load(os.path.join(D, "profile_meta.npy"))

C_S, C_INK, C_INK2, C_GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#dddbd4"
C_BODY, C_POCK, C_HI = "#9ec5f4", "#eda100", "#e34948"

fig, ax = plt.subplots(figsize=(7.6, 8.2), facecolor=C_S)
ax.set_facecolor(C_S)
ax.scatter(body[:, 0], body[:, 1], s=2, c=C_BODY, linewidths=0, label="G1 body (mid-section)")
ax.scatter(pock[:, 0], pock[:, 1], s=3, c=C_POCK, linewidths=0, label="pocket")

# 背面板が向き合う高さ帯で、体の前面がどこにあるかを1本の線にする
zs = np.linspace(oz + z_lo, oz + z_hi, 60)
front = []
for z in zs:
    m = body[np.abs(body[:, 1] - z) < 0.004]
    front.append(m[:, 0].max() if len(m) else np.nan)
front = np.array(front)
ax.plot(front, zs, color=C_INK2, lw=1.4, ls="--", label="front surface of the body")
ax.plot([ox, ox], [oz + z_lo, oz + z_hi], color=C_HI, lw=2.2, label="pocket back plate (flat)")

ok = ~np.isnan(front)
i = int(np.nanargmin(ox - front))
ax.annotate(f"touches here\n{(ox-front[i])*1000:.0f} mm",
            (front[i], zs[i]), xytext=(front[i] - 0.085, zs[i] + 0.012),
            fontsize=9.5, color=C_INK,
            arrowprops=dict(arrowstyle="->", color=C_INK2, lw=1.2))
j = int(np.nanargmax(ox - front))
ax.annotate(f"{(ox-front[j])*1000:.0f} mm away\n(body is narrower here)",
            (front[j], zs[j]), xytext=(front[j] - 0.115, zs[j] - 0.020),
            fontsize=9.5, color=C_INK,
            arrowprops=dict(arrowstyle="->", color=C_INK2, lw=1.2))
for k in range(0, len(zs), 5):
    if ok[k]:
        ax.plot([front[k], ox], [zs[k], zs[k]], color=C_HI, lw=0.7, alpha=0.45)

ax.set_xlabel("X  forward from torso [m]", fontsize=10, color=C_INK2)
ax.set_ylabel("Z  up [m]", fontsize=10, color=C_INK2)
ax.set_title("Why a flat back plate cannot sit flush on G1",
             fontsize=12.5, color=C_INK, loc="left", pad=18)
ax.text(0, 1.015, "side view through the middle — the body narrows going down, "
                  "so the gap grows toward the bottom",
        transform=ax.transAxes, fontsize=9, color=C_INK2)
ax.set_aspect("equal")
ax.grid(True, color=C_GRID, lw=0.6)
ax.set_axisbelow(True)
ax.tick_params(colors=C_INK2, labelsize=9)
for sp in ax.spines.values():
    sp.set_color(C_GRID)
ax.legend(frameon=False, fontsize=9, labelcolor=C_INK2, loc="upper left")
fig.tight_layout()
out = os.path.join(D, "profile.png")
fig.savefig(out, dpi=150, facecolor=C_S)
print("wrote", out)
print(f"接触 {np.nanmin(ox-front)*1000:.1f} mm / 最も離れる {np.nanmax(ox-front)*1000:.1f} mm")
