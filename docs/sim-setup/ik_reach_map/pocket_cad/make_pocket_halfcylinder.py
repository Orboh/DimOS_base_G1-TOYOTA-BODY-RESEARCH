#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""半円柱のポケットを生成する（正面から見て半円、それを前後に押し出した形）。

前後とも板で塞いだ容器で、開口は上面のみ。

座標の規約は pocket_v6.stl と同じにしてあるので mount_pocket.py にそのまま渡せる:
  X: 0 = 背面（G1 に当たる取付面）、+X が前（押し出し方向）
  Y: 左右（0 が中央）
  Z: 0 = 上端（開口部）、下に向かって負

既定寸法: 左右 170mm（半円の直径） × 深さ 85mm（半円の半径） × 奥行き 80mm、肉厚 3mm。
"""
from __future__ import annotations

import argparse
import math
import os
import struct

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--width", type=float, default=170.0, help="左右の幅＝半円の直径[mm]")
ap.add_argument("--depth", type=float, default=85.0, help="上端から底までの深さ＝半円の半径[mm]")
ap.add_argument("--extrude", type=float, default=80.0, help="前後の奥行き[mm]")
ap.add_argument("--thick", type=float, default=3.0, help="肉厚[mm]")
ap.add_argument("--seg", type=int, default=96, help="半円の分割数")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "pocket_halfcyl.stl"))
a = ap.parse_args()

Ro_y, Ro_z = a.width / 2.0, a.depth      # 外側（左右半径 / 深さ）。等しければ真円の半円
Ri_y, Ri_z = Ro_y - a.thick, Ro_z - a.thick
X0, X1 = 0.0, a.extrude

th = np.linspace(0.0, math.pi, a.seg + 1)          # 0=右上端 → π/2=底 → π=左上端


def arc(ry, rz):
    return np.stack([ry * np.cos(th), -rz * np.sin(th)], axis=1)   # (y, z)


OUT, IN = arc(Ro_y, Ro_z), arc(Ri_y, Ri_z)
tris: list[tuple] = []


def quad(p0, p1, p2, p3):
    """四角形を2枚の三角形に。頂点は反時計回りで渡す。"""
    tris.append((p0, p1, p2))
    tris.append((p0, p2, p3))


for i in range(a.seg):
    oy0, oz0 = OUT[i]; oy1, oz1 = OUT[i + 1]
    iy0, iz0 = IN[i];  iy1, iz1 = IN[i + 1]
    # 外壁（前から見て外を向く）
    quad((X0, oy0, oz0), (X1, oy0, oz0), (X1, oy1, oz1), (X0, oy1, oz1))
    # 内壁（法線を内向きにするため巻き順を逆に）
    quad((X0, iy0, iz0), (X0, iy1, iz1), (X1, iy1, iz1), (X1, iy0, iz0))
    # 前面の縁（外と内をつなぐリング）
    quad((X1, oy0, oz0), (X1, iy0, iz0), (X1, iy1, iz1), (X1, oy1, oz1))
    # 背面板（x=X0 を塞ぐ。G1 に当たる面）
    quad((X0, oy0, oz0), (X0, oy1, oz1), (X0, iy1, iz1), (X0, iy0, iz0))

# 背面の内側を塞ぐ。扇の中心は半円の中心（上端の中点 y=0,z=0）に置く。
# ここを半円の中心以外にすると背面が円錐状に凹み、正面から V 字の谷が見えてしまう。
for i in range(a.seg):
    iy0, iz0 = IN[i]; iy1, iz1 = IN[i + 1]
    tris.append(((X0, iy1, iz1), (X0, 0.0, 0.0), (X0, iy0, iz0)))   # 法線 -X（外向き）

# 前面（X1）も同じように塞ぐ。ここを開けると入れた物が前から落ちる
for i in range(a.seg):
    iy0, iz0 = IN[i]; iy1, iz1 = IN[i + 1]
    tris.append(((X1, iy0, iz0), (X1, 0.0, 0.0), (X1, iy1, iz1)))   # 法線 +X（外向き）

# 上端の縁（開口部のふち。外と内の肉厚ぶん）
for y_sign in (+1, -1):
    oy, iy = y_sign * Ro_y, y_sign * Ri_y
    if y_sign > 0:
        quad((X0, oy, 0.0), (X1, oy, 0.0), (X1, iy, 0.0), (X0, iy, 0.0))
    else:
        quad((X0, oy, 0.0), (X0, iy, 0.0), (X1, iy, 0.0), (X1, oy, 0.0))

T = np.array(tris, dtype=np.float64)
V = T.reshape(-1, 3)
n = len(T)
buf = bytearray(b"\0" * 80 + struct.pack("<I", n))
for t in T:
    nz = np.cross(t[1] - t[0], t[2] - t[0])
    ln = np.linalg.norm(nz)
    nz = nz / ln if ln > 1e-12 else np.array([0.0, 0.0, 1.0])
    buf += struct.pack("<12fH", *nz, *t[0], *t[1], *t[2], 0)
open(a.out, "wb").write(bytes(buf))

lo, hi = V.min(0), V.max(0)
print(f"三角形 {n}")
print(f"bbox  X {lo[0]:.1f}..{hi[0]:.1f}  Y {lo[1]:.1f}..{hi[1]:.1f}  Z {lo[2]:.1f}..{hi[2]:.1f} [mm]")
print(f"外形  奥行き {hi[0]-lo[0]:.1f} × 左右 {hi[1]-lo[1]:.1f} × 高さ {hi[2]-lo[2]:.1f} mm"
      f"  = {(hi[0]-lo[0])/10:.1f} × {(hi[1]-lo[1])/10:.1f} × {(hi[2]-lo[2])/10:.1f} cm")
print(f"肉厚  {a.thick} mm / 内寸 左右 {2*Ri_y:.1f} × 深さ {Ri_z:.1f} mm")
print(f"出力  {a.out}")
