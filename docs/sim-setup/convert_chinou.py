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

"""FARO 知能センター OBJ スキャン → 正しいスケールの USD へ変換。

処理:
  1) 複数 OBJ サブメッシュを読み込み（頂点+面、頂点色RGB）
  2) 結合
  3) mm→m: 頂点に ×0.001 をベイク（metersPerUnit=1.0 のまま＝参照時の単位ズレを回避）
  4) recenter: 鉛直軸(最小スパン軸)の下端を z=0、水平2軸を原点中心へ
  5) decimation（TARGET_TRIS 三角数まで、頂点色を保持）
  6) USD 出力（Z-up, displayColor primvar 付き）

env:
  IN_DIR      OBJ ディレクトリ（*.obj を全て使用）
  OUT_USD     出力 USD パス
  TARGET_TRIS 目標三角数（0=decimationしない）。既定 2,000,000
  SCALE       単位係数（既定 0.001 = mm→m）
  MESH_PATH   出力メッシュ prim パス（既定 /World/ChinouCenter。畑は /World/OkraField 等）
"""

import glob
import os
import time

import numpy as np

IN_DIR = os.environ["IN_DIR"]
OUT_USD = os.environ["OUT_USD"]
TARGET_TRIS = int(os.environ.get("TARGET_TRIS", "2000000"))
SCALE = float(os.environ.get("SCALE", "0.001"))
MESH_PATH = os.environ.get("MESH_PATH", "/World/ChinouCenter")

t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)


import trimesh

objs = sorted(glob.glob(os.path.join(IN_DIR, "*.obj")))
assert objs, f"no .obj in {IN_DIR}"
log("OBJ files:", [os.path.basename(o) for o in objs])


def parse_vertex_colors(path):
    """`v x y z r g b` の rgb 列(0-255)を頂点順で抽出。色が無ければ None。"""
    cols = []
    has = True
    with open(path, errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                if len(p) >= 7:
                    cols.append((float(p[4]), float(p[5]), float(p[6])))
                else:
                    has = False
                    break
    if not has or not cols:
        return None
    c = np.asarray(cols, dtype=np.float64)
    if c.max() > 1.5:  # 0-255 とみなす
        c = c / 255.0
    return c


allV, allF, allC = [], [], []
voff = 0
for o in objs:
    m = trimesh.load(o, process=False, maintain_order=True, skip_materials=True)
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)
    V = np.asarray(m.vertices, dtype=np.float64)
    F = np.asarray(m.faces, dtype=np.int64)
    C = parse_vertex_colors(o)
    if C is None or len(C) != len(V):
        log(f"  WARN {os.path.basename(o)}: vertex color missing/mismatch -> gray")
        C = np.full((len(V), 3), 0.7)
    allV.append(V)
    allF.append(F + voff)
    allC.append(C)
    voff += len(V)
    log(f"  {os.path.basename(o)}: V={len(V):,} F={len(F):,}")

V = np.vstack(allV)
F = np.vstack(allF)
C = np.vstack(allC)
log(f"merged: V={len(V):,} F={len(F):,}")

mn, mx = V.min(0), V.max(0)
span = mx - mn
log(f"native bbox span (mm): {span}  min={mn} max={mx}")

# mm -> m
V = V * SCALE
mn, mx = V.min(0), V.max(0)
span_m = mx - mn
log(f"after scale x{SCALE} -> bbox span (m): {np.round(span_m, 3)}")

# recenter: 鉛直軸 = 最小スパン軸
up = int(np.argmin(span_m))
log(f"up-axis (min span) = {'XYZ'[up]} (height {span_m[up]:.3f} m)")
center = (mn + mx) / 2.0
trans = -center.copy()
trans[up] = -mn[up]  # 鉛直は下端を0へ
V = V + trans
mn, mx = V.min(0), V.max(0)
log(f"after recenter -> min={np.round(mn, 3)} max={np.round(mx, 3)}  (floor at {('XYZ')[up]}=0)")

# decimation: fast-simplification 主 / open3d vertex-clustering 予備。色は最近傍で再マップ
if TARGET_TRIS and len(F) > TARGET_TRIS:
    log(f"decimating {len(F):,} -> {TARGET_TRIS:,} tris (fast-simplification) ...")
    Vd = Fd = None
    try:
        import fast_simplification

        Vd, Fd = fast_simplification.simplify(
            np.ascontiguousarray(V, dtype=np.float32),
            np.ascontiguousarray(F, dtype=np.int32),
            target_count=int(TARGET_TRIS),
        )
        Vd = np.asarray(Vd, dtype=np.float64)
        Fd = np.asarray(Fd)
        log(f"fast-simplification done: V={len(Vd):,} F={len(Fd):,}")
    except Exception as e:
        log(
            f"fast-simplification FAILED ({type(e).__name__}: {e}); fallback open3d vertex-clustering"
        )
        import open3d as o3d

        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(V)
        m.triangles = o3d.utility.Vector3iVector(F.astype(np.int32))
        vox = float(np.linalg.norm(V.max(0) - V.min(0))) / 800.0
        m = m.simplify_vertex_clustering(
            voxel_size=vox, contraction=o3d.geometry.SimplificationContraction.Average
        )
        Vd = np.asarray(m.vertices)
        Fd = np.asarray(m.triangles)
        log(f"vertex-clustering done (voxel {vox:.3f}): V={len(Vd):,} F={len(Fd):,}")
    # 新頂点 -> 最近傍の元頂点色
    from scipy.spatial import cKDTree

    _, idx = cKDTree(V).query(Vd, k=1, workers=-1)
    C = C[idx]
    V = Vd
    F = Fd
    log(f"decimated final: V={len(V):,} F={len(F):,}")
else:
    log("no decimation")

# USD 出力
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

stage = Usd.Stage.CreateNew(OUT_USD)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)  # 出力は Z-up に正規化
# もし元の up が Z でないなら頂点を回転して Z-up へ
if up != 2:
    # up軸 -> Z へ入れ替え（符号維持の単純swap）
    order = [0, 1, 2]
    order[2], order[up] = order[up], order[2]
    V = V[:, order]
    log(f"reordered axes so up->Z (order={order})")

xf = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(xf.GetPrim())
mesh_p = UsdGeom.Mesh.Define(stage, MESH_PATH)
mesh_p.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V.astype(np.float32)))
mesh_p.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, dtype=np.int32)))
mesh_p.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.astype(np.int32).reshape(-1)))
mesh_p.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
mn, mx = V.min(0), V.max(0)
mesh_p.CreateExtentAttr(
    [
        Gf.Vec3f(float(mn[0]), float(mn[1]), float(mn[2])),
        Gf.Vec3f(float(mx[0]), float(mx[1]), float(mx[2])),
    ]
)
pv = UsdGeom.PrimvarsAPI(mesh_p).CreatePrimvar(
    "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
)
pv.Set(Vt.Vec3fArray.FromNumpy(np.clip(C, 0, 1).astype(np.float32)))
stage.Save()
log(f"WROTE {OUT_USD}")
log(f"FINAL bbox (m): min={np.round(mn, 3)} max={np.round(mx, 3)} size={np.round(mx - mn, 3)}")
print("CONVERT_OK", flush=True)
