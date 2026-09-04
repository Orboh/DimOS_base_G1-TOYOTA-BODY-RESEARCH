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

"""Okra01.3mf（スライサープロジェクト, mm, 3個配置）から把持可能なオクラ USD を生成。

  - trimesh で 3mf を解決し、最もオクラらしい geom を 1 個選ぶ
  - 高ポリ(数十万面)を fast_simplification で減面
  - 原点中心 + 実寸（長軸 ~10cm）へ縮尺（mm→m）
  - RigidBody + convexHull コライダー + 質量 + grippy 物理マテリアルを付与
    （Dex1指 grip_mat μ1.4 と average 合成で実効 μ≈1.2 ＝掴んで落ちない）

実行（usd-core+trimesh+fast_simplification の venv）:
  <python> docs/sim-setup/make_okra_usd.py
"""

import fast_simplification
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
import trimesh

SRC = "/home/kota-ueda/Desktop/dimos-hackathon/usd_file/Okra01.3mf"
OUT = "/home/kota-ueda/Desktop/dimos-hackathon/usd_file/okra.usd"
GEOM = "1"  # 使うジオメトリ（細身ポッド）
TARGET_LEN_M = 0.10  # オクラ長軸 [m]（収穫オクラ ~7-12cm）
TARGET_FACES = 3000  # 減面後の面数（把持・描画に十分）
MASS_KG = 0.012  # オクラ1本 ~12g
OKRA_FRIC = dict(static=1.0, dynamic=0.9, restitution=0.0)  # grippy（指と合わせて掴める）

scene = trimesh.load(SRC, force="scene")
m = scene.geometry[GEOM]
V = np.asarray(m.vertices, dtype=np.float64)
F = np.asarray(m.faces, dtype=np.int64)
print(f"[okra] 元: verts={len(V)} faces={len(F)} bbox(mm)={np.round(m.extents, 1)}")

# 減面
Vd, Fd = fast_simplification.simplify(V, F, target_count=TARGET_FACES)
print(f"[okra] 減面後: verts={len(Vd)} faces={len(Fd)}")

# 原点中心 + 実寸へ（mm→m して長軸を TARGET_LEN_M に）
Vd = Vd - (Vd.min(0) + Vd.max(0)) / 2.0
ext_m = (Vd.max(0) - Vd.min(0)) / 1000.0
scale = TARGET_LEN_M / ext_m.max()
Vm = Vd * 0.001 * scale
print(f"[okra] 実寸(m): {np.round((Vm.max(0) - Vm.min(0)), 3)}  (長軸={TARGET_LEN_M}m)")

stage = Usd.Stage.CreateNew(OUT)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

okra = UsdGeom.Xform.Define(stage, "/Okra")
stage.SetDefaultPrim(okra.GetPrim())
UsdPhysics.RigidBodyAPI.Apply(okra.GetPrim())  # 動的剛体（落ちる・掴まれる）
UsdPhysics.MassAPI.Apply(okra.GetPrim()).CreateMassAttr(MASS_KG)

mesh = UsdGeom.Mesh.Define(stage, "/Okra/mesh")
mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in Vm.astype(float)]))
mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(Fd.flatten().tolist()))
mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(Fd)))
mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.30, 0.52, 0.18)]))  # オクラ緑
UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
    UsdPhysics.Tokens.convexHull
)

mat = UsdShade.Material.Define(stage, "/Okra/PhysicsMaterials/okra_mat")
api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
api.CreateStaticFrictionAttr(OKRA_FRIC["static"])
api.CreateDynamicFrictionAttr(OKRA_FRIC["dynamic"])
api.CreateRestitutionAttr(OKRA_FRIC["restitution"])
bind = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
bind.Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")

stage.GetRootLayer().Save()
print(
    f"[okra] wrote {OUT}  mass={MASS_KG}kg friction={OKRA_FRIC['static']}/{OKRA_FRIC['dynamic']} rest={OKRA_FRIC['restitution']}"
)
