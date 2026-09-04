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

"""chinou_center.usd を「床=z0 + コライダー + ライト」付きに再生成（ローカル usd-core, GPU不要）。

1) 既存 USD の幾何(頂点/面/頂点色)を読む
2) 本当の床高さ(支配的な低い水平面 ~0.49m)を検出し、床を z=0 へシフト（足の埋もれを解消）
3) 不可視の static コライダーを追加: 床ボックス(上面 z=0) + 4枚壁ボックス(室外周)
4) DomeLight + DistantLight 追加、メッシュ doubleSided=true
5) chinou_center.usd を上書き出力
"""

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, Vt

SRC = "/home/kota-ueda/Desktop/dimos-hackathon/usd_file/chinou_center.usd"
OUT = "/home/kota-ueda/Desktop/dimos-hackathon/usd_file/chinou_center.usd"

src = Usd.Stage.Open(SRC)
m = UsdGeom.Mesh(src.GetPrimAtPath("/World/ChinouCenter"))
P = np.array(m.GetPointsAttr().Get(), dtype=np.float64)
FVC = np.array(m.GetFaceVertexCountsAttr().Get())
FVI = np.array(m.GetFaceVertexIndicesAttr().Get())
pv = UsdGeom.PrimvarsAPI(m.GetPrim()).GetPrimvar("displayColor")
C = np.array(pv.Get(), dtype=np.float64) if pv and pv.Get() else None
print(f"read: V={len(P)} F={len(FVC)} color={'yes' if C is not None else 'no'}")

# 床高さ検出: 0.4-0.6m の支配的水平スラブの中央値
band = P[(P[:, 2] >= 0.45) & (P[:, 2] <= 0.54)]
floor_z = float(np.median(band[:, 2]))
print(f"detected floor z = {floor_z:.4f} m (band n={len(band)})")

# 床を z=0 へシフト
P[:, 2] -= floor_z
zmin, zmax = float(P[:, 2].min()), float(P[:, 2].max())
xmin, xmax = float(P[:, 0].min()), float(P[:, 0].max())
ymin, ymax = float(P[:, 1].min()), float(P[:, 1].max())
print(f"after shift: z[{zmin:.3f},{zmax:.3f}]  X[{xmin:.2f},{xmax:.2f}] Y[{ymin:.2f},{ymax:.2f}]")

# 出力ステージ
OUT_TMP = OUT.replace(".usd", "_tmp.usd")
stage = Usd.Stage.CreateNew(OUT_TMP)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())

# 視覚メッシュ
mesh = UsdGeom.Mesh.Define(stage, "/World/ChinouCenter")
mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(P.astype(np.float32)))
mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(FVC.astype(np.int32)))
mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(FVI.astype(np.int32)))
mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
mesh.CreateDoubleSidedAttr(True)  # 裏面の暗さ解消
mesh.CreateExtentAttr(
    [
        Gf.Vec3f(float(P[:, 0].min()), float(P[:, 1].min()), zmin),
        Gf.Vec3f(float(P[:, 0].max()), float(P[:, 1].max()), zmax),
    ]
)
if C is not None:
    UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
    ).Set(Vt.Vec3fArray.FromNumpy(np.clip(C, 0, 1).astype(np.float32)))


# コライダー(不可視 static box)
def collider_box(path, center, half):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(2.0)  # ±1 -> scale=half で ±half
    x = UsdGeom.XformCommonAPI(cube)
    x.SetTranslate(Gf.Vec3d(*center))
    x.SetScale(Gf.Vec3f(*half))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())  # static(剛体なし)= 動かない衝突体
    UsdGeom.Imageable(cube.GetPrim()).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    return cube


col = UsdGeom.Scope.Define(stage, "/World/Colliders")
t = 0.10  # 壁/床の半厚
wall_h = max(2.8, zmax)  # 壁高さ
hz = wall_h / 2.0
# 床(上面 z=0)
collider_box("/World/Colliders/floor", (0, 0, -t), (4.6, 4.6, t))
# 4枚壁(室外周, 内向き)
collider_box("/World/Colliders/wall_xp", (xmax, 0, hz), (t, (ymax - ymin) / 2, hz))
collider_box("/World/Colliders/wall_xn", (xmin, 0, hz), (t, (ymax - ymin) / 2, hz))
collider_box("/World/Colliders/wall_yp", (0, ymax, hz), ((xmax - xmin) / 2, t, hz))
collider_box("/World/Colliders/wall_yn", (0, ymin, hz), ((xmax - xmin) / 2, t, hz))

# ライト
lights = UsdGeom.Scope.Define(stage, "/World/Lights")
dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
dome.CreateIntensityAttr(1000.0)
dist = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
dist.CreateIntensityAttr(2500.0)
dist.CreateAngleAttr(1.0)
UsdGeom.XformCommonAPI(dist).SetRotate(Gf.Vec3f(315.0, 0.0, 0.0))  # 斜め上から

stage.GetRootLayer().Save()

# 置換
import os

os.replace(OUT_TMP, OUT)
print(f"WROTE {OUT}")
print(f"floor at z=0, ceiling ~{zmax:.2f}m, footprint {xmax - xmin:.2f}x{ymax - ymin:.2f}m")
print("colliders: floor + 4 walls (invisible static), lights: Dome+Distant, doubleSided=on")
print("BUILD_OK")
