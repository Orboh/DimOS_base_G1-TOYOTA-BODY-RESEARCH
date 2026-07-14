#!/usr/bin/env python3
"""オクラ畑 USD を「地面=z0 + 地面コライダー + 太陽光」付きに再生成（ローカル usd-core, GPU不要）。

`build_chinou_phys.py`（屋内=知能センター用）の屋外(畑)版。屋内との違い:
  - 壁/天井が無い屋外なので **4枚壁コライダーは付けず、地面コライダーのみ**。
  - 床高さは決め打ちバンドではなく **z ヒストグラムで支配的な水平スラブを自動検出**
    （どの畑スキャンでも通るように一般化）。
  - 照明は屋外想定で **DomeLight(環境光) + DistantLight(太陽)** を明るめに。

処理:
  1) 既存 USD の幾何(頂点/面/頂点色)を読む
  2) 地面高さ(最も密な低い水平スラブ)を検出し、地面を z=0 へシフト（足の埋もれ/浮き解消）
  3) 不可視 static 地面コライダー(上面 z=0, footprint 全体+マージン)を追加
  4) DomeLight + DistantLight(太陽) 追加、メッシュ doubleSided=true
  5) OUT へ出力（既定は SRC 上書き）

env:
  SRC          入力 USD（既定 usd_file/okra_field.usd）
  OUT          出力 USD（既定 = SRC 上書き）
  MESH_PATH    メッシュ prim パス（既定 /World/OkraField）
  SUN_INTENSITY DistantLight 強度（既定 3000, 屋外は chinou の 2500 より明るめ）
  DOME_INTENSITY DomeLight 強度（既定 1200）
"""
import os
import numpy as np
from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, Sdf, Vt, Gf

ROOT = "/home/kota-ueda/Desktop/dimos-hackathon"
SRC = os.environ.get("SRC", f"{ROOT}/usd_file/okra_field.usd")
OUT = os.environ.get("OUT", SRC)
MESH_PATH = os.environ.get("MESH_PATH", "/World/OkraField")
SUN_INTENSITY = float(os.environ.get("SUN_INTENSITY", "3000"))
DOME_INTENSITY = float(os.environ.get("DOME_INTENSITY", "1200"))

src = Usd.Stage.Open(SRC)
# メッシュ prim を探索（MESH_PATH が無ければ最初の Mesh を採用）
mp = src.GetPrimAtPath(MESH_PATH)
if not (mp and mp.IsValid()):
    mp = next((p for p in src.TraverseAll() if p.IsA(UsdGeom.Mesh)), None)
    assert mp is not None, f"no Mesh in {SRC}"
    MESH_PATH = str(mp.GetPath())
    print(f"MESH_PATH auto -> {MESH_PATH}")
m = UsdGeom.Mesh(mp)
P = np.array(m.GetPointsAttr().Get(), dtype=np.float64)
FVC = np.array(m.GetFaceVertexCountsAttr().Get())
FVI = np.array(m.GetFaceVertexIndicesAttr().Get())
pv = UsdGeom.PrimvarsAPI(m.GetPrim()).GetPrimvar("displayColor")
C = np.array(pv.Get(), dtype=np.float64) if pv and pv.Get() else None
print(f"read: V={len(P)} F={len(FVC)} color={'yes' if C is not None else 'no'}")

# --- 地面高さ自動検出: 下位40%レンジ内で最も頂点が密な z バンドの中央値 ---
z = P[:, 2]
zmin0, zmax0 = float(z.min()), float(z.max())
lo_cut = zmin0 + 0.40 * (zmax0 - zmin0)          # 下位40%だけを地面候補に
lo = z[z <= lo_cut]
bins = np.linspace(lo.min(), lo.max(), 200)       # ~細かい bin で密度ピーク検出
hist, edges = np.histogram(lo, bins=bins)
k = int(np.argmax(hist))                           # 最も密な水平スラブ
band_lo, band_hi = edges[k], edges[k + 1]
# ピーク bin の周辺（±1cm）中央値を地面 z とする
band = z[(z >= band_lo - 0.01) & (z <= band_hi + 0.01)]
floor_z = float(np.median(band))
print(f"detected ground z = {floor_z:.4f} m "
      f"(peak bin [{band_lo:.3f},{band_hi:.3f}] n={hist[k]}, band n={len(band)})")

# --- 地面を z=0 へシフト ---
P[:, 2] -= floor_z
zmin, zmax = float(P[:, 2].min()), float(P[:, 2].max())
xmin, xmax = float(P[:, 0].min()), float(P[:, 0].max())
ymin, ymax = float(P[:, 1].min()), float(P[:, 1].max())
print(f"after shift: z[{zmin:.3f},{zmax:.3f}]  "
      f"X[{xmin:.2f},{xmax:.2f}] Y[{ymin:.2f},{ymax:.2f}]  "
      f"footprint {xmax - xmin:.2f}x{ymax - ymin:.2f}m, height {zmax:.2f}m")

# --- 出力ステージ ---
OUT_TMP = OUT.replace(".usd", "_tmp.usd")
if os.path.exists(OUT_TMP):
    os.remove(OUT_TMP)
stage = Usd.Stage.CreateNew(OUT_TMP)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())

# 視覚メッシュ
mesh = UsdGeom.Mesh.Define(stage, MESH_PATH)
mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(P.astype(np.float32)))
mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(FVC.astype(np.int32)))
mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(FVI.astype(np.int32)))
mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
mesh.CreateDoubleSidedAttr(True)   # 裏面の暗さ解消（屋外スキャンの薄い葉/地面）
mesh.CreateExtentAttr([Gf.Vec3f(xmin, ymin, zmin), Gf.Vec3f(xmax, ymax, zmax)])
if C is not None:
    UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
    ).Set(Vt.Vec3fArray.FromNumpy(np.clip(C, 0, 1).astype(np.float32)))

# --- 地面コライダー(不可視 static box, 上面 z=0) ---
col = UsdGeom.Scope.Define(stage, "/World/Colliders")
cube = UsdGeom.Cube.Define(stage, "/World/Colliders/ground")
cube.CreateSizeAttr(2.0)  # ±1 -> scale=half で ±half
# footprint 全体 + 2m マージン、厚み 0.5m（上面 z=0）
hx = (xmax - xmin) / 2.0 + 2.0
hy = (ymax - ymin) / 2.0 + 2.0
tz = 0.5
cx = (xmin + xmax) / 2.0
cy = (ymin + ymax) / 2.0
xc = UsdGeom.XformCommonAPI(cube)
xc.SetTranslate(Gf.Vec3d(cx, cy, -tz))
xc.SetScale(Gf.Vec3f(hx, hy, tz))
UsdPhysics.CollisionAPI.Apply(cube.GetPrim())  # static(剛体なし)= 動かない衝突体
UsdGeom.Imageable(cube.GetPrim()).CreateVisibilityAttr(UsdGeom.Tokens.invisible)

# --- ライト（屋外: 環境光 Dome + 太陽 Distant）---
lights = UsdGeom.Scope.Define(stage, "/World/Lights")
dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
dome.CreateIntensityAttr(DOME_INTENSITY)
sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
sun.CreateIntensityAttr(SUN_INTENSITY)
sun.CreateAngleAttr(0.53)  # 太陽の見かけ角度 ~0.53deg
UsdGeom.XformCommonAPI(sun).SetRotate(Gf.Vec3f(315.0, 15.0, 0.0))  # 斜め上から

stage.GetRootLayer().Save()

# 置換
os.replace(OUT_TMP, OUT)
print(f"WROTE {OUT}")
print(f"ground at z=0, height ~{zmax:.2f}m, footprint {xmax - xmin:.2f}x{ymax - ymin:.2f}m")
print("collider: ground only (invisible static, no walls), lights: Dome+Sun, doubleSided=on")
print("BUILD_OK")
