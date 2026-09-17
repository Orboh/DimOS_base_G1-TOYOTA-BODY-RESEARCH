#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pocket_v6.stl を G1 の torso_link 前面に取り付けた USD を作り、確認用スクショを撮る。

元の G1 USD は改変しない（reference で参照し、ポケットを子として足すだけ）。

座標の対応（回転不要）:
  pocket STL : X+ = お椀が膨らむ向き / Y = 左右(中央0) / Z+ = 上   ※単位 mm、原点は背面の取付基準点
  torso_link : X+ = 前 / Y+ = 左 / Z+ = 上                        ※単位 m
→ スケール 0.001 と平行移動だけで載る。

「胴体に食い込ませない」ため、ポケット背面の当たる範囲で torso 前面の最大 X を求め、
そこから --gap だけ前に離して置く（--gap 0 で表面に接触、めり込みゼロ）。

実行:
  cd /isaac-sim/workspace/DimOS_base_G1-TOYOTA-BODY-
  conda run -n env_isaaclab_2 python /isaac-sim/workspace/sub/四次元ポケット_CAD/mount_pocket.py
"""
from __future__ import annotations

import argparse
import os
import struct

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/isaac-sim/workspace/DimOS_base_G1-TOYOTA-BODY-"
G1_USD = f"{REPO}/usd_file/g1-29dof-dex1-base-fix-usd/g1_29dof_with_dex1_base_fix1.usd"

ap = argparse.ArgumentParser()
ap.add_argument("--stl", default=os.path.join(HERE, "pocket_v6.stl"))
ap.add_argument("--out", default=os.path.join(HERE, "out"))
# 高さ: ZED相当位置は torso 相対 Z=0.20（sim の SIM_CAM_LOCAL_POS）。その下に置く。
ap.add_argument("--z", type=float, default=0.11, help="ポケット原点の torso 相対 Z[m]")
ap.add_argument("--gap", type=float, default=0.004, help="胴体表面とのすき間[m]（0で接触・めり込み無し）")
ap.add_argument("--pitch-deg", type=float, default=0.0, help="お椀を上向きに傾ける角度[deg]")
ap.add_argument("--sink-into-waist", type=float, default=None, metavar="M",
                help="丸い底が腰に指定量[m]めり込む高さを自動で探す（--z は探索の開始点）")
ap.add_argument("--no-shot", action="store_true")
ap.add_argument("--shoulder-pitch", type=float, default=0.0, help="肩pitch[rad]（腕を下ろす姿勢の微調整）")
ap.add_argument("--elbow", type=float, default=0.0, help="肘[rad]")
ap.add_argument("--shoulder-roll", type=float, default=0.06, help="肩roll[rad]（体から少し離す）")
args = ap.parse_args()

from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

import numpy as np
from pxr import Usd, UsdGeom, UsdShade, UsdLux, Gf, Sdf
from isaacsim.core.utils.stage import add_reference_to_stage
import omni.usd

os.makedirs(args.out, exist_ok=True)


def load_stl(path):
    d = open(path, "rb").read()
    n = struct.unpack("<I", d[80:84])[0]
    if 84 + 50 * n != len(d):
        raise ValueError(f"binary STL ではない: size={len(d)} n={n}")
    a = np.frombuffer(d[84:], dtype=np.uint8).reshape(n, 50)
    return a[:, 12:48].copy().view("<f4").reshape(n, 3, 3).astype(np.float64)


tri = load_stl(args.stl) * 0.001          # mm → m
V = tri.reshape(-1, 3)
print(f"[pocket] STL 読み込み: 三角形 {len(tri)}  サイズ {np.round(V.max(0)-V.min(0), 4)} m", flush=True)

if args.pitch_deg:                         # お椀を上向きに（torso Y 軸まわり、-方向が上向き）
    th = np.radians(-args.pitch_deg)
    R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    V = V @ R.T
back = V[V[:, 0] < 1e-6]                   # 背面（取付面）の当たる範囲
z_lo, z_hi = back[:, 2].min(), back[:, 2].max()
y_lo, y_hi = back[:, 1].min(), back[:, 1].max()
print(f"[pocket] 背面の当たり範囲: Y {y_lo:+.4f}..{y_hi:+.4f}  Z {z_lo:+.4f}..{z_hi:+.4f} m", flush=True)

# --- G1 を読み、背面が当たる高さ帯での torso 前面 X を調べる -------------------
add_reference_to_stage(usd_path=G1_USD, prim_path="/G1")
stage = omni.usd.get_context().get_stage()
root = stage.GetPrimAtPath("/G1")
cache = UsdGeom.XformCache()
torso = next(p for p in Usd.PrimRange(root) if p.GetName() == "torso_link")
M_inv = cache.GetLocalToWorldTransform(torso).GetInverse()

def collect(link_names):
    out = []
    for nm in link_names:
        lk = next((p for p in Usd.PrimRange(root) if p.GetName() == nm), None)
        if lk is None:
            continue
        for p in Usd.PrimRange(lk):
            if not p.IsA(UsdGeom.Mesh):
                continue
            v = UsdGeom.Mesh(p).GetPointsAttr().Get()
            if not v:
                continue
            M = cache.GetLocalToWorldTransform(p) * M_inv
            for q in v:
                w = M.Transform(Gf.Vec3d(q[0], q[1], q[2]))
                out.append([w[0], w[1], w[2]])
    return np.array(out)


P = collect(["torso_link"])                      # 背面が当たる相手（胴体）
# 腰まわり。丸い底をここに当てる
PW = collect(["torso_link", "waist_roll_link", "waist_yaw_link", "pelvis", "pelvis_contour_link"])
print(f"[pocket] 胴体 {len(P)} 点 / 胴体+腰 {len(PW)} 点", flush=True)

def overlap_at(x, z):
    """背面板（半円）が占める領域で、G1 表面が背面板より前に出ている最大量[m]。
    正 = その分だけポケットが G1 に食い込んでいる。頂点だけでなく領域で見るのが要点。"""
    dy, dz = PW[:, 1], PW[:, 2] - z
    inside = (dz <= 0.0) & (dz >= z_lo) & (dy >= y_lo) & (dy <= y_hi)
    if not inside.any():
        return -1e9
    return float(PW[inside, 0].max()) - x


SINK = args.sink_into_waist

band = P[(P[:, 2] >= args.z + z_lo) & (P[:, 2] <= args.z + z_hi) &
         (P[:, 1] >= y_lo) & (P[:, 1] <= y_hi)]
if len(band) == 0:
    raise SystemExit(f"[pocket] Z={args.z} は胴体メッシュの外。--z を見直してください")
front_x = float(band[:, 0].max())
print(f"[pocket] 背面が当たる帯の胴体前面 X = {front_x:+.4f} m（点 {len(band)}）", flush=True)

if SINK is None:
    origin_x = front_x + args.gap
    print(f"[pocket] → ポケット原点を xyz = ({origin_x:+.4f}, 0, {args.z:+.4f}) に置く"
          f"（すき間 {args.gap*1000:.1f} mm, めり込み 0）", flush=True)
else:
    # X を後ろへ動かすほどめり込みが増える。目標量になる X を二分探索する。
    # 領域で見た重なりが SINK になる X（単調なので直接求まる）
    origin_x = overlap_at(0.0, args.z) - SINK
    print(f"[pocket] → ポケット原点を xyz = ({origin_x:+.4f}, 0, {args.z:+.4f}) に置く"
          f"（最大めり込みが {SINK*1000:.1f} mm になる位置）", flush=True)

V = V + np.array([origin_x, 0.0, args.z])

# 実際にめり込んでいないかを確認（背面の各点と、その高さの胴体前面を突き合わせる）
worst = None
for zc in np.linspace(args.z + z_lo, args.z + z_hi, 25):
    s = P[(np.abs(P[:, 2] - zc) < 0.005) & (P[:, 1] >= y_lo) & (P[:, 1] <= y_hi)]
    if len(s) == 0:
        continue
    clear = origin_x - float(s[:, 0].max())
    if worst is None or clear < worst[1]:
        worst = (zc, clear)
# 背面板が向き合う範囲での「すき間」の分布。平面 vs 曲面なので必ずばらつく。
_dz = PW[:, 2] - args.z
_in = (_dz <= 0.0) & (_dz >= z_lo) & (PW[:, 1] >= y_lo) & (PW[:, 1] <= y_hi)
if _in.any():
    _g = origin_x - PW[_in, 0]
    print(f"[pocket] 背面板と胴体のすき間: 最小 {_g.min()*1000:+.2f} mm / "
          f"中央値 {np.median(_g)*1000:+.1f} mm / 最大 {_g.max()*1000:+.1f} mm", flush=True)

_ov = overlap_at(origin_x, args.z)
print(f"[pocket] 胴体・腰への最大めり込み {_ov*1000:+.2f} mm"
      f"（負なら浮いている / 背面板の領域全体で評価）", flush=True)

# --- ポケットを torso_link の子として足す（元 USD は無改変）-------------------
mesh_path = f"{torso.GetPath().pathString}/pocket_v6"
mesh = UsdGeom.Mesh.Define(stage, mesh_path)
mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in V])
mesh.CreateFaceVertexCountsAttr([3] * len(tri))
mesh.CreateFaceVertexIndicesAttr(list(range(len(V))))
mesh.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.55, 0.10)])   # 参考画像のオレンジ
mesh.CreateSubdivisionSchemeAttr("none")

# 正中断面（Y≈0）のデータを書き出す。胴体の輪郭とポケットの位置関係を図にするため。
_prof = PW[np.abs(PW[:, 1]) < 0.012]
np.save(os.path.join(args.out, "profile_body.npy"), _prof[:, [0, 2]])
np.save(os.path.join(args.out, "profile_pocket.npy"), V[np.abs(V[:, 1]) < 0.012][:, [0, 2]])
np.save(os.path.join(args.out, "profile_meta.npy"),
        np.array([origin_x, args.z, args.gap, z_lo, z_hi]))

out_usd = os.path.join(args.out, "g1_with_pocket.usd")
stage.GetRootLayer().Export(out_usd)
print(f"\n[pocket] USD を書き出し: {out_usd}", flush=True)

if not args.no_shot:
    from isaacsim.core.api import World
    from isaacsim.sensors.camera import Camera

    # headless のシーンには光源が無い（そのままだと真っ黒になる）
    dome = UsdLux.DomeLight.Define(stage, "/shot_dome")
    dome.CreateIntensityAttr(1200.0)
    key = UsdLux.DistantLight.Define(stage, "/shot_key")
    key.CreateIntensityAttr(2500.0)
    key.CreateAngleAttr(2.0)
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 0.0, 35.0))

    world = World(stage_units_in_meters=1.0)
    world.get_physics_context().set_gravity(0.0)   # 姿勢が崩れないように

    # 既定姿勢だと腕が前に出てポケットが隠れるので、腕を下ろした姿勢にしてから撮る
    from pxr import UsdPhysics
    try:
        from isaacsim.core.prims import SingleArticulation as ArtCls
    except Exception:  # noqa: BLE001
        from isaacsim.core.api.articulations import Articulation as ArtCls
    art_path = next(p.GetPath().pathString for p in Usd.PrimRange(root)
                    if p.HasAPI(UsdPhysics.ArticulationRootAPI))
    art = ArtCls(prim_path=art_path, name="g1")
    world.scene.add(art)
    world.reset()
    art.initialize()
    from isaacsim.core.utils.types import ArticulationAction
    names = list(art.dof_names)
    q = np.zeros(art.num_dof)
    for side, sign in (("left", +1.0), ("right", -1.0)):
        for jn, val in ((f"{side}_shoulder_pitch_joint", args.shoulder_pitch),
                        (f"{side}_shoulder_roll_joint", sign * args.shoulder_roll),
                        (f"{side}_elbow_joint", args.elbow)):
            if jn in names:
                q[names.index(jn)] = val
    art.set_joint_positions(q)
    art.set_joint_velocities(np.zeros_like(q))
    art.apply_action(ArticulationAction(joint_positions=q))
    for _ in range(5):
        world.step(render=False)
    print(f"[pocket] 腕を下ろした姿勢に設定（shoulder_pitch={args.shoulder_pitch} "
          f"roll=±{args.shoulder_roll} elbow={args.elbow}）", flush=True)

    ARM_PREFIX = ("left_shoulder", "left_elbow", "left_wrist", "left_hand", "left_rubber",
                  "right_shoulder", "right_elbow", "right_wrist", "right_hand", "right_rubber")

    def set_arms_visible(vis: bool) -> None:
        for p in Usd.PrimRange(root):
            if p.GetName().startswith(ARM_PREFIX) and p.IsA(UsdGeom.Imageable):
                im = UsdGeom.Imageable(p)
                im.MakeVisible() if vis else im.MakeInvisible()

    M_t = UsdGeom.XformCache().GetLocalToWorldTransform(torso)
    tw = M_t.ExtractTranslation()
    for nm, eye, tgt, res, hide_arms in [
        ("pocket_front", (1.30, 0.00, 0.30), (0.00, 0.0, 0.10), (900, 900), False),
        ("pocket_close", (0.70, -0.32, 0.22), (0.05, 0.0, 0.10), (900, 900), False),
        # 真横（X=0）から。腕を消さないと胴体とポケットの間に重なって半円が読めない
        ("pocket_side",  (0.12, -1.15, 0.07), (0.12, 0.0, 0.07), (900, 900), True),
        ("pocket_top",   (0.55, -0.30, 0.55), (0.06, 0.0, 0.09), (900, 900), True),
    ]:
        set_arms_visible(not hide_arms)
        e = Gf.Vec3d(tw[0] + eye[0], tw[1] + eye[1], tw[2] + eye[2])
        t = Gf.Vec3d(tw[0] + tgt[0], tw[1] + tgt[1], tw[2] + tgt[2])
        m = Gf.Matrix4d(); m.SetLookAt(e, t, Gf.Vec3d(0, 0, 1))
        q = m.GetInverse().ExtractRotationQuat()
        cam = Camera(prim_path=f"/shot_{nm}", resolution=res)
        cam.initialize()
        cam.set_world_pose(np.array([e[0], e[1], e[2]]),
                           np.array([q.GetReal(), *q.GetImaginary()]), camera_axes="usd")
        cam.set_clipping_range(0.02, 100.0)
        for _ in range(60):
            world.step(render=True)
        img = cam.get_rgba()
        if img is not None and img.size:
            import PIL.Image
            PIL.Image.fromarray((img[:, :, :3]).astype(np.uint8)).save(os.path.join(args.out, f"{nm}.png"))
            print(f"[pocket] スクショ: {os.path.join(args.out, nm)}.png", flush=True)
sim_app.close()
