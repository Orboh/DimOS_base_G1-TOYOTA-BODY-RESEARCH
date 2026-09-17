#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""到達できる点を緑、できない点を赤にして Isaac Sim 上に点群で出す。

pocket_reach_map.csv（最終判定つき）を読み、torso_link の子として点を置く。
1000点あっても UsdGeom.Points なら軽い。
"""
from __future__ import annotations

import argparse
import csv
import os
import struct

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

REPO = "/isaac-sim/workspace/DimOS_base_G1-TOYOTA-BODY-"
G1_USD = f"{REPO}/usd_file/g1-29dof-dex1-base-fix-usd/g1_29dof_with_dex1_base_fix1.usd"
DEF_CSV = "/isaac-sim/workspace/02_orita_tool/IK_doc/outAndPocket/koshi_plane/isaacsim/pocket_reach_map.csv"

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=DEF_CSV)
ap.add_argument("--size", type=float, default=0.016, help="点の直径[m]")
ap.add_argument("--depths", default=None, help="表示する距離をカンマ区切りで絞る（既定は全部）")
ap.add_argument("--by-state", action="store_true",
                help="緑/赤の2色ではなく、落ちた理由별に色分けする")
ap.add_argument("--lift", type=float, default=0.8)
ap.add_argument("--headless", action="store_true")
ap.add_argument("--shot", default=None)
ap.add_argument("--seconds", type=float, default=1800.0)
ap.add_argument("--orbit", default=None, metavar="GIF",
                help="点群のまわりをカメラで一周しながら撮って GIF にする")
ap.add_argument("--orbit-frames", type=int, default=72)
ap.add_argument("--orbit-radius", type=float, default=2.4, help="回る半径[m]")
ap.add_argument("--orbit-height", type=float, default=0.75, help="torso からの高さ[m]")
ap.add_argument("--orbit-center", default="0.30,-0.10,0.05", help="注視点（torso基準）")
ap.add_argument("--orbit-res", default="700,600")
ap.add_argument("--fps", type=int, default=15)
ap.add_argument("--pocket-stl", default="/isaac-sim/workspace/sub/四次元ポケット_CAD/pocket_halfcyl.stl")
ap.add_argument("--pocket-origin", default="0.0709,0,0.0770")
ap.add_argument("--cam", default="1.9,-2.2,1.2")
ap.add_argument("--cam-target", default="0.35,-0.15,0.05")
args = ap.parse_args()

rows = list(csv.DictReader(open(args.csv)))
if args.depths:
    keep = {f"{float(x):g}" for x in args.depths.split(",")}
    rows = [r for r in rows if f"{float(r['depth']):g}" in keep]
n_ok = sum(1 for r in rows if r["final_ok"] == "1")
print(f"[cloud] {len(rows)} 点を表示（到達 {n_ok} / 不到達 {len(rows)-n_ok}）", flush=True)

from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": args.headless})

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, Gf
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
import omni.usd

try:
    from isaacsim.core.prims import SingleArticulation as ArtCls
except Exception:  # noqa: BLE001
    from isaacsim.core.api.articulations import Articulation as ArtCls

world = World(stage_units_in_meters=1.0)
world.get_physics_context().set_gravity(0.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path=G1_USD, prim_path="/G1")
stage = omni.usd.get_context().get_stage()
root = stage.GetPrimAtPath("/G1")

xf = UsdGeom.Xformable(root)         # 原点が骨盤なので持ち上げないと脚が地面に埋まる
xf.ClearXformOpOrder()
xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, float(args.lift)))
torso = next(p for p in Usd.PrimRange(root) if p.GetName() == "torso_link")

# 落ちた理由別の色（--by-state のとき）。2色モードは緑／赤のみ
STATE_COL = {
    "reachable":             (0.00, 0.51, 0.00),
    "hits pocket":           (0.93, 0.63, 0.00),
    "model / self-contact":  (0.92, 0.41, 0.20),
    "IK fails":              (0.89, 0.29, 0.28),
    "outside workspace box": (0.73, 0.72, 0.69),
}
pts, cols = [], []
for r in rows:
    pts.append(Gf.Vec3f(float(r["tx"]), float(r["ty"]), float(r["tz"])))
    if args.by_state:
        cols.append(Gf.Vec3f(*STATE_COL.get(r["state"], (0.5, 0.5, 0.5))))
    else:
        cols.append(Gf.Vec3f(0.00, 0.51, 0.00) if r["final_ok"] == "1"
                    else Gf.Vec3f(0.85, 0.11, 0.11))

pc = UsdGeom.Points.Define(stage, f"{torso.GetPath().pathString}/reach_cloud")
pc.CreatePointsAttr(pts)
pc.CreateWidthsAttr([float(args.size)] * len(pts))
pc.CreateDisplayColorAttr(cols)
pc.GetDisplayColorPrimvar().SetInterpolation("vertex")
print(f"[cloud] 点群を配置（直径 {args.size*1000:.0f} mm）", flush=True)

# ポケットも置く（見た目のみ）
if os.path.exists(args.pocket_stl):
    po = np.array([float(v) for v in args.pocket_origin.split(",")])
    d = open(args.pocket_stl, "rb").read()
    n = struct.unpack("<I", d[80:84])[0]
    if 84 + 50 * n == len(d):
        a = np.frombuffer(d[84:], dtype=np.uint8).reshape(n, 50)
        tri = a[:, 12:48].copy().view("<f4").reshape(n, 3, 3).astype(float) * 0.001
        V = tri.reshape(-1, 3) + po
        m = UsdGeom.Mesh.Define(stage, f"{torso.GetPath().pathString}/pocket_view")
        m.CreatePointsAttr([Gf.Vec3f(*p) for p in V])
        m.CreateFaceVertexCountsAttr([3] * n)
        m.CreateFaceVertexIndicesAttr(list(range(len(V))))
        m.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.55, 0.10)])
        m.CreateSubdivisionSchemeAttr("none")

UsdLux.DomeLight.Define(stage, "/cloud_dome").CreateIntensityAttr(1300.0)
_k = UsdLux.DistantLight.Define(stage, "/cloud_key")
_k.CreateIntensityAttr(2200.0)
UsdGeom.Xformable(_k.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 0.0, 35.0))

art_path = next(p.GetPath().pathString for p in Usd.PrimRange(root)
                if p.HasAPI(UsdPhysics.ArticulationRootAPI))
art = ArtCls(prim_path=art_path, name="g1")
world.scene.add(art)
world.reset()
art.initialize()
q = np.zeros(art.num_dof)            # 基準姿勢で立たせておく

cache = UsdGeom.XformCache()
tw = cache.GetLocalToWorldTransform(torso).ExtractTranslation()
ce = [float(v) for v in args.cam.split(",")]
ct = [float(v) for v in args.cam_target.split(",")]
eye = Gf.Vec3d(tw[0] + ce[0], tw[1] + ce[1], tw[2] + ce[2])
tgt = Gf.Vec3d(tw[0] + ct[0], tw[1] + ct[1], tw[2] + ct[2])
if not args.headless:
    try:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=[eye[0], eye[1], eye[2]], target=[tgt[0], tgt[1], tgt[2]])
    except Exception as _e:  # noqa: BLE001
        print(f"[cloud] 視点設定に失敗: {_e}", flush=True)

if args.orbit:
    import math
    from isaacsim.sensors.camera import Camera
    import PIL.Image
    _res = tuple(int(v) for v in args.orbit_res.split(","))
    _oc = [float(v) for v in args.orbit_center.split(",")]
    ctr = Gf.Vec3d(tw[0] + _oc[0], tw[1] + _oc[1], tw[2] + _oc[2])
    cam = Camera(prim_path="/orbit_cam", resolution=_res)
    cam.initialize()
    cam.set_clipping_range(0.02, 100.0)
    for _ in range(40):                     # レンダラを暖める（最初の数枚が黒くなる）
        art.set_joint_positions(q)
        art.apply_action(ArticulationAction(joint_positions=q))
        world.step(render=True)
    N = args.orbit_frames
    # 1枚目の位置にカメラを置いた状態で暖機する（姿勢を入れる前に暖めても効かない）
    _m0 = Gf.Matrix4d()
    _m0.SetLookAt(Gf.Vec3d(ctr[0] + args.orbit_radius, ctr[1], tw[2] + args.orbit_height),
                  ctr, Gf.Vec3d(0, 0, 1))
    _q0 = _m0.GetInverse().ExtractRotationQuat()
    cam.set_world_pose(np.array([ctr[0] + args.orbit_radius, ctr[1], tw[2] + args.orbit_height]),
                       np.array([_q0.GetReal(), *_q0.GetImaginary()]), camera_axes="usd")
    for _ in range(30):
        art.set_joint_positions(q)
        art.apply_action(ArticulationAction(joint_positions=q))
        world.step(render=True)
        cam.get_rgba()
    frames = []
    print(f"[cloud] 一周を {N} フレームで撮影（{args.fps} fps）", flush=True)
    for i in range(N):
        th = 2.0 * math.pi * i / N
        # ロボットの正面(+X)から始めて、右回りに一周する
        eye = Gf.Vec3d(ctr[0] + args.orbit_radius * math.cos(th),
                       ctr[1] + args.orbit_radius * math.sin(th),
                       tw[2] + args.orbit_height)
        m = Gf.Matrix4d(); m.SetLookAt(eye, ctr, Gf.Vec3d(0, 0, 1))
        qq = m.GetInverse().ExtractRotationQuat()
        cam.set_world_pose(np.array([eye[0], eye[1], eye[2]]),
                           np.array([qq.GetReal(), *qq.GetImaginary()]), camera_axes="usd")
        art.set_joint_positions(q)
        art.apply_action(ArticulationAction(joint_positions=q))
        for _ in range(4):
            world.step(render=True)
        img = cam.get_rgba()
        if img is not None and img.size:
            im = PIL.Image.fromarray(img[:, :, :3].astype(np.uint8))
            frames.append(im.quantize(colors=128, method=PIL.Image.MEDIANCUT))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{N} フレーム", flush=True)
    if frames:
        frames[0].save(args.orbit, save_all=True, append_images=frames[1:],
                       duration=int(1000 / args.fps), loop=0, optimize=True)
        print(f"[cloud] 動画: {args.orbit}  （{len(frames)} フレーム）", flush=True)
    sim_app.close()
    raise SystemExit(0)

import time
t0 = time.time()
print(f"[cloud] 表示中。{args.seconds:.0f} 秒で終了します", flush=True)
while time.time() - t0 < args.seconds:
    art.set_joint_positions(q)
    art.apply_action(ArticulationAction(joint_positions=q))
    world.step(render=True)
    if args.shot and time.time() - t0 > 2.0:
        from isaacsim.sensors.camera import Camera
        mm = Gf.Matrix4d(); mm.SetLookAt(eye, tgt, Gf.Vec3d(0, 0, 1))
        qq = mm.GetInverse().ExtractRotationQuat()
        cam = Camera(prim_path="/cloud_cam", resolution=(1100, 900))
        cam.initialize()
        cam.set_world_pose(np.array([eye[0], eye[1], eye[2]]),
                           np.array([qq.GetReal(), *qq.GetImaginary()]), camera_axes="usd")
        cam.set_clipping_range(0.02, 100.0)
        for _ in range(45):
            art.set_joint_positions(q)
            art.apply_action(ArticulationAction(joint_positions=q))
            world.step(render=True)
        img = cam.get_rgba()
        if img is not None and img.size:
            import PIL.Image
            PIL.Image.fromarray(img[:, :, :3].astype(np.uint8)).save(args.shot)
            print(f"[cloud] スクショ: {args.shot}", flush=True)
        break
sim_app.close()
