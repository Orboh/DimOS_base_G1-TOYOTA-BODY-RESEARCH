#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""指定した1点の IK 解を G1 に入れて、目標点に赤丸を置いて見せる（Isaac Sim GUI）。

例: 0.7 m で唯一到達できる r0c9 を見る
  conda run -n env_isaaclab_2 python show_point.py \
      --csv .../reach_map.csv --depth 0.7 --row 0 --col 9
"""
from __future__ import annotations

import argparse
import csv
import os
import struct

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/isaac-sim/workspace/DimOS_base_G1-TOYOTA-BODY-"
G1_USD = f"{REPO}/usd_file/g1-29dof-dex1-base-fix-usd/g1_29dof_with_dex1_base_fix1.usd"

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="/isaac-sim/workspace/02_orita_tool/IK_doc/outAndPocket/koshi/isaacsim/reach_map.csv")
ap.add_argument("--depth", required=True)
ap.add_argument("--row", type=int, required=True)
ap.add_argument("--col", type=int, required=True)
ap.add_argument("--marker-r", type=float, default=0.012, help="赤丸の半径[m]")
ap.add_argument("--lift", type=float, default=0.8)
ap.add_argument("--headless", action="store_true", help="画面を出さずスクショだけ撮る")
ap.add_argument("--shot", default=None, help="スクショの保存先PNG")
ap.add_argument("--video", default=None, help="アニメーション1周期をGIFで保存（--animate と併用）")
ap.add_argument("--fps", type=int, default=15, help="GIF のフレームレート")
ap.add_argument("--pocket-stl", default="/isaac-sim/workspace/sub/四次元ポケット_CAD/pocket_halfcyl.stl")
ap.add_argument("--pocket-origin", default="0.0709,0,0.0770")
ap.add_argument("--seconds", type=float, default=600.0, help="GUI を開いておく秒数")
ap.add_argument("--fov-plane", type=float, default=None, metavar="D",
                help="画角グリッドを、体に正対する平面（--radius-center から前方D[m]）に描く")
ap.add_argument("--fov-grid", type=float, default=None, metavar="R",
                help="画角の10x10グリッドを、半径R[m]の球面上に線で描く（中心は --radius-center）")
ap.add_argument("--radius-center", default="0.00396,0,-0.044",
                help="半径の中心（torso座標）。既定は IK の基準座標 = pelvis")
ap.add_argument("--grid-n", type=int, default=10)
ap.add_argument("--highlight-cell", action="store_true",
                help="--row/--col のマスの枠を赤で太く描く（どのマスか一目で分かる）")
ap.add_argument("--cam-hfov", type=float, default=90.0)
ap.add_argument("--cam-res", default="640,360")
ap.add_argument("--cam-pos", default="0.08,0,0.20")
ap.add_argument("--cam-fwd", default="1,0,-0.35")
ap.add_argument("--animate", action="store_true",
                help="基準姿勢から目標姿勢へ腕を動かし、戻る動きを繰り返す")
ap.add_argument("--anim-sec", type=float, default=2.0, help="片道にかける秒数")
ap.add_argument("--hold-sec", type=float, default=0.8, help="伸ばしきった所と戻った所で止まる秒数")
ap.add_argument("--cam", default="1.9,-1.7,0.9", help="視点のオフセット（torso基準）")
ap.add_argument("--cam-target", default="0.25,-0.15,0.05", help="注視点のオフセット（torso基準）")
args = ap.parse_args()

row = next((r for r in csv.DictReader(open(args.csv))
            if r["depth"] == args.depth and int(r["row"]) == args.row and int(r["col"]) == args.col), None)
if row is None or row["reach_ok"] != "1":
    raise SystemExit(f"該当点が無いか到達不可: depth={args.depth} r{args.row}c{args.col}")
q_arm = [float(row[f"q{i}"]) for i in range(7)]
target = [float(row["tx"]), float(row["ty"]), float(row["tz"])]
print(f"[show] depth={args.depth} r{args.row}c{args.col}", flush=True)
print(f"[show] 目標(torso相対) = {target}   IK残差 = {row['err_m']} m", flush=True)

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

# 原点が骨盤なので、地面に置くと脚が埋まる。持ち上げる（判定には無関係）
xf = UsdGeom.Xformable(root)
xf.ClearXformOpOrder()
xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, float(args.lift)))

torso = next(p for p in Usd.PrimRange(root) if p.GetName() == "torso_link")

# 目標点の赤丸（torso_link の子にすれば torso 相対のまま置ける）
sph = UsdGeom.Sphere.Define(stage, f"{torso.GetPath().pathString}/target_marker")
sph.CreateRadiusAttr(float(args.marker_r))
sph.CreateDisplayColorAttr([Gf.Vec3f(0.80, 0.10, 0.10)])
UsdGeom.Xformable(sph.GetPrim()).AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
    Gf.Vec3d(*target))
print(f"[show] 赤丸を配置（半径 {args.marker_r*1000:.0f} mm）", flush=True)

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

# 画角の10x10グリッドを球面上に描く（マスがどの方向を見ているかの可視化）
if args.fov_grid or args.fov_plane:
    import math as _m
    _W, _H = (int(v) for v in args.cam_res.split(","))
    _lp = np.array([float(v) for v in args.cam_pos.split(",")])
    _fw = np.array([float(v) for v in args.cam_fwd.split(",")]); _fw /= np.linalg.norm(_fw)
    _up = np.array([0.0, 0.0, 1.0])
    _cz = -_fw
    _cx = np.cross(_up, _cz); _cx /= np.linalg.norm(_cx)
    _cy = np.cross(_cz, _cx)
    _R = np.column_stack([_cx, -_cy, _fw])
    _hf = _m.radians(args.cam_hfov)
    _fx = _fy = (_W / 2) / _m.tan(_hf / 2)
    _ccx, _ccy = _W / 2, _H / 2
    _rc = np.array([float(v) for v in args.radius_center.split(",")])
    _Rr = float(args.fov_grid) if args.fov_grid else 0.0

    def _on_sphere(u, v):
        """画素(u,v)の光線と、球 or 平面との交点（torso座標）。無ければ None"""
        d = _R @ np.array([(u - _ccx) / _fx, (v - _ccy) / _fy, 1.0])
        if args.fov_plane is not None:
            # 体に正対する平面（torso の X が一定）。評価に使った点の並びと揃う
            if abs(d[0]) < 1e-9:
                return None
            t = (_rc[0] + float(args.fov_plane) - _lp[0]) / d[0]
            return _lp + d * t if t > 0 else None
        o = _lp - _rc
        aa, bb, cc = float(d @ d), 2.0 * float(o @ d), float(o @ o) - _Rr * _Rr
        disc = bb * bb - 4 * aa * cc
        if disc < 0:
            return None
        t = (-bb + _m.sqrt(disc)) / (2 * aa)
        return _lp + d * t if t > 0 else None

    _pts, _counts = [], []
    _NS = 40                      # 1本の線をこの数で分割（球面なので曲がる）
    for i in range(args.grid_n + 1):                    # 横線
        _v = i * _H / args.grid_n
        seg = [_on_sphere(j * _W / _NS, _v) for j in range(_NS + 1)]
        seg = [x for x in seg if x is not None]
        if len(seg) > 1:
            _pts += seg; _counts.append(len(seg))
    for i in range(args.grid_n + 1):                    # 縦線
        _u = i * _W / args.grid_n
        seg = [_on_sphere(_u, j * _H / _NS) for j in range(_NS + 1)]
        seg = [x for x in seg if x is not None]
        if len(seg) > 1:
            _pts += seg; _counts.append(len(seg))
    # 対象のマスだけ枠を赤で太く描く。正面から見ると左右が反転するので、
    # 「画像の右上」がどこかを線だけで追うのは難しいため。
    if args.highlight_cell:
        _hp, _hc = [], []
        _u0, _u1 = args.col * _W / args.grid_n, (args.col + 1) * _W / args.grid_n
        _v0, _v1 = args.row * _H / args.grid_n, (args.row + 1) * _H / args.grid_n
        for _a, _b, _fixed in ((_u0, _u1, ("v", _v0)), (_u0, _u1, ("v", _v1)),
                               (_v0, _v1, ("u", _u0)), (_v0, _v1, ("u", _u1))):
            seg = []
            for j in range(13):
                t = _a + (_b - _a) * j / 12
                q_ = _on_sphere(t, _fixed[1]) if _fixed[0] == "v" else _on_sphere(_fixed[1], t)
                if q_ is not None:
                    seg.append(q_)
            if len(seg) > 1:
                _hp += seg; _hc.append(len(seg))
        if _hp:
            _hcv = UsdGeom.BasisCurves.Define(stage, f"{torso.GetPath().pathString}/fov_cell")
            _hcv.CreateTypeAttr("linear")
            _hcv.CreateCurveVertexCountsAttr(_hc)
            _hcv.CreatePointsAttr([Gf.Vec3f(*q_) for q_ in _hp])
            _hcv.CreateWidthsAttr([0.012] * len(_hp))
            _hcv.SetWidthsInterpolation("vertex")
            _hcv.CreateDisplayColorAttr([Gf.Vec3f(0.80, 0.10, 0.10)])
            print(f"[show] r{args.row}c{args.col} のマス枠を強調", flush=True)

    if _pts:
        _cv = UsdGeom.BasisCurves.Define(stage, f"{torso.GetPath().pathString}/fov_grid")
        _cv.CreateTypeAttr("linear")
        _cv.CreateCurveVertexCountsAttr(_counts)
        _cv.CreatePointsAttr([Gf.Vec3f(*q) for q in _pts])
        _cv.CreateWidthsAttr([0.004] * len(_pts))
        _cv.SetWidthsInterpolation("vertex")
        _cv.CreateDisplayColorAttr([Gf.Vec3f(0.16, 0.47, 0.84)])
        _w = (f"前方 {args.fov_plane} m の平面" if args.fov_plane is not None
              else f"半径 {_Rr} m の球面")
        print(f"[show] 画角グリッドを{_w}に描画（線 {len(_counts)} 本）", flush=True)

UsdLux.DomeLight.Define(stage, "/show_dome").CreateIntensityAttr(1200.0)
_k = UsdLux.DistantLight.Define(stage, "/show_key")
_k.CreateIntensityAttr(2500.0)
UsdGeom.Xformable(_k.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 0.0, 35.0))

art_path = next(p.GetPath().pathString for p in Usd.PrimRange(root)
                if p.HasAPI(UsdPhysics.ArticulationRootAPI))
art = ArtCls(prim_path=art_path, name="g1")
world.scene.add(art)
world.reset()
art.initialize()
try:
    c = art.get_articulation_controller()
    c.set_gains(kps=np.full(art.num_dof, 1e7), kds=np.full(art.num_dof, 1e5))
except Exception:  # noqa: BLE001
    pass

names = list(art.dof_names)
ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
       "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
q = np.zeros(art.num_dof)
for nm, v in zip(ARM, q_arm):
    q[names.index(nm)] = v

cache = UsdGeom.XformCache()
tw = cache.GetLocalToWorldTransform(torso).ExtractTranslation()
_ce = [float(v) for v in args.cam.split(",")]
_ct = [float(v) for v in args.cam_target.split(",")]
eye = Gf.Vec3d(tw[0] + _ce[0], tw[1] + _ce[1], tw[2] + _ce[2])
tgt = Gf.Vec3d(tw[0] + _ct[0], tw[1] + _ct[1], tw[2] + _ct[2])
if not args.headless:
    try:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=[eye[0], eye[1], eye[2]], target=[tgt[0], tgt[1], tgt[2]])
    except Exception as _e:  # noqa: BLE001
        print(f"[show] 視点設定に失敗: {_e}", flush=True)

if args.video:
    from isaacsim.sensors.camera import Camera
    import PIL.Image
    _mm = Gf.Matrix4d(); _mm.SetLookAt(eye, tgt, Gf.Vec3d(0, 0, 1))
    _qq = _mm.GetInverse().ExtractRotationQuat()
    _cam = Camera(prim_path="/video_cam", resolution=(760, 620))
    _cam.initialize()
    _cam.set_world_pose(np.array([eye[0], eye[1], eye[2]]),
                        np.array([_qq.GetReal(), *_qq.GetImaginary()]), camera_axes="usd")
    _cam.set_clipping_range(0.02, 100.0)
    _q_rest = np.zeros_like(q)
    _cycle = 2.0 * (args.anim_sec + args.hold_sec)
    _nf = max(2, int(_cycle * args.fps))
    print(f"[show] 動画を書き出し: {_nf} フレーム（1周期 {_cycle:.1f} 秒 / {args.fps} fps）", flush=True)
    for _ in range(40):                      # レンダラを暖める（最初の数枚が黒くなるため）
        art.set_joint_positions(_q_rest)
        art.apply_action(ArticulationAction(joint_positions=_q_rest))
        world.step(render=True)
    _frames = []
    for i in range(_nf):
        ph = _cycle * i / _nf
        if ph < args.anim_sec:
            u = ph / args.anim_sec
        elif ph < args.anim_sec + args.hold_sec:
            u = 1.0
        elif ph < 2 * args.anim_sec + args.hold_sec:
            u = 1.0 - (ph - args.anim_sec - args.hold_sec) / args.anim_sec
        else:
            u = 0.0
        u = u * u * (3.0 - 2.0 * u)
        qn = _q_rest + (q - _q_rest) * u
        art.set_joint_positions(qn)
        art.set_joint_velocities(np.zeros_like(qn))
        art.apply_action(ArticulationAction(joint_positions=qn))
        for _ in range(3):
            world.step(render=True)
        img = _cam.get_rgba()
        if img is not None and img.size:
            _im = PIL.Image.fromarray(img[:, :, :3].astype(np.uint8))
            _frames.append(_im.quantize(colors=128, method=PIL.Image.MEDIANCUT))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{_nf} フレーム", flush=True)
    if _frames:
        _frames[0].save(args.video, save_all=True, append_images=_frames[1:],
                        duration=int(1000 / args.fps), loop=0, optimize=True)
        print(f"[show] 動画: {args.video}  （{len(_frames)} フレーム）", flush=True)
    sim_app.close()
    raise SystemExit(0)

import time
t0 = time.time()
q_rest = np.zeros_like(q)          # 基準姿勢（全関節0）
cycle = 2.0 * (args.anim_sec + args.hold_sec)
if args.animate:
    print(f"[show] 動作中: 基準姿勢 → 目標 → 戻る を {cycle:.1f} 秒周期で繰り返し", flush=True)
print(f"[show] 表示中。{args.seconds:.0f} 秒で終了します", flush=True)

while time.time() - t0 < args.seconds:
    if args.animate:
        # 1周期の中の位置から補間率を決める。smoothstep で加減速をつける
        ph = (time.time() - t0) % cycle
        if ph < args.anim_sec:
            u = ph / args.anim_sec
        elif ph < args.anim_sec + args.hold_sec:
            u = 1.0
        elif ph < 2 * args.anim_sec + args.hold_sec:
            u = 1.0 - (ph - args.anim_sec - args.hold_sec) / args.anim_sec
        else:
            u = 0.0
        u = u * u * (3.0 - 2.0 * u)
        q_now = q_rest + (q - q_rest) * u
    else:
        q_now = q
    # 毎ステップ指令し直す（放っておくと駆動が保持しきれず姿勢が崩れる）
    art.set_joint_positions(q_now)
    art.set_joint_velocities(np.zeros_like(q_now))
    art.apply_action(ArticulationAction(joint_positions=q_now))
    world.step(render=True)
    if args.shot and time.time() - t0 > 2.0:
        from isaacsim.sensors.camera import Camera
        mm = Gf.Matrix4d(); mm.SetLookAt(eye, tgt, Gf.Vec3d(0, 0, 1))
        qq = mm.GetInverse().ExtractRotationQuat()
        cam = Camera(prim_path="/show_cam", resolution=(1100, 900))
        cam.initialize()
        cam.set_world_pose(np.array([eye[0], eye[1], eye[2]]),
                           np.array([qq.GetReal(), *qq.GetImaginary()]), camera_axes="usd")
        cam.set_clipping_range(0.02, 100.0)
        for _ in range(45):
            art.set_joint_positions(q)
            art.apply_action(ArticulationAction(joint_positions=q))  # スクショは到達姿勢で
            world.step(render=True)
        img = cam.get_rgba()
        if img is not None and img.size:
            import PIL.Image
            PIL.Image.fromarray(img[:, :, :3].astype(np.uint8)).save(args.shot)
            print(f"[show] スクショ: {args.shot}", flush=True)
        break
sim_app.close()
