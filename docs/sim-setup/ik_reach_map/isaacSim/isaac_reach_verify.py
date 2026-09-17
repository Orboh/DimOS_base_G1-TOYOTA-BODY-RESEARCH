#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""段階2（Isaac Sim, headless）: IK解を実際の G1 モデルに入れて幾何と自己干渉を確かめる。

段階1（`zed_fov_reach_map.py`, dimos .venv + pinocchio）が出した `reach_map.csv` の
右腕7関節角 q0..q6 を読み、Isaac Sim 上の G1（USD）に **直接設定**して:

  1. 指先が本当に目標 torso 座標へ行くか  … URDF(pinocchio) と USD(Isaac) の整合チェック
  2. そのとき体のどこかに当たっていないか … 自己干渉（段階1では原理的に判定できない）

関節は位置制御ではなく `set_joint_positions` で直接入れる。位置制御だと PD の追従誤差
（実測 0.15 rad）が幾何誤差に混ざってしまうため。

環境: env_isaaclab_2（isaacsim 5.1.0）。※ pinocchio 側の .venv とは別環境なので2段構成。
実行（段階1を先に走らせて reach_map.csv を作っておくこと）:
  cd /isaac-sim/workspace/DimOS_base_G1-TOYOTA-BODY-
  conda run -n env_isaaclab_2 python /isaac-sim/workspace/02_orita_tool/IK_doc/isaacSim/isaac_reach_verify.py
"""
from __future__ import annotations

import argparse
import csv
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE1 = os.path.dirname(HERE)   # 段階1の出力（reach_map.csv 等）は1つ上にある
REPO = os.environ.get("DIMOS_REPO", "/isaac-sim/workspace/DimOS_base_G1-TOYOTA-BODY-")
G1_USD = f"{REPO}/usd_file/g1-29dof-dex1-base-fix-usd/g1_29dof_with_dex1_base_fix1.usd"

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=os.path.join(STAGE1, "reach_map.csv"))
ap.add_argument("--out", default=os.path.join(HERE, "isaac_verify.csv"))
ap.add_argument("--limit", type=int, default=0, help="先頭N点だけ試す（0=全部）")
ap.add_argument("--gui", action="store_true")
ap.add_argument("--lift", type=float, default=None,
                help="G1 をこの高さ[m]に持ち上げる（原点が骨盤なので地面に置くと脚が埋まる）。"
                     "既定は GUI のとき 0.8 m、headless では持ち上げない")
ap.add_argument("--pause", type=float, default=0.0,
                help="各点でこの秒数だけ描画し続ける（GUI で様子を見る用）")
ap.add_argument("--settle", type=int, default=3, help="姿勢を入れてから静定させる物理ステップ数")
ap.add_argument("--stiff", type=float, default=1e7, help="位置保持の剛性。低いと指令姿勢が崩れる")
ap.add_argument("--selftest", action="store_true",
                help="接触検出が本当に機能しているかを、わざと干渉させて確認する")
# ik_approach.py の既定と同じ: wrist_yaw リンク座標系での指先オフセット
ap.add_argument("--tip-offset", default="0.1845,-0.003,0.0")
# 胸に付けたポケット（半円柱）。腕がこの空間に入る点も落とす
ap.add_argument("--pocket", action="store_true", help="ポケットとの当たりも判定する")
ap.add_argument("--pocket-origin", default="0.0709,0,0.0770",
                help="ポケット原点の torso 相対 xyz[m]（mount_pocket.py が出す値）")
ap.add_argument("--pocket-size", default="0.080,0.085,0.085",
                help="奥行き,左右半径,深さ[m]")
ap.add_argument("--pocket-samples", type=int, default=260, help="腕リンクあたりの判定点数")
ap.add_argument("--pocket-stl", default="/isaac-sim/workspace/sub/四次元ポケット_CAD/pocket_halfcyl.stl",
                help="表示用のポケット形状。判定自体は解析的に行うので見た目だけの用途")
args = ap.parse_args()

from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": not args.gui})

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema, PhysicsSchemaTools
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
import omni.usd
from omni.physx import get_physx_simulation_interface

try:
    from isaacsim.core.prims import SingleArticulation as ArtCls
except Exception:  # noqa: BLE001
    from isaacsim.core.api.articulations import Articulation as ArtCls

TIP_OFF = Gf.Vec3d(*[float(x) for x in args.tip_offset.split(",")])


def _actor_path(a):
    """contact report の actor（整数ID）を prim パス文字列にする。"""
    try:
        return str(PhysicsSchemaTools.intToSdfPath(a))
    except Exception:  # noqa: BLE001
        return str(a)

world = World(stage_units_in_meters=1.0)
world.get_physics_context().set_gravity(0.0)   # 幾何＋干渉だけ見る（重力たわみは誤差要因）
world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path=G1_USD, prim_path="/G1")

# G1 の原点は骨盤。地面(z=0)に置くと脚が地面にめり込んで見えるので、見る用に持ち上げる。
# 骨盤固定・重力0で評価しているので、判定結果には影響しない（すべて torso 相対で計算）。
_lift = args.lift if args.lift is not None else (0.8 if args.gui else 0.0)
if _lift:
    _g1 = omni.usd.get_context().get_stage().GetPrimAtPath("/G1")
    _xf = UsdGeom.Xformable(_g1)
    _xf.ClearXformOpOrder()
    _xf.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, float(_lift)))
    print(f"[isaac] G1 を z={_lift} m に持ち上げ（表示用。判定は torso 相対なので影響なし）", flush=True)
stage = omni.usd.get_context().get_stage()
root = stage.GetPrimAtPath("/G1")

art_path = None
for p in Usd.PrimRange(root):
    if p.HasAPI(UsdPhysics.ArticulationRootAPI):
        art_path = p.GetPath().pathString
        break

# --- 自己干渉を見るための下ごしらえ ------------------------------------------
ap_prim = stage.GetPrimAtPath(art_path)
PhysxSchema.PhysxArticulationAPI.Apply(ap_prim).GetEnabledSelfCollisionsAttr().Set(True)
print(f"[isaac] self-collision 有効化: {art_path}", flush=True)

bodies = [p.GetPath().pathString for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
for b in bodies:
    PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(b)).GetThresholdAttr().Set(0.0)
print(f"[isaac] contact report を {len(bodies)} リンクに適用", flush=True)

# 関節でつながっている親子ペア＝常に接触しうるので自己干渉から除外する
adjacent = set()
for p in Usd.PrimRange(root):
    if not p.IsA(UsdPhysics.Joint):
        continue
    j = UsdPhysics.Joint(p)
    b0 = [str(t) for t in j.GetBody0Rel().GetTargets()]
    b1 = [str(t) for t in j.GetBody1Rel().GetTargets()]
    for x in b0:
        for y in b1:
            adjacent.add(frozenset((x, y)))
print(f"[isaac] 隣接ペア {len(adjacent)} 組を自己干渉判定から除外", flush=True)

art = ArtCls(prim_path=art_path, name="g1")
world.scene.add(art)
world.reset()
art.initialize()

try:
    _ctrl = art.get_articulation_controller()
    _n = art.num_dof
    _ctrl.set_gains(kps=np.full(_n, args.stiff), kds=np.full(_n, args.stiff * 1e-2))
    print(f"[isaac] 位置保持ゲイン kp={args.stiff:g} を全{_n}DOFに設定", flush=True)
except Exception as _e:  # noqa: BLE001
    print(f"[isaac] ゲイン設定に失敗: {_e}", flush=True)

if args.gui:
    try:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=[1.1, -0.9, 1.1], target=[0.1, 0.0, 0.75])
        print("[isaac] GUI 視点を胸まわりに設定", flush=True)
    except Exception as _e:  # noqa: BLE001
        print(f"[isaac] 視点設定に失敗: {_e}", flush=True)

dof_names = list(art.dof_names)
arm_order = [l.strip() for l in open(os.path.join(STAGE1, "right_arm_joint_order.txt")) if l.strip()]
# ★ Isaac の DOF 順は正準順と違う（実測: index22 が肘）ので必ず名前で対応付ける
arm_idx = [dof_names.index(n) for n in arm_order]
print(f"[isaac] 右腕 DOF index（名前で対応付け）= {arm_idx}", flush=True)

# --- URDF(pinocchio) と USD(Isaac) の関節リミットを突き合わせる ---------------
# 食い違っていると「段階1でOKなのに Isaac では入らない」点が出る。
def _usd_limits(jname):
    """USD の RevoluteJoint リミット[rad]。PhysX が実際に使う値はこちら（属性は度）。"""
    import math as _m
    for _p in Usd.PrimRange(root):
        if _p.GetName() == jname and _p.IsA(UsdPhysics.RevoluteJoint):
            _j = UsdPhysics.RevoluteJoint(_p)
            _lo, _up = _j.GetLowerLimitAttr().Get(), _j.GetUpperLimitAttr().Get()
            if _lo is None or _up is None:
                return float("nan"), float("nan")
            return _m.radians(float(_lo)), _m.radians(float(_up))
    return float("nan"), float("nan")

try:
    lim = {n: _usd_limits(n) for n in arm_order}
    ref = {r["joint"]: (float(r["urdf_lower_rad"]), float(r["urdf_upper_rad"]))
           for r in csv.DictReader(open(os.path.join(STAGE1, "right_arm_limits.csv")))}
    print(f"\n[isaac] 関節リミット比較（URDF=段階1が使った値 / USD=Isaacが実際に使う値）", flush=True)
    print(f"  {'joint':<28} {'URDF lo':>9} {'URDF up':>9} | {'USD lo':>9} {'USD up':>9}  {'差':>6}", flush=True)
    for nm, idx in zip(arm_order, arm_idx):
        ul, uu = ref.get(nm, (float("nan"),) * 2)
        sl, su = lim[nm]
        gap = max(abs(ul - sl), abs(uu - su))
        print(f"  {nm:<28} {ul:>9.4f} {uu:>9.4f} | {sl:>9.4f} {su:>9.4f}  "
              f"{gap:>6.3f}{'  ★差あり' if gap > 1e-3 else ''}", flush=True)
except Exception as _e:  # noqa: BLE001
    print(f"[isaac] リミット比較スキップ: {_e}", flush=True)

torso_prim = None
for p in Usd.PrimRange(root):
    if p.GetName() == "torso_link":
        torso_prim = p
        break
wrist_prim = stage.GetPrimAtPath(f"{root.GetPath().pathString}/right_wrist_yaw_link")
print(f"[isaac] torso={torso_prim.GetPath()}  wrist={wrist_prim.GetPath()}", flush=True)

def _probe(qmap, label):
    q = np.zeros(art.num_dof)
    for nm, v in qmap.items():
        q[dof_names.index(nm)] = v
    art.set_joint_positions(q)
    art.set_joint_velocities(np.zeros_like(q))
    art.apply_action(ArticulationAction(joint_positions=q))
    for _ in range(3):
        world.step(render=False)
    get_physx_simulation_interface().get_contact_report()
    world.step(render=False)
    hs, _ = get_physx_simulation_interface().get_contact_report()
    pairs = set()
    for h in hs:
        a, b = _actor_path(h.actor0), _actor_path(h.actor1)
        if "/G1" in a and "/G1" in b and a != b and frozenset((a, b)) not in adjacent:
            pairs.add("|".join(sorted((a.split("/")[-1], b.split("/")[-1]))))
    print(f"  {label:<38} 接触 {len(pairs):2d} 組  {sorted(pairs)[:3]}", flush=True)
    return len(pairs)

if args.selftest:
    print("\n[selftest] わざと干渉させて接触検出が機能するか確認", flush=True)
    tot = 0
    tot += _probe({"right_shoulder_roll_joint": -2.2, "right_elbow_joint": 2.0}, "肩roll最大内側 + 肘全曲げ")
    tot += _probe({"right_shoulder_roll_joint": 1.55, "right_elbow_joint": 2.0}, "肩roll最大外側 + 肘全曲げ")
    tot += _probe({"right_shoulder_pitch_joint": 1.5, "right_shoulder_roll_joint": -2.2}, "腕を体の前へ回し込む")
    tot += _probe({"right_shoulder_pitch_joint": -3.0, "right_elbow_joint": 2.0}, "腕を真後ろへ + 肘全曲げ")
    tot += _probe({"left_shoulder_roll_joint": 2.2, "right_shoulder_roll_joint": -2.2,
                   "left_elbow_joint": 2.0, "right_elbow_joint": 2.0}, "両腕を体の中心で交差")
    print(f"[selftest] 合計 {tot} 組の接触を検出 → "
          f"{'接触検出は機能している' if tot > 0 else '★検出されない＝自己干渉判定は信用できない'}", flush=True)
    sim_app.close()
    raise SystemExit(0)

RIGHT_ARM_LINKS = ("right_shoulder", "right_elbow", "right_wrist", "right_hand", "right_rubber")


def _is_right_arm(link: str) -> bool:
    return link.startswith(RIGHT_ARM_LINKS)


def _contacts_now():
    """(右腕が関与する接触ペア集合, 最大めり込み深さ[m]) を返す。"""
    hs, data = get_physx_simulation_interface().get_contact_report()
    out, deepest = set(), 0.0
    for h in hs:
        a, b = _actor_path(h.actor0), _actor_path(h.actor1)
        if "/G1" not in a or "/G1" not in b or a == b:
            continue
        if frozenset((a, b)) in adjacent:
            continue
        la, lb = a.split("/")[-1], b.split("/")[-1]
        if not (_is_right_arm(la) or _is_right_arm(lb)):
            continue          # 右腕が関与しない接触は本検証の対象外
        out.add("|".join(sorted((la, lb))))
        try:
            for i in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
                deepest = max(deepest, -float(data[i].separation))   # separation<0 = めり込み
        except Exception:  # noqa: BLE001
            pass
    return out, deepest


q_zero = np.zeros(art.num_dof)
art.set_joint_positions(q_zero)
art.set_joint_velocities(q_zero)
art.apply_action(ArticulationAction(joint_positions=q_zero))
for _ in range(5):
    world.step(render=False)
_contacts_now()
world.step(render=False)
BASELINE, _ = _contacts_now()
print(f"[isaac] ホーム姿勢でも接触している {len(BASELINE)} 組をベースラインとして除外", flush=True)
for b in sorted(BASELINE)[:6]:
    print(f"          {b}", flush=True)

# --- ポケットの占める空間と、腕リンクの判定点を用意する -----------------------
POCK = None
if args.pocket:
    _po = np.array([float(v) for v in args.pocket_origin.split(",")], float)
    _pd, _pry, _prz = (float(v) for v in args.pocket_size.split(","))

    def in_pocket(P_t):
        """torso 座標の点群がポケットの外形ボリュームに入っているか。"""
        x, y, z = P_t[:, 0], P_t[:, 1], P_t[:, 2]
        dz = z - _po[2]
        return ((x >= _po[0]) & (x <= _po[0] + _pd) & (dz <= 0.0) &
                (((y - _po[1]) / _pry) ** 2 + (dz / _prz) ** 2 <= 1.0))

    arm_meshes = []          # (prim, ローカル頂点[N,3])
    for lk in Usd.PrimRange(root):
        if not lk.GetName().startswith(RIGHT_ARM_LINKS):
            continue
        for p in Usd.PrimRange(lk):
            if not p.IsA(UsdGeom.Mesh):
                continue
            v = UsdGeom.Mesh(p).GetPointsAttr().Get()
            if not v:
                continue
            a = np.array([[q[0], q[1], q[2]] for q in v], dtype=float)
            if len(a) > args.pocket_samples:      # 全頂点は重いので間引く
                a = a[np.linspace(0, len(a) - 1, args.pocket_samples).astype(int)]
            arm_meshes.append((p, a))
    # 画面で見るときにポケットが無いと様子が分からないので、表示用メッシュを置く
    # （当たり判定は上の in_pocket で解析的に行うので、衝突形状は付けない）
    if os.path.exists(args.pocket_stl):
        import struct as _st
        _d = open(args.pocket_stl, "rb").read()
        _n = _st.unpack("<I", _d[80:84])[0]
        if 84 + 50 * _n == len(_d):
            _a = np.frombuffer(_d[84:], dtype=np.uint8).reshape(_n, 50)
            _tri = _a[:, 12:48].copy().view("<f4").reshape(_n, 3, 3).astype(np.float64) * 0.001
            _V = _tri.reshape(-1, 3) + _po
            _mp = f"{torso_prim.GetPath().pathString}/pocket_view"
            _mesh = UsdGeom.Mesh.Define(stage, _mp)
            _mesh.CreatePointsAttr([Gf.Vec3f(*q) for q in _V])
            _mesh.CreateFaceVertexCountsAttr([3] * _n)
            _mesh.CreateFaceVertexIndicesAttr(list(range(len(_V))))
            _mesh.CreateDisplayColorAttr([Gf.Vec3f(0.85, 0.55, 0.10)])
            _mesh.CreateSubdivisionSchemeAttr("none")
            print(f"[isaac] ポケットを表示用に配置（当たり判定は解析的に実施）", flush=True)

    POCK = (in_pocket, arm_meshes)
    print(f"[isaac] ポケット判定: 原点 {_po} / 奥行き{_pd} 半径{_pry} 深さ{_prz} m、"
          f"右腕メッシュ {len(arm_meshes)} 個 × 最大{args.pocket_samples}点", flush=True)


def pocket_hit_count(cache_):
    """いま腕がポケットの空間に入れている点の数。"""
    if POCK is None:
        return 0
    in_pocket, meshes = POCK
    M_t_inv = cache_.GetLocalToWorldTransform(torso_prim).GetInverse()
    n = 0
    for pr, loc in meshes:
        M = cache_.GetLocalToWorldTransform(pr) * M_t_inv
        m = np.array(M, dtype=float)          # 行ベクトル規約: p' = p @ M
        P_t = loc @ m[:3, :3] + m[3, :3]
        n += int(in_pocket(P_t).sum())
    return n


rows = [r for r in csv.DictReader(open(args.csv)) if r["reach_ok"] == "1"]
if args.limit:
    rows = rows[: args.limit]
print(f"[isaac] 検証対象 {len(rows)} 点\n", flush=True)

q_home = np.array(art.get_joint_positions(), dtype=float) * 0.0
out = []
cache = UsdGeom.XformCache()

for i, r in enumerate(rows):
    q = q_home.copy()
    for k, idx in enumerate(arm_idx):
        q[idx] = float(r[f"q{k}"])
    art.set_joint_positions(q)
    art.set_joint_velocities(np.zeros_like(q))
    art.apply_action(ArticulationAction(joint_positions=q))  # ★drive目標も揃える
    for _ in range(args.settle):
        world.step(render=False)
    get_physx_simulation_interface().get_contact_report()    # 静定後に前回分を捨てる
    world.step(render=False)

    cache.Clear()
    M_t = cache.GetLocalToWorldTransform(torso_prim)
    M_w = cache.GetLocalToWorldTransform(wrist_prim)
    tip_w = M_w.Transform(TIP_OFF)                       # 指先 world
    tip_t = M_t.GetInverse().Transform(tip_w)            # → torso 相対
    tgt = np.array([float(r["tx"]), float(r["ty"]), float(r["tz"])])
    got = np.array([tip_t[0], tip_t[1], tip_t[2]])
    d = float(np.linalg.norm(got - tgt))

    # 実際に入った関節角（リミットや干渉で押し戻されていれば指令と食い違う）
    q_act = np.array(art.get_joint_positions(), dtype=float)
    q_dev = float(np.max(np.abs(q_act[arm_idx] - q[arm_idx])))

    if args.pause > 0:      # GUI で見えるように、その姿勢のまま少し回す
        # ★毎ステップ指令し直す。放っておくと駆動が保持しきれず姿勢が崩れていく
        import time as _t
        _t0 = _t.time()
        while _t.time() - _t0 < args.pause:
            art.set_joint_positions(q)
            art.set_joint_velocities(np.zeros_like(q))
            art.apply_action(ArticulationAction(joint_positions=q))
            world.step(render=True)
    _now, _depth = _contacts_now()
    hits = sorted(_now - BASELINE)      # その姿勢で新たに生じた接触＝自己干渉
    n_pk = pocket_hit_count(cache)

    out.append(dict(depth=r["depth"], row=r["row"], col=r["col"],
                    tx=r["tx"], ty=r["ty"], tz=r["tz"],
                    ik_err_m=r["err_m"],
                    isaac_tip_x=round(got[0], 4), isaac_tip_y=round(got[1], 4),
                    isaac_tip_z=round(got[2], 4),
                    isaac_vs_target_m=round(d, 4),
                    joint_deviation_rad=round(q_dev, 5),
                    self_collision=int(bool(hits)),
                    penetration_m=round(_depth, 5),
                    pocket_hit=int(n_pk > 0), pocket_pts=n_pk,
                    collision_pairs=";".join(hits)))
    if args.pause > 0:
        print(f"  [{i+1}/{len(rows)}] r{r['row']}c{r['col']} @{r['depth']}m  "
              f"{'ポケット接触' if n_pk else ('自己干渉' if hits else 'OK')}", flush=True)
    if (i + 1) % 40 == 0:
        print(f"  {i+1}/{len(rows)} 点", flush=True)

with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)

dd = np.array([o["isaac_vs_target_m"] for o in out])
qq = np.array([o["joint_deviation_rad"] for o in out])
ncol = sum(o["self_collision"] for o in out)
print(f"\n{'='*66}", flush=True)
print(f"検証 {len(out)} 点（段階1で到達と判定された点）")
print(f"{'='*66}")
print(f"指先と目標の距離   : 中央値 {np.median(dd)*1000:.1f} mm / 最大 {dd.max()*1000:.1f} mm")
print(f"  5mm以内 {int((dd<0.005).sum())}点 / 10mm以内 {int((dd<0.01).sum())}点 / 50mm超 {int((dd>0.05).sum())}点")
print(f"関節角の押し戻し   : 最大 {qq.max():.5f} rad（0に近いほど指令通りの姿勢が取れている）")
print(f"自己干渉あり       : {ncol}/{len(out)} 点", flush=True)
_pen = np.array([o["penetration_m"] for o in out if o["self_collision"]])
if _pen.size:
    print(f"  めり込み深さ     : 中央値 {np.median(_pen)*1000:.1f} mm / 最大 {_pen.max()*1000:.1f} mm", flush=True)
    print(f"  1mm未満（かすり程度）{int((_pen<0.001).sum())}点 / "
          f"5mm以上（明確なめり込み）{int((_pen>=0.005).sum())}点", flush=True)
if ncol:
    from collections import Counter
    c = Counter(p for o in out if o["collision_pairs"] for p in o["collision_pairs"].split(";"))
    print("  多い接触ペア:")
    for k, v in c.most_common(8):
        print(f"    {v:4d}  {k}")
if args.pocket:
    _ph = np.array([o["pocket_hit"] for o in out], bool)
    print(f"ポケットに当たる     : {int(_ph.sum())}/{len(out)} 点", flush=True)
    _clean = (~_ph) & np.array([not o["self_collision"] for o in out]) & (dd < 0.005)
    print(f"すべて良好           : {int(_clean.sum())}/{len(out)} 点"
          f"（指先誤差<5mm かつ 自己干渉なし かつ ポケットに当たらない）", flush=True)

print(f"\n出力: {args.out}", flush=True)
sim_app.close()
