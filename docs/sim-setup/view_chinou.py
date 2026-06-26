#!/usr/bin/env python3
"""chinou_center.usd（FARO実測室・正しいスケール）を Isaac Sim にロードし、
床コライダー(z=0) と G1 を置いてスケールを目視確認する確認用ローダー。

  open_stage(chinou_center.usd)  → /World/ChinouCenter（~8.2×8.4×3.2m, 床z=0）
  → default ground plane (z=0) を追加（G1 が立つ床コライダー）
  → G1 を /G1 に配置（fix_base, 足を床へ lift）
  → スクショ保存（G1 1.27m が 3.22m 天井の室内に正しい比率で立つか）

実行:
  cd <repo>
  PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES \
    ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/view_chinou.py [--gui]
"""
from __future__ import annotations
import argparse, os
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
ROOM_USD = f"{REPO}/usd_file/chinou_center.usd"
# 収穫構成モデル = 左手首にバスケット + 右手 Dex1（g1bag.usd）。
# 旧 g1_29dof_with_dex1_base_fix1.usd は両手 Dex1・バスケット無しなので使わない。
G1_USD = f"{REPO}/usd_file/g1-29dof-dex1-base-fix-usd/g1bag.usd"
OKRA_USD = f"{REPO}/usd_file/okra.usd"
OUT_DIR = f"{REPO}/docs/sim-setup"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--gravity", action="store_true",
                    help="通常重力で動力学を見る（既定は重力OFFで初期姿勢固定表示）")
    # 天井照明（研究室のシーリングライト再現）。SphereLight をグリッド配置。
    ap.add_argument("--ceil-lights", type=int, default=3,
                    help="天井 SphereLight のグリッド数 NxN（例 3 で 3x3=9灯）。0 で無効")
    ap.add_argument("--ceil-intensity", type=float, default=40000.0,
                    help="各天井ライトの intensity（暗ければ上げる。距離2乗で減衰）")
    ap.add_argument("--ceil-radius", type=float, default=0.15,
                    help="各天井ライトの半径[m]（大きいほど影が柔らかい）")
    ap.add_argument("--okra", type=int, default=3,
                    help="オクラ本数（--table 時は机上に格子配置、無指定時は右手前の把持圏）")
    ap.add_argument("--table", action="store_true",
                    help="ロボット前方に机を置き、机上に --okra 本のオクラを少し浮かせて配置（重力自動ON）")
    ap.add_argument("--table-h", type=float, default=0.72,
                    help="机の天板高さ[m]（G1がしゃがまず届く高さ。既定0.72）")
    args = ap.parse_args()

    from isaacsim import SimulationApp
    sim_app = SimulationApp({"headless": not args.gui})

    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.utils.stage import open_stage
    from isaacsim.core.utils.viewports import set_camera_view
    import omni.usd
    try:
        from isaacsim.core.prims import SingleArticulation as ArtCls
    except Exception:
        from isaacsim.core.api.articulations import Articulation as ArtCls  # type: ignore

    print(f"[chinou] open_stage: {ROOM_USD}", flush=True)
    open_stage(ROOM_USD)
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # 確認用ビューアなので重力 OFF。PD 剛性が弱く腕が垂れ下がる（→ 手首に付いた
    # バスケットごとズレる）のを防ぎ、腕・バスケットを初期姿勢で静止させる。
    # 動歩行や把持の動力学を見たい時は --gravity を付けて通常重力に戻す。
    if not (args.gravity or args.table):
        try:
            world.get_physics_context().set_gravity(0.0)  # [m/s^2]
            print("[chinou] gravity OFF（腕・バスケットを初期姿勢で固定表示）", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[chinou] set_gravity failed: {e}", flush=True)
    elif args.table:
        print("[chinou] gravity ON（--table: オクラを机に落とすため）", flush=True)

    # 室メッシュの実寸を確認（スケール検証の主目的）
    bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    room_prim = stage.GetPrimAtPath("/World/ChinouCenter") or stage.GetPrimAtPath("/World")
    rng = bbc.ComputeWorldBound(room_prim).ComputeAlignedRange()
    rmn, rmx = rng.GetMin(), rng.GetMax()
    rsz = tuple(round(rmx[i] - rmn[i], 3) for i in range(3))
    print(f"[chinou] ROOM world size (m) = {rsz}  min={tuple(round(v,2) for v in rmn)} max={tuple(round(v,2) for v in rmx)}", flush=True)

    # --- 天井照明: 研究室のシーリングライトを SphereLight グリッドで再現 ---
    # SphereLight は位置を持つ点光源（距離2乗で減衰）。天井(z=rmx[2])の少し下に
    # 壁際を避けて格子状に並べる。数/強度/半径は CLI で調整可。
    if args.ceil_lights > 0:
        from pxr import UsdLux
        n = args.ceil_lights
        cz = float(rmx[2]) - 0.15  # 天井から 15cm 下 [m]

        def _grid(a, b, k, inset=0.7):
            c = 0.5 * (a + b)
            h = 0.5 * (b - a) * inset  # 内側 70% に収めて壁から離す
            return [c] if k == 1 else [c - h + 2 * h * i / (k - 1) for i in range(k)]

        xs = _grid(float(rmn[0]), float(rmx[0]), n)
        ys = _grid(float(rmn[1]), float(rmx[1]), n)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                lt = UsdLux.SphereLight.Define(stage, f"/World/CeilingLights/sphere_{i}_{j}")
                lt.CreateRadiusAttr(float(args.ceil_radius))      # 光源半径 [m]
                lt.CreateIntensityAttr(float(args.ceil_intensity))
                lt.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.92))     # わずかに電球色
                UsdGeom.XformCommonAPI(lt.GetPrim()).SetTranslate(Gf.Vec3d(x, y, cz))
        print(f"[chinou] ceiling SphereLight {n}x{n}={n*n}灯 @z={cz:.2f}m "
              f"intensity={args.ceil_intensity} radius={args.ceil_radius}", flush=True)

    # 床/壁コライダーは chinou_center.usd に焼き込み済み（/World/Colliders, 不可視 static box）。
    # 念のため安全網のグリッド地面を z=0 に追加したい場合は下行を有効化:
    # world.scene.add_default_ground_plane(z_position=0.0)

    # G1 配置
    from isaacsim.core.utils.stage import add_reference_to_stage
    add_reference_to_stage(usd_path=G1_USD, prim_path="/G1")
    g1_prim = stage.GetPrimAtPath("/G1")
    min_z = float(bbc.ComputeWorldBound(g1_prim).ComputeAlignedRange().GetMin()[2])
    lift = -min_z
    UsdGeom.XformCommonAPI(g1_prim).SetTranslate(Gf.Vec3d(0.0, 0.0, lift))
    print(f"[chinou] G1 lift +{lift:.3f} m (room ceiling {rsz[2]} m, G1 ~1.27 m)", flush=True)

    art_root = None
    for prim in Usd.PrimRange(g1_prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_root = prim.GetPath().pathString
            break
    robot = ArtCls(prim_path=art_root or "/G1", name="g1")
    world.scene.add(robot)

    if args.table:
        # 机（天板＋台座）をロボット前方に配置。G1 がしゃがまず届く高さ（既定 0.72m）。
        TH = args.table_h        # 天板上面 [m]
        TCX = 0.50               # 机中心 x（前方）
        table_mat = PhysicsMaterial("/World/PM/table", static_friction=0.8,
                                    dynamic_friction=0.7, restitution=0.0)
        FixedCuboid(prim_path="/World/Table_top", name="table_top",
                    position=np.array([TCX, 0.0, TH - 0.02]), scale=np.array([0.65, 0.55, 0.04]),
                    color=np.array([0.55, 0.40, 0.25]), physics_material=table_mat)
        _bh = TH - 0.04          # 台座高（床→天板下）
        FixedCuboid(prim_path="/World/Table_base", name="table_base",
                    position=np.array([TCX, 0.0, _bh / 2.0]), scale=np.array([0.38, 0.32, _bh]),
                    color=np.array([0.45, 0.32, 0.20]), physics_material=table_mat)
        print(f"[chinou] 机を配置（天板高 {TH:.2f}m, 中心x={TCX}）", flush=True)

        # 机上にオクラを「直立（先端上）」で配置し、机に SINK だけめり込ませて socket 風に。
        # 各オクラを天板へ破断可能 FixedJoint で固定（上向き維持）。joint の collision 無効で
        # めり込みは押し出されない。ハンドで掴んで引く（breakForce 超）と外れて収穫できる。
        if args.okra > 0:
            cols = 5
            xs = np.linspace(TCX - 0.16, TCX + 0.12, cols)        # 手前寄り＝届きやすい
            rows = (args.okra + cols - 1) // cols
            ys = np.linspace(-0.15, 0.15, rows) if rows > 1 else np.array([0.0])
            zc = TH + 0.05               # オクラ中心 z（長さ10cm → base=TH=天板表面に直立）
            FLT_MAX = 3.4028234663852886e+38
            k = 0
            for r in range(rows):
                for c in range(cols):
                    if k >= args.okra:
                        break
                    xi, yi = float(xs[c]), float(ys[r])
                    pth = f"/Okra_{k}"
                    add_reference_to_stage(usd_path=OKRA_USD, prim_path=pth)
                    # 直立: 先端(+Y)を上(rotX+90°)→鉛直まわりに少し振る（順序重要: 倒れ/逆さ防止）
                    qd = (Gf.Rotation(Gf.Vec3d(1, 0, 0), 90.0)
                          * Gf.Rotation(Gf.Vec3d(0, 0, 1), float((k * 37) % 360))).GetQuat()
                    qf = Gf.Quatf(qd.GetReal(), Gf.Vec3f(*qd.GetImaginary()))
                    op = UsdGeom.Xformable(stage.GetPrimAtPath(pth))
                    op.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(xi, yi, zc))
                    op.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(qd)
                    # 破断可能 FixedJoint で直立保持。body0 未設定 = world アンカー（机アンカーは
                    # 剛性不足で倒れた）。collision 無効で接触インパルス破断を回避。
                    # ハンドで掴んで引く（breakForce 超）と外れて収穫できる。
                    j = UsdPhysics.FixedJoint.Define(stage, f"/World/OkraJoints/joint_{k}")
                    j.CreateBody1Rel().SetTargets([pth])
                    j.CreateLocalPos0Attr(Gf.Vec3f(xi, yi, zc))   # world 座標アンカー
                    j.CreateLocalRot0Attr(qf)
                    j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
                    j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
                    j.CreateBreakForceAttr(8.0)        # 引張 8N 超で外れる＝収穫（自重0.12Nでは外れない）
                    j.CreateBreakTorqueAttr(FLT_MAX)
                    j.CreateCollisionEnabledAttr(False)
                    k += 1
            print(f"[chinou] okra x{k} を机上に直立固定（world剛ジョイント）。引張8Nで収穫可", flush=True)

    elif args.okra > 0:
        # 机なし: 右手 Dex1（world ~0.28,-0.15,0.91）前方の把持圏に縦向きで配置。
        OKRA_POS = [
            (0.40, -0.15, 0.95), (0.43, -0.09, 0.86), (0.38, -0.21, 0.90),
            (0.46, -0.16, 1.00), (0.41, -0.12, 0.82),
        ]
        no = min(args.okra, len(OKRA_POS))
        for i in range(no):
            pth = f"/Okra_{i}"
            add_reference_to_stage(usd_path=OKRA_USD, prim_path=pth)
            api = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(pth))
            api.SetTranslate(Gf.Vec3d(*OKRA_POS[i]))
            api.SetRotate(Gf.Vec3f(-90.0, 0.0, float(i * 30)))
        print(f"[chinou] okra x{no} を右手前の把持圏に配置", flush=True)

    # カメラ: --table は机まわりを寄りで、通常は室内俯瞰
    if args.table:
        set_camera_view(eye=np.array([1.35, -1.05, 1.00]), target=np.array([0.45, 0.0, 0.78]))
    else:
        set_camera_view(eye=np.array([6.0, -6.0, 3.0]), target=np.array([0.0, 0.0, 0.8]))

    world.reset()
    try:
        robot.set_world_pose(position=np.array([0.0, 0.0, lift]), orientation=np.array([1.0, 0.0, 0.0, 0.0]))
    except Exception as e:
        print(f"[chinou] set_world_pose failed: {e}", flush=True)
    for _ in range(args.steps):
        world.step(render=True)

    try:
        bp, _ = robot.get_world_pose()
        print(f"[chinou] G1 base pos = {np.asarray(bp)} (no divergence if z stable)", flush=True)
    except Exception:
        pass

    import omni.kit.viewport.utility as vpu
    for _ in range(8):
        world.step(render=True)
    vp = vpu.get_active_viewport()
    out = f"{OUT_DIR}/chinou_scale_check.png"
    vpu.capture_viewport_to_file(vp, out)
    for _ in range(12):
        sim_app.update()
    print(f"[chinou] screenshot -> {out}", flush=True)
    print("[chinou] DONE", flush=True)

    if args.gui:
        # GUI 保持。fix_base で静止なので物理ステップは不要 → sim_app.update() だけ回し
        # 入力/描画を最優先にして視点操作を軽快にする（world.step だと重く反応が鈍い）。
        print(
            "[chinou] GUI 保持中。視点操作(Omniverse): "
            "右ドラッグ=見回す / Alt+左ドラッグ=旋回 / 中ドラッグ=平行移動 / "
            "ホイール=ズーム / 物体クリック後 F=フォーカス。"
            "※まずビューポートを1回クリックしてフォーカスを当てること。ウィンドウを閉じると終了。",
            flush=True,
        )
        while sim_app.is_running():
            sim_app.update()
    sim_app.close()


if __name__ == "__main__":
    main()
