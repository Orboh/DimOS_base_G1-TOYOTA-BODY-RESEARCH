"""sim シーン部品の共有ビルダー（isaac-sim env で使用）。

机上オクラ「A 配置」を単一ソース化したもの。`view_chinou.py --table --okra`（commit 896ff12ab）の
配置仕様をそのまま関数化し、ブリッジ（`sim_dds_bridge.py`）でも同一配置を再現するために共用する。
配置の正本はこの関数。view_chinou は将来この関数へ寄せて重複を解消する（TODO: 数値ドリフト防止）。
"""
from __future__ import annotations

import numpy as np

REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
OKRA_USD = f"{REPO}/usd_file/okra.usd"

# A 配置の定数（view_chinou.py と一致させること）
TABLE_CX = 0.50      # 机中心 x（前方）[m]
OKRA_COLS = 5
OKRA_X_MIN_OFF, OKRA_X_MAX_OFF = -0.16, 0.12   # 列の x オフセット（TCX 基準）
OKRA_Y_MIN, OKRA_Y_MAX = -0.15, 0.15           # 行の y
OKRA_Z_OFF = 0.05    # オクラ中心 z = 天板 + これ（長さ10cm→base=天板）
OKRA_BREAK_FORCE = 8.0  # [N] 引張でこれ超→破断＝収穫（自重0.12Nでは外れない）
_FLT_MAX = 3.4028234663852886e+38

# 天井照明（収穫シーン統一照明。view_chinou / bridge 共用の単一ソース）
CEIL_N = 3              # n×n グリッド
CEIL_INTENSITY = 6000.0
CEIL_RADIUS = 0.15     # [m]
CEIL_COLOR = (1.0, 0.98, 0.92)  # わずかに電球色


def build_table_okra(stage, *, table_h: float = 0.72, table_cx: float = TABLE_CX,
                     n_okra: int = 10, okra_usd: str = OKRA_USD) -> list[str]:
    """机（天板+台座, 静的+摩擦）＋直立オクラ N 本（world アンカー剛 FixedJoint, 8N 破断）を
    ``stage`` に追加する。戻り値=オクラ prim パスのリスト。world.reset() の前に呼ぶこと。"""
    from pxr import Gf, UsdGeom, UsdPhysics
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.utils.stage import add_reference_to_stage

    TH, TCX = float(table_h), float(table_cx)
    table_mat = PhysicsMaterial("/World/PM/table", static_friction=0.8,
                                dynamic_friction=0.7, restitution=0.0)
    FixedCuboid(prim_path="/World/Table_top", name="table_top",
                position=np.array([TCX, 0.0, TH - 0.02]), scale=np.array([0.65, 0.55, 0.04]),
                color=np.array([0.55, 0.40, 0.25]), physics_material=table_mat)
    _bh = TH - 0.04
    FixedCuboid(prim_path="/World/Table_base", name="table_base",
                position=np.array([TCX, 0.0, _bh / 2.0]), scale=np.array([0.38, 0.32, _bh]),
                color=np.array([0.45, 0.32, 0.20]), physics_material=table_mat)

    okra_paths: list[str] = []
    if n_okra <= 0:
        return okra_paths
    xs = np.linspace(TCX + OKRA_X_MIN_OFF, TCX + OKRA_X_MAX_OFF, OKRA_COLS)
    rows = (n_okra + OKRA_COLS - 1) // OKRA_COLS
    ys = np.linspace(OKRA_Y_MIN, OKRA_Y_MAX, rows) if rows > 1 else np.array([0.0])
    zc = TH + OKRA_Z_OFF
    k = 0
    for r in range(rows):
        for c in range(OKRA_COLS):
            if k >= n_okra:
                break
            xi, yi = float(xs[c]), float(ys[r])
            pth = f"/Okra_{k}"
            add_reference_to_stage(usd_path=okra_usd, prim_path=pth)
            # 直立: 先端(+Y)を上(rotX+90°)→鉛直まわりに少し振る（順序重要）
            qd = (Gf.Rotation(Gf.Vec3d(1, 0, 0), 90.0)
                  * Gf.Rotation(Gf.Vec3d(0, 0, 1), float((k * 37) % 360))).GetQuat()
            qf = Gf.Quatf(qd.GetReal(), Gf.Vec3f(*qd.GetImaginary()))
            op = UsdGeom.Xformable(stage.GetPrimAtPath(pth))
            op.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(xi, yi, zc))
            op.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(qd)
            # 破断可能 FixedJoint（world アンカー, collision 無効）で直立保持
            j = UsdPhysics.FixedJoint.Define(stage, f"/World/OkraJoints/joint_{k}")
            j.CreateBody1Rel().SetTargets([pth])
            j.CreateLocalPos0Attr(Gf.Vec3f(xi, yi, zc))
            j.CreateLocalRot0Attr(qf)
            j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
            j.CreateBreakForceAttr(OKRA_BREAK_FORCE)
            j.CreateBreakTorqueAttr(_FLT_MAX)
            j.CreateCollisionEnabledAttr(False)
            okra_paths.append(pth)
            k += 1
    return okra_paths


def add_ceiling_lights(stage, *, n: int = CEIL_N, intensity: float = CEIL_INTENSITY,
                       radius: float = CEIL_RADIUS, room_path: str | None = None) -> int:
    """部屋天井に SphereLight を n×n グリッド配置（収穫シーンの統一照明）。

    照明は「カメラごと」ではなく「シーン(stage)ごと」。同じ stage に置けば全カメラ・全ビューで
    共有される。view_chinou と bridge で見た目を揃えるため、両者がこの関数を呼ぶ（単一ソース）。
    部屋 bbox から天井高(z)と XY 広がりを算出。room_path 未指定なら /World/ChinouCenter→/World 探索。
    部屋が無ければ 0 を返す（天井が無いので置かない）。戻り値=設置灯数。
    冪等: 既に /World/CeilingLights がある（USD焼き込み等）なら二重配置せず 0 を返す。
    """
    if n <= 0:
        return 0
    from pxr import Gf, Usd, UsdGeom, UsdLux
    # 焼き込み済み（or 既に追加済み）なら二重配置しない
    cl = stage.GetPrimAtPath("/World/CeilingLights")
    if cl and cl.IsValid() and len(list(cl.GetChildren())) > 0:
        return 0
    bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rp = None
    for cand in ([room_path] if room_path else ["/World/ChinouCenter", "/World"]):
        p = stage.GetPrimAtPath(cand) if cand else None
        if p and p.IsValid():
            rp = p
            break
    if rp is None:
        return 0
    rng = bbc.ComputeWorldBound(rp).ComputeAlignedRange()
    rmn, rmx = rng.GetMin(), rng.GetMax()
    cz = float(rmx[2]) - 0.15  # 天井から 15cm 下

    def _grid(a: float, b: float, k: int, inset: float = 0.7) -> list[float]:
        c = 0.5 * (a + b)
        h = 0.5 * (b - a) * inset  # 内側 inset に収め壁から離す
        return [c] if k == 1 else [c - h + 2 * h * i / (k - 1) for i in range(k)]

    xs = _grid(float(rmn[0]), float(rmx[0]), n)
    ys = _grid(float(rmn[1]), float(rmx[1]), n)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            lt = UsdLux.SphereLight.Define(stage, f"/World/CeilingLights/sphere_{i}_{j}")
            lt.CreateRadiusAttr(float(radius))
            lt.CreateIntensityAttr(float(intensity))
            lt.CreateColorAttr(Gf.Vec3f(*CEIL_COLOR))
            UsdGeom.XformCommonAPI(lt.GetPrim()).SetTranslate(Gf.Vec3d(x, y, cz))
    return n * n


__all__ = ["build_table_okra", "add_ceiling_lights", "OKRA_USD", "TABLE_CX"]
