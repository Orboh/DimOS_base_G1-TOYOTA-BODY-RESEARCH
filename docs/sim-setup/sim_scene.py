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


__all__ = ["build_table_okra", "OKRA_USD", "TABLE_CX"]
