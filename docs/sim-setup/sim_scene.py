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

"""sim シーン部品の共有ビルダー（isaac-sim env で使用）。

机上オクラ「A 配置」を単一ソース化したもの。`view_chinou.py --table --okra`（commit 896ff12ab）の
配置仕様をそのまま関数化し、ブリッジ（`sim_dds_bridge.py`）でも同一配置を再現するために共用する。
配置の正本はこの関数。view_chinou は将来この関数へ寄せて重複を解消する（TODO: 数値ドリフト防止）。
"""

from __future__ import annotations

import numpy as np

REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
OKRA_USD = f"{REPO}/usd_file/okra.usd"

# 配置の定数。本番＝苗列に対し横移動(y)で収穫し基本手が届く → 机は「横長(y)・奥行き小(x)」、
# オクラは全て reach 内に置く（取れないオクラは置かない）。x=前方(奥行), y=横(横移動方向)。
TABLE_CX = 0.50  # 机中心 x（前方）[m]
# 机寸法 [m]: 横長・奥行き小（旧 0.65×0.55 → 奥行0.30×横1.00）
TABLE_DEPTH = 0.30  # x（奥行）方向の天板長
TABLE_WIDTH = 1.00  # y（横）方向の天板長
# オクラ配置: 横(y)に並べ、奥行(x)は浅く全て reach 内。
OKRA_LAT_N = 5  # 横(y)方向に並べる本数（横移動収穫の主方向）
OKRA_LAT_MIN, OKRA_LAT_MAX = -0.16, 0.14  # 横(y)範囲 [m]（右腕 reach 内＝全数到達, やや右寄せ）
OKRA_DEPTH_OFF_MIN, OKRA_DEPTH_OFF_MAX = (
    -0.08,
    -0.02,
)  # 奥行(x, TCX基準)＝0.42..0.48（浅く全て reach 内）
OKRA_Z_OFF = 0.05  # オクラ中心 z = 天板 + これ（長さ10cm→base=天板）
# [N] 引張でこれ超→破断＝収穫（自重0.12Nでは外れない）。磁石把持(既定)は bridge が
# アンカーを能動的に削除するため値は無関係だが、摩擣把持(SIM_GRASP_FRICTION=1)では
# 「指の摩擦で実際に引き抜く」ので低め(例 1.0N)に下げる＝SIM_OKRA_BREAK_N で上書き。
import os as _os_break

OKRA_BREAK_FORCE = float(_os_break.getenv("SIM_OKRA_BREAK_N", "8.0"))
_FLT_MAX = 3.4028234663852886e38

# 天井照明（収穫シーン統一照明。view_chinou / bridge 共用の単一ソース）
CEIL_N = 3  # n×n グリッド
CEIL_INTENSITY = 6000.0
CEIL_RADIUS = 0.15  # [m]
CEIL_COLOR = (1.0, 0.98, 0.92)  # わずかに電球色


def build_table_okra(
    stage,
    *,
    table_h: float = 0.72,
    table_cx: float = TABLE_CX,
    n_okra: int = 10,
    okra_usd: str = OKRA_USD,
) -> list[str]:
    """机（天板+台座, 静的+摩擦）＋直立オクラ N 本（world アンカー剛 FixedJoint, 8N 破断）を
    ``stage`` に追加する。戻り値=オクラ prim パスのリスト。world.reset() の前に呼ぶこと。"""
    import os as _os0

    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Gf, UsdGeom, UsdPhysics

    TH, TCX = float(table_h), float(table_cx)
    # SIM_TABLE_NOBODY=1: 机の実体（天板+台座 collider）を作らない。オクラは world テザーで浮くので
    # 机が無くても直立する。把持だけを切り分けて検証する用（腕が机フチに引っかかるのを排除）。
    if _os0.getenv("SIM_TABLE_NOBODY", "0") != "1":
        table_mat = PhysicsMaterial(
            "/World/PM/table", static_friction=0.8, dynamic_friction=0.7, restitution=0.0
        )
        FixedCuboid(
            prim_path="/World/Table_top",
            name="table_top",
            position=np.array([TCX, 0.0, TH - 0.02]),
            scale=np.array([TABLE_DEPTH, TABLE_WIDTH, 0.04]),  # 奥行(x)小・横(y)長
            color=np.array([0.55, 0.40, 0.25]),
            physics_material=table_mat,
        )
        _bh = TH - 0.04
        FixedCuboid(
            prim_path="/World/Table_base",
            name="table_base",
            position=np.array([TCX, 0.0, _bh / 2.0]),
            scale=np.array([TABLE_DEPTH * 0.6, TABLE_WIDTH * 0.4, _bh]),  # 台座は一回り小さく
            color=np.array([0.45, 0.32, 0.20]),
            physics_material=table_mat,
        )

    okra_paths: list[str] = []
    if n_okra <= 0:
        return okra_paths
    # 横(y)主・奥行(x)浅。横に lat_n 本、足りない分だけ奥行に薄く段を足す（既定は全て reach 内）。
    # cmd_vel/reposition 検証用に「届かない広い畝」を作るなら env で横範囲/本数を上書き:
    #   SIM_OKRA_LAT_MIN/MAX（横範囲[m]）, SIM_OKRA_LAT_N（横の本数）。
    import os as _os

    lat_min = float(_os.getenv("SIM_OKRA_LAT_MIN", str(OKRA_LAT_MIN)))
    lat_max = float(_os.getenv("SIM_OKRA_LAT_MAX", str(OKRA_LAT_MAX)))
    lat_n = int(_os.getenv("SIM_OKRA_LAT_N", str(OKRA_LAT_N)))
    ys = np.linspace(lat_min, lat_max, lat_n)  # 横(y) 主方向
    depth_rows = (n_okra + lat_n - 1) // lat_n
    xs = (
        np.linspace(TCX + OKRA_DEPTH_OFF_MIN, TCX + OKRA_DEPTH_OFF_MAX, depth_rows)
        if depth_rows > 1
        else np.array([TCX + 0.5 * (OKRA_DEPTH_OFF_MIN + OKRA_DEPTH_OFF_MAX)])
    )
    zc = TH + OKRA_Z_OFF
    k = 0
    for d in range(depth_rows):  # 奥行(x) 浅い段
        for c in range(lat_n):  # 横(y) に並べる
            if k >= n_okra:
                break
            xi, yi = float(xs[d]), float(ys[c])
            pth = f"/Okra_{k}"
            add_reference_to_stage(usd_path=okra_usd, prim_path=pth)
            # 直立: 先端(+Y)を上(rotX+90°)→鉛直まわりに少し振る（順序重要）
            qd = (
                Gf.Rotation(Gf.Vec3d(1, 0, 0), 90.0)
                * Gf.Rotation(Gf.Vec3d(0, 0, 1), float((k * 37) % 360))
            ).GetQuat()
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


# 立ち姿勢収穫用オクラ配置（机なし・torso 相対）。
# build_table_okra は机上ピック(M2/M3)用の配置で、机高さ0.72m・オクラ高さ0.82mは
# torso_link 絶対高さ(直立時 約0.80m)よりわずかに高いだけだが、胸カメラ(torso相対
# z=0.248m・ほぼ水平)から見ると「体のすぐ下」にあり視野角(垂直約58.7°)の外に出る
# （2026-09-12 実機画角検証で判明: 既定配置ではカメラに何も映らなかった）。
# 本関数は「株に実ったオクラ」を模し、torso_link の実 world 変換を使って
# torso 相対座標（IK の ws_x/ws_y/ws_z と同じ標準ROS torso frame）でオクラを直立
# 配置する。z をカメラ高さ(torso相対 z≈0.248m)に近づけることで、カメラ視野の中心
# 付近かつ IK reach box (ik_approach.py: ws_x=[0.05,0.65] ws_y=[-0.75,0.20]
# ws_z=[-0.35,0.85]) の内側に収める。株(茎)のジオメトリ自体はスコープ外
# （G1収穫設計書 SS-04 参照）— オクラ実体のみを world アンカーで直立させる。
STANDING_X_OFF = 0.40  # torso前方 [m]（reach内、カメラからも近すぎない距離）
STANDING_Z_OFF = 0.18  # torso相対高さ [m]（カメラ z=0.248m に近づけ画角内に収める）
STANDING_LAT_MIN, STANDING_LAT_MAX = -0.25, -0.05  # 横(y) [m]（右腕reach内、右寄せ）


def build_standing_okra(
    stage,
    torso_to_world,
    *,
    n_okra: int = 3,
    x_off: float = STANDING_X_OFF,
    z_off: float = STANDING_Z_OFF,
    z_jitter: tuple[float, float] = (0.0, 0.0),
    x_jitter: tuple[float, float] = (0.0, 0.0),
    lat_min: float = STANDING_LAT_MIN,
    lat_max: float = STANDING_LAT_MAX,
    okra_usd: str = OKRA_USD,
    seed: int | None = None,
    index_offset: int = 0,
) -> list[str]:
    """torso_link 相対座標でオクラ N 本を直立配置する（机なし、world アンカー剛
    FixedJoint, 破断力 OKRA_BREAK_FORCE）。``torso_to_world`` は
    ``UsdGeom.XformCache(...).GetLocalToWorldTransform(torso_link prim)`` の
    戻り値（G1 の直立姿勢に応じた実変換）。戻り値=オクラ prim パスのリスト。
    ``world.reset()`` の前後どちらでも呼べる（torso_link の xform は物理と独立）。

    ``z_jitter``: 各オクラの高さを ``z_off + uniform(z_jitter[0], z_jitter[1])`` で
    ランダムにばらつかせる[m]（既定 (0,0)=バラつきなし）。株ごとの実高さの違いを
    模した検証用（2026-09-12 要望: 既定高さから -5cm〜+15cm でばらつかせたい）。
    reach box の z 範囲(ik_approach.py 既定 [-0.35, 0.85])を超えないよう呼び出し側
    で調整すること。``seed`` を指定すると再現可能な配置になる。

    ``x_jitter``: 各オクラの前後位置を ``x_off + uniform(x_jitter[0], x_jitter[1])``
    でばらつかせる[m]（既定 (0,0)=バラつきなし＝全本同一距離）。既定のまま n_okra を
    増やすと、横(y)幅が狭い設定（既定 lat_min/max は右腕reach内に絞った0.20m幅）と
    相まって「全本が同じ奥行きの1枚の壁」のように詰まって見え、しかも胸カメラは
    最短0.3m弱まで近いため、ロボット自身の胴体・頭に重なって見えるほど密集する
    （2026-09-12 GUIスクリーンショットで実際に指摘・確認: 「オクラの量がG1の前
    だけ多い」）。奥行きにもばらつきを与えて実際の畑の株のように前後に散らす。
    reach box の x 範囲(ik_approach.py 既定 [0.05,0.65])を超えないよう呼び出し側で
    調整すること。

    ``index_offset``: prim パス ``/Okra_{index_offset+k}`` のオフセット。同一シーンに
    複数回（例: 右側の近距離グループ＋左側の探索用グループ）呼んでも prim パスが
    衝突しないようにするため（2026-09-12 要望: 右に既存配置＋左に10本・10m範囲を追加）。
    """
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Gf, UsdGeom, UsdPhysics

    okra_paths: list[str] = []
    if n_okra <= 0:
        return okra_paths
    lat_n = max(1, n_okra)
    ys = np.linspace(lat_min, lat_max, lat_n) if lat_n > 1 else np.array([0.5 * (lat_min + lat_max)])
    rng = np.random.default_rng(seed)
    zs = z_off + rng.uniform(z_jitter[0], z_jitter[1], size=n_okra)
    xs = x_off + rng.uniform(x_jitter[0], x_jitter[1], size=n_okra)
    print(
        f"[standing_okra] torso_to_world.Transform(0,0,0) world = "
        f"{tuple(round(v, 3) for v in torso_to_world.Transform(Gf.Vec3d(0, 0, 0)))} "
        f"z_jitter={z_jitter} x_jitter={x_jitter} index_offset={index_offset}",
        flush=True,
    )
    for k in range(n_okra):
        idx = k + index_offset
        y_t = float(ys[k % lat_n])
        z_k = float(zs[k])
        x_k = float(xs[k])
        p_world = torso_to_world.Transform(Gf.Vec3d(x_k, y_t, z_k))
        print(
            f"[standing_okra] Okra_{idx} torso_rel=({x_k:.3f},{y_t:.3f},{z_k:.3f}) "
            f"-> world={tuple(round(v, 3) for v in p_world)}",
            flush=True,
        )
        pth = f"/Okra_{idx}"
        add_reference_to_stage(usd_path=okra_usd, prim_path=pth)
        # 直立: 先端(+Y)を上(rotX+90°)→鉛直まわりに少し振る（build_table_okra と同じ規約）
        qd = (
            Gf.Rotation(Gf.Vec3d(1, 0, 0), 90.0)
            * Gf.Rotation(Gf.Vec3d(0, 0, 1), float((idx * 37) % 360))
        ).GetQuat()
        qf = Gf.Quatf(qd.GetReal(), Gf.Vec3f(*qd.GetImaginary()))
        op = UsdGeom.Xformable(stage.GetPrimAtPath(pth))
        op.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(p_world)
        op.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(qd)
        j = UsdPhysics.FixedJoint.Define(stage, f"/World/OkraJoints/joint_{idx}")
        j.CreateBody1Rel().SetTargets([pth])
        j.CreateLocalPos0Attr(Gf.Vec3f(float(p_world[0]), float(p_world[1]), float(p_world[2])))
        j.CreateLocalRot0Attr(qf)
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateBreakForceAttr(OKRA_BREAK_FORCE)
        j.CreateBreakTorqueAttr(_FLT_MAX)
        j.CreateCollisionEnabledAttr(False)
        okra_paths.append(pth)
    return okra_paths


def add_ceiling_lights(
    stage,
    *,
    n: int = CEIL_N,
    intensity: float = CEIL_INTENSITY,
    radius: float = CEIL_RADIUS,
    room_path: str | None = None,
) -> int:
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
    bbc = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    rp = None
    for cand in [room_path] if room_path else ["/World/ChinouCenter", "/World"]:
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


__all__ = ["OKRA_USD", "TABLE_CX", "add_ceiling_lights", "build_table_okra"]
