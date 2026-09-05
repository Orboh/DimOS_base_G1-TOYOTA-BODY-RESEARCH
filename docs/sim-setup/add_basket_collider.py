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

"""バスケットに「凹を保つ」コライダー、オクラに CCD を付与し、もぎ取ったオクラを籠で受ける。

バスケットは円筒カップ（軸=Y, 開口=+Y端, 底=y0, 半径~7.7cm, 高さ15cm）。
convexHull だと開口が塞がるため **convexDecomposition**（複数凸包・中空保持）を使う。
左手首リンク（動的剛体）の子コライダーゆえ triangle mesh(none) 不可＝凸分解が正解。
薄肉シェルを拾うため voxel 解像度↑・hull 数↑・shrinkWrap。さらにオクラに **CCD** を付けて
高速落下時のすり抜け（トンネリング）を防止。これで受けテスト 4/4（verify_basket_catch.py）。

★パラメータ(PhysxSchema)とCCDは Omniverse の python が要る:
  PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES \
    ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/add_basket_collider.py
（usd-core で実行するとコア collider+material のみ＝薄肉が粗い/トンネリングし得る）
"""

import shutil

from pxr import Usd, UsdPhysics, UsdShade

try:
    from pxr import PhysxSchema  # Omniverse/Isaac の python にのみ存在
except ImportError:
    PhysxSchema = None

BASKET = (
    "/home/kota-ueda/Desktop/dimos-hackathon/usd_file/g1-29dof-dex1-base-fix-usd/basket_physics.usd"
)
MESH = "/basket/mesh"

shutil.copy(BASKET, BASKET + ".pre-collider-bak")
s = Usd.Stage.Open(BASKET)
mp = s.GetPrimAtPath(MESH)

# 凹コライダー（convexDecomposition）
UsdPhysics.CollisionAPI.Apply(mp)
mc = UsdPhysics.MeshCollisionAPI.Apply(mp)
mc.CreateApproximationAttr(UsdPhysics.Tokens.convexDecomposition)
# 薄肉カップを拾う分解パラメータ＋オクラCCD（PhysxSchema があるときのみ＝isaac-sim python）
if PhysxSchema is not None:
    cd = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mp)
    cd.CreateMaxConvexHullsAttr(128)
    cd.CreateHullVertexLimitAttr(64)
    cd.CreateVoxelResolutionAttr(1000000)
    cd.CreateShrinkWrapAttr(True)
    cd.CreateErrorPercentageAttr(0.5)
    print("  convexDecomposition params 設定（maxHulls128, voxel1M, shrinkWrap, err0.5）")
    # オクラに CCD（高速落下でも薄壁をすり抜けない）
    OKRA = "/home/kota-ueda/Desktop/dimos-hackathon/usd_file/okra.usd"
    so = Usd.Stage.Open(OKRA)
    PhysxSchema.PhysxRigidBodyAPI.Apply(so.GetPrimAtPath("/Okra")).CreateEnableCCDAttr(True)
    so.GetRootLayer().Save()
    print("  okra.usd に CCD 有効化")
else:
    print(
        "  注意: PhysxSchema 無し→分解デフォルト＋CCD未設定（受けが粗い→isaac-sim python で再実行推奨）"
    )

# 物理マテリアル（受けたオクラが跳ねない・滑りすぎない）
mat = UsdShade.Material.Define(s, "/basket/PhysicsMaterials/basket_mat")
api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
api.CreateStaticFrictionAttr(0.7)
api.CreateDynamicFrictionAttr(0.6)
api.CreateRestitutionAttr(0.0)
bind = UsdShade.MaterialBindingAPI.Apply(mp)
bind.Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")

s.GetRootLayer().Save()
print("basket collider 追加: convexDecomposition + basket_mat(0.7/0.6/0)")
print("  保存:", BASKET, " バックアップ:", BASKET + ".pre-collider-bak")
