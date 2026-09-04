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

"""収穫シムの物理マテリアル（摩擦・弾性）を USD に焼き込む。

対象:
  chinou_center.usd  : 床・壁コライダー → floor_mat（よく効く床）
  g1bag.usd          : 足裏 → foot_mat / Dex1右手の指 → grip_mat（把持で滑らない）

物理マテリアル = collider に割り当てる「素材」。摩擦と反発(restitution)を持つ。
未割り当てだと PhysX 既定（摩擦≈0.5・反発0）で、指がオクラを掴めず滑り落ちる。
PhysX の摩擦合成は既定 average なので「両面とも高い」と接触摩擦が高くなる。

実行（usd-core か isaac-sim の python どちらでも可）:
  <python> docs/sim-setup/setup_physics_materials.py
"""

from __future__ import annotations

import shutil

from pxr import Usd, UsdPhysics, UsdShade

USD_DIR = "/home/kota-ueda/Desktop/dimos-hackathon/usd_file"
ROOM = f"{USD_DIR}/chinou_center.usd"
ROBOT = f"{USD_DIR}/g1-29dof-dex1-base-fix-usd/g1bag.usd"

# 物理マテリアル定数（摩擦係数は無次元 / restitution=反発, 0=跳ねない）
# 床・壁: 立位で滑らない程度
FLOOR = dict(static=1.0, dynamic=0.9, restitution=0.0)
# 足裏: 床と同等に高摩擦（接触摩擦は両面の平均で決まる）
FOOT = dict(static=1.0, dynamic=0.9, restitution=0.0)
# Dex1 指: 把持で滑らないよう高摩擦
GRIP = dict(static=1.4, dynamic=1.2, restitution=0.0)


def make_material(stage: Usd.Stage, path: str, p: dict) -> UsdShade.Material:
    """物理マテリアルを生成（UsdPhysics.MaterialAPI で摩擦・反発を付与）。"""
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(p["static"])
    api.CreateDynamicFrictionAttr(p["dynamic"])
    api.CreateRestitutionAttr(p["restitution"])
    return mat


def bind_physics(prim: Usd.Prim, mat: UsdShade.Material) -> None:
    """collider prim に物理マテリアルを bind（material purpose="physics"）。"""
    api = UsdShade.MaterialBindingAPI.Apply(prim)
    api.Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def setup_room() -> None:
    shutil.copy(ROOM, ROOM + ".pre-physmat-bak")
    s = Usd.Stage.Open(ROOM)
    mat = make_material(s, "/World/PhysicsMaterials/floor_mat", FLOOR)
    n = 0
    for p in s.TraverseAll():
        if p.HasAPI(UsdPhysics.CollisionAPI) and p.GetPath().pathString.startswith(
            "/World/Colliders/"
        ):
            bind_physics(p, mat)
            n += 1
    s.GetRootLayer().Save()
    print(
        f"[room] floor_mat を {n} collider に割当 (static={FLOOR['static']} dyn={FLOOR['dynamic']} rest={FLOOR['restitution']})"
    )


def setup_robot() -> None:
    shutil.copy(ROBOT, ROBOT + ".pre-physmat-bak")
    s = Usd.Stage.Open(ROBOT)
    root = "/g1_29dof_with_hand_rev_1_0"
    foot_mat = make_material(s, f"{root}/PhysicsMaterials/foot_mat", FOOT)
    grip_mat = make_material(s, f"{root}/PhysicsMaterials/grip_mat", GRIP)
    nf = ng = 0
    for p in s.TraverseAll():
        if not p.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path = p.GetPath().pathString
        if "ankle_roll" in path:
            bind_physics(p, foot_mat)
            nf += 1
        elif "right_hand" in path:  # Dex1 右手（base + 指リンク）
            bind_physics(p, grip_mat)
            ng += 1
    s.GetRootLayer().Save()
    print(
        f"[robot] foot_mat→{nf}（足裏） grip_mat→{ng}（Dex1指）割当 "
        f"foot(static={FOOT['static']}) grip(static={GRIP['static']})"
    )


if __name__ == "__main__":
    setup_room()
    setup_robot()
    print("done.")
