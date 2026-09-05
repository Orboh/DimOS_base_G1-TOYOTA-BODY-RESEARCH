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

"""部屋(全体) と G1(接地) を Isaac Sim GUI で開き、ウィンドウを閉じるまで保持する目視確認用。

  - room.usd を open_stage で *全体* ロード（壁/奥床も含む）
  - G1 を /G1 に配置し、足が床(z=0)に来るよう持ち上げ
  - GUI を閉じるまで物理ステップを回し続ける（マウスで視点操作・寸法計測が可能）

実行:
  conda activate isaac-sim
  cd ~/Desktop/dimos-hackathon
  PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES python docs/sim-setup/view_scene.py
"""

import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp

sim_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage, open_stage
from isaacsim.core.utils.viewports import set_camera_view
import numpy as np
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics

try:
    from isaacsim.core.prims import SingleArticulation as ArtCls
except Exception:
    from isaacsim.core.api.articulations import Articulation as ArtCls  # type: ignore

REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
ROOM = f"{REPO}/usd_file/room.usd"
G1 = f"{REPO}/usd_file/g1-29dof-dex1-base-fix-usd/g1_29dof_with_dex1_base_fix1.usd"

print(f"[view] open full room: {ROOM}", flush=True)
open_stage(ROOM)
world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

add_reference_to_stage(usd_path=G1, prim_path="/G1")
g1 = stage.GetPrimAtPath("/G1")
bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
lift = -float(bbc.ComputeWorldBound(g1).ComputeAlignedRange().GetMin()[2])
UsdGeom.XformCommonAPI(g1).SetTranslate(Gf.Vec3d(0.0, 0.0, lift))
print(f"[view] G1 lift = +{lift:.3f} m", flush=True)

art = None
for p in Usd.PrimRange(g1):
    if p.HasAPI(UsdPhysics.ArticulationRootAPI):
        art = p.GetPath().pathString
        break
robot = ArtCls(prim_path=art or "/G1", name="g1")
world.scene.add(robot)

set_camera_view(eye=np.array([5.5, -3.5, 3.2]), target=np.array([0.0, 1.5, 0.5]))
world.reset()
try:
    robot.set_world_pose(position=np.array([0.0, 0.0, lift]))
except Exception as e:
    print(f"[view] set_world_pose failed: {e}", flush=True)

print("[view] GUI を閉じるまで保持します。マウスで視点操作・計測可能。", flush=True)
while sim_app.is_running():
    world.step(render=True)
sim_app.close()
