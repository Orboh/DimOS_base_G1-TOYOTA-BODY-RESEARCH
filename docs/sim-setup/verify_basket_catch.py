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

"""バスケットがオクラを受けられるか検証（落下→カップ内に収まるか）。

  basket_physics.usd を開口上向き(rotX90)で静置し、上から okra.usd を投入。
  最終位置がカップ内（底付近・半径内）なら「受けた」。convexDecomposition collider +
  okra CCD で 4/4 受ける（add_basket_collider.py 済み前提）。

実行:
  PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES \
    ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/verify_basket_catch.py [--gui]
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
_ap = argparse.ArgumentParser()
_ap.add_argument("--gui", action="store_true", help="GUI 表示＋繰り返し投入デモ")
ARGS = _ap.parse_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": not ARGS.gui})

from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
import numpy as np

try:
    from isaacsim.core.prims import SingleRigidPrim as RP
except Exception:
    from isaacsim.core.prims import RigidPrim as RP
import omni.usd
from pxr import Gf, UsdGeom

BASKET = (
    "/home/kota-ueda/Desktop/dimos-hackathon/usd_file/g1-29dof-dex1-base-fix-usd/basket_physics.usd"
)
OKRA = "/home/kota-ueda/Desktop/dimos-hackathon/usd_file/okra.usd"
ZB = 0.50  # バスケット底の world z
DROPS = [(0.0, 0.0, 0.75), (0.03, 0.0, 0.83), (-0.03, 0.02, 0.91), (0.0, -0.03, 0.99)]

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
stage = omni.usd.get_context().get_stage()

# バスケットを開口上向き(rotX90: local+Y→world+Z)で静置（mesh collider 有, RigidBody無=static）
add_reference_to_stage(usd_path=BASKET, prim_path="/Basket")
api = UsdGeom.XformCommonAPI(stage.GetPrimAtPath("/Basket"))
api.SetTranslate(Gf.Vec3d(0.0, 0.0, ZB))
api.SetRotate(Gf.Vec3f(90.0, 0.0, 0.0))

prims = []
for k, (x, y, z) in enumerate(DROPS):
    pth = f"/Okra_{k}"
    add_reference_to_stage(usd_path=OKRA, prim_path=pth)
    UsdGeom.XformCommonAPI(stage.GetPrimAtPath(pth)).SetTranslate(Gf.Vec3d(x, y, z))
    prims.append(RP(pth))

if ARGS.gui:
    from isaacsim.core.utils.viewports import set_camera_view

    set_camera_view(eye=np.array([0.6, -0.6, 0.85]), target=np.array([0.0, 0.0, 0.55]))

world.reset()
for _ in range(300):
    world.step(render=ARGS.gui)

lines = ["===== バスケット受けテスト ====="]
caught = 0
for k, p in enumerate(prims):
    pos = np.asarray(p.get_world_pose()[0]).flatten()
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    rad = (x * x + y * y) ** 0.5
    inside = (ZB - 0.03 <= z <= ZB + 0.18) and rad < 0.10
    caught += inside
    lines.append(
        f"okra{k}: pos=({x:+.3f},{y:+.3f},{z:.3f}) r={rad:.3f} -> "
        f"{'受けた(中)' if inside else ('床に抜けた' if z < 0.10 else '外/縁')}"
    )
lines.append(f"判定: {caught}/{len(prims)} 個がカップ内 ({'OK' if caught >= 4 else '要改善'})")
text = "\n".join(lines)
with open("/tmp/basket_catch.txt", "w") as f:
    f.write(text + "\n")
print("\n" + text + "\n", flush=True)

if ARGS.gui:
    print("[gui] 繰り返し投入デモ。ウィンドウを閉じると終了。", flush=True)
    period, k = 320, 0
    while app.is_running():
        if k % period == 0:
            for prim, (x, y, z) in zip(prims, DROPS, strict=False):
                prim.set_world_pose(position=np.array([x, y, z]))
                try:
                    prim.set_linear_velocity(np.zeros(3))
                    prim.set_angular_velocity(np.zeros(3))
                except Exception:
                    pass
        world.step(render=True)
        k += 1
app.close()
