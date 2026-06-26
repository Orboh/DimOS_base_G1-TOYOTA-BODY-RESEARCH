#!/usr/bin/env python3
"""M2 phase A（isaac-sim env）: chinou+g1bag を headless で建て、A配置オクラ10本の
**torso_link 相対座標**を JSON 出力する。dimos は import しない（structlog/langgraph 依存回避）。
IK 判定は phase B（.venv の verify_m2_reach_ik.py）で行う。

実行:
  M2_OUT=<path.json> PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES \
    ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/_dump_torso_m2.py
"""
import json
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
ROOM = f"{REPO}/usd_file/chinou_center.usd"
G1 = f"{REPO}/usd_file/g1-29dof-dex1-base-fix-usd/g1bag.usd"
OUT = os.environ.get("M2_OUT", "/tmp/m2_okra_torso.json")

import numpy as np

# A 配置（view_chinou.py --table --okra 10 と同一）
TCX, TH = 0.50, 0.72
xs = np.linspace(TCX - 0.16, TCX + 0.12, 5)
ys = np.array([-0.15, 0.15])
ZC = TH + 0.05
okra = []
k = 0
for r in range(2):
    for c in range(5):
        if k >= 10:
            break
        okra.append((r, c, float(xs[c]), float(ys[r]), ZC))
        k += 1

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from pxr import Gf, Usd, UsdGeom, UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage, open_stage
import omni.usd

try:
    from isaacsim.core.prims import SingleArticulation as ArtCls
except Exception:  # noqa: BLE001
    from isaacsim.core.api.articulations import Articulation as ArtCls  # type: ignore

open_stage(ROOM)
world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
add_reference_to_stage(usd_path=G1, prim_path="/G1")
g1 = stage.GetPrimAtPath("/G1")
bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
lift = -float(bbc.ComputeWorldBound(g1).ComputeAlignedRange().GetMin()[2])
UsdGeom.XformCommonAPI(g1).SetTranslate(Gf.Vec3d(0.0, 0.0, lift))
art = None
for p in Usd.PrimRange(g1):
    if p.HasAPI(UsdPhysics.ArticulationRootAPI):
        art = p.GetPath().pathString
        break
robot = ArtCls(prim_path=art or "/G1", name="g1")
world.scene.add(robot)
world.reset()
try:
    robot.set_world_pose(position=np.array([0.0, 0.0, lift]), orientation=np.array([1.0, 0.0, 0.0, 0.0]))
except Exception:  # noqa: BLE001
    pass
for _ in range(10):
    world.step(render=False)

torso = None
for p in Usd.PrimRange(g1):
    if p.GetName() == "torso_link":
        torso = p
        break
if torso is None:
    for p in Usd.PrimRange(g1):
        if "torso" in p.GetName().lower():
            torso = p
            break
xc = UsdGeom.XformCache(Usd.TimeCode.Default())
M = xc.GetLocalToWorldTransform(torso)
Minv = M.GetInverse()
tp = M.ExtractTranslation()

out = {"lift": lift, "torso_world": [tp[0], tp[1], tp[2]], "okra": []}
for r, c, wx, wy, wz in okra:
    pt = Minv.Transform(Gf.Vec3d(wx, wy, wz))  # world -> torso_link frame
    out["okra"].append({"row": r, "col": c, "world": [wx, wy, wz], "torso": [pt[0], pt[1], pt[2]]})
with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
print(f"[m2A] torso_world=({tp[0]:.3f},{tp[1]:.3f},{tp[2]:.3f}) lift={lift:.3f} okra={len(out['okra'])} -> {OUT}", flush=True)
app.close()
