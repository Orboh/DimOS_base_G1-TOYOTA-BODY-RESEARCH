#!/usr/bin/env python3
"""room.usd と G1 USD のスケール診断（SimulationApp 内で pxr 使用、描画なし）。"""
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, Gf
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
import omni.usd

REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
ROOM = f"{REPO}/usd_file/room.usd"
G1 = f"{REPO}/usd_file/g1-29dof-dex1-base-fix-usd/g1_29dof_with_dex1_base_fix1.usd"


def native_info(path):
    s = Usd.Stage.Open(path)
    return UsdGeom.GetStageMetersPerUnit(s), str(UsdGeom.GetStageUpAxis(s)), (s.GetDefaultPrim().GetPath() if s.GetDefaultPrim() else None)


print("\n=== native metersPerUnit / upAxis / defaultPrim ===", flush=True)
for nm, p in (("room", ROOM), ("g1", G1)):
    mpu, up, dp = native_info(p)
    print(f"  {nm}: metersPerUnit={mpu}  up={up}  defaultPrim={dp}", flush=True)

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
print(f"\n=== composed stage metersPerUnit = {UsdGeom.GetStageMetersPerUnit(stage)} ===", flush=True)

add_reference_to_stage(usd_path=ROOM, prim_path="/World/Room")
add_reference_to_stage(usd_path=G1, prim_path="/World/G1")


def wbb(prim_path):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    prim = stage.GetPrimAtPath(prim_path)
    b = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    mn, mx = b.GetMin(), b.GetMax()
    return mn, mx, Gf.Vec3d(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])


print("\n=== world bbox in COMPOSED stage (stage units == meters) ===", flush=True)
for nm, pp in (("Room", "/World/Room"), ("G1", "/World/G1")):
    mn, mx, sz = wbb(pp)
    print(f"  {nm}: size=({sz[0]:.3f}, {sz[1]:.3f}, {sz[2]:.3f})  min={tuple(round(v,2) for v in mn)} max={tuple(round(v,2) for v in mx)}", flush=True)

print("\n[diag] DONE", flush=True)
app.close()
