#!/usr/bin/env python3
"""room.usd 全体(pseudoroot) と defaultPrim(/World) の bbox を比較し、欠落範囲を特定。"""
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, Gf
import omni.usd

REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
ROOM = f"{REPO}/usd_file/room.usd"

ctx = omni.usd.get_context()
ctx.open_stage(ROOM)
stage = ctx.get_stage()
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])


def bb(prim):
    r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    mn, mx = r.GetMin(), r.GetMax()
    return mn, mx, Gf.Vec3d(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2])


print(f"\ndefaultPrim = {stage.GetDefaultPrim().GetPath()}", flush=True)

print("\n=== root-level prims (children of '/') ===", flush=True)
for p in stage.GetPseudoRoot().GetChildren():
    try:
        mn, mx, s = bb(p)
        print(f"  /{p.GetName():18s} [{p.GetTypeName()}] size=({s[0]:.2f},{s[1]:.2f},{s[2]:.2f}) y=[{mn[1]:.2f},{mx[1]:.2f}] z=[{mn[2]:.2f},{mx[2]:.2f}]", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  /{p.GetName()} bbox err {e}", flush=True)

mn, mx, s = bb(stage.GetPseudoRoot())
print(f"\n=== FULL stage bbox: size=({s[0]:.2f},{s[1]:.2f},{s[2]:.2f}) min={tuple(round(v,2) for v in mn)} max={tuple(round(v,2) for v in mx)} ===", flush=True)
wp = stage.GetPrimAtPath("/World")
if wp and wp.IsValid():
    mn2, mx2, s2 = bb(wp)
    print(f"=== /World only bbox: size=({s2[0]:.2f},{s2[1]:.2f},{s2[2]:.2f}) ===", flush=True)

print("\n[diag] DONE", flush=True)
app.close()
