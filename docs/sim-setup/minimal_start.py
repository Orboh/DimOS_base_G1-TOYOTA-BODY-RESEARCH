#!/usr/bin/env python3
"""Isaac Sim 4.5 最小起動テスト（DoD-1）。

SimulationApp を headless で起動し、物理を数ステップ回して閉じるだけ。
当該 8GB GPU で Kit/RTX が立ち上がるか、API 名前空間、起動時間を確認する。
"""
import os
import time

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

t0 = time.time()
from isaacsim import SimulationApp

sim_app = SimulationApp({"headless": True})
print(f"[min] SimulationApp up in {time.time() - t0:.1f}s", flush=True)

# API 名前空間の確認
api = None
try:
    from isaacsim.core.api import World  # noqa: F401

    api = "isaacsim.core.api"
except Exception as e:  # noqa: BLE001
    try:
        from omni.isaac.core import World  # noqa: F401  # type: ignore

        api = "omni.isaac.core"
    except Exception as e2:  # noqa: BLE001
        api = f"NONE ({e} / {e2})"
print(f"[min] World API namespace = {api}", flush=True)

from isaacsim.core.api import World

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
world.reset()
for i in range(60):
    world.step(render=True)
print(f"[min] stepped 60 frames, total {time.time() - t0:.1f}s", flush=True)
print("[min] DONE", flush=True)
sim_app.close()
