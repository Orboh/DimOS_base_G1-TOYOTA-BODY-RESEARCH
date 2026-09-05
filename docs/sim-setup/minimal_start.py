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
    from isaacsim.core.api import World

    api = "isaacsim.core.api"
except Exception as e:
    try:
        from omni.isaac.core import World  # type: ignore

        api = "omni.isaac.core"
    except Exception as e2:
        api = f"NONE ({e} / {e2})"
print(f"[min] World API namespace = {api}", flush=True)

from isaacsim.core.api import World

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
world.reset()
for _i in range(60):
    world.step(render=True)
print(f"[min] stepped 60 frames, total {time.time() - t0:.1f}s", flush=True)
print("[min] DONE", flush=True)
sim_app.close()
