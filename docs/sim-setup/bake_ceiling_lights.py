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

"""天井 SphereLight を chinou_center.usd に焼き込む（究極の単一ソース照明）。

これを実行すると、`chinou_center.usd` を読む**全プログラム**（view_chinou / sim_dds_bridge /
将来の任意ツール）が、起動時に何もしなくても同じ天井照明を得る。
sim_scene.add_ceiling_lights は冪等なので、各スクリプトの実行時呼び出しは焼き込み後 no-op になる。

実行（usd-core でも可。pxr のみ使用）:
  <python> docs/sim-setup/bake_ceiling_lights.py
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # docs/sim-setup を import path に
from pxr import Usd
import sim_scene

ROOM = "/home/kota-ueda/Desktop/dimos-hackathon/usd_file/chinou_center.usd"

shutil.copy(ROOM, ROOM + ".pre-ceil-bak")
stage = Usd.Stage.Open(ROOM)
# 既存の CeilingLights があれば一旦消して焼き直す（再実行で値を更新できるように）
cl = stage.GetPrimAtPath("/World/CeilingLights")
if cl and cl.IsValid():
    stage.RemovePrim(cl.GetPath())
n = sim_scene.add_ceiling_lights(stage)  # /World/ChinouCenter を見て /World/CeilingLights に配置
stage.GetRootLayer().Save()
print(
    f"baked {n} ceiling SphereLights into chinou_center.usd "
    f"(intensity={sim_scene.CEIL_INTENSITY}, n={sim_scene.CEIL_N}x{sim_scene.CEIL_N})"
)
print("backup:", ROOM + ".pre-ceil-bak")
