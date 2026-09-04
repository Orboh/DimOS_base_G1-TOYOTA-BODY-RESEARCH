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

"""物理マテリアルの効きをヘッドレスで実測する落下テスト。

  反発(restitution): floor_mat(e=0) 上に
     - 実オクラ okra.usd（okra_mat e=0）→ 跳ねない（rebound≈0）
     - 対照の弾性ボール（e=0.9）       → 跳ねる（rebound 大）
  摩擦(friction): 実オクラに水平速度 1m/s を与え、停止距離を測る
     - 高摩擦(μ≈0.9)なら理論停止距離 v^2/(2 μ g) ≈ 5-6cm で止まる

実行:
  PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES \
    ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/verify_physics_drop.py
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

_ap = argparse.ArgumentParser()
_ap.add_argument("--gui", action="store_true", help="GUI で落下を表示し、繰り返し落下デモを保持")
ARGS = _ap.parse_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": not ARGS.gui})

from isaacsim.core.api import World
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects import DynamicSphere, FixedCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
import numpy as np
import omni.usd
from pxr import PhysxSchema, UsdLux

try:
    from isaacsim.core.prims import SingleRigidPrim as RigidPrim
except Exception:
    from isaacsim.core.prims import RigidPrim  # type: ignore

OKRA = "/home/kota-ueda/Desktop/dimos-hackathon/usd_file/okra.usd"
DROP_Z = 0.30  # 落下開始高さ [m]
STEPS = 300  # 60Hz で 5 秒
G = 9.81

world = World(stage_units_in_meters=1.0)

# 照明（このシーンは既定で暗いので Dome+Distant を追加。物理には無影響）
_stage = omni.usd.get_context().get_stage()
UsdLux.DomeLight.Define(_stage, "/World/DomeLight").CreateIntensityAttr(1500.0)
UsdLux.DistantLight.Define(_stage, "/World/SunLight").CreateIntensityAttr(3000.0)

# 床（floor_mat 相当: static1.0/dyn0.9/rest0）
floor_mat = PhysicsMaterial(
    "/World/PM/floor", static_friction=1.0, dynamic_friction=0.9, restitution=0.0
)
FixedCuboid(
    "/World/Floor",
    position=np.array([0, 0, -0.05]),
    scale=np.array([2.0, 2.0, 0.1]),
    physics_material=floor_mat,
)

# 実オクラ（okra_mat e=0 を保持）— 反発テスト
add_reference_to_stage(usd_path=OKRA, prim_path="/OkraReal")
okra_real = RigidPrim("/OkraReal")
# 対照の弾性ボール（e=0.9）
bouncy = PhysicsMaterial(
    "/World/PM/bouncy", static_friction=0.5, dynamic_friction=0.5, restitution=0.9
)
# 対照を確実に跳ねさせる: floor(e=0) と接触しても反発を消さないよう combine=max にする
PhysxSchema.PhysxMaterialAPI.Apply(
    omni.usd.get_context().get_stage().GetPrimAtPath("/World/PM/bouncy")
).CreateRestitutionCombineModeAttr("max")
ball = DynamicSphere(
    "/Ball", position=np.array([0.2, 0, DROP_Z]), radius=0.03, mass=0.05, physics_material=bouncy
)
# 摩擦テスト用オクラ（水平速度を与える）
add_reference_to_stage(usd_path=OKRA, prim_path="/OkraSlide")
okra_slide = RigidPrim("/OkraSlide")

world.reset()
if ARGS.gui:
    from isaacsim.core.utils.viewports import set_camera_view

    set_camera_view(eye=np.array([0.6, -0.6, 0.4]), target=np.array([0.0, 0.0, 0.05]))
okra_real.set_world_pose(position=np.array([0.0, 0.0, DROP_Z]))
okra_slide.set_world_pose(position=np.array([-0.4, 0.0, 0.010]))  # 床のすぐ上

PUSH_AT = 40  # この step まで床に静置 → 押す
z_real, z_ball = [], []
slide_x0 = None
for i in range(STEPS):
    if i == PUSH_AT:
        slide_x0 = float(okra_slide.get_world_pose()[0][0])
        okra_slide.set_linear_velocity(np.array([1.0, 0.0, 0.0]))  # vx=1 m/s を付与
    world.step(render=ARGS.gui)
    z_real.append(float(okra_real.get_world_pose()[0][2]))
    z_ball.append(float(ball.get_world_pose()[0][2]))

xf = float(okra_slide.get_world_pose()[0][0])
zr = np.array(z_real)
zb = np.array(z_ball)


def rebound(z):
    """最初の接地(最小)後の最大高さ - 静止高さ = リバウンド量[m]"""
    imin = int(np.argmin(z[: len(z) // 2]))
    rest = z[-1]
    peak = float(z[imin:].max())
    return max(0.0, peak - rest), rest


rb_o, rest_o = rebound(zr)
rb_b, rest_b = rebound(zb)
slide_dist = abs(xf - (slide_x0 if slide_x0 is not None else -0.4))
theory = 1.0**2 / (2 * 0.9 * G)

lines = [
    "================= 落下テスト結果 =================",
    f"[反発] 実オクラ(okra_mat e=0): 静止z={rest_o * 1000:.1f}mm リバウンド={rb_o * 1000:.1f}mm",
    f"[反発] 弾性ボール(e=0.9)     : 静止z={rest_b * 1000:.1f}mm リバウンド={rb_b * 1000:.1f}mm",
    f"  → 判定: {'OK 反発材が効いている（オクラは跳ねず・ボールは跳ねる）' if rb_o < 0.005 and rb_b > rb_o + 0.01 else '要確認'}",
    f"[摩擦] オクラ vx=1m/s → 停止距離={slide_dist * 1000:.1f}mm （理論 μ0.9 で {theory * 1000:.0f}mm 付近で停止）",
    f"  → 判定: {'OK 摩擦が効いて短距離で停止' if slide_dist < 0.25 else '滑りすぎ=要確認'}",
    "=================================================",
]
text = "\n".join(lines)
# Isaac Sim の app.close() は stdout を flush せず終了するためファイルに確実に書く
with open("/tmp/okra_drop_result.txt", "w") as f:
    f.write(text + "\n")
print("\n" + text + "\n", flush=True)

if ARGS.gui:
    # 繰り返し落下デモ + ウィンドウ保持（左=オクラ落下 / 中=弾性ボール / 右下=横滑りオクラ）
    print("[gui] 落下デモをループ表示中。ウィンドウを閉じると終了。", flush=True)
    DROP_POS = {
        okra_real: [0.0, 0.0, DROP_Z],
        ball: [0.2, 0.0, DROP_Z],
        okra_slide: [-0.4, 0.0, 0.01],
    }
    period, k = 260, 0
    while app.is_running():
        ph = k % period
        if ph == 0:  # 周期的に再投下
            for prim, pos in DROP_POS.items():
                prim.set_world_pose(position=np.array(pos))
                try:
                    prim.set_linear_velocity(np.zeros(3))
                    prim.set_angular_velocity(np.zeros(3))
                except Exception:
                    pass
        elif ph == 40:  # 横滑りオクラに速度
            okra_slide.set_linear_velocity(np.array([1.0, 0.0, 0.0]))
        world.step(render=True)
        k += 1

app.close()
