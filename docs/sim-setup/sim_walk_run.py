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

"""公式 sim2sim 歩行シム（unitree_mujoco ラッパー）。

unitree_mujoco.py 相当の MuJoCo + DDS ブリッジを起動しつつ、
elastic band（吊り）を **ファイル /tmp/sim_band.txt とキー 7/8/9 の両方**で制御できる。
ゲームパッドが無いので wireless_remote は bridge がファイル /tmp/sim_joy.bin から注入する
（SIM_JOY_FILE で変更可）。状態遷移と速度指令は orchestrator (sim_walk_joy.py) が書く。

構成（公式の3端末フローを自動化）:
  [端末1] 本スクリプト             … MuJoCo シム + DDS(domain 0, lo) + elastic band
  [端末2] g1_ctrl                  … 公式C++ deploy（FSM/velocity policy）
  [端末3] sim_walk_joy.py          … FixStand→band解除→Velocity→前進 の時系列注入

実行（.venv の python、mujoco 3.5 + unitree_sdk2py 入り）:
  cd ~/Desktop/dimos-hackathon
  .venv/bin/python docs/sim-setup/sim_walk_run.py
"""

from __future__ import annotations

import os
import sys
import threading
import time

# unitree_mujoco の simulate_python（config / bridge）を import 可能にする
MJ_DIR = os.path.expanduser("~/Desktop/unitree_mujoco/simulate_python")
sys.path.insert(0, MJ_DIR)

import config
import mujoco
import mujoco.viewer
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import ElasticBand, UnitreeSdk2Bridge

BAND_FILE = os.getenv("SIM_BAND_FILE", "/tmp/sim_band.txt")  # "length enable(0/1)"
SCENE = os.path.join(MJ_DIR, config.ROBOT_SCENE)  # ../unitree_robots/g1/scene_29dof.xml

locker = threading.Lock()


def read_band_file(band: ElasticBand) -> None:
    """/tmp/sim_band.txt から band の length / enable を反映（"<length> <0|1>"）。"""
    try:
        with open(BAND_FILE) as f:
            parts = f.read().split()
        if parts:
            band.length = float(parts[0])
        if len(parts) > 1:
            band.enable = bool(int(parts[1]))
    except (FileNotFoundError, ValueError, OSError):
        pass


def main() -> int:
    mj_model = mujoco.MjModel.from_xml_path(SCENE)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = config.SIMULATE_DT

    band = ElasticBand()
    # 初期: torso を立位高さ(≈0.83m)で支持。band 力=K(200)*(dist-length)=重量(344N) となる
    # length=0.445（質量35.1kg実測）。脱力落下を防ぐ。
    band.length = float(os.getenv("SIM_BAND_LEN0", "0.445"))
    band.enable = True
    band_link = mj_model.body("torso_link").id

    viewer = mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=band.MujuocoKeyCallback)
    time.sleep(0.2)

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)  # domain 0, lo
    bridge = UnitreeSdk2Bridge(mj_model, mj_data)
    if config.PRINT_SCENE_INFORMATION:
        bridge.PrintSceneInformation()

    print(
        f"[walk-sim] DDS domain={config.DOMAIN_ID} iface={config.INTERFACE} "
        f"band_link=torso_link len0={band.length} joy_file={bridge._virtual_remote_path}",
        flush=True,
    )
    print(
        "[walk-sim] g1_ctrl と sim_walk_joy.py を起動してください（band は /tmp/sim_band.txt でも制御可）",
        flush=True,
    )

    def sim_thread() -> None:
        while viewer.is_running():
            t0 = time.perf_counter()
            locker.acquire()
            read_band_file(band)
            if band.enable:
                mj_data.xfrc_applied[band_link, :3] = band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )
            else:
                mj_data.xfrc_applied[band_link, :3] = 0.0
            mujoco.mj_step(mj_model, mj_data)
            locker.release()
            dt = mj_model.opt.timestep - (time.perf_counter() - t0)
            if dt > 0:
                time.sleep(dt)

    def view_thread() -> None:
        while viewer.is_running():
            locker.acquire()
            viewer.sync()
            locker.release()
            time.sleep(config.VIEWER_DT)

    vt = threading.Thread(target=view_thread, daemon=True)
    st = threading.Thread(target=sim_thread, daemon=True)
    vt.start()
    st.start()
    try:
        t0 = time.time()
        while viewer.is_running():
            time.sleep(1.0)
            x, y, z = (float(mj_data.qpos[0]), float(mj_data.qpos[1]), float(mj_data.qpos[2]))
            fell = z < 0.45  # 転倒判定（立位 base≈0.78m）
            print(
                f"[walk-sim] t={time.time() - t0:4.0f}s base=({x:+.2f},{y:+.2f},{z:+.2f}) "
                f"band.enable={band.enable} len={band.length:.2f}"
                f"{'  ⚠FALL' if fell else ''}",
                flush=True,
            )
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
