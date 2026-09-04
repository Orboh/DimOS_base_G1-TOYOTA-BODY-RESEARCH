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

# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""RuntimeParameters.enable_depth の動的トグルを実測（診断用、物理動作なし）。

「常時 RGB のみ（軽い）/ IK 時だけ深度（重い）」設計の数字を取る:
  1. depth OFF (enable_depth=False) を N 回 grab → FPS
  2. depth ON  (enable_depth=True)  を N 回 grab → FPS
  3. OFF→ON 切り替え直後、深度が安定する（有効点率が立ち上がる）までのフレーム数
depth_mode は NEURAL 固定で open したまま（エンジン常駐）、grab ごとに enable_depth
だけを切り替える＝カメラ再 open なし。

    .venv/bin/python scripts/measure_zed_depth_toggle.py --mode NEURAL --n 60
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pyzed.sl as sl


def _fps(cam, runtime, n, want_depth, depth_mat):
    """n 回 grab して FPS と（深度ありなら）平均有効点率を返す。"""
    runtime.enable_depth = want_depth
    # ウォームアップ 5 フレーム（計測に含めない）
    for _ in range(5):
        cam.grab(runtime)
    t0 = time.time()
    valid = []
    for _ in range(n):
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue
        if want_depth:
            cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            d = depth_mat.get_data()
            valid.append(float(np.isfinite(d).mean()))
    dt = time.time() - t0
    return n / dt if dt > 0 else 0.0, (float(np.mean(valid)) if valid else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="NEURAL", help="PERFORMANCE/NEURAL_LIGHT/NEURAL/NEURAL_PLUS")
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    cam = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 60  # 上限を上げ、深度計算がボトルネックになる様子を見る
    init.depth_mode = getattr(sl.DEPTH_MODE, args.mode.upper())
    init.coordinate_units = sl.UNIT.METER
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("ZED open failed")
    runtime = sl.RuntimeParameters()
    depth_mat = sl.Mat()

    print(f"depth_mode={args.mode}  resolution=HD720  n={args.n}\n")

    fps_off, _ = _fps(cam, runtime, args.n, False, depth_mat)
    print(f"[depth OFF (RGB only)] {fps_off:5.1f} FPS")
    fps_on, valid_on = _fps(cam, runtime, args.n, True, depth_mat)
    print(
        f"[depth ON  ({args.mode})] {fps_on:5.1f} FPS   valid-depth pixels={valid_on * 100:4.1f}%"
    )
    print(f"  -> depth ON は OFF の約 {fps_off / fps_on:.1f}x 遅い\n" if fps_on > 0 else "")

    # OFF→ON 切り替え後、有効点率が安定するまでのフレーム数
    runtime.enable_depth = False
    for _ in range(10):
        cam.grab(runtime)
    print("OFF -> ON に切り替え、各フレームの有効深度率:")
    runtime.enable_depth = True
    t_switch = time.time()
    stable_frame = None
    prev = 0.0
    for i in range(20):
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue
        cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
        v = float(np.isfinite(depth_mat.get_data()).mean())
        dt_ms = (time.time() - t_switch) * 1000
        mark = ""
        if stable_frame is None and v > 0.3 and abs(v - prev) < 0.03:
            stable_frame = i
            mark = "  <- 安定"
        prev = v
        print(f"  frame {i:2d}  t={dt_ms:6.0f}ms  valid={v * 100:4.1f}%{mark}")
    cam.close()
    print(
        f"\n安定まで: frame {stable_frame}（OFF->ON 後）"
        if stable_frame is not None
        else "\n20フレーム内で安定判定に達せず"
    )


if __name__ == "__main__":
    main()
