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

import time

import numpy as np
import pyzed.sl as sl

MODES_TO_TEST = [
    sl.DEPTH_MODE.PERFORMANCE,
    sl.DEPTH_MODE.QUALITY,
    sl.DEPTH_MODE.ULTRA,
    sl.DEPTH_MODE.NEURAL_LIGHT,
    sl.DEPTH_MODE.NEURAL,
    sl.DEPTH_MODE.NEURAL_PLUS,
]

FRAMES_PER_MODE = 90  # ~3s at 30fps
WARMUP_FRAMES = 10


def test_mode(mode):
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode = mode
    init.coordinate_units = sl.UNIT.METER
    init.depth_minimum_distance = 0.2

    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        return {"mode": str(mode), "error": str(status)}

    runtime = sl.RuntimeParameters()
    depth = sl.Mat()

    # warm-up (first frames include model load / stereo init cost, skip from timing)
    for _ in range(WARMUP_FRAMES):
        zed.grab(runtime)

    grab_times = []
    coverages = []
    means = []

    for _ in range(FRAMES_PER_MODE):
        t0 = time.perf_counter()
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        grab_times.append(time.perf_counter() - t0)

        d = depth.get_data()
        valid = np.isfinite(d) & (d > 0)
        coverages.append(100.0 * valid.sum() / d.size)
        if valid.any():
            means.append(float(d[valid].mean()))

    zed.close()

    return {
        "mode": str(mode),
        "error": None,
        "fps": 1.0 / np.mean(grab_times) if grab_times else float("nan"),
        "coverage_pct": np.mean(coverages) if coverages else float("nan"),
        "mean_depth_m": np.mean(means) if means else float("nan"),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("ZED Mini Depth Mode Comparison (live, on this camera/scene)")
    print(f"{FRAMES_PER_MODE} frames per mode after {WARMUP_FRAMES}-frame warm-up")
    print("=" * 70)

    results = []
    for mode in MODES_TO_TEST:
        print(f"\nTesting {mode} ...")
        r = test_mode(mode)
        results.append(r)
        if r["error"]:
            print(f"  FAILED: {r['error']}")
        else:
            print(
                f"  FPS: {r['fps']:.1f}  |  coverage: {r['coverage_pct']:.1f}%  |  mean depth: {r['mean_depth_m']:.3f}m"
            )
        time.sleep(1)  # let the camera fully release before reopening

    print("\n" + "=" * 70)
    print(f"{'Mode':<18}{'FPS':>10}{'Coverage %':>14}{'Mean Depth (m)':>18}")
    print("-" * 70)
    for r in results:
        if r["error"]:
            print(f"{r['mode']:<18}{'FAILED':>10}")
        else:
            print(
                f"{r['mode']:<18}{r['fps']:>10.1f}{r['coverage_pct']:>14.1f}{r['mean_depth_m']:>18.3f}"
            )
    print("=" * 70)
