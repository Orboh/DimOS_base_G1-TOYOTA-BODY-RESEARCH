#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ZEDCamera.set_depth_streaming() の動的トグルを実機で検証（物理動作なし）。

ZEDCamera を直接 open し、capture-loop と同じ経路で
  set_depth_streaming(False) → grab → depth は計算されない (FPS高)
  set_depth_streaming(True)  → grab → depth が出る (FPS低)
を確認する。深度の有無は retrieve_measure の有効点率で判定。

    .venv/bin/python scripts/test_zed_depth_streaming_toggle.py
"""

from __future__ import annotations

import time

import numpy as np
import pyzed.sl as sl

from dimos.hardware.sensors.camera.zed.camera import ZEDCamera


def _grab_n(zed, runtime, depth_mat, n):
    """n 回 grab。runtime.enable_depth に従い深度有効点率と FPS を返す。"""
    for _ in range(5):
        zed.grab(runtime)  # warmup
    t0 = time.time()
    valid = []
    for _ in range(n):
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue
        if runtime.enable_depth:
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            valid.append(float(np.isfinite(depth_mat.get_data()).mean()))
    dt = time.time() - t0
    return (n / dt if dt > 0 else 0.0), (float(np.mean(valid)) if valid else 0.0)


def main() -> None:
    # ZEDCamera の config 経路を通して _depth_on の初期値ロジックも確認
    # （Module は kwargs を直接受け取り、内部で ZEDCameraConfig を構築する）
    cam = ZEDCamera(depth_mode="NEURAL", depth_streaming=False)
    print(f"config.enable_depth={cam.config.enable_depth}  config.depth_streaming={cam.config.depth_streaming}")
    print(f"初期 _depth_on (=enable_depth and depth_streaming): {cam._depth_on}  ← False 期待\n")

    # ZED を直接開いて capture-loop と同じ grab 経路を再現
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 60
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("ZED open failed")
    runtime = sl.RuntimeParameters()
    depth_mat = sl.Mat()

    # set_depth_streaming(False) 相当 → _depth_on=False → grab は depth 無し
    cam.set_depth_streaming(False)
    runtime.enable_depth = cam._depth_on
    fps_off, _ = _grab_n(zed, runtime, depth_mat, 60)
    print(f"set_depth_streaming(False): _depth_on={cam._depth_on}  ->  {fps_off:5.1f} FPS (RGBのみ)")

    # set_depth_streaming(True) → _depth_on=True → grab に depth
    cam.set_depth_streaming(True)
    runtime.enable_depth = cam._depth_on
    fps_on, valid = _grab_n(zed, runtime, depth_mat, 60)
    print(f"set_depth_streaming(True) : _depth_on={cam._depth_on}  ->  {fps_on:5.1f} FPS, depth有効={valid*100:4.1f}%")

    zed.close()
    print("\n判定:",
          "OK — トグルで depth ON/OFF が効き、OFF が明確に高速"
          if (fps_off > fps_on and valid > 0.5) else "要確認")


if __name__ == "__main__":
    main()
