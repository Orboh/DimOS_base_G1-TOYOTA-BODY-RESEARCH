#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ZED + YOLO + depth 検出の単体検証（実機 harvest フローと同じ知覚部品）。

ZED-M から color + depth を 1 フレーム取り、HarvestModule と **同じ**
``make_yolo_detect_okra`` + ZED ``depth_getter`` で banana を検出し、各検出の
3D 位置 [m] と reach box 内かどうかを表示する。フロー全体を起動せずに
「YOLO が banana を拾うか / ZED 深度が妥当か / 3D 位置が手の届く箱に入るか」を
切り分けて確認するための診断ツール（物理動作なし）。

    .venv/bin/python scripts/verify_zed_banana_detect.py
    .venv/bin/python scripts/verify_zed_banana_detect.py --classes banana,orange --frames 5
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pyzed.sl as sl

from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.perception.detection.detectors.yolo import Yolo2DDetector
from dimos.robot.unitree.g1.harvest.blackboard import HarvestConfig
from dimos.robot.unitree.g1.harvest.detect_yolo import make_yolo_detect_okra

_FALLBACK_DEPTH_M = 0.45  # [m] HarvestModule と同じフォールバック


def _open_zed() -> sl.Camera:
    cam = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 15
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # NEURAL は TensorRT 必須（ブループリントと同じ）
    init.coordinate_units = sl.UNIT.METER
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {status}")
    return cam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="banana", help="検出対象クラス（カンマ区切り）")
    ap.add_argument("--frames", type=int, default=10, help="検出を試みる最大フレーム数")
    ap.add_argument("--model", default="yolo11n.pt", help="YOLO 重み（models_yolo 配下）")
    args = ap.parse_args()
    targets = {c.strip() for c in args.classes.split(",") if c.strip()}

    cfg = HarvestConfig()
    reach = cfg.reach  # Box3D（手の届く範囲 [m]）
    print(f"target classes : {targets}")
    print(f"reach box [m]  : x[{reach.x_min:.2f},{reach.x_max:.2f}] "
          f"y[{reach.y_min:.2f},{reach.y_max:.2f}] z[{reach.z_min:.2f},{reach.z_max:.2f}]")

    cam = _open_zed()
    info = cam.get_camera_information()
    print(f"ZED opened     : {info.camera_model} S/N {info.serial_number} "
          f"{info.camera_configuration.resolution.width}x"
          f"{info.camera_configuration.resolution.height}\n")

    runtime = sl.RuntimeParameters()
    img_mat = sl.Mat()
    depth_mat = sl.Mat()

    # HarvestModule と同じ frame_getter / depth_getter を組み、同じ検出器に渡す。
    state: dict = {"color": None, "depth": None}

    def frame_getter() -> Image | None:
        return state["color"]

    def depth_getter(u: float, v: float) -> float:
        arr = state["depth"]
        if arr is None:
            return _FALLBACK_DEPTH_M
        h, w = arr.shape[:2]
        d = float(arr[int(np.clip(v, 0, h - 1)), int(np.clip(u, 0, w - 1))])
        return d if np.isfinite(d) and 0.05 < d < 10.0 else _FALLBACK_DEPTH_M

    detector = Yolo2DDetector(model_name=args.model)
    detect_fn = make_yolo_detect_okra(
        frame_getter, target_classes=targets, detector=detector, depth_getter=depth_getter,
    )

    found = False
    for i in range(args.frames):
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            time.sleep(0.05)
            continue
        cam.retrieve_image(img_mat, sl.VIEW.LEFT)
        bgra = img_mat.get_data()
        rgb = np.ascontiguousarray(bgra[:, :, :3][:, :, ::-1])  # BGRA -> RGB
        state["color"] = Image(data=rgb, format=ImageFormat.RGB, frame_id="zed", ts=time.time())

        cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
        state["depth"] = depth_mat.get_data()  # (H, W) float32 [m]

        okras = detect_fn()  # -> list[Okra]（本番と同じ経路）
        print(f"[frame {i}] detections={len(okras)}")
        for ok in okras:
            p = ok.pos_3d
            in_reach = reach.contains(p)
            print(f"  - {ok.id:16s} region={ok.img_region} "
                  f"pos[m] x={p['x']:+.3f} y={p['y']:+.3f} z={p['z']:+.3f} "
                  f"ripeness={ok.ripeness:.2f} reach={'YES' if in_reach else 'no'}")
        if okras:
            found = True
            break

    cam.close()
    print("\nRESULT:", "✅ 検出あり" if found else "❌ 検出なし（カメラ前に対象物を置いて再実行）")


if __name__ == "__main__":
    main()
