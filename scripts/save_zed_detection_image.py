#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ZED 1 フレームに YOLO 検出ボックス + クラス名 + ZED 深度[m] を描いて保存（診断用）。

全クラスを描画する（target フィルタなし）ので、banana が正しく捉えられているか /
誤検出がないかを目視確認できる。物理動作なし。

    .venv/bin/python scripts/save_zed_detection_image.py --out /tmp/zed_det.jpg
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import pyzed.sl as sl

from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.perception.detection.detectors.yolo import Yolo2DDetector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/zed_det.jpg")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--model", default="yolo11n.pt")
    args = ap.parse_args()

    cam = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 15
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.METER
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("ZED open failed")

    detector = Yolo2DDetector(model_name=args.model)
    runtime = sl.RuntimeParameters()
    img_mat, depth_mat = sl.Mat(), sl.Mat()

    bgr = None
    dets = []
    depth = None
    for _ in range(args.frames):
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            time.sleep(0.05)
            continue
        cam.retrieve_image(img_mat, sl.VIEW.LEFT)
        bgr = np.ascontiguousarray(img_mat.get_data()[:, :, :3])  # BGRA -> BGR
        cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
        depth = depth_mat.get_data()  # (H, W) float32 [m]

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image(data=rgb, format=ImageFormat.RGB, frame_id="zed", ts=time.time())
        dets = list(detector.process_image(image))
        if dets:
            break

    cam.close()
    if bgr is None:
        raise RuntimeError("no frame grabbed")

    h, w = depth.shape[:2]
    for det in dets:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        u, v = (x1 + x2) // 2, (y1 + y2) // 2
        d = float(depth[int(np.clip(v, 0, h - 1)), int(np.clip(u, 0, w - 1))])
        d_txt = f"{d:.2f}m" if np.isfinite(d) and d > 0 else "n/a"
        name = str(getattr(det, "name", "?"))
        conf = float(getattr(det, "confidence", 0.0))
        is_banana = name.lower() == "banana"
        color = (0, 0, 255) if is_banana else (0, 200, 0)  # banana=red, others=green
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        cv2.circle(bgr, (u, v), 4, color, -1)
        label = f"{name} {conf:.2f} {d_txt}"
        cv2.putText(bgr, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    cv2.imwrite(args.out, bgr)
    print(f"detections={len(dets)} saved={args.out}")
    for det in dets:
        print(f"  {getattr(det, 'name', '?')} conf={float(getattr(det, 'confidence', 0)):.2f} "
              f"bbox={[int(v) for v in det.bbox]}")


if __name__ == "__main__":
    main()
