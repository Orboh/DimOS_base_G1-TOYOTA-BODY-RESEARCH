#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""点群 3D 検出(B2)の可視化: フィルタ後の物体点群を画像に再投影してオーバーレイ（診断用）。

各 banana の bbox(赤) と、``Detection3DPC.from_2d`` がフィルタ後に残した点群を
画像に投影して点(シアン)で描き、centroid を黄丸 + 距離[m] で示す。背景が落ちて
バナナ本体に点が集まる様子が一目で分かる。物理動作なし。
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import pyzed.sl as sl

from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.perception.detection.detectors.yolo import Yolo2DDetector
from dimos.perception.detection.type.detection3d.pointcloud import Detection3DPC


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="banana")
    ap.add_argument("--out", default="/tmp/zed_pc3d.jpg")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--model", default="yolo11n.pt")
    args = ap.parse_args()
    targets = {c.strip().lower() for c in args.classes.split(",") if c.strip()}

    cam = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 15
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.METER
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("ZED open failed")

    info = cam.get_camera_information()
    w = info.camera_configuration.resolution.width
    h = info.camera_configuration.resolution.height
    calib = info.camera_configuration.calibration_parameters.left_cam
    fx, fy, cx, cy = calib.fx, calib.fy, calib.cx, calib.cy
    K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    camera_info = CameraInfo(height=h, width=w, distortion_model="plumb_bob",
                             D=list(calib.disto), K=K, P=P, frame_id="zed_optical")
    ident = Transform.identity()
    detector = Yolo2DDetector(model_name=args.model)

    runtime = sl.RuntimeParameters()
    img_mat, pc_mat = sl.Mat(), sl.Mat()

    bgr = None
    for _ in range(args.frames):
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            time.sleep(0.05)
            continue
        ts = time.time()
        cam.retrieve_image(img_mat, sl.VIEW.LEFT)
        bgr = np.ascontiguousarray(img_mat.get_data()[:, :, :3])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image(data=rgb, format=ImageFormat.RGB, frame_id="zed_optical", ts=ts)

        cam.retrieve_measure(pc_mat, sl.MEASURE.XYZRGBA)
        xyz = pc_mat.get_data()[:, :, :3].reshape(-1, 3)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        pc = PointCloud2.from_numpy(xyz, frame_id="zed_optical", timestamp=ts)

        dets = [d for d in detector.process_image(image) if str(d.name).lower() in targets]
        if dets:
            break
    cam.close()
    if bgr is None:
        raise RuntimeError("no frame")

    def project(pts: np.ndarray) -> np.ndarray:
        z = np.clip(pts[:, 2], 1e-6, None)
        u = fx * pts[:, 0] / z + cx
        v = fy * pts[:, 1] / z + cy
        return np.stack([u, v], axis=1)

    for det in dets:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)  # bbox red
        d3d = Detection3DPC.from_2d(det, pc, camera_info, ident, filters=None)  # filtered
        if d3d is None:
            continue
        obj = np.asarray(d3d.pointcloud.pointcloud.points, dtype=np.float64)  # (N,3) filtered pts
        uv = project(obj).astype(int)
        for pu, pv in uv:
            if 0 <= pu < w and 0 <= pv < h:
                bgr[pv, pu] = (255, 255, 0)  # filtered points = cyan
        c = d3d.center
        cu, cv_ = int(fx * c.x / max(c.z, 1e-6) + cx), int(fy * c.y / max(c.z, 1e-6) + cy)
        cv2.circle(bgr, (cu, cv_), 8, (0, 255, 255), -1)  # centroid yellow
        cv2.putText(bgr, f"{det.name} z={c.z:.2f}m ({len(obj)}pts)", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(args.out, bgr)
    print(f"saved={args.out} detections={len(dets)}")


if __name__ == "__main__":
    main()
