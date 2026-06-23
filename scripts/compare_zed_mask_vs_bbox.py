#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""bbox 点群(B2) vs セグメンテーションマスク点群(C) の 3D 比較 + 可視化（診断用）。

YOLO-seg(yolo11n-seg) で banana のピクセルマスクを得て、

  B2  bbox + 公式フィルタ（Detection3DPC.from_2d）   ＝ 矩形ベース（既存）
  C   マスク内に投影される点群のみ → centroid       ＝ 輪郭ベース（新）

を同じフレームで比較する。C はバナナの輪郭で切り取るので、矩形に入る背景
（隙間・奥の物）を最初から除外できる。物理動作なし。

    .venv/bin/python scripts/compare_zed_mask_vs_bbox.py --out /tmp/zed_mask.jpg
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO  # type: ignore[attr-defined]

from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.perception.detection.type.detection2d.imageDetections2D import ImageDetections2D
from dimos.perception.detection.type.detection3d.pointcloud import Detection3DPC
from dimos.utils.data import get_data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="banana")
    ap.add_argument("--out", default="/tmp/zed_mask.jpg")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--model", default="yolo11n-seg.pt")
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
    model = YOLO(get_data("models_yolo") / args.model)  # seg 重み → segment タスク

    runtime = sl.RuntimeParameters()
    img_mat, pc_mat = sl.Mat(), sl.Mat()

    print(f"ZED {info.camera_model} {w}x{h}  model={args.model}\n")

    bgr = None
    for i in range(args.frames):
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            time.sleep(0.05)
            continue
        ts = time.time()
        cam.retrieve_image(img_mat, sl.VIEW.LEFT)
        bgr = np.ascontiguousarray(img_mat.get_data()[:, :, :3])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image(data=rgb, format=ImageFormat.RGB, frame_id="zed_optical", ts=ts)

        cam.retrieve_measure(pc_mat, sl.MEASURE.XYZRGBA)
        xyz_full = pc_mat.get_data()[:, :, :3].reshape(-1, 3)
        finite = np.isfinite(xyz_full).all(axis=1)
        xyz = xyz_full[finite]
        pc = PointCloud2.from_numpy(xyz, frame_id="zed_optical", timestamp=ts)

        results = model.track(source=bgr, persist=True, conf=0.5, iou=0.6, verbose=False)
        dets = [d for d in ImageDetections2D.from_ultralytics_result(image, results)
                if str(d.name).lower() in targets]
        if dets:
            break
    cam.close()
    if bgr is None or not dets:
        print("検出なし（カメラ前に対象物を置いて再実行）")
        return

    # 全 finite 点を一度だけ 2D 投影（identity transform）
    z = np.clip(xyz[:, 2], 1e-6, None)
    uu = (fx * xyz[:, 0] / z + cx).astype(int)
    vv = (fy * xyz[:, 1] / z + cy).astype(int)
    in_img = (xyz[:, 2] > 0) & (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)

    print(f"banana detections={len(dets)} (scene {len(xyz)} pts)\n")
    for det in dets:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)  # bbox red

        # --- B2: bbox + 公式フィルタ ---
        b2 = Detection3DPC.from_2d(det, pc, camera_info, ident, filters=None)
        if b2 is not None:
            c = b2.center
            print(f"  [B2 bbox+filter ] optical(x={c.x:+.3f}, y={c.y:+.3f}, z={c.z:+.3f}) m  "
                  f"fwd z={c.z:.3f} m  ({len(b2.pointcloud.pointcloud.points)} pts)")

        # --- C: マスク内の点群のみ ---
        mask = getattr(det, "mask", None)
        if mask is None:
            print("  [C mask         ] このモデルはマスク無し（seg 重みを使うこと）")
            continue
        sel = in_img.copy()
        sel[in_img] = mask[vv[in_img], uu[in_img]] > 0
        obj = xyz[sel]
        if obj.shape[0] == 0:
            print("  [C mask         ] マスク内に有効点なし")
            continue
        c_mean = obj.mean(axis=0)
        c_med = np.median(obj, axis=0)
        mask_px = int((mask > 0).sum())
        print(f"  [C mask         ] optical(x={c_med[0]:+.3f}, y={c_med[1]:+.3f}, z={c_med[2]:+.3f}) m  "
              f"fwd z={c_med[2]:.3f} m  (mask {mask_px}px → {obj.shape[0]} pts, median)")
        print(f"  {' '*18}mean z={c_mean[2]:.3f} m")

        # 可視化: マスク点群を画像に重ねる（シアン）+ centroid（黄）
        u_obj = uu[sel]
        v_obj = vv[sel]
        for pu, pv in zip(u_obj, v_obj):
            bgr[pv, pu] = (255, 255, 0)
        cu = int(fx * c_med[0] / max(c_med[2], 1e-6) + cx)
        cvp = int(fy * c_med[1] / max(c_med[2], 1e-6) + cy)
        cv2.circle(bgr, (cu, cvp), 8, (0, 255, 255), -1)
        cv2.putText(bgr, f"{det.name} mask z={c_med[2]:.2f}m", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        print()

    cv2.imwrite(args.out, bgr)
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
