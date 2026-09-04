#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ZED 深度→3D 化の 2 方式を同じフレームで並べて比較する（診断用、物理動作なし）。

方式 A（現 harvest）: ``unitree-g1-okra-harvest-zed`` の方式。bbox 中心の **深度 1
  ピクセル**を読み、``default_pixel_to_base`` のピンホールで base 系 3D に変換。
方式 B（IK 用 / 公式）: ``Detection3DModule`` の方式。ZED の **点群**を bbox に投影
  して箱内の点群を集め、その **centroid** を 3D 位置とする（``Detection3DPC.from_2d``）。

両者を ZED 光学系（optical: x=右, y=下, z=前方）に揃えて並べ、カメラからの前方距離
（A=中心深度, B=centroid z）と、点群方式が使った有効点数を表示する。

    .venv/bin/python scripts/compare_zed_depth_methods.py --classes banana
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pyzed.sl as sl

from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.perception.detection.detectors.yolo import Yolo2DDetector
from dimos.perception.detection.type.detection3d.pointcloud import Detection3DPC

_FALLBACK_DEPTH_M = 0.45  # [m] harvest と同じフォールバック


def _build_camera_info(cam: sl.Camera, w: int, h: int) -> CameraInfo:
    calib = cam.get_camera_information().camera_configuration.calibration_parameters.left_cam
    K = [calib.fx, 0.0, calib.cx, 0.0, calib.fy, calib.cy, 0.0, 0.0, 1.0]
    P = [calib.fx, 0.0, calib.cx, 0.0, 0.0, calib.fy, calib.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return CameraInfo(
        height=h, width=w, distortion_model="plumb_bob",
        D=list(calib.disto), K=K, P=P, frame_id="zed_optical",
    )


def _center_depth(depth: np.ndarray, u: float, v: float) -> float:
    """方式 A: bbox 中心 1 ピクセルの ZED 深度 [m]（harvest の depth_getter と同じ）。"""
    h, w = depth.shape[:2]
    d = float(depth[int(np.clip(v, 0, h - 1)), int(np.clip(u, 0, w - 1))])
    return d if np.isfinite(d) and 0.05 < d < 10.0 else _FALLBACK_DEPTH_M


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="banana")
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
    # ZED デフォルト座標系 = IMAGE（x=右, y=下, z=前方）= optical frame
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("ZED open failed")

    info = cam.get_camera_information()
    w = info.camera_configuration.resolution.width
    h = info.camera_configuration.resolution.height
    camera_info = _build_camera_info(cam, w, h)
    ident = Transform.identity()  # 点群は既に optical 系なので world->optical は恒等
    detector = Yolo2DDetector(model_name=args.model)

    runtime = sl.RuntimeParameters()
    img_mat, depth_mat, pc_mat = sl.Mat(), sl.Mat(), sl.Mat()

    print(f"ZED {info.camera_model} S/N {info.serial_number} {w}x{h}")
    print(f"intrinsics fx={camera_info.K[0]:.1f} fy={camera_info.K[4]:.1f} "
          f"cx={camera_info.K[2]:.1f} cy={camera_info.K[5]:.1f}\n")

    for i in range(args.frames):
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            time.sleep(0.05)
            continue
        ts = time.time()
        cam.retrieve_image(img_mat, sl.VIEW.LEFT)
        rgb = np.ascontiguousarray(img_mat.get_data()[:, :, :3][:, :, ::-1])
        image = Image(data=rgb, format=ImageFormat.RGB, frame_id="zed_optical", ts=ts)

        cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
        depth = depth_mat.get_data()  # (H, W) float32 [m]

        cam.retrieve_measure(pc_mat, sl.MEASURE.XYZRGBA)
        xyz = pc_mat.get_data()[:, :, :3].reshape(-1, 3)  # (N, 3) [m], optical 系
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        pc = PointCloud2.from_numpy(xyz, frame_id="zed_optical", timestamp=ts)

        dets = [d for d in detector.process_image(image) if str(d.name).lower() in targets]
        if not dets:
            continue

        print(f"[frame {i}] {args.classes} detections={len(dets)}  "
              f"(scene pointcloud: {len(xyz)} pts)\n")
        for det in dets:
            x1, y1, x2, y2 = det.bbox
            u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            # --- 方式 A: 中心 1 ピクセル深度 ---
            dA = _center_depth(depth, u, v)
            raw = float(depth[int(np.clip(v, 0, h - 1)), int(np.clip(u, 0, w - 1))])
            a_valid = np.isfinite(raw) and 0.05 < raw < 10.0
            print(f"  {det.name} conf={det.confidence:.2f} bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
                  f"center=({u:.0f},{v:.0f})")
            print(f"    [A 中心1px      ] fwd z = {dA:.3f} m"
                  f"{'' if a_valid else '  (中心が無効→0.45m フォールバック)'}")
            # --- 方式 B1: 点群 centroid（フィルタなし＝箱内の全点） ---
            b_raw = Detection3DPC.from_2d(det, pc, camera_info, ident, filters=[])
            if b_raw is not None:
                c = b_raw.center
                n = len(b_raw.pointcloud.pointcloud.points)
                print(f"    [B1 点群raw     ] optical(x={c.x:+.3f}, y={c.y:+.3f}, z={c.z:+.3f}) m"
                      f"  fwd z={c.z:.3f} m  (box内 {n} pts, 背景混入)")
            else:
                print("    [B1 点群raw     ] None")
            # --- 方式 B2: 点群 centroid（公式デフォルトフィルタ: raycast/外れ値/statistical） ---
            b_flt = Detection3DPC.from_2d(det, pc, camera_info, ident, filters=None)
            if b_flt is not None:
                c = b_flt.center
                n = len(b_flt.pointcloud.pointcloud.points)
                print(f"    [B2 点群filtered] optical(x={c.x:+.3f}, y={c.y:+.3f}, z={c.z:+.3f}) m"
                      f"  fwd z={c.z:.3f} m  (残 {n} pts, 背景除去後)")
            else:
                print("    [B2 点群filtered] None（フィルタで全点除去）")
            print()
        cam.close()
        return

    cam.close()
    print("検出なし（カメラ前に対象物を置いて再実行）")


if __name__ == "__main__":
    main()
