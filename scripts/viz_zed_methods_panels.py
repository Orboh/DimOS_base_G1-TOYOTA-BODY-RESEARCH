#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""banana 3D 検出 3 手法を同一フレームで可視化し、横に並べた 1 枚にする（実験記録用）。

  パネル A  : bbox + 中心 1px（赤点）         — 現 harvest use_zed_depth
  パネル B2 : bbox + フィルタ後点群（シアン） — Detection3DPC.from_2d（IK 経路）
  パネル C  : bbox + マスク点群（シアン）     — YOLO-seg 輪郭（新）

各パネル上部に手法名 + 計測 z[m]、全体上部に距離ラベルの帯を付ける。物理動作なし。

    .venv/bin/python scripts/viz_zed_methods_panels.py --label 30cm --out /tmp/panels_30cm.jpg
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

_FALLBACK = 0.45
_CYAN, _RED, _YELLOW, _WHITE, _BLACK = (255, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255), (0, 0, 0)


def _banner(img: np.ndarray, text: str, color: tuple = _WHITE, hpx: int = 46) -> np.ndarray:
    w = img.shape[1]
    cv2.rectangle(img, (0, 0), (w, hpx), _BLACK, -1)
    cv2.putText(img, text, (12, hpx - 14), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="", help="距離ラベル（例: 30cm）")
    ap.add_argument("--classes", default="banana")
    ap.add_argument("--out", default="/tmp/zed_panels.jpg")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--model", default="yolo11n-seg.pt")
    ap.add_argument("--scale", type=float, default=0.5, help="出力縮小率")
    ap.add_argument("--depth_mode", default="PERFORMANCE",
                    help="ZED 深度モード: PERFORMANCE / NEURAL / NEURAL_LIGHT / NEURAL_PLUS")
    args = ap.parse_args()
    targets = {c.strip().lower() for c in args.classes.split(",") if c.strip()}

    cam = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 15
    init.depth_mode = getattr(sl.DEPTH_MODE, args.depth_mode.upper())
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
    model = YOLO(get_data("models_yolo") / args.model)

    runtime = sl.RuntimeParameters()
    img_mat, depth_mat, pc_mat = sl.Mat(), sl.Mat(), sl.Mat()

    bgr = depth = None
    dets: list = []
    for _ in range(args.frames):
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            time.sleep(0.05)
            continue
        ts = time.time()
        cam.retrieve_image(img_mat, sl.VIEW.LEFT)
        bgr = np.ascontiguousarray(img_mat.get_data()[:, :, :3])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image(data=rgb, format=ImageFormat.RGB, frame_id="zed_optical", ts=ts)
        cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
        depth = depth_mat.get_data()
        cam.retrieve_measure(pc_mat, sl.MEASURE.XYZRGBA)
        xyz = pc_mat.get_data()[:, :, :3].reshape(-1, 3)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        pc = PointCloud2.from_numpy(xyz, frame_id="zed_optical", timestamp=ts)
        results = model.track(source=bgr, persist=True, conf=0.5, iou=0.6, verbose=False)
        dets = [d for d in ImageDetections2D.from_ultralytics_result(image, results)
                if str(d.name).lower() in targets]
        if dets:
            break
    cam.close()
    if bgr is None or not dets:
        print("検出なし（カメラ前に banana を置いて再実行）")
        return

    z = np.clip(xyz[:, 2], 1e-6, None)
    uu = (fx * xyz[:, 0] / z + cx).astype(int)
    vv = (fy * xyz[:, 1] / z + cy).astype(int)
    in_img = (xyz[:, 2] > 0) & (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)

    det = max(dets, key=lambda d: d.confidence)  # 最も確からしい banana
    x1, y1, x2, y2 = (int(v) for v in det.bbox)
    u, v = (x1 + x2) // 2, (y1 + y2) // 2

    # --- パネル A: 中心 1px ---
    pA = bgr.copy()
    cv2.rectangle(pA, (x1, y1), (x2, y2), _RED, 2)
    dA = float(depth[int(np.clip(v, 0, h - 1)), int(np.clip(u, 0, w - 1))])
    a_ok = np.isfinite(dA) and 0.05 < dA < 10.0
    zA = dA if a_ok else _FALLBACK
    cv2.circle(pA, (u, v), 7, _YELLOW, -1)
    _banner(pA, f"A center-1px  z={zA:.3f}m" + ("" if a_ok else " (fallback)"), _YELLOW)

    # --- パネル B2: bbox + フィルタ点群 ---
    pB = bgr.copy()
    cv2.rectangle(pB, (x1, y1), (x2, y2), _RED, 2)
    b2 = Detection3DPC.from_2d(det, pc, camera_info, ident, filters=None)
    zB, nB = float("nan"), 0
    if b2 is not None:
        obj = np.asarray(b2.pointcloud.pointcloud.points, dtype=np.float64)
        zb = np.clip(obj[:, 2], 1e-6, None)
        ub = (fx * obj[:, 0] / zb + cx).astype(int)
        vb = (fy * obj[:, 1] / zb + cy).astype(int)
        ok = (ub >= 0) & (ub < w) & (vb >= 0) & (vb < h)
        pB[vb[ok], ub[ok]] = _CYAN
        c = b2.center
        zB, nB = c.z, len(obj)
        cv2.circle(pB, (int(fx * c.x / max(c.z, 1e-6) + cx), int(fy * c.y / max(c.z, 1e-6) + cy)),
                   8, _YELLOW, -1)
    _banner(pB, f"B2 bbox+filter  z={zB:.3f}m ({nB}pts)", _YELLOW)

    # --- パネル C: マスク点群 ---
    pC = bgr.copy()
    cv2.rectangle(pC, (x1, y1), (x2, y2), _RED, 2)
    zC, nC = float("nan"), 0
    mask = getattr(det, "mask", None)
    if mask is not None:
        sel = in_img.copy()
        sel[in_img] = mask[vv[in_img], uu[in_img]] > 0
        obj = xyz[sel]
        if obj.shape[0]:
            pC[vv[sel], uu[sel]] = _CYAN
            cm = np.median(obj, axis=0)
            zC, nC = cm[2], obj.shape[0]
            cv2.circle(pC, (int(fx * cm[0] / max(cm[2], 1e-6) + cx), int(fy * cm[1] / max(cm[2], 1e-6) + cy)),
                       8, _YELLOW, -1)
    _banner(pC, f"C seg-mask  z={zC:.3f}m ({nC}pts)", _YELLOW)

    panels = np.hstack([pA, pB, pC])
    tag = f"{args.depth_mode.upper()}" + (f" {args.label}" if args.label else "")
    title = f"banana 3D depth: A vs B2 vs C  [{tag}]"
    header = np.zeros((56, panels.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, title, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, _WHITE, 2, cv2.LINE_AA)
    out = np.vstack([header, panels])
    if args.scale != 1.0:
        out = cv2.resize(out, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(args.out, out)
    print(f"label={args.label or '(none)'}  A={zA:.3f}  B2={zB:.3f}  C={zC:.3f}  saved={args.out}")


if __name__ == "__main__":
    main()
