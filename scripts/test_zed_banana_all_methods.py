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

"""banana 検出 3D 化を 3 手法で同時テスト（診断用、物理動作なし）。

  A  中心 1px 深度      : bbox 中心 1 ピクセルの ZED 深度（現 harvest use_zed_depth）
  B2 bbox 点群+filter   : Detection3DPC.from_2d（公式 Detection3DModule / IK 経路）
  C  マスク点群         : YOLO-seg のマスク内点群（輪郭で切り取り、新）

同じ ZED フレームで 3 手法の前方距離 z[m] と 3D を並べ、C のマスク点群を画像に
重ねて保存する。

    .venv/bin/python scripts/test_zed_banana_all_methods.py --out /tmp/zed_all.jpg
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

_FALLBACK_DEPTH_M = 0.45


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="banana")
    ap.add_argument("--out", default="/tmp/zed_all.jpg")
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
    camera_info = CameraInfo(
        height=h,
        width=w,
        distortion_model="plumb_bob",
        D=list(calib.disto),
        K=K,
        P=P,
        frame_id="zed_optical",
    )
    ident = Transform.identity()
    model = YOLO(get_data("models_yolo") / args.model)

    runtime = sl.RuntimeParameters()
    img_mat, depth_mat, pc_mat = sl.Mat(), sl.Mat(), sl.Mat()

    print(f"ZED {info.camera_model} {w}x{h}  model={args.model}\n")

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
        depth = depth_mat.get_data()  # (H, W) float32 [m]

        cam.retrieve_measure(pc_mat, sl.MEASURE.XYZRGBA)
        xyz = pc_mat.get_data()[:, :, :3].reshape(-1, 3)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        pc = PointCloud2.from_numpy(xyz, frame_id="zed_optical", timestamp=ts)

        results = model.track(source=bgr, persist=True, conf=0.5, iou=0.6, verbose=False)
        dets = [
            d
            for d in ImageDetections2D.from_ultralytics_result(image, results)
            if str(d.name).lower() in targets
        ]
        if dets:
            break
    cam.close()
    if bgr is None or not dets:
        print("検出なし（カメラ前に banana を置いて再実行）")
        return

    # finite 点を 1 度だけ 2D 投影（identity）
    z = np.clip(xyz[:, 2], 1e-6, None)
    uu = (fx * xyz[:, 0] / z + cx).astype(int)
    vv = (fy * xyz[:, 1] / z + cy).astype(int)
    in_img = (xyz[:, 2] > 0) & (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)

    print(f"banana detections={len(dets)} (scene {len(xyz)} pts)\n")
    for det in dets:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        u, v = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
        print(f"  banana conf={det.confidence:.2f} bbox=({x1},{y1},{x2},{y2})")

        # A: 中心 1px 深度
        dA = float(depth[int(np.clip(v, 0, h - 1)), int(np.clip(u, 0, w - 1))])
        a_ok = np.isfinite(dA) and 0.05 < dA < 10.0
        zA = dA if a_ok else _FALLBACK_DEPTH_M
        print(
            f"    [A 中心1px      ] z={zA:.3f} m" + ("" if a_ok else "  (無効→0.45mフォールバック)")
        )

        # B2: bbox 点群 + フィルタ
        b2 = Detection3DPC.from_2d(det, pc, camera_info, ident, filters=None)
        if b2 is not None:
            c = b2.center
            print(
                f"    [B2 bbox+filter ] z={c.z:.3f} m  optical(x={c.x:+.3f}, y={c.y:+.3f})  "
                f"({len(b2.pointcloud.pointcloud.points)} pts)"
            )
        else:
            print("    [B2 bbox+filter ] None")

        # C: マスク点群
        mask = getattr(det, "mask", None)
        if mask is None:
            print("    [C mask         ] マスク無し（seg 重みを使うこと）")
            print()
            continue
        sel = in_img.copy()
        sel[in_img] = mask[vv[in_img], uu[in_img]] > 0
        obj = xyz[sel]
        if obj.shape[0]:
            cm = np.median(obj, axis=0)
            print(
                f"    [C mask         ] z={cm[2]:.3f} m  optical(x={cm[0]:+.3f}, y={cm[1]:+.3f})  "
                f"(mask{int((mask > 0).sum())}px → {obj.shape[0]} pts)"
            )
            for pu, pv in zip(uu[sel], vv[sel], strict=False):
                bgr[pv, pu] = (255, 255, 0)
            ccu = int(fx * cm[0] / max(cm[2], 1e-6) + cx)
            ccv = int(fy * cm[1] / max(cm[2], 1e-6) + cy)
            cv2.circle(bgr, (ccu, ccv), 8, (0, 255, 255), -1)
            cv2.putText(
                bgr,
                f"banana z={cm[2]:.2f}m",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            print("    [C mask         ] マスク内に有効点なし")
        print()

    cv2.imwrite(args.out, bgr)
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
