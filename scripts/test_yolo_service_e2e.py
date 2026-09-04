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

"""dimos 側から YOLO-GPU サービス(ZMQ)を叩く end-to-end 検証（dimos venv で実行）。

ZED で連続フレームを取り、各フレームを JPEG にして ZMQ で YOLO-GPU サービスへ送り、
検出(bbox/mask/track_id)を受け取って実効 FPS を測る。サービス分離した検出パイプラインの
実周期を測る目的。物理動作なし。

  .venv/bin/python scripts/test_yolo_service_e2e.py --classes banana --n 60
"""

from __future__ import annotations

import argparse
import time

import cv2
import msgpack
import numpy as np
import pyzed.sl as sl
import zmq

ENDPOINT = "tcp://127.0.0.1:5702"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="banana")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--depth_mode", default="NEURAL")
    ap.add_argument("--endpoint", default=ENDPOINT)
    args = ap.parse_args()
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    cam = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 60
    init.depth_mode = getattr(sl.DEPTH_MODE, args.depth_mode.upper())
    init.coordinate_units = sl.UNIT.METER
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("ZED open failed")
    runtime = sl.RuntimeParameters()
    img_mat, depth_mat = sl.Mat(), sl.Mat()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.connect(args.endpoint)

    def call(jpeg: bytes, reset: bool) -> dict:
        sock.send(
            msgpack.packb(
                {"image_jpeg": jpeg, "classes": classes, "reset": reset}, use_bin_type=True
            )
        )
        return msgpack.unpackb(sock.recv(), raw=False)

    # warmup
    for _ in range(5):
        if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(img_mat, sl.VIEW.LEFT)
            ok, enc = cv2.imencode(".jpg", img_mat.get_data()[:, :, :3])
            call(enc.tobytes(), reset=True)

    t_grab, t_jpeg, t_zmq, t_3d, t_total, ndet = [], [], [], [], [], []
    last_dets = []
    for _ in range(args.n):
        s0 = time.time()
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue
        cam.retrieve_image(img_mat, sl.VIEW.LEFT)
        bgr = np.ascontiguousarray(img_mat.get_data()[:, :, :3])
        cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
        depth = depth_mat.get_data()
        s1 = time.time()
        ok, enc = cv2.imencode(".jpg", bgr)
        jpeg = enc.tobytes()
        s2 = time.time()
        resp = call(jpeg, reset=False)
        s3 = time.time()
        dets = resp.get("detections", [])
        h, w = depth.shape[:2]
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)
            dz = float(depth[int(np.clip(v, 0, h - 1)), int(np.clip(u, 0, w - 1))])
            d["depth_m"] = dz if np.isfinite(dz) and 0.05 < dz < 10.0 else None
        s4 = time.time()
        last_dets = dets
        t_grab.append(s1 - s0)
        t_jpeg.append(s2 - s1)
        t_zmq.append(s3 - s2)
        t_3d.append(s4 - s3)
        t_total.append(s4 - s0)
        ndet.append(len(dets))

    cam.close()

    def ms(x):
        return f"{np.mean(x) * 1000:5.1f} ms"

    print(
        f"depth_mode={args.depth_mode} classes={classes} frames={len(t_total)} avg_det={np.mean(ndet):.1f}\n"
    )
    print(f"  grab+depth retrieve : {ms(t_grab)}")
    print(f"  JPEG encode         : {ms(t_jpeg)}")
    print(f"  ZMQ + YOLO(GPU)     : {ms(t_zmq)}")
    print(f"  depth lookup(3D)    : {ms(t_3d)}")
    print(f"  ---- 1サイクル合計  : {ms(t_total)}")
    print(f"  ===> 実効 {1.0 / np.mean(t_total):.1f} FPS（周期 {np.mean(t_total) * 1000:.0f} ms）")
    if last_dets:
        print("\n  最終フレームの検出:")
        for d in last_dets:
            print(
                f"    {d['name']} id={d['track_id']} conf={d['confidence']:.2f} "
                f"depth={d.get('depth_m')} mask={'有' if d.get('mask_polygon') else '無'}"
            )


if __name__ == "__main__":
    main()
