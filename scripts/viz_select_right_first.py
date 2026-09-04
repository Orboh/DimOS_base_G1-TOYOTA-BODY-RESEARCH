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

"""右優先 select を実機で可視化（物理動作なし）。

本番の detect→select を1フレームで再現する:
  YOLO-seg で banana を複数検出 → 各々 ZED 深度で 3D 位置(base系) → reach box
  内か判定 → in-box の中で最も右(x最大)を TARGET として強調。

各 banana に track_id / 3D / reach(YES/no) を描き、TARGET を赤で強調した画像を保存。

    .venv/bin/python scripts/viz_select_right_first.py --depth_mode NEURAL --out /tmp/sel.jpg
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO  # type: ignore[attr-defined]

from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.perception.detection.type.detection2d.imageDetections2D import ImageDetections2D
from dimos.robot.unitree.g1.harvest.blackboard import HarvestConfig
from dimos.robot.unitree.g1.harvest.detect_yolo import default_pixel_to_base
from dimos.utils.data import get_data

_FALLBACK_DEPTH_M = 0.45


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="banana")
    ap.add_argument("--out", default="/tmp/sel.jpg")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--model", default="yolo11n-seg.pt")
    ap.add_argument("--depth_mode", default="NEURAL")
    ap.add_argument(
        "--ignore-reach",
        action="store_true",
        help="reach box 判定を無視し、検出全 banana 中で最も右を TARGET に（机上検証用）",
    )
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

    reach = HarvestConfig().reach  # base系 Box3D: x=lateral(+右), y=depth(+前), z=height
    print(
        f"reach box [m]: x[{reach.x_min:.2f},{reach.x_max:.2f}] "
        f"y[{reach.y_min:.2f},{reach.y_max:.2f}] z[{reach.z_min:.2f},{reach.z_max:.2f}]"
    )
    model = YOLO(get_data("models_yolo") / args.model)
    runtime = sl.RuntimeParameters()
    img_mat, depth_mat = sl.Mat(), sl.Mat()

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
        image = Image(data=rgb, format=ImageFormat.RGB, frame_id="zed", ts=ts)
        cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
        depth = depth_mat.get_data()
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

    h, w = depth.shape[:2]
    okras = []
    for det in dets:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        u, v = (x1 + x2) // 2, (y1 + y2) // 2
        d = float(depth[int(np.clip(v, 0, h - 1)), int(np.clip(u, 0, w - 1))])
        d = d if np.isfinite(d) and 0.05 < d < 10.0 else _FALLBACK_DEPTH_M
        pos = default_pixel_to_base(u, v, image_w=w, image_h=h, depth_m=d)
        in_reach = reach.contains(pos)
        tid = getattr(det, "track_id", None)
        okras.append(
            {"bbox": (x1, y1, x2, y2), "uv": (u, v), "pos": pos, "in_reach": in_reach, "tid": tid}
        )

    # 右優先 select（本番 graph.select と同じ: reach box 内で x 最大）
    # --ignore-reach: 机上検証用に reach 判定を外し、検出全 banana 中で最も右を選ぶ
    candidates = okras if args.ignore_reach else [o for o in okras if o["in_reach"]]
    in_box = [o for o in okras if o["in_reach"]]
    target = max(candidates, key=lambda o: o["pos"]["x"]) if candidates else None

    print(f"\nbanana detections={len(okras)}  in_reach={len(in_box)}")
    for o in okras:
        p = o["pos"]
        star = " ★TARGET(右優先)" if o is target else ""
        print(
            f"  id={o['tid']} pos[m] x={p['x']:+.3f} y={p['y']:+.3f} z={p['z']:+.3f} "
            f"reach={'YES' if o['in_reach'] else 'no '}{star}"
        )

    # 描画
    for o in okras:
        x1, y1, x2, y2 = o["bbox"]
        is_t = o is target
        color = (0, 0, 255) if is_t else ((0, 200, 0) if o["in_reach"] else (160, 160, 160))
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 3 if is_t else 2)
        cv2.circle(bgr, o["uv"], 5, color, -1)
        p = o["pos"]
        label = f"id{o['tid']} x{p['x']:+.2f} y{p['y']:+.2f} z{p['z']:+.2f} {'REACH' if o['in_reach'] else 'far'}"
        cv2.putText(
            bgr, label, (x1, max(14, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA
        )
        if is_t:
            cv2.putText(
                bgr,
                "TARGET (right-most in reach)",
                (x1, min(h - 8, y2 + 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    banner = (
        f"depth={args.depth_mode}  detections={len(okras)}  in_reach={len(in_box)}  "
        f"(green=in reach, gray=out, red=TARGET)"
    )
    cv2.rectangle(bgr, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.putText(
        bgr, banner, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
    )
    cv2.imwrite(args.out, bgr)
    print(
        f"saved={args.out}"
        + ("  TARGET=id{}".format(target["tid"]) if target else "  TARGET=なし(reach box内に無し)")
    )


if __name__ == "__main__":
    main()
