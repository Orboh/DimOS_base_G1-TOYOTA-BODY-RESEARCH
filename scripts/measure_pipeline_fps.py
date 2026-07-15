#!/usr/bin/env python3
"""検出パイプライン（grab → YOLO-seg track → 3D化）の連続実効FPSを実測（診断用）。

本番 harvest の detect 相当を連続 N フレーム回し、各段のレイテンシと実効FPSを測る。
物理動作なし。

    .venv/bin/python measure_pipeline_fps.py --depth_mode NEURAL --n 60
"""
from __future__ import annotations
import argparse, time
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO  # type: ignore
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.perception.detection.type.detection2d.imageDetections2D import ImageDetections2D
from dimos.robot.unitree.g1.harvest.detect_yolo import default_pixel_to_base
from dimos.utils.data import get_data

_FB = 0.45


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth_mode", default="NEURAL")
    ap.add_argument("--classes", default="banana")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--model", default="yolo11n-seg.pt")
    args = ap.parse_args()
    targets = {c.strip().lower() for c in args.classes.split(",") if c.strip()}

    cam = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 60
    init.depth_mode = getattr(sl.DEPTH_MODE, args.depth_mode.upper())
    init.coordinate_units = sl.UNIT.METER
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("ZED open failed")
    model = YOLO(get_data("models_yolo") / args.model)
    rt = sl.RuntimeParameters()
    img_mat, depth_mat = sl.Mat(), sl.Mat()

    # warmup
    for _ in range(5):
        if cam.grab(rt) == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(img_mat, sl.VIEW.LEFT)
            model.track(source=img_mat.get_data()[:, :, :3], persist=True, conf=0.5, verbose=False)

    t_grab, t_yolo, t_3d, t_total, ndet = [], [], [], [], []
    for _ in range(args.n):
        s0 = time.time()
        if cam.grab(rt) != sl.ERROR_CODE.SUCCESS:
            continue
        cam.retrieve_image(img_mat, sl.VIEW.LEFT)
        bgr = np.ascontiguousarray(img_mat.get_data()[:, :, :3])
        cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
        depth = depth_mat.get_data()
        s1 = time.time()
        rgb = bgr[:, :, ::-1]
        image = Image(data=np.ascontiguousarray(rgb), format=ImageFormat.RGB, frame_id="z", ts=s1)
        results = model.track(source=bgr, persist=True, conf=0.5, iou=0.6, verbose=False)
        dets = [d for d in ImageDetections2D.from_ultralytics_result(image, results)
                if str(d.name).lower() in targets]
        s2 = time.time()
        h, w = depth.shape[:2]
        for det in dets:
            x1, y1, x2, y2 = det.bbox
            u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)
            d = float(depth[int(np.clip(v, 0, h - 1)), int(np.clip(u, 0, w - 1))])
            d = d if np.isfinite(d) and 0.05 < d < 10.0 else _FB
            default_pixel_to_base(u, v, image_w=w, image_h=h, depth_m=d)
        s3 = time.time()
        t_grab.append(s1 - s0); t_yolo.append(s2 - s1); t_3d.append(s3 - s2); t_total.append(s3 - s0)
        ndet.append(len(dets))

    def ms(x): return f"{np.mean(x)*1000:5.1f} ms"
    print(f"depth_mode={args.depth_mode}  model={args.model}  frames={len(t_total)}  avg_det={np.mean(ndet):.1f}\n")
    print(f"  grab+depth retrieve : {ms(t_grab)}")
    print(f"  YOLO-seg track      : {ms(t_yolo)}")
    print(f"  3D化(検出数ぶん)    : {ms(t_3d)}")
    print(f"  ---- 1サイクル合計  : {ms(t_total)}")
    print(f"  ===> 実効 {1.0/np.mean(t_total):.1f} FPS（周期 {np.mean(t_total)*1000:.0f} ms）")


if __name__ == "__main__":
    main()
