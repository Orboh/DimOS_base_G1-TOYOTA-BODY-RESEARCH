#!/usr/bin/env python3
"""ラップトップ直結の ZED を、カラー画像・3D点群・okra検出3D位置つきで
ブラウザ(localhost)へライブ配信する。rerun 0.32 の同梱Webビューアを使うので
インターネット不要（rerun/pyzed/AIモデルが導入済みならオフラインで動く）。

前提:
  - repo .venv に pyzed + rerun + ultralytics + torch(cuda) 導入済み
  - data/models_yolo/okra11n-seg.pt 配置済み
  - ZED を USB3 でラップトップに接続
  - NEURAL をオフラインで使うなら事前に:  ZED_Diagnostic -nrlo

実行:  ./.venv/bin/python scripts/zed_rerun_stream_local.py [stride] [yolo_every] [imgsz] [conf] [depth]
表示:  ブラウザで  http://localhost:9090
"""
from __future__ import annotations
import sys, os, time
import numpy as np
import cv2
import pyzed.sl as sl
import rerun as rr
from ultralytics import YOLO

HOST      = "localhost"        # 同一マシン表示
GRPC_PORT = 9876
WEB_PORT  = 9090
HERE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL     = os.path.join(HERE, "data/models_yolo/okra11n-seg.pt")

STRIDE     = int(sys.argv[1]) if len(sys.argv) > 1 else 3
YOLO_EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 2      # GPU なので密でも可
IMGSZ      = int(sys.argv[3]) if len(sys.argv) > 3 else 1280
CONF       = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
DEPTH_PREF = (sys.argv[5].upper() if len(sys.argv) > 5 else "NEURAL")

# ---- rerun web viewer -------------------------------------------------------
rr.init("okra_zed_live_local")
rr.serve_grpc(grpc_port=GRPC_PORT, cors_allow_origin=["*"])
uri = "rerun+http://%s:%d/proxy" % (HOST, GRPC_PORT)
rr.serve_web_viewer(web_port=WEB_PORT, open_browser=False, connect_to=uri)
print("=" * 64, flush=True)
print("  WEB VIEWER:  http://%s:%d" % (HOST, WEB_PORT), flush=True)
print("  fallback:    http://%s:%d/?url=%s" % (HOST, WEB_PORT, uri), flush=True)
print("=" * 64, flush=True)

rr.log("world", rr.ViewCoordinates.RDF, static=True)

# ---- ZED（NEURAL 失敗時は PERFORMANCE に自動フォールバック）-----------------
def open_zed(mode_name):
    cam = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 15
    init.depth_mode = getattr(sl.DEPTH_MODE, mode_name)
    init.coordinate_units = sl.UNIT.METER
    st = cam.open(init)
    return (cam, st)

cam, st = open_zed(DEPTH_PREF)
used_mode = DEPTH_PREF
if st != sl.ERROR_CODE.SUCCESS and DEPTH_PREF == "NEURAL":
    print("NEURAL 起動失敗(%s)。PERFORMANCE にフォールバック（オフライン時など）" % st, flush=True)
    cam, st = open_zed("PERFORMANCE")
    used_mode = "PERFORMANCE"
if st != sl.ERROR_CODE.SUCCESS:
    raise SystemExit("ZED open failed: %s" % st)

info = cam.get_camera_information()
calib = info.camera_configuration.calibration_parameters.left_cam
fx, fy, cx, cy = calib.fx, calib.fy, calib.cx, calib.cy
W = info.camera_configuration.resolution.width
H = info.camera_configuration.resolution.height
rr.log("world/cam", rr.Pinhole(focal_length=[fx, fy], principal_point=[cx, cy],
                               width=W, height=H), static=True)
print("ZED %s S/N %s %dx%d depth=%s | streaming..."
      % (info.camera_model, info.serial_number, W, H, used_mode), flush=True)

model = YOLO(MODEL)
rt = sl.RuntimeParameters()
img_mat, pc_mat = sl.Mat(), sl.Mat()


def okra_3d(res, pc):
    out = []
    if res.boxes is None or len(res.boxes) == 0:
        return out
    polys = res.masks.xy if res.masks is not None else [None] * len(res.boxes)
    for i in range(len(res.boxes)):
        conf = float(res.boxes.conf[i])
        poly = polys[i] if i < len(polys) else None
        if poly is not None and len(poly) >= 3:
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [poly.astype(np.int32)], 1)
            pts = pc[m.astype(bool)]
        else:
            x1, y1, x2, y2 = [int(t) for t in res.boxes.xyxy[i]]
            pts = pc[y1:y2, x1:x2].reshape(-1, 4)
        xyz = pts[:, :3]
        good = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0.1) & (xyz[:, 2] < 10)
        if good.sum() < 10:
            continue
        X, Y, Z = np.median(xyz[good], axis=0)
        u = float(res.boxes.xywh[i][0]); v = float(res.boxes.xywh[i][1])
        out.append((u, v, float(X), float(Y), float(Z), conf))
    return out


frame = 0
last = []
try:
    while True:
        if cam.grab(rt) != sl.ERROR_CODE.SUCCESS:
            time.sleep(0.01); continue
        cam.retrieve_image(img_mat, sl.VIEW.LEFT)
        bgr = np.ascontiguousarray(img_mat.get_data()[:, :, :3])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cam.retrieve_measure(pc_mat, sl.MEASURE.XYZRGBA)
        pc = pc_mat.get_data()

        rr.set_time("frame", sequence=frame)
        rr.log("world/cam/image", rr.Image(rgb).compress(jpeg_quality=80))

        subxyz = pc[::STRIDE, ::STRIDE, :3].reshape(-1, 3)
        subcol = rgb[::STRIDE, ::STRIDE, :].reshape(-1, 3)
        g = np.isfinite(subxyz).all(axis=1) & (subxyz[:, 2] > 0.1) & (subxyz[:, 2] < 8)
        rr.log("world/points", rr.Points3D(subxyz[g], colors=subcol[g], radii=0.004))

        if frame % YOLO_EVERY == 0:
            res = model.predict(bgr, conf=CONF, iou=0.6, imgsz=IMGSZ, verbose=False)[0]
            last = okra_3d(res, pc)

        if last:
            pos = np.array([[d[2], d[3], d[4]] for d in last], np.float32)
            rr.log("world/okra", rr.Points3D(pos, colors=[255, 40, 40], radii=0.02,
                                             labels=["%.2f / %.2fm" % (d[5], d[4]) for d in last]))
            rr.log("world/cam/image/okra2d",
                   rr.Points2D(np.array([[d[0], d[1]] for d in last], np.float32),
                               colors=[255, 40, 40], radii=6,
                               labels=["%.2fm" % d[4] for d in last]))
        frame += 1
except KeyboardInterrupt:
    pass
finally:
    cam.close()
