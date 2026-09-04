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

"""YOLO-seg 検出サービス（GPU）を dimos へ ZMQ でブリッジする。

専用 venv（/mnt/ssd/yolo_gpu_venv, Python 3.10 + Jetson CUDA torch）で動く。
重い torch/ultralytics 依存をこちらに閉じ込め、dimos venv(3.12, CPU torch) は
触らずに、中立な ZMQ ワイヤ(msgpack)で検出だけを GPU 化する（act_service と同型）。

なぜ別プロセス: dimos venv は uv sync で CPU 版 torch が入り YOLO 推論が CPU で
~367ms（実測）。この venv の GPU torch では ~30ms（約12倍, 2026-06-24 実測）。

起動には LD_LIBRARY_PATH が必須（systemの CUDA 12.6 cublas + venv の cudss）:
  run_yolo_service.sh 経由で起動すること。

ワイヤプロトコル（ZMQ REP, tcp://127.0.0.1:5702）:
  request  (msgpack): {"image_jpeg": <jpeg bytes>,
                       "conf": 0.5, "iou": 0.6,        # optional
                       "classes": ["okra"],            # optional, 名前フィルタ
                       "reset": <bool>}                # optional, tracker をリセット
  response (msgpack): {"width": W, "height": H,
                       "detections": [
                          {"name": str, "class_id": int, "confidence": float,
                           "track_id": int|null,
                           "bbox": [x1,y1,x2,y2],
                           "mask_polygon": [[x,y],...] | null}  # seg 輪郭(画像座標)
                       ]}
  3D 化は dimos 側（ZED depth を持つのは dimos）。本サービスは 2D 検出+mask まで。

実行（run_yolo_service.sh が LD_LIBRARY_PATH を設定して呼ぶ）:
  serve   : python yolo_service.py --serve --model <path>/yolo11n-seg.pt
  selftest: python yolo_service.py --selftest --model <path>/yolo11n-seg.pt
"""

from __future__ import annotations

import argparse

import cv2
import msgpack
import numpy as np
from ultralytics import YOLO

ENDPOINT = "tcp://127.0.0.1:5702"
DEFAULT_MODEL = "/mnt/ssd/workspace/DimOS_base_G1-TOYOTA-BODY-/data/models_yolo/yolo11n-seg.pt"


class YoloService:
    def __init__(self, model_path: str = DEFAULT_MODEL, device: str = "0") -> None:
        self.model = YOLO(model_path)
        self.device = device
        # GPU 常駐 + ウォームアップ（初回推論の遅延を起動時に吸収）
        warm = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.model.predict(warm, device=device, verbose=False)
        import torch

        print(
            f"[yolo_service] model={model_path} device={device} "
            f"cuda={torch.cuda.is_available()} "
            f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
            flush=True,
        )

    def infer(self, req: dict) -> dict:
        buf = np.frombuffer(req["image_jpeg"], dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        conf = float(req.get("conf", 0.5))
        iou = float(req.get("iou", 0.6))
        want = {c.lower() for c in req.get("classes", [])} or None
        # persist=True でフレーム間トラッキング（track_id 維持）。reset でトラッカ初期化。
        results = self.model.track(
            source=bgr,
            device=self.device,
            conf=conf,
            iou=iou,
            persist=not req.get("reset", False),
            verbose=False,
        )
        out = []
        r = results[0]
        if r.boxes is not None:
            polys = r.masks.xy if r.masks is not None else None
            names = r.names
            for i in range(len(r.boxes)):
                cls = int(r.boxes.cls[i].item())
                name = str(names.get(cls, cls)) if isinstance(names, dict) else str(names[cls])
                if want is not None and name.lower() not in want:
                    continue
                xyxy = r.boxes.xyxy[i].tolist()
                tid = None
                if r.boxes.id is not None:
                    tid = int(r.boxes.id[i].item())
                poly = None
                if polys is not None and i < len(polys):
                    poly = [[float(x), float(y)] for x, y in polys[i].tolist()]
                out.append(
                    {
                        "name": name,
                        "class_id": cls,
                        "confidence": float(r.boxes.conf[i].item()),
                        "track_id": tid,
                        "bbox": [float(v) for v in xyxy],
                        "mask_polygon": poly,
                    }
                )
        return {"width": int(w), "height": int(h), "detections": out}

    def serve(self, endpoint: str = ENDPOINT) -> None:
        import zmq

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REP)
        sock.bind(endpoint)
        print(f"[yolo_service] serving on {endpoint}", flush=True)
        while True:
            req = msgpack.unpackb(sock.recv(), raw=False)
            try:
                resp = self.infer(req)
                sock.send(msgpack.packb(resp, use_bin_type=True))
            except Exception as exc:
                sock.send(msgpack.packb({"error": str(exc)}, use_bin_type=True))


def _selftest(model_path: str) -> int:
    svc = YoloService(model_path)
    img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    ok, enc = cv2.imencode(".jpg", img)
    import time

    t = []
    for _ in range(20):
        s = time.time()
        resp = svc.infer({"image_jpeg": enc.tobytes()})
        t.append(time.time() - s)
    print(
        f"[selftest] {resp['width']}x{resp['height']} det={len(resp['detections'])} "
        f"infer {np.mean(t) * 1000:.1f} ms ({1 / np.mean(t):.1f} FPS)"
    )
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--endpoint", default=ENDPOINT)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(_selftest(args.model))
    svc = YoloService(args.model, device=args.device)
    svc.serve(args.endpoint)


if __name__ == "__main__":
    main()
