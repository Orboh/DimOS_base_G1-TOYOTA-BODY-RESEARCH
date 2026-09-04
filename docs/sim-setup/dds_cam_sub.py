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

"""検証用: ブリッジの ego_view ZMQ カメラ配信を受信（color + depth + intrinsics）。

dimos `ZmqCamera` は color(images[topic]) のみ読む（後方互換）。本受信機は depth/intrinsics/
cam_to_torso も復号して中身を確認する（full okra 検出の sim 取り込み用の前段検証）。

実行:
  SIM_CAM_HOST=127.0.0.1 ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/dds_cam_sub.py
  （Jetson から: SIM_CAM_HOST=<laptop 100.x>）
"""

import base64
import os
import time

import cv2
import msgpack
import numpy as np
import zmq

HOST = os.getenv("SIM_CAM_HOST", "127.0.0.1")
PORT = int(os.getenv("SIM_CAM_PORT", "5555"))
TOPIC = os.getenv("SIM_CAM_TOPIC", "ego_view")
OUTDIR = os.getenv(
    "SIM_CAM_OUTDIR",
    "/tmp/claude-1000/-home-kota-ueda-Desktop-dimos-hackathon/1487315e-e890-48a9-8754-b1a3f80b5f18/scratchpad/sim_out",
)
SECS = float(os.getenv("SUB_SECS", "12"))


def _b(x):
    return base64.b64decode(x) if isinstance(x, str) else bytes(x)


ctx = zmq.Context.instance()
s = ctx.socket(zmq.SUB)
s.connect(f"tcp://{HOST}:{PORT}")
s.setsockopt(zmq.SUBSCRIBE, b"")
s.setsockopt(zmq.RCVTIMEO, 1000)
print(f"[cam-sub] SUB tcp://{HOST}:{PORT} topic={TOPIC} for {SECS}s", flush=True)

n = 0
t0 = time.time()
last_bgr = None
last_depth = None
shown_meta = False
while time.time() - t0 < SECS:
    try:
        payload = s.recv()
    except zmq.Again:
        continue
    msg = msgpack.unpackb(payload, raw=False)
    cb = (msg.get("images") or {}).get(TOPIC)
    if cb is None:
        continue
    bgr = cv2.imdecode(np.frombuffer(_b(cb), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        continue
    n += 1
    last_bgr = bgr
    # depth
    db = (msg.get("depth") or {}).get(TOPIC)
    if db is not None:
        d16 = cv2.imdecode(np.frombuffer(_b(db), np.uint8), cv2.IMREAD_UNCHANGED)
        if d16 is not None:
            last_depth = d16.astype(np.float32) * float(msg.get("depth_scale", 0.001))
    if not shown_meta:
        K = (msg.get("intrinsics") or {}).get(TOPIC)
        print(f"[cam-sub] intrinsics K={K}", flush=True)
        print(f"[cam-sub] cam_to_torso={msg.get('cam_to_torso')}", flush=True)
        print(f"[cam-sub] color={bgr.shape} depth={'yes' if db is not None else 'no'}", flush=True)
        shown_meta = True
    if n % 30 == 0 and last_depth is not None:
        valid = last_depth[(last_depth > 0.05) & (last_depth < 20)]
        if valid.size:
            print(
                f"[cam-sub] frame {n}: depth m min/med/max = {valid.min():.2f}/{np.median(valid):.2f}/{valid.max():.2f}",
                flush=True,
            )

os.makedirs(OUTDIR, exist_ok=True)
if last_bgr is not None:
    cv2.imwrite(f"{OUTDIR}/cam_color.png", last_bgr)
if last_depth is not None:
    vis = np.clip(
        last_depth / max(0.1, float(np.percentile(last_depth[last_depth > 0], 95) or 1)) * 255,
        0,
        255,
    ).astype(np.uint8)
    cv2.imwrite(f"{OUTDIR}/cam_depth.png", cv2.applyColorMap(vis, cv2.COLORMAP_JET))
print(f"[cam-sub] DONE received={n}, saved color/depth -> {OUTDIR}", flush=True)
