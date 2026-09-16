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

"""``compare_ik_approach.py`` の実行と並行して回す、連続フレームキャプチャ（.venv）。

``dds_cam_sub.py``（検証用: 受信した最後の1枚だけ保存）の連続版。bridge の
``SIM_PUB_CAMERA=1`` の ZMQ カメラ(``ego_view``既定)を購読し、``INTERVAL_S`` 秒
おきに1枚ずつ ``OUTDIR`` へ保存する。above/front の軌道の見た目比較
（フィルムストリップ）に使う想定 — ``compare_ik_approach.py`` の実行と重ねて
別プロセスで起動しておく。

実行（bridge起動済み・SIM_PUB_CAMERA=1 前提。compare_ik_approach.py と並行実行）:
  SECS=15 INTERVAL_S=1.0 OUTDIR=/tmp/sim_frames/above \
    .venv/bin/python docs/sim-setup/capture_ik_approach_frames.py
"""

from __future__ import annotations

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
OUTDIR = os.getenv("OUTDIR", "/tmp/sim_frames")
SECS = float(os.getenv("SECS", "15"))
INTERVAL_S = float(os.getenv("INTERVAL_S", "1.0"))


def _b(x: str | bytes) -> bytes:
    return base64.b64decode(x) if isinstance(x, str) else bytes(x)


def main() -> int:
    os.makedirs(OUTDIR, exist_ok=True)
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.SUB)
    s.connect(f"tcp://{HOST}:{PORT}")
    s.setsockopt(zmq.SUBSCRIBE, b"")
    s.setsockopt(zmq.RCVTIMEO, 1000)
    print(
        f"[capture] SUB tcp://{HOST}:{PORT} topic={TOPIC} for {SECS}s "
        f"every {INTERVAL_S}s -> {OUTDIR}",
        flush=True,
    )

    t0 = time.time()
    next_save = t0
    idx = 0
    n = 0
    while time.time() - t0 < SECS:
        try:
            payload = s.recv()
        except zmq.Again:
            continue
        msg = msgpack.unpackb(payload, raw=False)
        cb = (msg.get("images") or {}).get(TOPIC)
        if cb is None:
            continue
        n += 1
        now = time.time()
        if now >= next_save:
            bgr = cv2.imdecode(np.frombuffer(_b(cb), np.uint8), cv2.IMREAD_COLOR)
            if bgr is not None:
                t_rel = now - t0
                path = f"{OUTDIR}/frame_{idx:03d}_t{t_rel:05.1f}s.png"
                cv2.imwrite(path, bgr)
                print(f"[capture] saved {path}", flush=True)
                idx += 1
            next_save = now + INTERVAL_S
    print(f"[capture] DONE received={n} saved={idx} -> {OUTDIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
