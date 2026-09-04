#!/usr/bin/env python3
# Copyright 2025-2026 Dimensional Inc.
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

"""Live viewer for the wrist camera (the ACT model's only input) on the laptop.

Read-only LCM subscriber — runs ALONGSIDE the okra-harvest / okra-collect blueprint
(no arm_sdk, no contention). Use it to confirm the okra stays in the wrist FOV through
the whole grasp (a key check for the ACT pipeline: if the okra leaves the frame, the
policy cannot visually servo to it).

  /camera/right_wrist_color (Image) -> a live cv2 window (the exact stream ACT sees).

Keys (in the window): 's' = save the current frame, 'q' = quit. Overlay shows the
running FPS + resolution; if no frames arrive it says so (publisher / network issue).

Run (laptop, same LCM bus as the blueprint — the launchers export LCM_DEFAULT_URL):
  LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1' \
    .venv/bin/python scripts/wrist_cam_view.py
  # other camera: --topic /camera/color_image  (head D435i)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from dimos.msgs.sensor_msgs.Image import Image
from dimos.protocol.pubsub.impl.lcmpubsub import LCM, Topic

LCM_URL = os.getenv("LCM_DEFAULT_URL", "udpm://239.255.76.67:7667?ttl=1")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topic", default="/camera/right_wrist_color", help="Image LCM topic")
    ap.add_argument("--save-dir", default=str(Path.home() / "okra_collect" / "wrist_view"))
    ap.add_argument("--scale", type=float, default=1.0, help="window scale factor")
    args = ap.parse_args()
    save_dir = Path(args.save_dir)

    lock = threading.Lock()
    latest = {"bgr": None, "t": 0.0, "n": 0}

    def on_img(msg, _t):  # type: ignore[no-untyped-def]
        try:
            bgr = msg.to_opencv()
        except Exception:
            return
        with lock:
            latest["bgr"] = bgr
            latest["t"] = time.time()
            latest["n"] += 1

    lc = LCM(url=LCM_URL)
    lc.start()
    lc.subscribe(Topic(args.topic, Image), on_img)
    print(f"[wrist-view] LCM {LCM_URL}  topic={args.topic}")
    print("[wrist-view] 's' = save frame, 'q' = quit. (Run alongside the harvest/collect app.)")

    win = f"wrist_cam_view {args.topic}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    last_n = 0
    fps = 0.0
    last_fps_t = time.time()
    saved = 0
    try:
        while True:
            with lock:
                bgr = None if latest["bgr"] is None else latest["bgr"].copy()
                n = latest["n"]
                age = time.time() - latest["t"] if latest["t"] else 1e9
            now = time.time()
            if now - last_fps_t >= 0.5:
                fps = (n - last_n) / (now - last_fps_t)
                last_n, last_fps_t = n, now
            if bgr is None:
                canvas = np.zeros((240, 640, 3), dtype=np.uint8)
                cv2.putText(
                    canvas,
                    f"waiting for {args.topic} ...",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 220, 255),
                    2,
                )
                cv2.imshow(win, canvas)
            else:
                h, w = bgr.shape[:2]
                view = bgr
                if args.scale != 1.0:
                    view = cv2.resize(bgr, (int(w * args.scale), int(h * args.scale)))
                stale = age > 1.0
                txt = f"{w}x{h}  {fps:4.1f} fps" + ("  STALE!" if stale else "")
                color = (0, 0, 255) if stale else (0, 220, 0)
                cv2.putText(view, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.imshow(win, view)
            k = cv2.waitKey(30) & 0xFF
            if k == ord("q"):
                break
            if k == ord("s") and bgr is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                p = save_dir / f"wrist_{saved:03d}.jpg"
                cv2.imwrite(str(p), bgr)
                print(f"[wrist-view] saved {p}")
                saved += 1
    finally:
        cv2.destroyAllWindows()
        try:
            lc.stop()
        except Exception:
            pass
    print("[wrist-view] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
