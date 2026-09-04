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

"""Standalone wrist UVC camera -> LCM color publisher for the okra-harvest pipeline.

Runs ON THE ROBOT'S JETSON (NX). The right-wrist camera is a **UVC webcam**
(JSK-WDR; enumerates as /dev/video6, NOT a RealSense — so use V4L2/opencv, not
pyrealsense2). It is a separate USB device from the head D435i, so this runs
alongside ik_camera_standalone with no contention.

Publishes color frames over LCM on ``/camera/right_wrist_color`` (Image) so the
laptop okra-harvest app feeds them to ACT as ``observation.images.cam_right_wrist``.
Uses the same low-level ``LCM().publish(Topic(name, Image), img)`` as the head
ik_camera_standalone, so the laptop's ``LCMTransport("/camera/right_wrist_color",
Image)`` subscriber decodes it byte-identically.

PRECONDITIONS:
- The wrist UVC camera is enumerated (``v4l2-ctl --list-devices``; default /dev/video6).
  If missing, re-plug into a different USB port (FleetSeek exp_01KTX4QESEN6GT24M6ZAC9058S).
- Run in an env with opencv + the dimos checkout (e.g. the ``ik_cam`` conda env).
- Set ``LCM_DEFAULT_URL`` + the eth0 multicast route so frames egress to the laptop.

Run (on the NX):
    LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1' \
    WRIST_CAMERA_DEV=/dev/video6 python wrist_camera_standalone.py
"""

from __future__ import annotations

import os
import signal
import time

import cv2

from dimos.msgs.sensor_msgs.Image import Image, ImageFormat

# NB: low-level LCM directly (NOT dimos.core.transport, which pulls in cyclonedds
# absent on the Jetson cam env) — same wire format as LCMTransport.
from dimos.protocol.pubsub.impl.lcmpubsub import LCM, Topic

DEV = os.getenv("WRIST_CAMERA_DEV", "/dev/video6")
WIDTH = int(os.getenv("WRIST_CAMERA_WIDTH", "640"))
HEIGHT = int(os.getenv("WRIST_CAMERA_HEIGHT", "480"))
FPS = float(os.getenv("WRIST_CAMERA_FPS", "15"))
LCM_URL = os.getenv("LCM_DEFAULT_URL", "udpm://239.255.76.67:7667?ttl=1")
# Frame id mirrors the training camera key (observation.images.cam_right_wrist).
FRAME_ID = os.getenv("WRIST_CAMERA_FRAME", "cam_right_wrist")

_running = True


def _stop(*_a):  # type: ignore[no-untyped-def]
    global _running
    _running = False


def main() -> int:
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    # UVC via V4L2; request MJPG (the JSK-WDR exposes MJPG 640x480@30/60).
    cap = cv2.VideoCapture(DEV, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    if not cap.isOpened():
        print(
            f"[wrist-cam] FAILED to open {DEV} (check v4l2-ctl --list-devices / re-plug USB)",
            flush=True,
        )
        return 1
    print(
        f"[wrist-cam] {DEV} {WIDTH}x{HEIGHT}@{FPS} -> LCM {LCM_URL} /camera/right_wrist_color "
        f"frame_id={FRAME_ID}",
        flush=True,
    )

    lc = LCM(url=LCM_URL)
    lc.start()
    topic = Topic("/camera/right_wrist_color", Image)
    interval = 1.0 / max(1.0, FPS)
    last = 0.0
    n = 0
    try:
        while _running:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                time.sleep(0.01)
                continue
            now = time.time()
            if now - last < interval:
                continue
            last = now
            # Publish RGB (same convention as ik_camera_standalone's color_img); the
            # laptop ActBridge does image.to_opencv() -> BGR -> jpeg for the ACT service.
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = Image(data=rgb, format=ImageFormat.RGB, frame_id=FRAME_ID, ts=now)
            lc.publish(topic, img)
            n += 1
            if n % 30 == 0:
                print(f"[wrist-cam] {n} frames, last shape={rgb.shape}", flush=True)
    finally:
        try:
            lc.stop()
        except Exception:
            pass
        cap.release()
        print("[wrist-cam] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
