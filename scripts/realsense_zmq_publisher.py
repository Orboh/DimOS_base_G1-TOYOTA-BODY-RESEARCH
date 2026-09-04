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

"""RealSense → ZMQ publisher for the G1 NX.

Runs on the G1's onboard NX (Jetson). Opens the RealSense color stream,
JPEG-encodes each frame, wraps in the NVIDIA GEAR-SONIC msgpack schema,
and publishes on `tcp://*:5555`.

Message format (matches
`gear_sonic.camera.composed_camera`):

    {
        "timestamps": {"ego_view": <float seconds>},
        "images":     {"ego_view": <base64-encoded JPEG bytes>},
    }

Launch:
    ~/.venv_cam/bin/python ~/realsense_zmq_publisher.py [--width 640]
    [--height 480] [--fps 30] [--port 5555] [--jpeg-quality 80]

Stop with Ctrl-C or `tmux kill-session -t cam`.
"""

from __future__ import annotations

import argparse
import base64
import signal
import sys
import time

import cv2  # type: ignore[import]
import msgpack  # type: ignore[import]
import numpy as np  # type: ignore[import]
import pyrealsense2 as rs  # type: ignore[import]
import zmq  # type: ignore[import]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--topic", type=str, default="ego_view")
    args = parser.parse_args()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{args.port}")
    print(f"[cam] ZMQ PUB bound on tcp://*:{args.port}", flush=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    pipeline.start(config)
    print(f"[cam] RealSense started: {args.width}x{args.height}@{args.fps} BGR8", flush=True)

    stopping = False

    def _handle_signal(signum, frame):  # type: ignore[no-untyped-def]
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]
    frames_sent = 0
    t_report = time.monotonic()

    try:
        while not stopping:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            color = frames.get_color_frame()
            if not color:
                continue
            bgr = np.asanyarray(color.get_data())
            ok, jpeg = cv2.imencode(".jpg", bgr, encode_params)
            if not ok:
                continue
            payload = msgpack.packb(
                {
                    "timestamps": {args.topic: time.time()},
                    "images": {args.topic: base64.b64encode(bytes(jpeg)).decode("ascii")},
                },
                use_bin_type=True,
            )
            sock.send(payload)
            frames_sent += 1
            now = time.monotonic()
            if now - t_report >= 5.0:
                print(
                    f"[cam] sent {frames_sent} frames, ~{frames_sent / (now - t_report):.1f} fps",
                    flush=True,
                )
                frames_sent = 0
                t_report = now
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        try:
            sock.close(linger=0)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
