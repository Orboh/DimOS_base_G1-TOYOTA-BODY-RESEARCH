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

"""UVC (V4L2) → ZMQ publisher for the G1 NX — pyrealsense2-free fallback.

Same wire format as ``realsense_zmq_publisher.py`` (GEAR-SONIC msgpack
schema on ``tcp://*:5555``), but reads the D435i's RGB sensor as a plain
UVC camera via OpenCV/V4L2. Use this when pyrealsense2 has no wheel for
the NX's Python (e.g. py3.13/aarch64) — color-only, no depth needed.

    {
        "timestamps": {"ego_view": <float seconds>},
        "images":     {"ego_view": <base64-encoded JPEG bytes>},
    }

Launch:
    ~/.venv_cam/bin/python ~/uvc_zmq_publisher.py            # auto-detect node
    ~/.venv_cam/bin/python ~/uvc_zmq_publisher.py --device 4 # explicit /dev/video4

Stop with Ctrl-C or `tmux kill-session -t cam`.
"""

from __future__ import annotations

import argparse
import base64
import signal
import sys
import time

import cv2  # type: ignore[import]
import zmq  # type: ignore[import]

try:
    import msgpack  # type: ignore[import]

    def _packb(obj: dict) -> bytes:
        return msgpack.packb(obj, use_bin_type=True)
except ImportError:  # NX ships u-msgpack-python (module name `umsgpack`) instead
    import umsgpack  # type: ignore[import]

    def _packb(obj: dict) -> bytes:
        return umsgpack.packb(obj)  # always bin-type on py3; no kwarg supported


# The D435i exposes ~6 V4L2 nodes (depth/IR/RGB/metadata); only the RGB
# node yields a valid 3-channel YUYV frame, which auto-detection relies on.
MAX_PROBE_DEVICES = 10
# Verified on the G1 NX (Sota, 2026-06-03): /dev/video4 = RGB 640x480.
DEFAULT_COLOR_DEVICE = 4
# Frames to discard at startup so auto-exposure settles before publishing.
WARMUP_FRAMES = 30


def _open_capture(device: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    """Open one V4L2 device configured for YUYV color at the given size."""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def _autodetect_device(width: int, height: int, fps: int) -> int | None:
    """Probe /dev/video0..N and return the first index that yields a 3-channel frame."""
    for idx in range(MAX_PROBE_DEVICES):
        cap = _open_capture(idx, width, height, fps)
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None and frame.ndim == 3 and frame.shape[2] == 3:
            print(f"[cam] auto-detected color node /dev/video{idx}", flush=True)
            return idx
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        type=int,
        default=DEFAULT_COLOR_DEVICE,
        help="/dev/videoN index (default: verified G1 NX RGB node), -1 = auto-detect",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--topic", type=str, default="ego_view")
    args = parser.parse_args()

    device = args.device
    if device < 0:
        detected = _autodetect_device(args.width, args.height, args.fps)
        if detected is None:
            print("[cam] ERROR: no usable color V4L2 device found", flush=True)
            return 1
        device = detected

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{args.port}")
    print(f"[cam] ZMQ PUB bound on tcp://*:{args.port}", flush=True)

    cap = _open_capture(device, args.width, args.height, args.fps)
    if not cap.isOpened():
        print(f"[cam] ERROR: cannot open /dev/video{device}", flush=True)
        return 1
    print(
        f"[cam] UVC started: /dev/video{device} {args.width}x{args.height}@{args.fps}", flush=True
    )

    # Discard warmup frames so auto-exposure settles before the first publish.
    for _ in range(WARMUP_FRAMES):
        cap.read()
    print(f"[cam] warmup done ({WARMUP_FRAMES} frames discarded)", flush=True)

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
            ok, bgr = cap.read()
            if not ok or bgr is None:
                time.sleep(0.01)
                continue
            ok, jpeg = cv2.imencode(".jpg", bgr, encode_params)
            if not ok:
                continue
            payload = _packb(
                {
                    "timestamps": {args.topic: time.time()},
                    "images": {args.topic: base64.b64encode(bytes(jpeg)).decode("ascii")},
                }
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
        cap.release()
        try:
            sock.close(linger=0)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
