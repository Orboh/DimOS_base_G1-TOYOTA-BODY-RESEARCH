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

"""ZMQ subscriber for RealSense camera frames from the G1 NX.

Matches the msgpack schema used by NVIDIA's GEAR-SONIC reference
(`gear_sonic.camera.composed_camera`):

    {
        "timestamps": {"ego_view": <float seconds>, ...},
        "images":     {"ego_view": <base64-encoded JPEG bytes>, ...},
    }

The publisher side runs on the G1 NX (where the RealSense is attached)
and emits on `tcp://*:5555` by default. This module is the laptop-side
subscriber used by `examples/g1_vla_bringup.py` stage 3 and, later,
`VLASkillContainer` for stage 7.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import threading
import time
from typing import Any

import cv2
import msgpack
import numpy as np
from reactivex import Observable
from reactivex.subject import Subject
import zmq

from dimos.msgs.sensor_msgs.Image import Image, ImageFormat


@dataclass
class ZmqCameraConfig:
    host: str = "192.168.123.164"
    port: int = 5555
    topic: str = "ego_view"
    recv_timeout_ms: int = 1000
    # Keep only the newest frame in the receive queue (ZMQ_CONFLATE). A live
    # camera consumer never wants a backlog: without this, a consumer slower
    # than the publisher accumulates stale frames and the view lags behind
    # reality by an unbounded amount.
    conflate: bool = True


class ZmqImageSource:
    """Subscribes to msgpack+base64-JPEG frames from the NX publisher."""

    def __init__(self, config: ZmqCameraConfig | None = None) -> None:
        self.config = config or ZmqCameraConfig()
        self._ctx: zmq.Context[zmq.Socket[bytes]] | None = None
        self._socket: zmq.Socket[bytes] | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._subject: Subject[Image] = Subject()
        self._latest: Image | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.SUB)
        if self.config.conflate:
            # Must be set BEFORE connect to take effect.
            self._socket.setsockopt(zmq.CONFLATE, 1)
        endpoint = f"tcp://{self.config.host}:{self.config.port}"
        self._socket.connect(endpoint)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.setsockopt(zmq.RCVTIMEO, self.config.recv_timeout_ms)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="zmq-cam")
        self._thread.start()

    def _run(self) -> None:
        assert self._socket is not None
        while not self._stop_event.is_set():
            try:
                payload = self._socket.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break
            img = self._decode_payload(payload)
            if img is None:
                continue
            self._latest = img
            self._subject.on_next(img)

    def _decode_payload(self, payload: bytes) -> Image | None:
        try:
            msg: dict[str, Any] = msgpack.unpackb(payload, raw=False)
            images = msg.get("images") or {}
            jpeg_b64 = images.get(self.config.topic)
            if jpeg_b64 is None:
                return None
            if isinstance(jpeg_b64, str):
                jpeg_bytes = base64.b64decode(jpeg_b64)
            else:
                jpeg_bytes = bytes(jpeg_b64)
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ts = float((msg.get("timestamps") or {}).get(self.config.topic, time.time()))
            return Image.from_numpy(rgb, format=ImageFormat.RGB, frame_id=self.config.topic, ts=ts)
        except Exception:
            return None

    def wait_for_first_frame(self, timeout_s: float) -> Image | None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._latest is not None:
                return self._latest
            time.sleep(0.02)
        return self._latest

    def video_stream(self) -> Observable[Image]:
        return self._subject

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None
