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

"""Module wrapper around :class:`ZmqImageSource` (RealSense-on-NX camera).

The G1's head D435i is USB-wired to the onboard NX and is NOT reachable
from an external PC (no WebRTC camera server — verified 2026-06-05, see
docs/platforms/humanoid/g1/index_orboh_make.md). The working path is the
GEAR-SONIC ZMQ transport from the Orboh add-vla branch:

    NX:  scripts/realsense_zmq_publisher.py  (pyrealsense2 → JPEG → tcp://*:5555)
    PC:  this module                          (ZMQ SUB → color_image)
"""

from __future__ import annotations

import os
import threading

from pydantic import Field

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.msgs.sensor_msgs.Image import Image
from dimos.robot.unitree.g1.camera.zmq_image_source import ZmqCameraConfig, ZmqImageSource
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# How often to report receive stats / warn about a silent publisher. [s]
STATS_INTERVAL_S = 5.0


class ZmqCameraModuleConfig(ModuleConfig):
    host: str = Field(default_factory=lambda: os.getenv("ZMQ_CAMERA_HOST", "192.168.123.164"))
    port: int = Field(default_factory=lambda: int(os.getenv("ZMQ_CAMERA_PORT", "5555")))
    topic: str = "ego_view"


class ZmqCamera(Module):
    """Publishes RealSense frames from the NX ZMQ publisher on color_image."""

    config: ZmqCameraModuleConfig
    color_image: Out[Image]

    _source: ZmqImageSource | None = None
    _frame_count: int = 0
    _monitor_thread: threading.Thread | None = None
    _monitor_stop: threading.Event | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self._source = ZmqImageSource(
            ZmqCameraConfig(
                host=self.config.host,
                port=self.config.port,
                topic=self.config.topic,
            )
        )
        self._source.start()
        self._frame_count = 0
        self.register_disposable(self._source.video_stream().subscribe(self._on_frame))
        self._monitor_stop = threading.Event()
        self._monitor_thread = threading.Thread(
            target=self._monitor, daemon=True, name="zmq-cam-monitor"
        )
        self._monitor_thread.start()
        logger.info(
            "ZmqCamera started",
            endpoint=f"tcp://{self.config.host}:{self.config.port}",
            topic=self.config.topic,
        )

    def _on_frame(self, image: Image) -> None:
        self._frame_count += 1
        if self._frame_count == 1:
            logger.info(
                "ZmqCamera first frame received — publishing on color_image",
                size=f"{image.width}x{image.height}",
            )
        self.color_image.publish(image)

    def _monitor(self) -> None:
        """Report receive rate periodically; warn when the publisher is silent."""
        assert self._monitor_stop is not None
        last_count = 0
        while not self._monitor_stop.wait(STATS_INTERVAL_S):
            received = self._frame_count - last_count
            last_count = self._frame_count
            if received == 0:
                logger.warning(
                    "ZmqCamera: NO frames received — is the NX publisher running?",
                    endpoint=f"tcp://{self.config.host}:{self.config.port}",
                    hint="on the NX: nohup setsid python3 ~/uvc_zmq_publisher.py &",
                )
            else:
                logger.info(
                    "ZmqCamera receiving",
                    fps=round(received / STATS_INTERVAL_S, 1),
                    total=self._frame_count,
                )

    @rpc
    def stop(self) -> None:
        if self._monitor_stop is not None:
            self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
        if self._source is not None:
            self._source.stop()
            self._source = None
        super().stop()


__all__ = ["ZmqCamera", "ZmqCameraModuleConfig"]
