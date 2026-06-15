# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Module wrapper around :class:`TeleimagerImageSource`.

Drop-in replacement for :class:`ZmqCamera`: it publishes a G1 camera's frames
on the SAME ``color_image`` stream, so downstream modules (nav viewer,
ActBridge) need no change — only the camera *source* differs (teleimager's
``image_server`` instead of the GEAR-SONIC ``ego_view`` publisher).

    NX:  teleimager-server --rs   (config :60000, frames :55555+)
    PC:  this module               (ImageClient -> color_image)

The ``camera`` config (env ``TELEIMAGER_CAMERA``) selects which teleimager
camera to publish: "head" (default), "left_wrist" or "right_wrist". Instantiate
one module per camera in a blueprint and remap each ``color_image`` to a
distinct stream (e.g. cam_left_high / cam_right_wrist).

Why teleimager: the okra ACT policy was trained/deployed against teleimager's
exact image format, so feeding it the same frames avoids any format guesswork.
A single boot publisher (teleimager) also removes the single-D435i contention
between the GEAR-SONIC publisher and teleimager.
"""

from __future__ import annotations

import os
import threading

from pydantic import Field

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.msgs.sensor_msgs.Image import Image
from dimos.robot.unitree.g1.camera.teleimager_image_source import (
    TeleimagerCameraConfig,
    TeleimagerImageSource,
)
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# How often to report receive stats / warn about a silent server. [s]
STATS_INTERVAL_S = 5.0


class TeleimagerCameraModuleConfig(ModuleConfig):
    host: str = Field(default_factory=lambda: os.getenv("TELEIMAGER_HOST", "192.168.123.164"))
    request_port: int = Field(default_factory=lambda: int(os.getenv("TELEIMAGER_PORT", "60000")))
    fps: float = Field(default_factory=lambda: float(os.getenv("TELEIMAGER_FPS", "30")))
    # Which teleimager camera to publish: "head" | "left_wrist" | "right_wrist".
    camera: str = Field(default_factory=lambda: os.getenv("TELEIMAGER_CAMERA", "head"))


class TeleimagerCamera(Module):
    """Publishes a teleimager camera's frames on color_image (RGB)."""

    config: TeleimagerCameraModuleConfig
    color_image: Out[Image]

    _source: TeleimagerImageSource | None = None
    _frame_count: int = 0
    _monitor_thread: threading.Thread | None = None
    _monitor_stop: threading.Event | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self._source = TeleimagerImageSource(
            TeleimagerCameraConfig(
                host=self.config.host,
                request_port=self.config.request_port,
                fps=self.config.fps,
                camera=self.config.camera,
            )
        )
        self._source.start()
        self._frame_count = 0
        self.register_disposable(self._source.video_stream().subscribe(self._on_frame))
        self._monitor_stop = threading.Event()
        self._monitor_thread = threading.Thread(
            target=self._monitor, daemon=True, name="teleimager-cam-monitor"
        )
        self._monitor_thread.start()
        logger.info(
            "TeleimagerCamera started",
            host=self.config.host,
            request_port=self.config.request_port,
            fps=self.config.fps,
            camera=self.config.camera,
        )

    def _on_frame(self, image: Image) -> None:
        self._frame_count += 1
        if self._frame_count == 1:
            logger.info(
                "TeleimagerCamera first frame received — publishing on color_image",
                camera=self.config.camera,
                size=f"{image.width}x{image.height}",
            )
        self.color_image.publish(image)

    def _monitor(self) -> None:
        """Report receive rate periodically; warn when the server is silent."""
        assert self._monitor_stop is not None
        last_count = 0
        while not self._monitor_stop.wait(STATS_INTERVAL_S):
            received = self._frame_count - last_count
            last_count = self._frame_count
            if received == 0:
                logger.warning(
                    "TeleimagerCamera: NO frames received — is teleimager-server running on the NX?",
                    host=self.config.host,
                    camera=self.config.camera,
                    hint="on the NX: teleimager-server --rs",
                )
            else:
                logger.info(
                    "TeleimagerCamera receiving",
                    camera=self.config.camera,
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


class RightWristTeleimagerCamera(TeleimagerCamera):
    """Distinct subclass of :class:`TeleimagerCamera` for the right-wrist camera.

    ``autoconnect()`` deduplicates blueprint atoms by module *class*, and the
    remapping API keys on (module class, stream). To run the head and a wrist
    TeleimagerCamera together in one blueprint, the wrist must be a distinct
    class so it is not deduped and its ``color_image`` Out can be remapped to a
    separate stream (e.g. cam_right_wrist). Instantiate with camera="right_wrist".
    """


__all__ = [
    "TeleimagerCamera",
    "TeleimagerCameraModuleConfig",
    "RightWristTeleimagerCamera",
]
