# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Image source backed by the Unitree/teleimager ``image_server``.

This is the teleimager counterpart of :class:`ZmqImageSource` (GEAR-SONIC).
It reuses teleimager's own ``ImageClient`` (the exact client unitree_lerobot's
``eval_g1.py`` uses) so the wire format is guaranteed correct — we do NOT
re-implement teleimager's protocol.

    NX (server): teleimager-server --rs        (config REQ-REP :60000, frames PUB :55555+)
    PC (client): this source                    (ImageClient -> color_image)

The ``camera`` field selects which teleimager camera to pull — "head",
"left_wrist" or "right_wrist" — mapping to ImageClient.get_<camera>_frame().
This lets one module type serve any G1 camera; a blueprint instantiates one
per camera and remaps the output stream.

``ImageClient`` is imported lazily inside :meth:`start` so the rest of DimOS
(and the blueprint) can import this module even when teleimager is not
installed in the active venv. To use it, install teleimager into the DimOS
venv (``pip install teleimager``) or point ``TELEIMAGER_IMAGECLIENT`` at an
importable ``module.path:ClassName``.
"""

from __future__ import annotations

import importlib
import os
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
from reactivex import Observable
from reactivex.subject import Subject

from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# Candidate import paths for teleimager's ImageClient, tried in order. Override
# with TELEIMAGER_IMAGECLIENT="module.path:ClassName".
_IMAGECLIENT_CANDIDATES = (
    "teleimager.image_client:ImageClient",
    "teleimager:ImageClient",
    "image_server.image_client:ImageClient",
    "unitree_lerobot.eval_robot.image_server.image_client:ImageClient",
)


def _load_image_client_cls() -> type:
    """Resolve teleimager's ImageClient class lazily, with a helpful error."""
    override = os.getenv("TELEIMAGER_IMAGECLIENT")
    candidates = (override, *_IMAGECLIENT_CANDIDATES) if override else _IMAGECLIENT_CANDIDATES
    errors: list[str] = []
    for spec in candidates:
        mod_name, _, cls_name = spec.partition(":")
        if not cls_name:
            cls_name = "ImageClient"
        try:
            module = importlib.import_module(mod_name)
            return getattr(module, cls_name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{spec}: {exc}")
    raise ImportError(
        "Could not import teleimager's ImageClient. Install teleimager into the "
        "DimOS venv (pip install teleimager) or set TELEIMAGER_IMAGECLIENT="
        "'module.path:ClassName'.\nTried:\n  " + "\n  ".join(errors)
    )


@dataclass
class TeleimagerCameraConfig:
    host: str = "192.168.123.164"
    request_port: int = 60000  # teleimager config REQ-REP port
    fps: float = 30.0  # frame poll rate [Hz]
    camera: str = "head"  # which camera to pull: "head" | "left_wrist" | "right_wrist"
    frame_id: str = ""  # default: same as `camera`
    # teleimager decodes to BGR when request_bgr=True (frame.bgr populated); we
    # then convert to RGB to match what the okra ACT policy was trained on.
    request_bgr: bool = True


class TeleimagerImageSource:
    """Polls a teleimager camera frame and emits RGB :class:`Image` frames."""

    def __init__(self, config: TeleimagerCameraConfig | None = None) -> None:
        self.config = config or TeleimagerCameraConfig()
        self._client: object | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._subject: Subject[Image] = Subject()
        self._latest: Image | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        image_client_cls = _load_image_client_cls()
        self._client = image_client_cls(host=self.config.host, request_bgr=self.config.request_bgr)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="teleimager-cam")
        self._thread.start()

    def _run(self) -> None:
        assert self._client is not None
        period = 1.0 / max(1e-3, self.config.fps)
        # Select the per-camera getter: get_head_frame / get_left_wrist_frame / get_right_wrist_frame
        getter = getattr(self._client, f"get_{self.config.camera}_frame")  # type: ignore[attr-defined]
        frame_id = self.config.frame_id or self.config.camera
        next_t = time.perf_counter()
        while not self._stop_event.is_set():
            img = self._poll_frame(getter, frame_id)
            if img is not None:
                self._latest = img
                self._subject.on_next(img)
            next_t += period
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                self._stop_event.wait(sleep_s)
            else:
                next_t = time.perf_counter()

    def _poll_frame(self, getter, frame_id) -> Image | None:  # noqa: ANN001
        try:
            frame = getter()
            if frame is None:
                return None
            bgr = getattr(frame, "bgr", None)
            # Some teleimager builds return the ndarray directly.
            if bgr is None and isinstance(frame, np.ndarray):
                bgr = frame
            if bgr is None:
                return None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ts = float(getattr(frame, "timestamp", None) or time.time())
            return Image.from_numpy(rgb, format=ImageFormat.RGB, frame_id=frame_id, ts=ts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("TeleimagerImageSource: frame poll failed", error=str(exc))
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
        if self._client is not None:
            try:
                self._client.close()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            self._client = None


__all__ = ["TeleimagerCameraConfig", "TeleimagerImageSource"]
