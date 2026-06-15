# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Interim ``detect_okra`` backed by the DimOS YOLO 2D detector (head camera).

This is the perception leg of the real wiring (the ``detect_fn`` injected into
``DimosHarvestSkills``). It runs YOLO on the **head camera** frame (same view the
okra-ACT policy uses), keeps the configured target class(es), and maps each
box to an :class:`Okra` with a RELATIVE 3D position.

⚠️ Honest status / known gaps (interim, not robot-verified):
* **Stock ``yolo11n.pt`` is COCO — it has no "okra" class.** Use it now either
  with a PROXY class (e.g. ``{"banana"}``) to validate the
  detect→3D→select→grasp plumbing, or swap in an okra-fine-tuned weight +
  ``target_classes={"okra"}`` later (design v0.7 = YOLO fine-tune). For real
  okra with no training, an open-vocab VLM (Moondream) is the alternative.
* **3D position needs calibration + depth.** The graph only uses ``pos_3d`` (not
  ``Okra.reachable``), so accuracy here drives grasping. :func:`default_pixel_to_base`
  is a rough pinhole estimate at an ASSUMED depth — replace with real intrinsics
  + head-cam→base extrinsics (and a depth/pointcloud lookup) before a real run.
* **Ripeness is a placeholder** (all detections = ripe). Real ripeness needs a
  classifier / VLM on the crop (``det.cropped_image()``); inject ``ripeness_fn``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from dimos.robot.unitree.g1.harvest.blackboard import Okra

# Rough head-camera placeholders for the pinhole estimate (REPLACE with the
# measured D435i intrinsics + mounting transform). [deg] / [m].
_HFOV_DEG = 69.0  # D435i color HFOV (approx)
_VFOV_DEG = 42.0
_CAM_HEIGHT_M = 1.10  # head camera height off the ground (approx, G1 ~1.27 m)
_CAM_FORWARD_M = 0.05  # camera forward offset from the base origin
_ASSUMED_DEPTH_M = 0.45  # fallback fruit depth when no depth source is wired


def default_pixel_to_base(
    u: float,
    v: float,
    *,
    image_w: int,
    image_h: int,
    depth_m: float = _ASSUMED_DEPTH_M,
    hfov_deg: float = _HFOV_DEG,
    vfov_deg: float = _VFOV_DEG,
    cam_height_m: float = _CAM_HEIGHT_M,
    cam_forward_m: float = _CAM_FORWARD_M,
) -> dict[str, float]:
    """Rough pinhole pixel→base-frame {x,y,z} [m]. PLACEHOLDER — needs calibration.

    Frame: x=lateral(+right), y=depth(+forward), z=height(+up). Image +u=right,
    +v=down. At ``depth_m`` forward, lateral/vertical come from the pixel angle.
    """
    nx = (u / image_w - 0.5) * 2.0  # [-1, 1], +right
    ny = (v / image_h - 0.5) * 2.0  # [-1, 1], +down
    h_ang = math.radians(nx * hfov_deg / 2.0)
    v_ang = math.radians(ny * vfov_deg / 2.0)
    lateral = depth_m * math.tan(h_ang)
    vertical_drop = depth_m * math.tan(v_ang)
    return {
        "x": lateral,
        "y": depth_m + cam_forward_m,
        "z": cam_height_m - vertical_drop,
    }


class YoloOkraDetector:
    """Runs a 2D detector on the head frame and emits :class:`Okra` detections.

    Args:
        detector: anything with ``process_image(Image) -> iterable`` of detections
            exposing ``.name``, ``.bbox`` (x1,y1,x2,y2), ``.confidence`` and
            ``.track_id`` (the DimOS ``Yolo2DDetector`` satisfies this).
        frame_getter: returns the latest head-camera ``Image`` (e.g. the
            ``color_image`` stream's last frame).
        target_classes: class names to keep (proxy now, ``{"okra"}`` later).
        pixel_to_base: maps (u, v, det) -> relative {x,y,z} [m]. Defaults to the
            calibration-free :func:`default_pixel_to_base` (with a depth_getter if
            given, else an assumed depth).
        depth_getter: optional ``(u, v) -> depth_m`` from a depth/pointcloud
            stream; without it the assumed depth is used.
        ripeness_fn: optional ``det -> ripeness`` in [0,1]; defaults to 1.0.
        min_confidence: drop detections below this YOLO confidence.
    """

    def __init__(
        self,
        detector: Any,
        frame_getter: Callable[[], Any],
        target_classes: set[str],
        pixel_to_base: Callable[[float, float, Any], dict[str, float]] | None = None,
        depth_getter: Callable[[float, float], float] | None = None,
        ripeness_fn: Callable[[Any], float] | None = None,
        min_confidence: float = 0.5,
    ) -> None:
        self._detector = detector
        self._frame_getter = frame_getter
        self._targets = {c.lower() for c in target_classes}
        self._pixel_to_base = pixel_to_base
        self._depth_getter = depth_getter
        self._ripeness_fn = ripeness_fn
        self._min_conf = min_confidence

    def detect(self) -> list[Okra]:
        frame = self._frame_getter()
        if frame is None:
            return []
        dets = self._detector.process_image(frame)
        out: list[Okra] = []
        for idx, det in enumerate(dets):
            name = str(getattr(det, "name", "")).lower()
            if name not in self._targets:
                continue
            if float(getattr(det, "confidence", 1.0)) < self._min_conf:
                continue
            x1, y1, x2, y2 = getattr(det, "bbox")
            u = (x1 + x2) / 2.0
            v = (y1 + y2) / 2.0
            pos = self._pos_3d(u, v, det, frame)
            ripeness = float(self._ripeness_fn(det)) if self._ripeness_fn else 1.0
            track = getattr(det, "track_id", None)
            okra_id = f"okra_{track}" if track is not None else f"okra_{name}_{idx}"
            out.append(
                Okra(
                    id=okra_id,
                    img_region="R" if pos["x"] >= 0 else "L",
                    pos_3d=pos,
                    ripeness=ripeness,
                    reachable=False,  # graph recomputes from cfg.reach.contains(pos_3d)
                )
            )
        return out

    def _pos_3d(self, u: float, v: float, det: Any, frame: Any) -> dict[str, float]:
        if self._pixel_to_base is not None:
            return self._pixel_to_base(u, v, det)
        w = int(getattr(frame, "width", 640))
        h = int(getattr(frame, "height", 480))
        depth = self._depth_getter(u, v) if self._depth_getter else _ASSUMED_DEPTH_M
        return default_pixel_to_base(u, v, image_w=w, image_h=h, depth_m=depth)


def make_yolo_detect_okra(
    frame_getter: Callable[[], Any],
    target_classes: set[str] | None = None,
    *,
    model_name: str = "yolo11n.pt",
    detector: Any = None,
    **kwargs: Any,
) -> Callable[[], list[Okra]]:
    """Build a ``detect_fn`` (the ``detect_okra`` injectable) backed by YOLO.

    Defaults ``target_classes`` to ``{"banana"}`` — a COCO proxy for okra so the
    pipeline can be exercised before an okra-fine-tuned weight exists. Pass
    ``target_classes={"okra"}`` with an okra weight for the real thing.

    The DimOS ``Yolo2DDetector`` (ultralytics) is imported lazily so this module
    stays importable without that dependency.
    """
    if detector is None:
        from dimos.perception.detection.detectors.yolo import Yolo2DDetector

        detector = Yolo2DDetector(model_name=model_name)
    yolo = YoloOkraDetector(
        detector=detector,
        frame_getter=frame_getter,
        target_classes=target_classes or {"banana"},
        **kwargs,
    )
    return yolo.detect


__all__ = ["YoloOkraDetector", "default_pixel_to_base", "make_yolo_detect_okra"]
