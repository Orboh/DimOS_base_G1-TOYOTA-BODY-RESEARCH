# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Offline checks for the segmentation path of detect_okra (Step 4).

A stub detector returns detections WITH a binary mask; verify the mask centroid
(not bbox centre) drives (u,v) and the mask-internal median depth (not a point
sample) drives the 3D depth. No robot, no ultralytics, no graph import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dimos.robot.unitree.g1.harvest.detect_yolo import (
    YoloOkraDetector,
    _mask_centroid,
    make_zed_pixel_to_base,
)


@dataclass
class _SegDet:
    name: str
    bbox: tuple[float, float, float, float]
    mask: Any = None
    confidence: float = 0.9
    track_id: int | None = None


class _StubDetector:
    def __init__(self, dets):
        self._dets = dets

    def process_image(self, _frame):
        return self._dets


class _Frame:
    width = 640
    height = 480


def _square_mask(x0: int, y0: int, x1: int, y1: int, h: int = 480, w: int = 640):
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0:y1, x0:x1] = 255
    return m


def test_mask_centroid_offset_from_bbox_centre() -> None:
    """Centroid of an L-shaped mask differs from the bbox centre."""
    m = np.zeros((480, 640), dtype=np.uint8)
    # Fill only the left half of a wide bbox -> centroid pulled left of bbox centre.
    m[200:260, 300:340] = 255
    det = _SegDet(name="okra", bbox=(300, 200, 400, 260), mask=m)
    uv = _mask_centroid(det)
    assert uv is not None
    u, _ = uv
    bbox_centre_u = (300 + 400) / 2.0
    assert u < bbox_centre_u  # centroid sits on the filled (left) part, not bbox middle


def test_no_mask_returns_none() -> None:
    assert _mask_centroid(_SegDet(name="okra", bbox=(0, 0, 10, 10), mask=None)) is None


def test_seg_uses_mask_median_depth() -> None:
    """Depth comes from the median of mask-internal samples, not a single point."""
    # depth_getter returns the column index /1000 -> varies across the mask; the
    # median over the mask columns [300,340) is ~0.320 m, stable to outliers.
    def depth_getter(u: float, v: float) -> float:
        return u / 1000.0

    mask = _square_mask(300, 200, 340, 260)
    det = _SegDet(name="okra", bbox=(300, 200, 340, 260), mask=mask)
    yolo = YoloOkraDetector(
        detector=_StubDetector([det]),
        frame_getter=lambda: _Frame(),
        target_classes={"okra"},
        depth_getter=depth_getter,
    )
    okra = yolo.detect()
    assert len(okra) == 1
    # median column in [300,340) ~ 319-320 -> depth ~0.32 m -> y (depth+forward) ~0.32+offset
    y = okra[0].pos_3d["y"]
    assert 0.30 < y < 0.42  # near the median depth, not an extreme single-pixel value


def test_seg_falls_back_to_point_depth_without_mask() -> None:
    """No mask -> bbox centre + point depth (legacy path) still works."""
    def depth_getter(u: float, v: float) -> float:
        return 0.5

    det = _SegDet(name="okra", bbox=(300, 220, 340, 260), mask=None)
    yolo = YoloOkraDetector(
        detector=_StubDetector([det]),
        frame_getter=lambda: _Frame(),
        target_classes={"okra"},
        depth_getter=depth_getter,
    )
    okra = yolo.detect()
    assert len(okra) == 1
    assert okra[0].pos_3d["y"] == 0.5 + 0.05  # point depth + cam forward offset


def test_mask_median_ignores_invalid_depths() -> None:
    """Zeros / out-of-range depths inside the mask are dropped before the median."""
    calls = {"n": 0}

    def depth_getter(u: float, v: float) -> float:
        calls["n"] += 1
        # Half the samples are invalid (0.0), half are a valid 0.40 m.
        return 0.40 if int(u) % 2 == 0 else 0.0

    mask = _square_mask(300, 200, 360, 260)
    det = _SegDet(name="okra", bbox=(300, 200, 360, 260), mask=mask)
    yolo = YoloOkraDetector(
        detector=_StubDetector([det]),
        frame_getter=lambda: _Frame(),
        target_classes={"okra"},
        depth_getter=depth_getter,
    )
    okra = yolo.detect()
    assert len(okra) == 1
    # Only the valid 0.40 m samples survive -> median 0.40 -> y = 0.45.
    assert okra[0].pos_3d["y"] == 0.40 + 0.05


def test_zed_pixel_to_base_uses_real_intrinsics() -> None:
    """With intrinsics available, lateral/vertical come from proper pinhole
    back-projection (fx,fy,cx,cy) instead of the D435i angular guess."""
    fx, fy, cx, cy = 700.0, 700.0, 640.0, 360.0
    depth = 0.40
    pixel_to_base = make_zed_pixel_to_base(
        depth_getter=lambda u, v: depth,
        intrinsics_getter=lambda: (fx, fy, cx, cy),
    )
    det = _SegDet(name="okra", bbox=(0, 0, 10, 10), mask=None)
    # A pixel 70px right and 35px below the principal point.
    u, v = cx + 70.0, cy + 35.0
    pos = pixel_to_base(u, v, det)
    assert pos["y"] == depth
    assert pos["x"] == (u - cx) * depth / fx
    assert pos["z"] == -((v - cy) * depth / fy)
    # Right-of-centre pixel -> lateral > 0 (+right); below-centre -> height < 0.
    assert pos["x"] > 0
    assert pos["z"] < 0


def test_zed_pixel_to_base_returns_none_without_intrinsics() -> None:
    """camera_info 未受信 → 推測せず None。

    以前は default_pixel_to_base（D435i 定数・地面原点）へフォールバックしていたが、
    下流の cam_to_torso 変換でカメラ高さが二重計上され、画像中心で z が約 110cm
    ずれていた。camera_info は毎秒1回配信なので、最大1秒待つ方が安全。
    """
    pixel_to_base = make_zed_pixel_to_base(
        depth_getter=lambda u, v: 0.5,
        intrinsics_getter=lambda: None,
    )
    det = _SegDet(name="okra", bbox=(0, 0, 10, 10), mask=None)
    assert pixel_to_base(320.0, 240.0, det) is None


def test_zed_pixel_to_base_returns_none_without_usable_depth() -> None:
    """深度が取れない/範囲外 → 定数(_ASSUMED_DEPTH_M=0.45)を当てず None。"""
    intr = (700.0, 700.0, 320.0, 240.0)
    det = _SegDet(name="okra", bbox=(0, 0, 10, 10), mask=None)

    # depth_getter 自体が無い
    assert make_zed_pixel_to_base(depth_getter=None, intrinsics_getter=lambda: intr)(
        320.0, 240.0, det
    ) is None
    # 深度が範囲外（ZED が 0 を返す = 無効値）
    assert make_zed_pixel_to_base(
        depth_getter=lambda u, v: 0.0, intrinsics_getter=lambda: intr
    )(320.0, 240.0, det) is None
    # 深度が NaN
    assert make_zed_pixel_to_base(
        depth_getter=lambda u, v: float("nan"), intrinsics_getter=lambda: intr
    )(320.0, 240.0, det) is None


def test_detect_skips_okra_without_3d_position() -> None:
    """3D 化できない検出は Okra リストに載らない（腕を動かさない）。"""
    mask = _square_mask(300, 200, 340, 260)
    seg_det = _SegDet(name="okra", bbox=(300, 200, 340, 260), mask=mask)
    yolo = YoloOkraDetector(
        detector=_StubDetector([seg_det]),
        frame_getter=lambda: _Frame(),
        target_classes={"okra"},
        pixel_to_base=make_zed_pixel_to_base(
            depth_getter=lambda u, v: 0.4,
            intrinsics_getter=lambda: None,  # camera_info 未受信
        ),
    )
    assert yolo.detect() == []


def test_zed_pixel_to_base_uses_mask_median_depth() -> None:
    """intrinsics が揃っていればマスク内 median 深度で 3D 化される。"""

    def depth_getter(u: float, v: float) -> float:
        return u / 1000.0

    mask = _square_mask(300, 200, 340, 260)
    seg_det = _SegDet(name="okra", bbox=(300, 200, 340, 260), mask=mask)
    yolo = YoloOkraDetector(
        detector=_StubDetector([seg_det]),
        frame_getter=lambda: _Frame(),
        target_classes={"okra"},
        pixel_to_base=make_zed_pixel_to_base(
            depth_getter=depth_getter, intrinsics_getter=lambda: (700.0, 700.0, 320.0, 240.0)
        ),
    )
    okra = yolo.detect()
    assert len(okra) == 1
    # Still driven by the mask-median depth (same behaviour as the legacy path).
    assert 0.30 < okra[0].pos_3d["y"] < 0.42
