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

"""``IsaacZmqDepthCamera`` のデコードロジックのテスト。

``docs/sim-setup/sim_dds_bridge.py`` の ego_view ZMQ wire format（msgpack, JPEG/PNG16）
と同じ手順でメッセージを組み立て、``_handle_message`` に直接渡す（実 ZMQ ソケットは使わない）。
"""

from __future__ import annotations

import base64

import cv2
import numpy as np
import pytest

from dimos.simulation.engines.isaac_zmq_camera import (
    IsaacZmqDepthCamera,
    _parse_cam_to_torso,
)

_TOPIC = "ego_view"


def _make_msg(
    *,
    width: int = 8,
    height: int = 6,
    with_depth: bool = True,
    depth_mm: int = 1500,
    cam_to_torso: str = "0.1090,0.0300,0.2480,-0.49475,0.49475,-0.50520,0.50520",
) -> dict:
    """sim_dds_bridge.py と同じ手順でエンコードした ego_view msgpack 相当の dict を作る。"""
    bgr = np.zeros((height, width, 3), dtype=np.uint8)
    ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    assert ok
    msg: dict = {
        "images": {_TOPIC: base64.b64encode(jpg.tobytes()).decode("ascii")},
        "timestamps": {_TOPIC: 123.456},
        "intrinsics": {_TOPIC: [320.0, 0.0, width / 2.0, 0.0, 320.0, height / 2.0, 0.0, 0.0, 1.0]},
        "cam_to_torso": cam_to_torso,
    }
    if with_depth:
        d16 = np.full((height, width), depth_mm, dtype=np.uint16)
        okd, dpng = cv2.imencode(".png", d16)
        assert okd
        msg["depth"] = {_TOPIC: base64.b64encode(dpng.tobytes()).decode("ascii")}
        msg["depth_scale"] = 0.001  # mm -> m
    return msg


@pytest.fixture
def make_camera():  # type: ignore[no-untyped-def]
    """IsaacZmqDepthCamera を作り、テスト終了時に stop() する（Module.__init__ が
    RPC/TF 用のバックグラウンドスレッドを起動するため、stop() し忘れると
    conftest.py の monitor_threads チェックに引っかかる）。
    """
    cameras: list[IsaacZmqDepthCamera] = []

    def _make(**kwargs) -> IsaacZmqDepthCamera:  # type: ignore[no-untyped-def]
        cam = IsaacZmqDepthCamera(zmq_topic=_TOPIC, **kwargs)
        cameras.append(cam)
        return cam

    yield _make
    for cam in cameras:
        cam.stop()


def test_handle_message_publishes_color_and_builds_camera_info(make_camera) -> None:  # type: ignore[no-untyped-def]
    cam = make_camera(width=8, height=6, enable_depth=False)
    msg = _make_msg(width=8, height=6, with_depth=False)

    published: list = []
    cam.color_image.subscribe(lambda img: published.append(img))

    assert cam._handle_message(msg) is True
    assert len(published) == 1
    assert published[0].data.shape[:2] == (6, 8)

    assert cam._color_camera_info is not None
    assert cam._color_camera_info.K[0] == 320.0  # fx
    assert cam._color_camera_info.K[4] == 320.0  # fy
    assert cam._color_camera_info.K[2] == 4.0  # cx
    assert cam._color_camera_info.K[5] == 3.0  # cy


def test_handle_message_decodes_depth_with_scale(make_camera) -> None:  # type: ignore[no-untyped-def]
    cam = make_camera(width=8, height=6, enable_depth=True)
    msg = _make_msg(width=8, height=6, with_depth=True, depth_mm=1500)

    depths: list = []
    cam.depth_image.subscribe(lambda img: depths.append(img))

    assert cam._handle_message(msg) is True
    assert len(depths) == 1
    # 1500mm * depth_scale(0.001) = 1.5m（PNG16はロスレスなので厳密一致）
    assert np.allclose(depths[0].data, 1.5)


def test_handle_message_skips_depth_when_disabled(make_camera) -> None:  # type: ignore[no-untyped-def]
    cam = make_camera(width=8, height=6, enable_depth=False)
    msg = _make_msg(width=8, height=6, with_depth=True)

    depths: list = []
    cam.depth_image.subscribe(lambda img: depths.append(img))

    assert cam._handle_message(msg) is True
    assert depths == []  # enable_depth=False なら depth を publish しない


def test_handle_message_stores_cam_to_torso(make_camera) -> None:  # type: ignore[no-untyped-def]
    cam = make_camera(width=8, height=6, enable_depth=False)
    msg = _make_msg(width=8, height=6, with_depth=False)

    assert cam._handle_message(msg) is True
    assert cam._cam_to_torso is not None
    assert cam._cam_to_torso.translation.x == 0.1090


def test_handle_message_returns_false_when_topic_missing(make_camera) -> None:  # type: ignore[no-untyped-def]
    cam = make_camera(width=8, height=6)
    msg = _make_msg(width=8, height=6, with_depth=False)
    msg["images"] = {}  # 対象トピックが無い
    assert cam._handle_message(msg) is False


def test_parse_cam_to_torso_roundtrip() -> None:
    t = _parse_cam_to_torso("0.1,0.2,0.3,0.0,0.0,0.0,1.0")
    assert t is not None
    assert (t.translation.x, t.translation.y, t.translation.z) == (0.1, 0.2, 0.3)


def test_parse_cam_to_torso_empty_returns_none() -> None:
    assert _parse_cam_to_torso("") is None
    assert _parse_cam_to_torso("   ") is None


def test_parse_cam_to_torso_wrong_count_returns_none() -> None:
    assert _parse_cam_to_torso("0.1,0.2,0.3") is None  # 7値でない
