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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from dimos.hardware.sensors.camera.zed import compat as zed
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo

# camera.py は `import pyzed.sl as sl` をトップレベルで行うため、SDK 未インストール
# 環境ではこの import 自体が失敗する。既存パターン(HAS_ZED_SDK)に合わせ、対象
# テストにだけ skipif を付けて遅延 import する。


@pytest.mark.skipif(not zed.HAS_ZED_SDK, reason="ZED SDK not installed")
def test_zed_import_and_calibration_access() -> None:
    """Test that zed module can be imported and calibrations accessed."""
    # Test that CameraInfo is accessible
    assert hasattr(zed, "CameraInfo")

    # Test snake_case access
    camera_info_snake = zed.CameraInfo.single_webcam
    assert isinstance(camera_info_snake, CameraInfo)
    assert camera_info_snake.width == 640
    assert camera_info_snake.height == 376
    assert camera_info_snake.distortion_model == "plumb_bob"

    # Test PascalCase access
    camera_info_pascal = zed.CameraInfo.SingleWebcam
    assert isinstance(camera_info_pascal, CameraInfo)
    assert camera_info_pascal.width == 640
    assert camera_info_pascal.height == 376

    # Verify both access methods return the same cached object
    assert camera_info_snake is camera_info_pascal

    print("✓ ZED import and calibration access test passed!")


# _capture_loop の異常系ログ（2026-09-15）。
# grab() が例外を投げるとループが無言で終了し、以降フレーム配信が完全に止まる
# サイレント故障だった（実機LIVEで「1個収穫後にviewerの映像が止まった」原因調査
# で発覚）。stop() 経由（_running=False）の正常終了とは区別してログすることを
# 確認する。ZEDCamera 自体は Module の重い依存を持つため、_capture_loop が
# 使う属性だけを持つ軽量なフェイクに直接バインドして呼び出す。


def _fake_zed_camera(*, running: bool) -> SimpleNamespace:
    zed_mock = MagicMock()
    zed_mock.grab.side_effect = RuntimeError("usb disconnect (simulated)")
    return SimpleNamespace(_running=running, _zed=zed_mock, _depth_on=False, _runtime_params=None)


@pytest.mark.skipif(not zed.HAS_ZED_SDK, reason="ZED SDK not installed")
def test_capture_loop_logs_when_grab_raises_while_running() -> None:
    """running中にgrab()が例外を投げるのは異常系 -> ログに残す。"""
    from dimos.hardware.sensors.camera.zed.camera import ZEDCamera

    fake = _fake_zed_camera(running=True)
    with patch("dimos.hardware.sensors.camera.zed.camera.logger") as mock_logger:
        ZEDCamera._capture_loop(fake)
    mock_logger.exception.assert_called_once()


@pytest.mark.skipif(not zed.HAS_ZED_SDK, reason="ZED SDK not installed")
def test_capture_loop_silent_when_grab_raises_after_stop() -> None:
    """stop()済み(_running=False)後の例外は正常系 -> ログしない。"""
    from dimos.hardware.sensors.camera.zed.camera import ZEDCamera

    fake = _fake_zed_camera(running=False)
    with patch("dimos.hardware.sensors.camera.zed.camera.logger") as mock_logger:
        ZEDCamera._capture_loop(fake)
    mock_logger.exception.assert_not_called()
