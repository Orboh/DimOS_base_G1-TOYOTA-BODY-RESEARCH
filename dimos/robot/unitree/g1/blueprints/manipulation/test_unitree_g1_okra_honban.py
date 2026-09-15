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

"""``unitree_g1_okra_honban`` のカメラソース切替（``OKRA_CAMERA_SOURCE``）テスト。

``ZEDCamera`` は pyzed をトップレベル import するため、Isaac Sim 実行ホスト
（pyzed 未インストール）でも honban.py を import できることが sim→ZED互換Module
移行（``IsaacZmqDepthCamera``）の前提。既定 "zed" が実機の後方互換を壊していない
ことも併せて確認する。
"""

from __future__ import annotations

import importlib

import pytest

from dimos.hardware.sensors.camera.zed import compat as zed_compat

# honban.py の既定("zed")分岐は camera.py の実物 ZEDCamera を直接importする
# （pyzed をトップレベルimportするクラス）。SDK 未インストール環境（CI等）では
# ここで ImportError が起き、importlib.reload() が失敗してモジュールが壊れた
# 状態のまま残り、後続の sim 系テストまで巻き添えで落ちる（2026-09-15 CIで発覚）。
# test_zed.py と同じ HAS_ZED_SDK パターンでこのテストだけスキップする。


def _reload_honban():  # type: ignore[no-untyped-def]
    import dimos.robot.unitree.g1.blueprints.manipulation.unitree_g1_okra_honban as m

    return importlib.reload(m)


@pytest.mark.skipif(not zed_compat.HAS_ZED_SDK, reason="ZED SDK not installed")
def test_default_camera_source_is_zed(monkeypatch: pytest.MonkeyPatch) -> None:
    """OKRA_CAMERA_SOURCE 未指定＝既定は実機 ZED（後方互換）。"""
    monkeypatch.delenv("OKRA_CAMERA_SOURCE", raising=False)
    m = _reload_honban()
    assert m._CAMERA_SOURCE == "zed"

    atoms = m._camera_module.blueprints
    assert len(atoms) == 1
    from dimos.hardware.sensors.camera.zed.camera import ZEDCamera

    assert atoms[0].module is ZEDCamera


def test_sim_camera_source_uses_isaac_zmq_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    """OKRA_CAMERA_SOURCE=sim は ZEDCamera(pyzed依存) を import せず、
    IsaacZmqDepthCamera（perception.DepthCamera 互換の sim カメラ）へ切り替わる。
    """
    monkeypatch.setenv("OKRA_CAMERA_SOURCE", "sim")
    m = _reload_honban()
    assert m._CAMERA_SOURCE == "sim"

    atoms = m._camera_module.blueprints
    assert len(atoms) == 1
    from dimos.simulation.engines.isaac_zmq_camera import IsaacZmqDepthCamera

    assert atoms[0].module is IsaacZmqDepthCamera
    # HarvestModule(use_zed_depth=True) が要求する3出力を perception.DepthCamera として
    # 提供できていること（ZEDCamera と同じストリーム形状）。
    out_names = {s.name for s in atoms[0].streams if s.direction == "out"}
    assert {"color_image", "depth_image", "camera_info"} <= out_names


def test_sim_camera_source_reads_zmq_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """OKRA_SIM_CAM_HOST/PORT/TOPIC が IsaacZmqDepthCamera の blueprint kwargs に渡ること。"""
    monkeypatch.setenv("OKRA_CAMERA_SOURCE", "sim")
    monkeypatch.setenv("OKRA_SIM_CAM_HOST", "100.64.0.5")
    monkeypatch.setenv("OKRA_SIM_CAM_PORT", "6001")
    monkeypatch.setenv("OKRA_SIM_CAM_TOPIC", "chest_cam")
    m = _reload_honban()

    kwargs = m._camera_module.blueprints[0].kwargs
    assert kwargs["zmq_host"] == "100.64.0.5"
    assert kwargs["zmq_port"] == 6001
    assert kwargs["zmq_topic"] == "chest_cam"
