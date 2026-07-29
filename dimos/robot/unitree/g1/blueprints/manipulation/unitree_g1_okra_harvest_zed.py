#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Blueprint: オクラ収穫 + ZED-M カメラ（color_image + depth + camera_info → 実 3D 検出）。

    dimos run unitree-g1-okra-harvest-zed

ZEDCamera (AGX Orin の ZED-M) を color + depth + camera_info で使用する。腕・グリッパー・
右手首カメラ（teleimager、別パッケージ依存）は含まないので、G1本体の電源なしでも検出だけを
検証できる最小構成（アーム/歩行なし、grasp=DUMMY）。

depth_image は HarvestModule に配線され、YoloOkraDetector がマスク内の ZED 実深度 median を
使用する。camera_info（ZED 実測の焦点距離・画像中心）も配線され、ピクセル→3D の左右・上下位置は
D435i 頭部カメラ用の画角当て推量ではなく、実 intrinsics による正しい逆投影で計算される
（``detect_yolo.make_zed_pixel_to_base``）。

前提条件:
  - ZED SDK + pyzed が実行ホスト（AGX Orin）にインストール済み
  - ZED-M カメラが USB3 で接続済み
  - Ollama + qwen3-vl:2b が実行中（verify 用、任意。OKRA_VLM_MODEL="" で無効化可）
    ``ollama pull qwen3-vl:2b``
  - OKRA_YOLO_MODEL（既定 "okra11n-seg.pt"、data/models_yolo/ 配下）
  - OKRA_TARGET（既定 "okra"）
"""

from __future__ import annotations

import os

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.hardware.sensors.camera.zed.camera import ZEDCamera
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.robot.unitree.g1.harvest.harvest_module import HarvestModule

unitree_g1_okra_harvest_zed = (
    autoconnect(
        ZEDCamera.blueprint(depth_mode=os.getenv("ZED_DEPTH_MODE", "NEURAL")),
        HarvestModule.blueprint(
            use_dummy=False,
            use_zed_depth=True,
            use_g1_speaker=True,
            vlm_model=os.getenv("OKRA_VLM_MODEL", "qwen3-vl:2b"),
            # オクラ専用 seg 重み（data/models_yolo/okra11n-seg.pt）。
            yolo_model=os.getenv("OKRA_YOLO_MODEL", "okra11n-seg.pt"),
            target_classes=os.getenv("OKRA_TARGET", "okra"),
        ),
    )
    .remappings(
        [
            (HarvestModule, "color_image", "color_image"),
            (HarvestModule, "depth_image", "depth_image"),
            (HarvestModule, "camera_info", "camera_info"),
        ]
    )
    .transports(
        {
            ("color_image", Image): LCMTransport("/color_image", Image),
            ("depth_image", Image): LCMTransport("/depth_image", Image),
            ("camera_info", CameraInfo): LCMTransport("/camera_info", CameraInfo),
        }
    )
)

__all__ = ["unitree_g1_okra_harvest_zed"]
