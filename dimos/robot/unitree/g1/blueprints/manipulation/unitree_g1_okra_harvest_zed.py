#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Blueprint: オクラ収穫 + ZED-M カメラ（color_image + depth → 実 3D 検出）。

    dimos run unitree-g1-okra-harvest-zed

ZEDCamera (AGX Orin の ZED-M) を color + depth で使用する。
depth_image は HarvestModule に配線され、YoloOkraDetector が仮定深度ピンホール推定の代わりに
検出ピクセルでの ZED 実深度を使用できるようになる。

前提条件:
  - ZED SDK + pyzed が実行ホスト（AGX Orin）にインストール済み
  - ZED-M カメラが USB3 で接続済み
  - Ollama + qwen3-vl:2b が実行中（verify 用）
    ``ollama pull qwen3-vl:2b``
  - 検出プロキシ: COCO yolo11n（"banana"）→ オクラファインチューニング済み重みで置き換え予定
"""

from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.hardware.sensors.camera.zed.camera import ZEDCamera
from dimos.msgs.sensor_msgs.Image import Image
from dimos.robot.unitree.g1.harvest.harvest_module import HarvestModule

unitree_g1_okra_harvest_zed = (
    autoconnect(
        ZEDCamera.blueprint(depth_mode="PERFORMANCE"),  # TensorRT 不要の軽量深度モード
        HarvestModule.blueprint(
            use_dummy=False,
            use_zed_depth=True,
            use_g1_speaker=True,
            vlm_model="qwen3-vl:2b",
            target_classes="banana",  # COCO プロキシ — オクラ重み準備後に "okra" へ変更
        ),
    )
    .remappings(
        [
            (HarvestModule, "color_image", "color_image"),
            (HarvestModule, "depth_image", "depth_image"),
        ]
    )
    .transports(
        {
            ("color_image", Image): LCMTransport("/color_image", Image),
            ("depth_image", Image): LCMTransport("/depth_image", Image),
        }
    )
)

__all__ = ["unitree_g1_okra_harvest_zed"]
