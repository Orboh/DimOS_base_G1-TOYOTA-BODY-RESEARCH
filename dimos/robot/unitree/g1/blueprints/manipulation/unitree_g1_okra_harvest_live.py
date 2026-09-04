#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Blueprint: okra-harvest with REAL head-camera detection (first live cut).

    dimos run unitree-g1-okra-harvest-live

Wires the head camera (teleimager) into ``HarvestModule(use_dummy=False)``, so
``detect_okra`` runs real YOLO on the live ``color_image`` stream and the flow
selects targets from real detections.

REAL now: head-camera YOLO detection, Japanese G1-speaker announcements, and
``verify_harvest`` via a local Ollama vision model (``moondream``, ~1 s). Still
``[LIVE-TODO]``: base ``move`` / ``grasp`` (okra ACT) / nav — so the robot does
NOT move yet (grasp = stoppable DummyGraspModule). Next: the okra-ACT GraspModule
(arm reach), real safety checks, then base/nav.

Prereqs: NX teleimager-server running; ``ROBOT_INTERFACE`` set (G1 audio);
local **Ollama** running with the vision model (``ollama pull moondream``).
See ``dimos/robot/unitree/g1/harvest/README.md``.
"""

from __future__ import annotations

import os
from typing import Any

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.msgs.sensor_msgs.Image import Image
from dimos.robot.unitree.g1.camera.teleimager_camera_module import TeleimagerCamera
from dimos.robot.unitree.g1.camera.zmq_camera_module import ZmqCamera
from dimos.robot.unitree.g1.harvest.harvest_module import HarvestModule


def _camera_blueprint() -> Any:
    """Head-camera source. Defaults to teleimager (okra's training format)."""
    source = os.getenv("DIMOS_CAMERA_SOURCE", "teleimager").strip().lower()
    return ZmqCamera.blueprint() if source == "zmq" else TeleimagerCamera.blueprint()


unitree_g1_okra_harvest_live = autoconnect(
    _camera_blueprint(),
    HarvestModule.blueprint(use_dummy=False, use_g1_speaker=True, vlm_model="moondream"),
).transports(
    {
        ("color_image", Image): LCMTransport("/color_image", Image),
    }
)

__all__ = ["unitree_g1_okra_harvest_live"]
