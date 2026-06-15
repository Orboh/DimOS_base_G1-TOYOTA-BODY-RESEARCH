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

"""``unitree-g1-nav-laptop`` plus the G1 head camera (RealSense via NX ZMQ).

The head D435i is USB-wired to the NX and unreachable from an external PC
(no WebRTC camera server on this hardware — verified 2026-06-05, see
docs/platforms/humanoid/g1/index_orboh_make.md). The camera therefore comes
from the GEAR-SONIC ZMQ publisher running on the NX:

    # on the NX (one-time, see scripts/realsense_zmq_publisher.py)
    ~/.venv_cam/bin/python ~/realsense_zmq_publisher.py

    # on the laptop
    dimos run unitree-g1-nav-laptop-cam   # ZMQ_CAMERA_HOST/PORT to override

Locomotion stays DDS-only (G1HighLevelDdsSdk from the base blueprint); the
camera module has no control path at all.
"""

from __future__ import annotations

import os
from typing import Any

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.msgs.sensor_msgs.Image import Image
from dimos.robot.unitree.g1.blueprints.navigation.unitree_g1_nav_laptop import (
    build_nav_laptop,
)
from dimos.robot.unitree.g1.camera.teleimager_camera_module import TeleimagerCamera
from dimos.robot.unitree.g1.camera.zmq_camera_module import ZmqCamera


def _camera_blueprint() -> Any:
    """Head-camera source, switchable via the ``DIMOS_CAMERA_SOURCE`` env var.

    - ``"zmq"`` (default): GEAR-SONIC ``ego_view`` publisher (scripts/uvc_zmq_publisher.py)
    - ``"teleimager"``:     teleimager ``image_server`` (``teleimager-server --rs``)

    Both emit on ``color_image`` so downstream (nav viewer, ActBridge) is unchanged.
    """
    source = os.getenv("DIMOS_CAMERA_SOURCE", "zmq").strip().lower()
    if source == "teleimager":
        return TeleimagerCamera.blueprint()
    return ZmqCamera.blueprint()


def _nav_cam_rerun_blueprint() -> Any:
    """Camera panel + nav 3D view, mirroring uintree_g1_primitive_no_nav.

    The nav default layout is 3D-only; without an explicit 2D view the
    /color_image feed is logged but never displayed.
    """
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="world/color_image", name="Camera"),
            rrb.Spatial3DView(origin="world", name="3D"),
            column_shares=[1, 2],
        ),
    )


unitree_g1_nav_laptop_cam = autoconnect(
    build_nav_laptop(rerun_blueprint=_nav_cam_rerun_blueprint),
    _camera_blueprint(),
).transports(
    {
        # LCM (not pSHM) so the LCM-listening RerunBridge shows the feed,
        # matching uintree_g1_primitive_no_nav's camera transport.
        ("color_image", Image): LCMTransport("/color_image", Image),
    }
)

__all__ = ["unitree_g1_nav_laptop_cam"]
