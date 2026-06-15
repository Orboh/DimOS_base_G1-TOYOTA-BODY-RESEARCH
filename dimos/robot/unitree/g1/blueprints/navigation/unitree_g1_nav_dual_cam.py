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

"""``unitree-g1-nav-laptop`` plus BOTH G1 cameras (head + right wrist) via teleimager.

Same as ``unitree-g1-nav-laptop-cam`` but shows two camera panels in Rerun:

    head        -> color_image      (LCM /color_image)        -> world/color_image
    right_wrist -> cam_right_wrist   (LCM /cam_right_wrist)    -> world/cam_right_wrist

Both come from teleimager's image_server on the NX (head_camera + right_wrist_camera
enabled in cam_config_server.yaml). The wrist uses a distinct module subclass
(:class:`RightWristTeleimagerCamera`) because ``autoconnect()`` deduplicates blueprint
atoms by class and remappings key on (class, stream) — so the wrist can be wired and
remapped independently of the head.

    # on the NX: teleimager-server --rs   (head + right_wrist serving)
    # on the laptop:
    .venv/bin/dimos run unitree-g1-nav-dual-cam
"""

from __future__ import annotations

from typing import Any

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.msgs.sensor_msgs.Image import Image
from dimos.robot.unitree.g1.blueprints.navigation.unitree_g1_nav_laptop import (
    build_nav_laptop,
)
from dimos.robot.unitree.g1.camera.teleimager_camera_module import (
    RightWristTeleimagerCamera,
    TeleimagerCamera,
)


def _dual_cam_rerun_blueprint() -> Any:
    """Rerun layout: head + right-wrist 2D panels (left, stacked) and the nav 3D view."""
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="world/color_image", name="Head"),
                rrb.Spatial2DView(origin="world/cam_right_wrist", name="Right Wrist"),
            ),
            rrb.Spatial3DView(origin="world", name="3D"),
            column_shares=[1, 2],
        ),
    )


unitree_g1_nav_dual_cam = (
    autoconnect(
        build_nav_laptop(rerun_blueprint=_dual_cam_rerun_blueprint),
        TeleimagerCamera.blueprint(camera="head"),
        RightWristTeleimagerCamera.blueprint(camera="right_wrist"),
    )
    .remappings(
        [
            # Wrist instance publishes color_image too; rename it so it does not
            # collide with the head and gets its own viewer panel.
            (RightWristTeleimagerCamera, "color_image", "cam_right_wrist"),
        ]
    )
    .transports(
        {
            ("color_image", Image): LCMTransport("/color_image", Image),
            ("cam_right_wrist", Image): LCMTransport("/cam_right_wrist", Image),
        }
    )
)

__all__ = ["unitree_g1_nav_dual_cam"]
