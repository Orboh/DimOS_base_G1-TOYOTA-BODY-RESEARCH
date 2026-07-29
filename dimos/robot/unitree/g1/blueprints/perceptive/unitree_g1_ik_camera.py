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

"""Jetson-side camera node for the IK-reach experiment: D435i -> LCM point cloud.

Runs ON THE ROBOT'S JETSON (the D435i is local USB there). Opens the D435i with
DimOS's own ``RealSenseCamera`` and publishes the color-aligned point cloud,
color image, and intrinsics over LCM so the laptop app (``unitree-g1-ik-reach``)
can render the cloud for the human to click. The cloud is rooted at ``torso_link``
via the URDF d435_joint mount transform.

PRECONDITIONS (see plan):
- ``g1-teleimager.service`` MUST be stopped first to free the D435i (Sota/NX
  coordination; restore afterward). Verify: ``rs-enumerate-devices | grep 405622072808``.
- Run inside an env with pyrealsense2 2.50.0 (e.g. teleimager_relobot) + a minimal
  DimOS install. Set ``LCM_DEFAULT_URL=udpm://239.255.76.67:7667?ttl=1`` and a
  NIC-scoped multicast route on eth0 (NOT loopback) so the cloud egresses to the laptop.

Run:
    LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1' dimos run unitree-g1-ik-camera

Plain ``LCMTransport(topic, MsgType)`` (binary LCM) is used deliberately so the
laptop RerunBridge's default ``LCM()`` subscriber can decode + render the streams
(pickle/jpeg transports would be silently dropped by that subscriber).
"""

from __future__ import annotations

import os

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.transport import LCMTransport
from dimos.hardware.sensors.camera.realsense.camera import RealSenseCamera
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2

# D435i serial verified on the robot (2026-06-16). Override via env if it changes.
_D435_SERIAL = os.getenv("IK_CAMERA_SERIAL", "405622072808")

# URDF d435_joint extrinsic torso_link -> d435_link (g1.urdf:564-568): the cloud is
# published rooted at torso_link so the laptop receives it in a frame the IK uses.
_D435_MOUNT = Transform(
    translation=Vector3(0.0576235, 0.01753, 0.42987),
    rotation=Quaternion.from_euler(Vector3(0.0, 0.8307767239493009, 0.0)),
)

# Conservative cloud rate (blocker #9: bandwidth / UDP fragmentation). One-shot
# human-click pipeline, so a low rate is fine.
_PC_FPS = float(os.getenv("IK_CAMERA_PC_FPS", "3.0"))

unitree_g1_ik_camera = autoconnect(
    RealSenseCamera.blueprint(
        serial_number=_D435_SERIAL,
        enable_depth=True,
        enable_pointcloud=True,
        align_depth_to_color=True,
        pointcloud_fps=_PC_FPS,
        camera_info_fps=1.0,
        base_frame_id="torso_link",
        base_transform=_D435_MOUNT,
    ),
).transports(
    {
        ("pointcloud", PointCloud2): LCMTransport("/camera/pointcloud", PointCloud2),
        ("color_image", Image): LCMTransport("/camera/color_image", Image),
        ("camera_info", CameraInfo): LCMTransport("/camera/camera_info", CameraInfo),
    }
)

__all__ = ["unitree_g1_ik_camera"]
