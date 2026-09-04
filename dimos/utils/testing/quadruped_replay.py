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

"""Sensor geometry for the recorded quadruped datasets under ``data/``.

The recordings (``unitree_go2_office_walk2``, ``unitree_go2_lidar_corrected``,
``unitree_go2_bigoffice*``, …) are still the fixtures for the mapping /
perception / memory tests, but the robot driver that produced them is no longer
in this repo. The camera intrinsics and the static mount chain that used to live
on ``GO2Connection`` are kept here so those replay tests stay runnable.
"""

from pathlib import Path

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo

_FRONT_CAMERA_720_YAML = Path(__file__).with_name("front_camera_720.yaml")


def camera_info() -> CameraInfo:
    """Intrinsics of the front camera used for the recordings (720p)."""
    return CameraInfo.from_yaml(str(_FRONT_CAMERA_720_YAML))


def odom_to_tf(odom: PoseStamped) -> list[Transform]:
    """Static mount chain of the recordings: odom -> base_link -> camera_optical."""
    camera_link = Transform(
        translation=Vector3(0.3, 0.0, 0.0),
        rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
        frame_id="base_link",
        child_frame_id="camera_link",
        ts=odom.ts,
    )

    camera_optical = Transform(
        translation=Vector3(0.0, 0.0, 0.0),
        rotation=Quaternion(-0.5, 0.5, -0.5, 0.5),
        frame_id="camera_link",
        child_frame_id="camera_optical",
        ts=odom.ts,
    )

    return [
        Transform.from_pose("base_link", odom),
        camera_link,
        camera_optical,
    ]


__all__ = ["camera_info", "odom_to_tf"]
