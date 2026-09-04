#!/usr/bin/env python3
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

"""G1 Mid-360 FAST-LIO diagnostic stack.

This intentionally avoids the nav stack and local planner, so it does not
need the G1 precomputed local-planner LFS archive.
"""

from __future__ import annotations

import os

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.hardware.sensors.lidar.fastlio2.module import FastLio2
from dimos.navigation.nav_stack.main import nav_stack_rerun_config
from dimos.robot.unitree.g1.config import G1
from dimos.robot.unitree.g1.g1_rerun import (
    g1_odometry_tf_override,
    g1_static_robot,
)
from dimos.visualization.vis_module import vis_module

unitree_g1_mid360_fastlio = autoconnect(
    FastLio2.blueprint(
        host_ip=os.getenv("LIDAR_HOST_IP", "192.168.123.164"),
        lidar_ip=os.getenv("LIDAR_IP", "192.168.123.120"),
        mount=G1.internal_odom_offsets["mid360_link"],
        map_freq=1.0,
        config="default.yaml",
        build_command="nix --extra-experimental-features 'nix-command flakes' build .#fastlio2_native",
    ),
    vis_module(
        viewer_backend=global_config.viewer,
        rerun_config=nav_stack_rerun_config(
            {
                "visual_override": {"world/odometry": g1_odometry_tf_override},
                "static": {"world/tf/robot": g1_static_robot},
                "memory_limit": "1GB",
            },
            vis_throttle=0.5,
        ),
    ),
).global_config(n_workers=2, robot_model="unitree_g1")


__all__ = ["unitree_g1_mid360_fastlio"]
