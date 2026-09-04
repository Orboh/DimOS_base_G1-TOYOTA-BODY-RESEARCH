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

"""Full G1 stack with agentic skills and onboard Mid-360 LiDAR."""

import os

from dimos.core.coordination.blueprints import autoconnect
from dimos.hardware.sensors.lidar.fastlio2.module import FastLio2
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner
from dimos.robot.unitree.g1.blueprints.agentic._agentic_skills import _agentic_skills
from dimos.robot.unitree.g1.blueprints.perceptive.unitree_g1 import unitree_g1
from dimos.robot.unitree.g1.config import G1
from dimos.robot.unitree.g1.odometry_bridge import G1FastLioOdometryBridge

_g1_fastlio_navigation = autoconnect(
    FastLio2.blueprint(
        host_ip=os.getenv("LIDAR_HOST_IP", "192.168.123.164"),
        lidar_ip=os.getenv("LIDAR_IP", "192.168.123.120"),
        mount=G1.internal_odom_offsets["mid360_link"],
        map_freq=1.0,
        config="default.yaml",
        build_command="nix --extra-experimental-features 'nix-command flakes' build .#fastlio2_native",
    ),
    G1FastLioOdometryBridge.blueprint(),
    ReplanningAStarPlanner.blueprint(),
    MovementManager.blueprint(),
).remappings(
    [
        # Keep FastLio2's native accumulated map visible without competing with
        # VoxelGridMapper.global_map, which feeds the existing G1 costmap path.
        (FastLio2, "global_map", "global_map_fastlio"),
    ]
)

unitree_g1_agentic = autoconnect(
    unitree_g1,
    _g1_fastlio_navigation,
    _agentic_skills,
)

__all__ = ["unitree_g1_agentic"]
