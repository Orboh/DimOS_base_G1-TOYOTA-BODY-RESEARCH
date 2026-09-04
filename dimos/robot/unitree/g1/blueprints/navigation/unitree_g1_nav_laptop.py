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

"""G1 navigation stack run from an operator laptop — nothing installed on the robot.

Same stack as ``unitree-g1-nav-onboard``, with networking defaults for a laptop
wired into the G1's internal ethernet segment (192.168.123.0/24):

- FastLio2 ``host_ip`` auto-resolves to this machine's address on the robot
  subnet (the Mid-360 streams UDP to it). Override with ``LIDAR_HOST_IP``.
- The DDS interface for ``G1HighLevelDdsSdk`` auto-resolves to the NIC holding
  that address. Override with ``ROBOT_INTERFACE``.

Usage::

    # Laptop wired to the G1, static IP on the robot subnet (e.g. 192.168.123.100/24)
    dimos run unitree-g1-nav-laptop
"""

from __future__ import annotations

from collections.abc import Callable
import ipaddress
import os
from typing import Any

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.hardware.sensors.lidar.fastlio2.module import FastLio2
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.navigation.nav_stack.main import create_nav_stack, nav_stack_rerun_config
from dimos.robot.unitree.g1.config import G1, G1_LOCAL_PLANNER_PRECOMPUTED_PATHS
from dimos.robot.unitree.g1.effectors.high_level.dds_sdk import G1HighLevelDdsSdk
from dimos.robot.unitree.g1.g1_rerun import (
    g1_odometry_tf_override,
    g1_static_robot,
)
from dimos.utils.generic import get_local_ips
from dimos.visualization.vis_module import vis_module

# G1 internal wired segment: NX at .164, Mid-360 LiDAR at .120
_G1_SUBNET = ipaddress.IPv4Network("192.168.123.0/24")
_G1_LIDAR_IP = "192.168.123.120"
# Documented laptop static IP (docs/platforms/humanoid/g1/index.md)
_FALLBACK_LAPTOP_IP = "192.168.123.100"


def _detect_robot_link() -> tuple[str, str]:
    """Return ``(host_ip, interface)`` for this machine's address on the G1 subnet.

    Falls back to the documented laptop defaults when no NIC is on the robot
    subnet (e.g. importing this module without the robot connected).
    """
    for ip, iface in get_local_ips():
        if ipaddress.IPv4Address(ip) in _G1_SUBNET:
            return ip, iface
    return _FALLBACK_LAPTOP_IP, "eth0"


_host_ip, _interface = _detect_robot_link()


def build_nav_laptop(rerun_blueprint: Callable[[], Any] | None = None) -> Any:
    """Build the laptop nav blueprint; variants may override the Rerun layout.

    ``rerun_blueprint`` replaces the nav default (3D-only) viewport — e.g.
    ``unitree_g1_nav_laptop_cam`` adds a camera panel for ``/color_image``.
    """
    rerun_user_config: dict[str, Any] = {
        "visual_override": {"world/odometry": g1_odometry_tf_override},
        "static": {"world/tf/robot": g1_static_robot},
        "memory_limit": "1GB",
    }
    if rerun_blueprint is not None:
        rerun_user_config["blueprint"] = rerun_blueprint
    # Nav stack parameters mirror unitree_g1_nav_onboard — keep the two in sync.
    return (
        autoconnect(
            FastLio2.blueprint(
                host_ip=os.getenv("LIDAR_HOST_IP", _host_ip),
                lidar_ip=os.getenv("LIDAR_IP", _G1_LIDAR_IP),
                mount=G1.internal_odom_offsets["mid360_link"],
                map_freq=1.0,
                config="default.yaml",
            ),
            create_nav_stack(
                planner="simple",
                vehicle_height=G1.height_clearance,
                max_speed=0.6,
                far_planner={
                    "is_static_env": False,
                },
                terrain_analysis={
                    "obstacle_height_threshold": 0.01,
                    "ground_height_threshold": 0.01,
                    "sensor_range": 40,  # meters
                },
                local_planner={
                    "paths_dir": str(G1_LOCAL_PLANNER_PRECOMPUTED_PATHS),
                    "publish_free_paths": False,
                },
                simple_planner={
                    "cell_size": 0.2,
                    "obstacle_height_threshold": 0.10,
                    "inflation_radius": 0.5,
                    "lookahead_distance": 2.0,
                    "replan_rate": 5.0,
                    "replan_cooldown": 2.0,
                },
            ),
            MovementManager.blueprint(),
            # ROBOT_INTERFACE pins cyclonedds to a NIC; required on multi-NIC hosts.
            G1HighLevelDdsSdk.blueprint(
                network_interface=os.getenv("ROBOT_INTERFACE", _interface),
            ),
            vis_module(
                viewer_backend=global_config.viewer,
                rerun_config=nav_stack_rerun_config(
                    rerun_user_config,
                    vis_throttle=0.5,
                ),
            ),
        )
        .remappings(
            [
                # FastLio2 outputs "lidar"; SmartNav modules expect "registered_scan"
                (FastLio2, "lidar", "registered_scan"),
                (FastLio2, "global_map", "global_map_fastlio"),
                # Planner owns way_point — disconnect MovementManager's click relay
                (MovementManager, "way_point", "_mgr_way_point_unused"),
            ]
        )
        .global_config(n_workers=12, robot_model="unitree_g1")
    )


# Wrapped in autoconnect() so the registry generator's AST scan
# (test_all_blueprints_generation.py) recognizes this as a blueprint —
# a bare build_nav_laptop() call would be dropped from all_blueprints.py.
unitree_g1_nav_laptop = autoconnect(build_nav_laptop())

__all__ = ["build_nav_laptop", "unitree_g1_nav_laptop"]
