#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""unitree_g1_nav_isaac — Unity を使わず Isaac 収穫 sim で dimos ナビを回すブループリント。

``unitree_g1_nav_sim`` から **UnityBridgeModule を外した**もの。registered_scan / odometry /
tf は、別プロセスの ``docs/sim-setup/nav_isaac_adapter.py`` が Isaac（生 LCM）から翻訳して
同じ LCM バスへ流す。cmd_vel はアダプタが Isaac へ戻す。本ブループリントは nav_stack
（terrain_analysis→local_planner→path_follower）＋ MovementManager ＋可視化のみ。

起動: dimos run unitree-g1-nav-isaac -o terrainanalysis.auto_build=true \
        -o localplanner.auto_build=true -o pathfollower.auto_build=true -o pgo.auto_build=true
（別途: Isaac 側 sim_nav_bridge.py と .venv の nav_isaac_adapter.py を起動。lo マルチキャスト必須）
"""

from __future__ import annotations

from typing import Any

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.navigation.nav_stack.main import create_nav_stack, nav_stack_rerun_config
from dimos.robot.unitree.g1.config import G1, G1_LOCAL_PLANNER_PRECOMPUTED_PATHS
from dimos.robot.unitree.g1.g1_rerun import g1_static_robot
from dimos.visualization.vis_module import vis_module

# nav_sim と同一設定（数値ドリフト防止のため値は unitree_g1_nav_sim と一致させること）。
nav_config: dict[str, Any] = dict(
    planner="simple",
    vehicle_height=G1.height_clearance,
    max_speed=2.0,
    terrain_analysis={
        "ground_height_threshold": 0.05,
        "min_relative_z": -1.5,
    },
    terrain_map_ext={
        "decay_time": 120,
    },
    local_planner={
        "paths_dir": str(G1_LOCAL_PLANNER_PRECOMPUTED_PATHS),
        "min_relative_z": -1.5,
        "freeze_ang": 180.0,
        "obstacle_height_threshold": 0.02,
        "publish_free_paths": True,
    },
    path_follower={
        "max_acceleration": 2.0,
        "max_yaw_rate": 60.0,
    },
)

unitree_g1_nav_isaac = (
    autoconnect(
        create_nav_stack(**nav_config),
        MovementManager.blueprint(),
        vis_module(
            viewer_backend=global_config.viewer,
            rerun_config=nav_stack_rerun_config(
                {
                    "static": {
                        "world/tf/robot": g1_static_robot,
                    },
                },
                vis_throttle=0.1,
            ),
        ),
    )
    .remappings(
        [
            # Planner owns way_point — disconnect MovementManager's click relay
            (MovementManager, "way_point", "_mgr_way_point_unused"),
        ]
    )
    .global_config(n_workers=8, robot_model="unitree_g1", simulation=True)
)
