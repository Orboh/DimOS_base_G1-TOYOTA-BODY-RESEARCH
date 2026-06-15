#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Blueprint: run the okra-harvest workflow with ``dimos run unitree-g1-okra-harvest``.

Starts the LangGraph harvest flow (detect → select → grasp → verify → record →
sweep/revisit/station/basket), with the SafetyMonitor and Japanese announcements.

⚠️ Defaults to **DUMMY** skills — no robot, every action logs ``[DUMMY]`` and the
spoken lines print with 🔊. This is the breadth-first end-to-end skeleton; swap
the dummy skills for the real ones (okra-ACT GraspModule / YOLO detect / VLM
verify / DimOS nav) when ready (see ``dimos/robot/unitree/g1/harvest/README.md``).

    dimos run unitree-g1-okra-harvest
"""

from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.robot.unitree.g1.harvest.harvest_module import HarvestModule

unitree_g1_okra_harvest = autoconnect(HarvestModule.blueprint())

__all__ = ["unitree_g1_okra_harvest"]
