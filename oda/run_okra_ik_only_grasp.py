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

"""Fallback standalone launcher for ``unitree_g1_okra_ik_only_grasp``.

The blueprint is now registered in ``dimos/robot/all_blueprints.py`` as
``unitree-g1-okra-ik-only-grasp`` (see that module's docstring), so ``dimos run
unitree-g1-okra-ik-only-grasp`` is the normal way to launch it --
``oda/start_okra_ik_only_grasp.sh`` does exactly that. This script is kept only
as a fallback that doesn't need the registry/CLI: it replicates what
``dimos/robot/cli/dimos.py``'s ``run()`` does internally (``autoconnect`` ->
``ModuleCoordinator.build`` -> ``start_rpc_service`` -> ``loop()``), minus the
CLI-only run-registry/watchdog bookkeeping used for ``dimos status``/``dimos
stop``.

``ModuleCoordinator.loop()`` already catches KeyboardInterrupt and calls
``self.stop()`` in a ``finally`` block (module_coordinator.py:558-565), so
Ctrl-C cleanly tears everything down.

Run: python oda/run_okra_ik_only_grasp.py
(or, preferred: dimos run unitree-g1-okra-ik-only-grasp /
oda/start_okra_ik_only_grasp.sh, which also handles the Jetson camera
publisher + laptop network setup + viewer.)
"""

from __future__ import annotations

import os

from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.robot.unitree.g1.blueprints.manipulation.unitree_g1_okra_ik_only_grasp import (
    unitree_g1_okra_ik_only_grasp,
)
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


def main() -> int:
    logger.info("Starting unitree_g1_okra_ik_only_grasp (fallback standalone launcher)")
    coordinator = ModuleCoordinator.build(unitree_g1_okra_ik_only_grasp, {})
    if os.environ.get("DIMOS_SKIP_COORDINATOR_RPC") != "1":
        coordinator.start_rpc_service()
    coordinator.loop()  # blocks; Ctrl-C -> KeyboardInterrupt -> coordinator.stop() (finally)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
