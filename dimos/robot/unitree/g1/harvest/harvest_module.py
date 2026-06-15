# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DimOS Module that runs the okra-harvest LangGraph flow on ``start()``.

Wraps the harvest orchestrator (graph + skills + SafetyMonitor + Japanese voice)
as a deployable Module so ``dimos run unitree-g1-okra-harvest`` starts the whole
flow. Defaults to the **DUMMY** skills (no robot) — every action logs ``[DUMMY]``
and the spoken lines print with a 🔊 prefix.

To drive the real robot, ``use_dummy=False`` is reserved but NOT yet wired (the
real grasp = a stoppable okra-ACT GraspModule, detect = YOLO+depth, etc. — see
``README.md``); it raises ``NotImplementedError`` for now rather than pretend.
"""

from __future__ import annotations

from threading import Thread
from typing import Any

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.robot.unitree.g1.harvest.announce import CallableAnnouncer
from dimos.robot.unitree.g1.harvest.blackboard import HarvestConfig, initial_state
from dimos.robot.unitree.g1.harvest.dummy_skills import DummyHarvestSkills
from dimos.robot.unitree.g1.harvest.graph import build_harvest_graph
from dimos.robot.unitree.g1.harvest.safety import SafetyCheck, SafetyMonitor
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class HarvestModuleConfig(ModuleConfig):
    use_dummy: bool = True  # DUMMY skills, no robot (the only mode wired today)
    num_okra: int = 3  # size of the dummy field
    stations: int = 1  # number of dummy work stations
    recursion_limit: int = 400  # LangGraph step budget (the loop revisits nodes)


class HarvestModule(Module):
    """Runs the okra-harvest LangGraph flow in a worker thread when deployed."""

    config: HarvestModuleConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._thread: Thread | None = None
        self._monitor: SafetyMonitor | None = None
        self._app: Any = None

    @rpc
    def start(self) -> None:
        super().start()
        if not self.config.use_dummy:
            raise NotImplementedError(
                "HarvestModule: real skills are not wired yet — run with use_dummy=True. "
                "Real grasp = stoppable okra-ACT GraspModule, detect = YOLO+depth, "
                "verify = VLM, nav = DimOS nav. See dimos/robot/unitree/g1/harvest/README.md."
            )

        skills = DummyHarvestSkills(num_okra=self.config.num_okra, stations=self.config.stations)
        # Print the Japanese announcements to the console (no audio hardware here).
        voice = CallableAnnouncer(lambda text: logger.info(f"🔊 {text}"))
        # DUMMY always-safe check so the supervisor structure is live (never trips).
        self._monitor = SafetyMonitor(
            [SafetyCheck("dummy_person_clear", lambda: True)],
            on_pause=lambda reason: skills.grasp_module.stop(),
            announcer=voice,
        )
        self._monitor.start()
        self._app = build_harvest_graph(
            skills, HarvestConfig(), announcer=voice, safety=self._monitor.gate
        )
        self._thread = Thread(target=self._run, daemon=True, name="okra-harvest")
        self._thread.start()
        logger.info("HarvestModule started — running the DUMMY okra-harvest flow (no real robot)")

    def _run(self) -> None:
        try:
            final = self._app.invoke(
                initial_state(), {"recursion_limit": self.config.recursion_limit}
            )
            logger.info(f"HarvestModule: harvest flow COMPLETE — picks={final.get('picks')}")
        except Exception:  # noqa: BLE001
            logger.exception("HarvestModule: harvest flow errored")

    @rpc
    def stop(self) -> None:
        if self._monitor is not None:
            self._monitor.stop()
            self._monitor = None
        if self._thread is not None:
            self._thread.join(timeout=3.0)  # dummy flow finishes quickly; daemon otherwise
            self._thread = None
        super().stop()


__all__ = ["HarvestModule", "HarvestModuleConfig"]
