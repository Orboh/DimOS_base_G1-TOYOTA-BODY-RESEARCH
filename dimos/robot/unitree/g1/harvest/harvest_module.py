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

import os
import threading
from threading import Thread
from typing import Any

from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.robot.unitree.g1.harvest.announce import CallableAnnouncer
from dimos.robot.unitree.g1.harvest.blackboard import HarvestConfig, initial_state
from dimos.robot.unitree.g1.harvest.dummy_skills import DummyHarvestSkills
from dimos.robot.unitree.g1.harvest.graph import build_harvest_graph
from dimos.robot.unitree.g1.harvest.real_skills import build_live_harvest_skills
from dimos.robot.unitree.g1.harvest.safety import SafetyCheck, SafetyMonitor
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class HarvestModuleConfig(ModuleConfig):
    use_dummy: bool = True  # True = DUMMY (no robot); False = LIVE (real YOLO detect on camera)
    num_okra: int = 3  # size of the dummy field (dummy mode only)
    stations: int = 1  # number of dummy work stations (dummy mode only)
    # LIVE detect target class(es). Stock yolo11n is COCO ("okra" needs a fine-tuned
    # weight) — "banana" is a proxy to exercise the real camera→detect→select path.
    target_classes: str = "banana"
    recursion_limit: int = 400  # LangGraph step budget (the loop revisits nodes)
    # LIVE: speak Japanese through the G1 speaker (pyopenjtalk + PlayStream).
    # False = log the lines to the console (no robot / no audio deps).
    use_g1_speaker: bool = False
    network_interface: str = ""  # NIC for the G1 audio DDS (defaults to ROBOT_INTERFACE)
    # LIVE verify_harvest via a local Ollama vision model (e.g. "moondream",
    # "qwen2.5vl"). Empty = placeholder verify (always True). See ollama_vlm.py.
    vlm_model: str = ""
    ollama_host: str = ""  # Ollama base URL (empty = ChatOllama default localhost:11434)
    # LIVE: drive the real base for reposition/sweep via cmd_vel (SDK LocoClient).
    # ⚠️ THE ROBOT WALKS — default off; enable only with real safety checks + operator.
    use_base_move: bool = False
    # LIVE: real okra-ACT reach for grasp (stoppable ActGraspModule). ⚠️ THE ARM
    # MOVES — default off; needs act_service + arm/gripper connections wired.
    use_act_grasp: bool = False
    act_endpoint: str = "tcp://127.0.0.1:5701"  # okra-ACT inference service (ZMQ REP)
    grasp_max_steps: int = 120  # ACT reach episode length cap
    # §6 real safety (used when real motion is on). File e-stop: `touch` to pause.
    safety_estop_file: str = "/tmp/okra_estop"
    torque_limit: float = 0.0  # [N·m] arm-torque contact guard; 0 = off (needs tuning)


class HarvestModule(Module):
    """Runs the okra-harvest LangGraph flow in a worker thread when deployed.

    ``use_dummy=True`` (default): fully self-contained DUMMY flow, no robot.
    ``use_dummy=False`` (LIVE): real YOLO detection on the head-camera
    ``color_image`` stream; verify/move/grasp/nav are still ``[LIVE-TODO]``
    placeholders (VLM verify, okra-ACT GraspModule, base motion and nav are
    follow-ups), so it runs real perception without real motion.
    """

    config: HarvestModuleConfig
    color_image: In[Image]  # head camera (LIVE mode); unused in dummy mode
    cmd_vel: Out[Twist]  # base velocity (LIVE + use_base_move) -> G1Connection
    # Arm streams (LIVE + use_act_grasp) -> G1ArmSdkConnection / G1GripperConnection.
    motor_states: In[JointState]
    right_gripper_state: In[JointState]
    arm_target: Out[JointState]
    gripper_target: Out[JointState]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._thread: Thread | None = None
        self._monitor: SafetyMonitor | None = None
        self._app: Any = None
        self._voice: Any = None
        self._lock = threading.Lock()
        self._latest_image: Image | None = None
        self._latest_state: JointState | None = None
        self._latest_gripper: float = 0.0

    def _on_state(self, state: JointState) -> None:
        with self._lock:
            self._latest_state = state

    def _on_gripper(self, state: JointState) -> None:
        pos = list(state.position)
        if pos:
            with self._lock:
                self._latest_gripper = float(pos[0])

    def _build_voice(self) -> Any:
        """Console-log announcer by default; the real G1 speaker if use_g1_speaker."""
        if self.config.use_g1_speaker:
            from dimos.robot.unitree.g1.harvest.g1_speaker import make_g1_playstream_announcer

            nic = self.config.network_interface or os.getenv("ROBOT_INTERFACE", "")
            try:  # the deployment may or may not have initialised DDS already
                return make_g1_playstream_announcer(nic, init_dds=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("G1 speaker init_dds=True failed; retry init_dds=False", error=str(exc))
                try:
                    return make_g1_playstream_announcer(init_dds=False)
                except Exception as exc2:  # noqa: BLE001
                    logger.warning("G1 speaker unavailable; using console voice", error=str(exc2))
        return CallableAnnouncer(lambda text: logger.info(f"🔊 {text}"))

    def _build_safety_checks(self) -> list[SafetyCheck]:
        """Real §6 checks when real motion is on; else a dummy always-safe check."""
        real_motion = self.config.use_act_grasp or self.config.use_base_move
        if not real_motion:
            return [SafetyCheck("dummy_person_clear", lambda: True)]
        from dimos.robot.unitree.g1.harvest.safety_checks import FileEStop, make_torque_check

        checks = [FileEStop(self.config.safety_estop_file).as_check()]
        if self.config.torque_limit > 0:
            checks.append(make_torque_check(lambda: self._latest_state, limit=self.config.torque_limit))
        logger.info(
            f"SafetyMonitor real checks: file e-stop={self.config.safety_estop_file!r} "
            f"(touch to pause), torque_limit={self.config.torque_limit}"
        )
        return checks

    def _on_image(self, image: Image) -> None:
        with self._lock:
            self._latest_image = image

    @rpc
    def start(self) -> None:
        super().start()
        voice = self._build_voice()
        self._voice = voice

        if self.config.use_dummy:
            skills: Any = DummyHarvestSkills(
                num_okra=self.config.num_okra, stations=self.config.stations
            )
            grasp_module = skills.grasp_module
            mode = "DUMMY (no robot)"
        else:
            self.register_disposable(Disposable(self.color_image.subscribe(self._on_image)))
            targets = {c.strip() for c in self.config.target_classes.split(",") if c.strip()}
            verify_fn = None
            verify_note = "verify=[LIVE-TODO] placeholder"
            if self.config.vlm_model:
                from dimos.robot.unitree.g1.harvest.ollama_vlm import make_ollama_verify

                verify_fn = make_ollama_verify(
                    lambda: self._latest_image,
                    model=self.config.vlm_model,
                    host=self.config.ollama_host or None,
                )
                verify_note = f"verify=Ollama:{self.config.vlm_model}"
            move_cmd = None
            move_note = "move=[LIVE-TODO] placeholder"
            if self.config.use_base_move:
                from dimos.robot.unitree.g1.harvest.nav_skills import make_twist_move_cmd

                move_cmd = make_twist_move_cmd(self.cmd_vel.publish)
                move_note = "move=cmd_vel(SDK)"

            grasp_override = None
            grasp_note = "grasp=DUMMY"
            if self.config.use_act_grasp:
                from dimos.robot.unitree.g1.harvest.act_grasp import ActGraspModule

                self.register_disposable(Disposable(self.motor_states.subscribe(self._on_state)))
                self.register_disposable(
                    Disposable(self.right_gripper_state.subscribe(self._on_gripper))
                )
                grasp_override = ActGraspModule(
                    image_getter=lambda: self._latest_image,
                    state_getter=lambda: self._latest_state,
                    gripper_getter=lambda: self._latest_gripper,
                    publish_arm=self.arm_target.publish,
                    publish_gripper=self.gripper_target.publish,
                    act_endpoint=self.config.act_endpoint,
                    max_steps=self.config.grasp_max_steps,
                )
                grasp_note = "grasp=okra-ACT"

            skills, grasp_module = build_live_harvest_skills(
                frame_getter=lambda: self._latest_image,
                target_classes=targets,
                verify_fn=verify_fn,
                move_cmd=move_cmd,
                grasp_module=grasp_override,
            )
            mode = f"LIVE — real YOLO detect; {verify_note}; {move_note}; {grasp_note}"

        # Real §6 checks when real motion is on (file e-stop + torque), else dummy.
        self._monitor = SafetyMonitor(
            self._build_safety_checks(),
            on_pause=lambda reason: grasp_module.stop(),
            announcer=voice,
        )
        self._monitor.start()
        self._app = build_harvest_graph(
            skills, HarvestConfig(), announcer=voice, safety=self._monitor.gate
        )
        self._thread = Thread(target=self._run, daemon=True, name="okra-harvest")
        self._thread.start()
        logger.info(f"HarvestModule started — {mode}")

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
        if self._voice is not None and hasattr(self._voice, "stop"):
            self._voice.stop()
        self._voice = None
        super().stop()


__all__ = ["HarvestModule", "HarvestModuleConfig"]
