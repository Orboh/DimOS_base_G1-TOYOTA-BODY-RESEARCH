# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Real-robot wiring of :class:`HarvestSkills` + the G1-speaker announcer.

This connects the (verified-in-mock) harvest orchestrator to the live robot.
⚠️ It is NOT runtime-verified — it needs the G1 and live DimOS modules, which
cannot run on a laptop. Per the project rule, the **operator launches robot
motion**; this module only assembles the wiring. All robot/Unitree imports are
lazy so the harvest package stays importable without those deps.

Wiring status per skill:

* ``relative_move`` — IMPLEMENTED here: the G1's ``move`` is velocity-based, so a
  relative displacement [m] is issued as a timed velocity command. (Confirm the
  lateral sign on your robot; see ``lateral_to_y_sign``.)
* ``record_harvest`` — IMPLEMENTED (optional sink; defaults to no-op / memory2).
* ``detect_okra`` / ``grasp_okra`` / ``verify_harvest`` /
  ``go_to_next_station`` / ``swap_basket`` — INJECTED callables. Each needs a
  real subsystem (VLM+depth perception, the okra-ACT episode, the Dex1/VLM
  check, nav route planning). Their contracts are documented on
  :func:`build_dimos_harvest_skills`; provide them from your live blueprint.

The G1 onboard speaker is wired concretely in :func:`make_g1_speaker_announcer`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from dimos.robot.unitree.g1.harvest.announce import CallableAnnouncer
from dimos.robot.unitree.g1.harvest.blackboard import Okra
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

# Default base translation speed for converting a displacement [m] into a timed
# velocity command. Keep conservative near the crop. [m/s]
_BASE_SPEED = 0.15
_MIN_MOVE_M = 1e-3  # ignore sub-millimetre moves


class DimosHarvestSkills:
    """Adapts the harvest graph's ``HarvestSkills`` to live DimOS/robot calls.

    ``relative_move`` is implemented against a velocity ``move`` command; the
    perception/manipulation/nav steps are injected callables (see
    :func:`build_dimos_harvest_skills` for their contracts).
    """

    def __init__(
        self,
        *,
        move_cmd: Callable[[float, float, float, float], Any],
        detect_fn: Callable[[], list[Okra]],
        grasp_fn: Callable[[Okra, float], None],
        verify_fn: Callable[[], bool],
        next_station_fn: Callable[[], bool],
        swap_fn: Callable[[], None],
        record_fn: Callable[[dict[str, Any]], None] | None = None,
        base_speed: float = _BASE_SPEED,
        lateral_to_y_sign: float = -1.0,  # harvest +x=right -> G1 move +y=left, so -1
    ) -> None:
        self._move_cmd = move_cmd
        self._detect_fn = detect_fn
        self._grasp_fn = grasp_fn
        self._verify_fn = verify_fn
        self._next_station_fn = next_station_fn
        self._swap_fn = swap_fn
        self._record_fn = record_fn
        self._base_speed = base_speed
        self._lat_sign = lateral_to_y_sign

    # --- perception / manipulation / nav: delegate to injected subsystems -----

    def detect_okra(self) -> list[Okra]:
        return self._detect_fn()

    def grasp_okra(self, okra: Okra, force: float) -> None:
        self._grasp_fn(okra, force)

    def verify_harvest(self) -> bool:
        return bool(self._verify_fn())

    def go_to_next_station(self) -> bool:
        return bool(self._next_station_fn())

    def swap_basket(self) -> None:
        self._swap_fn()

    def record_harvest(self, record: dict[str, Any]) -> None:
        if self._record_fn is not None:
            self._record_fn(record)

    # --- base motion: implemented via the velocity `move` command -------------

    def relative_move(self, lateral: float, forward: float = 0.0, yaw: float = 0.0) -> None:
        """Issue a relative base displacement [m] as timed velocity commands.

        ``move_cmd(vx, vy, vyaw, duration)`` matches the G1 ``move`` skill
        (vx=forward, vy=left/right). Forward then lateral are issued in turn so
        the geometry stays simple; each blocks for its duration.
        """
        # forward axis -> vx
        if abs(forward) >= _MIN_MOVE_M:
            dur = abs(forward) / self._base_speed
            vx = self._base_speed * (1.0 if forward > 0 else -1.0)
            self._move_cmd(vx, 0.0, 0.0, dur)
            time.sleep(dur)
        # lateral axis -> vy (sign per robot convention)
        if abs(lateral) >= _MIN_MOVE_M:
            dur = abs(lateral) / self._base_speed
            vy = self._lat_sign * self._base_speed * (1.0 if lateral > 0 else -1.0)
            self._move_cmd(0.0, vy, 0.0, dur)
            time.sleep(dur)


def build_dimos_harvest_skills(
    *,
    move_cmd: Callable[[float, float, float, float], Any],
    detect_fn: Callable[[], list[Okra]],
    grasp_fn: Callable[[Okra, float], None],
    verify_fn: Callable[[], bool],
    next_station_fn: Callable[[], bool],
    swap_fn: Callable[[], None],
    record_fn: Callable[[dict[str, Any]], None] | None = None,
    **kwargs: Any,
) -> DimosHarvestSkills:
    """Assemble :class:`DimosHarvestSkills` from live DimOS handles.

    Contracts for the injected callables (build these from your live blueprint):

    * ``move_cmd(vx, vy, vyaw, duration)`` — the G1 ``move`` skill / connection
      (``UnitreeG1SkillContainer.move`` or ``G1Connection.move``).
    * ``detect_fn() -> list[Okra]`` — run the head-camera VLM
      (``module3D.ask_vlm`` / ``nav_vlm`` or a detector) + depth, returning each
      okra with RELATIVE ``pos_3d`` {x:lateral, y:depth, z:height} [m],
      ``ripeness`` in [0,1], and ``reachable`` (in the arm's reach box).
    * ``grasp_fn(okra, force)`` — run ONE okra-ACT grasp episode
      (``unitree-g1-act-arm`` / ActBridge with ``dry_run=False``) and the Dex1
      ``force``; returns when the episode ends (success checked separately).
    * ``verify_fn() -> bool`` — Dex1 hold state + a VLM "is it picked?" check.
    * ``next_station_fn() -> bool`` — nav to the next work pose (route planning);
      False when the field is done.
    * ``swap_fn()`` — nav to the collection point, swap an empty basket, return.
    * ``record_fn(record)`` — optional; persist the §9 record (e.g. memory2).
    """
    return DimosHarvestSkills(
        move_cmd=move_cmd,
        detect_fn=detect_fn,
        grasp_fn=grasp_fn,
        verify_fn=verify_fn,
        next_station_fn=next_station_fn,
        swap_fn=swap_fn,
        record_fn=record_fn,
        **kwargs,
    )


def build_live_harvest_skills(
    *,
    frame_getter: Callable[[], Any],
    target_classes: set[str] | None = None,
    detector: Any = None,
    ask_vlm: Callable[[str], str] | None = None,
    move_cmd: Callable[[float, float, float, float], Any] | None = None,
    grasp_module: Any = None,
    pixel_to_base: Callable[[float, float, Any], dict[str, float]] | None = None,
    depth_getter: Callable[[float, float], float] | None = None,
    verify_fn: Callable[[], bool] | None = None,
) -> tuple[DimosHarvestSkills, Any]:
    """Assemble a :class:`DimosHarvestSkills` for the LIVE robot (first cut).

    REAL now: ``detect_okra`` via the YOLO detector on the head-camera
    ``frame_getter`` (and ``verify_harvest`` via a VLM if ``ask_vlm`` is given).
    PLACEHOLDER (logged ``[LIVE-TODO]``) until wired:
      * ``relative_move`` / ``go_to_next_station`` / ``swap_basket`` — base motion.
        In *motion control mode* the legs run on the locomotion policy (``cmd_vel``
        walk) while the upper body is driven via ``rt/arm_sdk``, so walking and the
        arm reach can run CONCURRENTLY — no mode switch. Wiring = a ``cmd_vel``
        publisher (short reposition/sweep) + the DimOS nav stack (station-to-station).
        Pass ``move_cmd`` to enable;
      * ``grasp_okra`` — defaults to the stoppable :class:`DummyGraspModule`;
        replace with the real okra-ACT ``GraspModule`` (the cancellable reach).

    Returns ``(skills, grasp_module)`` — the module is exposed so a SafetyMonitor
    can stop a running grasp (``on_pause = grasp_module.stop``).
    """
    from dimos.robot.unitree.g1.harvest.detect_yolo import (
        YoloOkraDetector,
        make_yolo_detect_okra,
    )
    from dimos.robot.unitree.g1.harvest.dummy_skills import (
        DummyGraspModule,
        make_vlm_verify_harvest,
    )

    if detector is not None:
        detect_fn = YoloOkraDetector(
            detector=detector,
            frame_getter=frame_getter,
            target_classes=target_classes or {"banana"},
            pixel_to_base=pixel_to_base,
            depth_getter=depth_getter,
        ).detect
    else:
        detect_fn = make_yolo_detect_okra(
            frame_getter,
            target_classes=target_classes,
            pixel_to_base=pixel_to_base,
            depth_getter=depth_getter,
        )

    if verify_fn is None:
        if ask_vlm is not None:
            verify_fn = make_vlm_verify_harvest(ask_vlm)
        else:
            def verify_fn() -> bool:  # noqa: E306
                logger.info("[LIVE-TODO] verify_harvest placeholder (wire a VLM) -> True")
                return True

    grasp = grasp_module or DummyGraspModule()  # replace with the real okra-ACT GraspModule

    def _placeholder_move(vx: float, vy: float, vyaw: float, dur: float) -> None:
        # In motion control mode the legs walk (cmd_vel) while the arm runs on
        # rt/arm_sdk — concurrent, no mode switch. To enable, pass move_cmd wired
        # to a cmd_vel publisher (small moves) / the DimOS nav stack (big moves).
        logger.info(
            f"[LIVE-TODO] base move ({vx:.2f},{vy:.2f},{vyaw:.2f}) {dur:.2f}s "
            "not wired — needs a cmd_vel publisher / nav stack"
        )

    def _placeholder_next_station() -> bool:
        logger.info("[LIVE-TODO] go_to_next_station placeholder -> False (nav not wired)")
        return False

    def _placeholder_swap() -> None:
        logger.info("[LIVE-TODO] swap_basket placeholder (nav not wired)")

    # While base move is a placeholder, use a high base_speed so DimosHarvestSkills'
    # distance->timed-velocity sleeps are negligible (no real motion happens anyway).
    skills = build_dimos_harvest_skills(
        move_cmd=move_cmd or _placeholder_move,
        detect_fn=detect_fn,
        grasp_fn=grasp.run_episode,
        verify_fn=verify_fn,
        next_station_fn=_placeholder_next_station,
        swap_fn=_placeholder_swap,
        base_speed=1000.0 if move_cmd is None else _BASE_SPEED,
    )
    return skills, grasp


def make_g1_speaker_announcer(
    network_interface: str,
    speaker_id: int = 0,
    volume: int | None = None,
    init_dds: bool = True,
) -> CallableAnnouncer:
    """Return a :class:`CallableAnnouncer` that speaks via the G1 onboard speaker.

    Uses ``unitree_sdk2py.g1.audio.AudioClient.TtsMaker``. Call once at startup.

    Args:
        network_interface: the wired NIC to the robot (same as ``ROBOT_INTERFACE``).
        speaker_id: TTS voice id. Confirm a Japanese-capable id on your firmware;
            if none speaks Japanese, synthesise Japanese audio off-board (DimOS
            ``OpenAITTSNode`` / ``PyTTSNode``) and push PCM with
            ``AudioClient.PlayStream`` instead of ``TtsMaker``.
        volume: 0-100; left unchanged if None.
        init_dds: call ``ChannelFactoryInitialize``. Set False if the DDS channel
            is already initialised in this process (e.g. the okra-ACT stack did
            it) to avoid a double-init.

    ⚠️ Not runtime-verified (needs the robot). Speaks non-blocking-friendly:
    ``TtsMaker`` returns a status code quickly; the harvest loop is slow anyway.
    """
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

    if init_dds:
        ChannelFactoryInitialize(0, network_interface)
    client = AudioClient()
    client.SetTimeout(10.0)
    client.Init()
    if volume is not None:
        client.SetVolume(volume)

    def speak(text: str) -> None:
        client.TtsMaker(text, speaker_id)

    return CallableAnnouncer(speak)


__all__ = [
    "DimosHarvestSkills",
    "build_dimos_harvest_skills",
    "build_live_harvest_skills",
    "make_g1_speaker_announcer",
]
