# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Wire the harvest's base motion to DimOS / the Unitree SDK.

Two granularities (see README):

* **Walking (locomotion)** = just the Unitree SDK velocity command. In DimOS that
  is ``G1Connection.move`` / its ``cmd_vel: In[Twist]`` stream (which calls
  ``LocoClient.SetVelocity``). :func:`make_twist_move_cmd` turns the harvest's
  ``relative_move`` into a ``cmd_vel`` Twist (drive for a duration, then stop) —
  enough for the small reposition / sweep moves. Works concurrently with the arm
  (``rt/arm_sdk``) in motion-control mode.
* **Navigation (where to go + avoid obstacles)** = the DimOS nav_stack (SLAM /
  planners / local_planner+terrain for obstacle avoidance) behind a nav skill
  (``navigate_with_text`` / ``navigate_to``). :func:`make_navigate_stations`
  turns ``go_to_next_station`` into a call to an injected ``navigate_fn`` — wire
  that to the DimOS nav skill once the nav_stack + field map are deployed.

⚠️ ``make_twist_move_cmd`` drives the REAL base (the robot walks) — gate it behind
the SafetyMonitor with real checks and the operator's go-ahead.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Any

from dimos.utils.logging_config import setup_logger

logger = setup_logger()


def make_twist_move_cmd(
    publish_twist: Callable[[Any], None],
    *,
    settle_s: float = 0.15,
) -> Callable[[float, float, float, float], None]:
    """Build a ``move_cmd(vx, vy, vyaw, dur)`` that drives the base via ``cmd_vel``.

    Publishes a velocity Twist (vx=forward, vy=left/right, vyaw=turn), holds it for
    ``dur`` seconds, then publishes a zero Twist to stop. ``publish_twist`` is the
    harvest module's ``cmd_vel`` Out (→ ``G1Connection`` → ``LocoClient.SetVelocity``).
    """
    from dimos.msgs.geometry_msgs.Twist import Twist
    from dimos.msgs.geometry_msgs.Vector3 import Vector3

    def _twist(vx: float, vy: float, vyaw: float) -> Any:
        return Twist(linear=Vector3(vx, vy, 0.0), angular=Vector3(0.0, 0.0, vyaw))

    def move_cmd(vx: float, vy: float, vyaw: float, dur: float) -> None:
        publish_twist(_twist(vx, vy, vyaw))
        time.sleep(max(0.0, dur))
        publish_twist(_twist(0.0, 0.0, 0.0))  # stop
        time.sleep(settle_s)

    return move_cmd


def make_navigate_stations(
    navigate_fn: Callable[[Any], Any],
    stations: Iterable[Any],
) -> Callable[[], bool]:
    """Build a ``go_to_next_station()`` that drives DimOS navigation.

    Calls ``navigate_fn(station)`` for each station in turn (``navigate_fn`` =
    DimOS ``navigate_with_text`` / ``navigate_to``, which runs the nav_stack:
    plan + follow + obstacle-avoid). Returns False once the stations are
    exhausted (field done). Wire ``navigate_fn`` to the live nav skill once the
    nav_stack + field map are deployed.
    """
    pending = list(stations)
    state = {"i": 0}

    def go_to_next_station() -> bool:
        if state["i"] >= len(pending):
            logger.info("[nav] no more stations")
            return False
        target = pending[state["i"]]
        state["i"] += 1
        logger.info(f"[nav] navigate to station {state['i']}: {target!r}")
        navigate_fn(target)
        return True

    return go_to_next_station


__all__ = ["make_twist_move_cmd", "make_navigate_stations"]
