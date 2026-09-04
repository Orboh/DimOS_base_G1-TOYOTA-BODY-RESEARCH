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

from collections.abc import Callable, Iterable
import time
from typing import Any

from dimos.utils.logging_config import setup_logger

logger = setup_logger()


def make_twist_move_cmd(
    publish_twist: Callable[[Any], None],
    *,
    publish_hz: float = 10.0,
    settle_s: float = 0.15,
) -> Callable[[float, float, float, float], None]:
    """Build a ``move_cmd(vx, vy, vyaw, dur)`` that drives the base via ``cmd_vel``.

    G1HighLevelDdsSdk uses a watchdog pattern: if no cmd_vel arrives within
    ``cmd_vel_timeout`` (0.2 s), it auto-stops the robot. A single publish is
    therefore not enough — we must stream at ``publish_hz`` (≥ 5 Hz) for the
    full ``dur`` seconds to keep the robot moving, then publish a zero-velocity
    to stop explicitly.
    """
    from dimos.msgs.geometry_msgs.Twist import Twist
    from dimos.msgs.geometry_msgs.Vector3 import Vector3

    interval = 1.0 / publish_hz

    def _twist(vx: float, vy: float, vyaw: float) -> Any:
        return Twist(linear=Vector3(vx, vy, 0.0), angular=Vector3(0.0, 0.0, vyaw))

    def move_cmd(vx: float, vy: float, vyaw: float, dur: float) -> None:
        if dur <= 0.0:
            return
        elapsed = 0.0
        while elapsed < dur:
            publish_twist(_twist(vx, vy, vyaw))
            time.sleep(interval)
            elapsed += interval
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


def make_search_forward(
    move_cmd: Callable[[float, float, float, float], Any],
    *,
    step_m: float = 0.30,
    speed: float = 0.5,
    max_advances: int = 3,
) -> Callable[[], bool]:
    """Build a ``go_to_next_station()`` that WALKS FORWARD to keep searching.

    Interim stand-in for the nav stack: instead of finishing after one spot when
    no okra is found, drive the base FORWARD by ``step_m`` and return True so the
    graph loops back to ``detect`` at the new position (okra may be ahead, out of
    detection range). Bounded by ``max_advances``; returns False once exhausted
    (field done) so the run still terminates. ``move_cmd(vx, vy, vyaw, dur)`` owns
    timing (see :func:`make_twist_move_cmd`); ``vx>0`` is forward.

    Combined with the §5 left-sweep this gives a forward-and-sweep search:
    sweep left → step forward → sweep left → ... until ``max_advances``.
    """
    state = {"i": 0}

    def go_to_next_station() -> bool:
        if state["i"] >= max_advances:
            logger.info(f"[search] forward advances exhausted ({max_advances}) -> field done")
            return False
        state["i"] += 1
        dur = abs(step_m) / max(1e-3, speed)
        move_cmd(speed, 0.0, 0.0, dur)  # vx = forward
        logger.info(f"[search] step forward {step_m:.2f} m (advance {state['i']}/{max_advances})")
        return True

    return go_to_next_station


__all__ = ["make_navigate_stations", "make_search_forward", "make_twist_move_cmd"]
