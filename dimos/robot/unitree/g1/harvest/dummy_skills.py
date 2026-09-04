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

"""DUMMY HarvestSkills for end-to-end pipeline bring-up — NO ROBOT.

⚠️⚠️ EVERYTHING HERE IS A DUMMY / PLACEHOLDER. It does not touch the robot. ⚠️⚠️
Every action logs with the ``[DUMMY]`` prefix so it is obvious in the trace that
nothing real happened. The point is to run the WHOLE pipeline (detect → select →
grasp → verify → record → station/basket) with no hardware, AND to demonstrate
that the SafetyMonitor can actually cancel a grasp mid-reach.

Swap each piece for a real implementation when ready:
* grasp → a real okra-ACT ``GraspModule`` (this DummyGraspModule shows the
  stoppable shape it must have);
* detect → ``detect_yolo.make_yolo_detect_okra`` (head/gripper cam);
* verify → :func:`make_vlm_verify_harvest` (route to a VLM);
* move / station / basket → ``real_skills`` / DimOS nav.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import threading
from typing import Any

from dimos.robot.unitree.g1.harvest.blackboard import Okra
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

_DUMMY = "[DUMMY]"


class DummyGraspModule:
    """⚠️ DUMMY stoppable stand-in for the okra-ACT reach.

    Simulates a multi-step reach to the cut point; ``stop()`` aborts it mid-reach
    — this is the exact cancellation path the SafetyMonitor drives
    (``on_pause = grasp_module.stop``). A real GraspModule must have this same
    shape (run an ACT episode, stop on ``_stop_event``, ramp the arm weight down
    safely). This dummy just sleeps through ``steps`` — it does NOT move the arm.
    """

    def __init__(self, steps: int = 20, step_s: float = 0.05) -> None:
        self._steps = steps
        self._step_s = step_s
        self._stop = threading.Event()
        # (okra_id, completed_steps, cancelled) per episode — for assertions/trace.
        self.episodes: list[tuple[str, int, bool]] = []

    def stop(self) -> None:
        """Cancel the in-flight reach (called by SafetyMonitor.on_pause)."""
        self._stop.set()

    def run_episode(self, okra: Okra, force: float) -> bool:
        """Run one dummy reach. Returns True if it completed, False if cancelled."""
        self._stop.clear()
        logger.info(
            f"{_DUMMY} GraspModule: ACT reach START (okra={okra.id}, force={force}) — NO real arm"
        )
        done = 0
        for i in range(self._steps):
            if self._stop.wait(self._step_s):  # stop() was called mid-reach
                logger.warning(f"{_DUMMY} GraspModule: STOPPED mid-reach at step {i}/{self._steps}")
                self.episodes.append((okra.id, i, True))
                return False
            done = i + 1
        logger.info(f"{_DUMMY} GraspModule: reach COMPLETE ({done} steps)")
        self.episodes.append((okra.id, done, False))
        return True


class DummyHarvestSkills:
    """⚠️ DUMMY HarvestSkills — runs the full pipeline with no robot.

    detect returns a small in-reach field, grasp uses a stoppable
    :class:`DummyGraspModule`, verify reports whether the reach completed, and
    nav/basket are no-ops. All log ``[DUMMY]``. ``self.grasp_module`` is exposed
    so a SafetyMonitor can stop a running grasp (``on_pause = skills.grasp_module.stop``).
    """

    def __init__(
        self,
        num_okra: int = 2,
        stations: int = 1,
        grasp_module: DummyGraspModule | None = None,
    ) -> None:
        self.grasp_module = grasp_module or DummyGraspModule()
        self._num_okra = num_okra
        self._stations_left = max(0, stations - 1)
        self._picked: set[str] = set()
        self._field = self._make_field()
        self._last: tuple[str, bool] | None = None
        self.records: list[dict[str, Any]] = []

    def _make_field(self) -> list[Okra]:
        # All in the reach box (centre ~x0.30,y0.45,z0.80) so they grasp directly.
        xs = [0.25, 0.35, 0.20, 0.40, 0.30]
        return [
            Okra(
                id=f"dummy_okra_{i}",
                img_region="R",
                pos_3d={"x": xs[i % len(xs)], "y": 0.45, "z": 0.80},
                ripeness=0.9,
                reachable=True,
            )
            for i in range(self._num_okra)
        ]

    def detect_okra(self) -> list[Okra]:
        remaining = [deepcopy(o) for o in self._field if o.id not in self._picked]
        logger.info(f"{_DUMMY} detect_okra: {len(remaining)} okra (fake field)")
        return remaining

    def relative_move(self, lateral: float, forward: float = 0.0, yaw: float = 0.0) -> None:
        logger.info(f"{_DUMMY} relative_move(lat={lateral:.2f}, fwd={forward:.2f}) — no robot")

    def grasp_okra(self, okra: Okra, force: float) -> None:
        reached = self.grasp_module.run_episode(okra, force)
        self._last = (okra.id, reached)

    def verify_harvest(self) -> bool:
        if self._last is None:
            return False
        okra_id, reached = self._last
        if reached:
            self._picked.add(okra_id)  # only a completed reach counts as picked
        logger.info(
            f"{_DUMMY} verify_harvest: {reached} (reach {'completed' if reached else 'cancelled'})"
        )
        return reached

    def go_to_next_station(self) -> bool:
        if self._stations_left <= 0:
            logger.info(f"{_DUMMY} go_to_next_station: no more stations")
            return False
        self._stations_left -= 1
        self._picked.clear()
        self._field = self._make_field()
        logger.info(f"{_DUMMY} go_to_next_station: moved to next station — no robot")
        return True

    def swap_basket(self) -> None:
        logger.info(f"{_DUMMY} swap_basket: emptied — no robot")

    def record_harvest(self, record: dict[str, Any]) -> None:
        self.records.append(record)


def make_vlm_verify_harvest(
    ask_vlm: Callable[[str], str],
    prompt: str = "Is the robot's gripper holding a picked okra? Answer yes or no.",
) -> Callable[[], bool]:
    """Wire ``verify_harvest`` to a VLM — i.e. "just ask the VLM if it's picked".

    ``ask_vlm(question) -> str`` is the DimOS VLM call (e.g. ``module3D.ask_vlm`` /
    Moondream). Returns True iff the answer is affirmative. This is a REAL wiring
    helper (not a dummy), kept here next to the dummies for convenience.
    """

    def verify() -> bool:
        answer = (ask_vlm(prompt) or "").strip().lower()
        return answer.startswith(("yes", "y", "true", "はい", "ある", "holding"))

    return verify


__all__ = ["DummyGraspModule", "DummyHarvestSkills", "make_vlm_verify_harvest"]
