# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Background safety monitor + pause gate for the harvest workflow (handbook §6).

A **parallel supervisor** that runs alongside the harvest graph and can preempt
it. Cheap checks (person proximity, contact, self-diagnosis, balance) run every
tick; expensive checks (a VLM "is the task still going right?" judgement) run on
a slower cadence. On a hazard it **trips a shared gate**, calls a stop hook (on
the real robot: cancel the running ACT/cut **Module** — that is why those must be
stoppable Modules, not blocking ``@skill``s), and announces it in Japanese. When
all checks are safe again it clears the gate and the workflow resumes.

The harvest graph consults the gate at each motion node via
:meth:`SafetyGate.checkpoint`, which blocks while paused — so a hazard halts the
robot at the next safe point and resume re-observes from the phase preconditions.

Everything here is dependency-light and testable with no robot: drive the checks
and assert the gate trips/clears and the hooks fire (see ``test_safety.py``).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from dimos.robot.unitree.g1.harvest import announce
from dimos.robot.unitree.g1.harvest.announce import Announcer, NullAnnouncer
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


@runtime_checkable
class SafetyGate(Protocol):
    """A pause flag the workflow consults at safe points."""

    def checkpoint(self, timeout: float | None = None) -> bool:
        """Block while paused; return True once clear (or if not paused)."""
        ...

    def is_paused(self) -> bool:
        ...


class NullSafetyGate:
    """Never pauses — the default when no monitor is wired."""

    def checkpoint(self, timeout: float | None = None) -> bool:
        return True

    def is_paused(self) -> bool:
        return False


class PauseGate:
    """Thread-safe pause flag: tripped by the monitor, awaited by the workflow."""

    def __init__(self) -> None:
        self._paused = False
        self._reason = ""
        self._lock = threading.Lock()
        self._clear = threading.Event()
        self._clear.set()  # starts clear (not paused)

    def trip(self, reason: str) -> None:
        with self._lock:
            self._paused = True
            self._reason = reason
            self._clear.clear()

    def clear(self) -> None:
        with self._lock:
            self._paused = False
            self._reason = ""
            self._clear.set()

    def is_paused(self) -> bool:
        return self._paused

    @property
    def reason(self) -> str:
        return self._reason

    def checkpoint(self, timeout: float | None = None) -> bool:
        return self._clear.wait(timeout)


@dataclass
class SafetyCheck:
    """A named safety condition. ``is_safe()`` returns True when OK.

    ``expensive`` checks (e.g. a VLM judgement) run on the slower cadence; cheap
    checks (force, proximity, self-diagnosis) run every tick.
    """

    name: str
    is_safe: Callable[[], bool]
    expensive: bool = False


class SafetyMonitor:
    """Parallel supervisor: evaluates :class:`SafetyCheck`s and drives a gate.

    Args:
        checks: the safety conditions to watch.
        on_pause: called with the reason when a hazard first trips the gate.
            On the real robot, stop the running grasp/cut Module here.
        on_resume: called when all checks are safe again.
        announcer: speaks the Japanese stop/resume lines (handbook §6 HMI).
        tick_s: cheap-check cadence [s]. vlm_every: run expensive checks every
            Nth tick.
    """

    def __init__(
        self,
        checks: list[SafetyCheck],
        *,
        on_pause: Callable[[str], None] | None = None,
        on_resume: Callable[[], None] | None = None,
        announcer: Announcer | None = None,
        tick_s: float = 0.2,
        vlm_every: int = 10,
    ) -> None:
        self._checks = list(checks)
        self._on_pause = on_pause
        self._on_resume = on_resume
        self._voice: Announcer = announcer or NullAnnouncer()
        self._tick_s = tick_s
        self._vlm_every = max(1, vlm_every)
        self.gate = PauseGate()
        self._safe: dict[str, bool] = {c.name: True for c in self._checks}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def step(self, include_expensive: bool = True) -> list[str]:
        """Run one evaluation pass; trip/clear the gate and fire hooks.

        Returns the list of currently-unsafe check names. Cheap ticks
        (``include_expensive=False``) refresh only cheap checks; expensive checks
        keep their last verdict, so a partial pass never spuriously resumes.
        """
        for c in self._checks:
            if include_expensive or not c.expensive:
                try:
                    self._safe[c.name] = bool(c.is_safe())
                except Exception as exc:  # noqa: BLE001 — a flaky check = treat as unsafe
                    logger.warning("safety check raised; treating as unsafe", check=c.name, error=str(exc))
                    self._safe[c.name] = False
        unsafe = [name for name, ok in self._safe.items() if not ok]

        if unsafe and not self.gate.is_paused():
            reason = ", ".join(unsafe)
            self.gate.trip(reason)
            logger.warning("SAFETY trip — pausing", reason=reason)
            self._voice.say(announce.safety_stop(reason))
            if self._on_pause is not None:
                self._on_pause(reason)
        elif not unsafe and self.gate.is_paused():
            self.gate.clear()
            logger.info("SAFETY clear — resuming")
            self._voice.say(announce.safety_resume())
            if self._on_resume is not None:
                self._on_resume()
        return unsafe

    def _loop(self) -> None:
        tick = 0
        while not self._stop.wait(self._tick_s):
            self.step(include_expensive=(tick % self._vlm_every == 0))
            tick += 1

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="safety-monitor")
        self._thread.start()
        logger.info("SafetyMonitor started", checks=[c.name for c in self._checks])

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


__all__ = ["SafetyGate", "NullSafetyGate", "PauseGate", "SafetyCheck", "SafetyMonitor"]
