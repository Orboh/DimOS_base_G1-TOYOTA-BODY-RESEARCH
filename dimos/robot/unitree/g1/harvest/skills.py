# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Skill interface the harvest graph drives, plus a spatial mock for dry runs.

The LangGraph nodes never touch the robot directly — they call methods on a
:class:`HarvestSkills` implementation. This keeps the *workflow logic* (fixed
sequence, retries, base-repositioning, routing) fully testable on a laptop with
no robot, and lets the real robot be plugged in later by providing a concrete
implementation.

Two implementations are intended:

* :class:`MockHarvestSkills` (here) — a 3D "field": okra sit at fixed absolute
  positions and the robot has a pose; moving the base changes which okra are in
  view (FOV) and in reach (the :class:`Box3D` reach volume). Deterministic;
  exercises the full graph (approach / sweep / skip) with no hardware.
* A real ``DimosHarvestSkills`` (not yet written — see ``README.md``) mapping
  ``detect_okra`` to a VLM skill, ``grasp_okra`` to the okra-ACT stack, and
  ``relative_move`` to the DimOS navigation skill. Per the project rule, the
  real base/arm motion is launched by the operator, not by this code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from dimos.robot.unitree.g1.harvest.blackboard import Box3D, Okra


@runtime_checkable
class HarvestSkills(Protocol):
    """The robot/perception capabilities the harvest graph depends on.

    A faithful slice of ``okra_harvest_workflow.md`` §2. For the MVP,
    ``detect_okra`` returns okra already annotated with ripeness, relative 3D
    position and reachability (detection + ``assess_ripeness`` +
    ``estimate_pose_3d`` folded together, as a VLM + depth call naturally would).
    """

    def detect_okra(self) -> list[Okra]:
        """Observe the current view and return every okra in it (relative pose)."""
        ...

    def relative_move(self, lateral: float, forward: float = 0.0, yaw: float = 0.0) -> None:
        """Move the base: ``lateral`` (+right) and ``forward`` (+forward), metres."""
        ...

    def grasp_okra(self, okra: Okra, force: float) -> None:
        """Reach and grasp ``okra`` with a normalized ``force`` in [0, 1].

        On the real robot this triggers the okra-ACT policy; it does not block
        on success — the result is checked separately by :meth:`verify_harvest`.
        """
        ...

    def verify_harvest(self) -> bool:
        """Return True iff the fruit is held and separated from the plant."""
        ...

    def record_harvest(self, record: dict[str, Any]) -> None:
        """Persist one §9 harvest record."""
        ...


@dataclass
class FieldOkra:
    """One okra in the (mock) field, at a fixed ABSOLUTE position [m]."""

    id: str
    x: float  # absolute lateral
    y: float  # absolute depth from the row origin
    z: float  # height
    ripeness: float


class MockHarvestSkills:
    """Deterministic 3D field for dry runs and tests.

    Okra sit at absolute positions; the robot starts at ``(robot_x, robot_y)``
    and moves via :meth:`relative_move`. ``detect_okra`` reports only okra inside
    the camera FOV, each with its position relative to the current robot pose and
    a ``reachable`` flag (inside the reach box). ``verify_harvest`` succeeds and
    removes the fruit, except for ids in ``flaky_verifies`` which fail the given
    number of times first (to exercise the §7 re-grasp recovery).
    """

    def __init__(
        self,
        field: list[FieldOkra],
        reach: Box3D,
        fov: Box3D,
        flaky_verifies: dict[str, int] | None = None,
        robot_x: float = 0.0,
        robot_y: float = 0.0,
    ) -> None:
        self._field = list(field)
        self._reach = reach
        self._fov = fov
        self._flaky = dict(flaky_verifies or {})
        self._picked: set[str] = set()
        self._rx = robot_x
        self._ry = robot_y
        self._last_grasped: str | None = None
        # Observable side effects, handy for assertions in tests.
        self.grasp_calls: list[tuple[str, float]] = []
        self.move_calls: list[tuple[float, float]] = []
        self.records: list[dict[str, Any]] = []

    def _relative(self, f: FieldOkra) -> dict[str, float]:
        """Position of okra ``f`` relative to the current robot pose."""
        return {"x": f.x - self._rx, "y": f.y - self._ry, "z": f.z}

    def detect_okra(self) -> list[Okra]:
        out: list[Okra] = []
        for f in self._field:
            if f.id in self._picked:
                continue
            rel = self._relative(f)
            if not self._fov.contains(rel):
                continue  # outside the camera view — not seen from here
            out.append(
                Okra(
                    id=f.id,
                    img_region="R" if rel["x"] >= 0 else "L",
                    pos_3d=rel,
                    ripeness=f.ripeness,
                    reachable=self._reach.contains(rel),
                )
            )
        return out

    def relative_move(self, lateral: float, forward: float = 0.0, yaw: float = 0.0) -> None:
        self._rx += lateral
        self._ry += forward
        self.move_calls.append((round(lateral, 3), round(forward, 3)))

    def grasp_okra(self, okra: Okra, force: float) -> None:
        self._last_grasped = okra.id
        self.grasp_calls.append((okra.id, force))

    def verify_harvest(self) -> bool:
        okra_id = self._last_grasped
        if okra_id is None:
            return False
        remaining_failures = self._flaky.get(okra_id, 0)
        if remaining_failures > 0:
            self._flaky[okra_id] = remaining_failures - 1
            return False
        self._picked.add(okra_id)
        return True

    def record_harvest(self, record: dict[str, Any]) -> None:
        self.records.append(record)


__all__ = ["HarvestSkills", "FieldOkra", "MockHarvestSkills"]
