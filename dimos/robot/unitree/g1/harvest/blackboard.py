# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Blackboard (shared world state) and config for the okra-harvest workflow.

This is the data half of the LangGraph orchestrator. The workflow handbook
(``okra_harvest_workflow.md`` §3) describes a JSON "blackboard" the agent
carries through every phase; :class:`HarvestState` is that blackboard as a
LangGraph ``State`` (a ``TypedDict`` whose fields each graph node may read and
return-merge).

Design rule (from the handbook): the *sequence* is fixed in code; only the
*judgment* steps (detection / ripeness / target choice / harvest verification)
defer to an LLM/VLM. Tunable thresholds therefore live in :class:`HarvestConfig`
as named constants with physical units — never hard-coded inside a node.

Spatial model (3D, robot base frame, metres):
    x = lateral  (+ right)
    y = depth    (+ forward, away from the robot)
    z = height   (+ up)
A fruit is graspable only inside the :class:`Box3D` reach volume. When no fruit
is in reach, the robot moves its base to bring one in: forward/back fixes the
depth (too far → forward, too close → back off the ridge), left/right fixes the
lateral offset. Height is reach-limited only: the G1 cannot squat in this build,
so an okra whose ``z`` is outside the reach box is skipped, not chased.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

# R = right half of the camera view, L = left half (informational; the real
# graspability gate is the 3D reach box below).
ImgRegion = Literal["R", "L"]

OkraStatus = Literal["target", "picked", "skipped_unripe", "skipped_height", "left_pending", "failed"]

# High-level mode mirrors the handbook's §3 `mode`. Harvest progresses
# right→left across the row, so the discovery sweep advances LEFT.
Mode = Literal["harvest", "reposition", "advance_left", "done", "paused_safe"]


@dataclass(frozen=True)
class Box3D:
    """An axis-aligned 3D window in the robot base frame [m] (x,y,z as above)."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @property
    def x_center(self) -> float:
        return 0.5 * (self.x_min + self.x_max)

    @property
    def y_center(self) -> float:
        return 0.5 * (self.y_min + self.y_max)

    def contains(self, p: dict[str, float]) -> bool:
        """True iff point ``p={'x','y','z'}`` is inside the box on all 3 axes."""
        return (
            self.x_min <= p.get("x", 0.0) <= self.x_max
            and self.y_min <= p.get("y", 0.0) <= self.y_max
            and self.z_min <= p.get("z", 0.0) <= self.z_max
        )

    def z_contains(self, p: dict[str, float]) -> bool:
        """True iff the height of ``p`` is within reach (x/y may still be off)."""
        return self.z_min <= p.get("z", 0.0) <= self.z_max

    def move_to_center(self, p: dict[str, float]) -> tuple[float, float]:
        """Base move (lateral, forward) [m] that lands ``p`` at the box centre.

        ``today's position − desired (centre) position``: positive forward means
        drive forward (the fruit was too far); negative means back off (too close).
        Same for lateral (positive = right). Re-detection after the move corrects
        any pose-estimate error (the "compute once, then verify" scheme).
        """
        return (p.get("x", 0.0) - self.x_center, p.get("y", 0.0) - self.y_center)


@dataclass
class Okra:
    """A single detected okra fruit (one entry of the handbook's `okra_visible`).

    ``pos_3d`` is the fruit position RELATIVE to the robot, metres ({x,y,z}).
    ``reachable`` is computed by the perception/robot layer (is it in the reach
    box?), so the orchestrator never re-derives kinematics.
    """

    id: str
    img_region: ImgRegion = "R"
    pos_3d: dict[str, float] = field(default_factory=dict)
    ripeness: float = 0.0  # ripeness score in [0, 1]; >= threshold means "ripe"
    reachable: bool = False
    status: OkraStatus = "target"


@dataclass(frozen=True)
class HarvestConfig:
    """Tunable parameters for the harvest workflow (no magic numbers in nodes).

    The geometry defaults are placeholders for the mock; replace with the G1's
    measured arm-reach volume and camera FOV when wiring the real robot.
    """

    ripeness_threshold: float = 0.6  # [0..1] minimum score to treat an okra as ripe
    grasp_force: float = 0.3  # [0..1] normalized Dex1 force; low, so as not to crush
    max_grasp_retries: int = 3  # §7: re-grasp attempts before marking an okra failed
    max_harvest_iterations: int = 80  # safety cap on total detect→(move|pick) loops
    basket_capacity: int = 30  # number of fruits before the basket is full

    # --- geometry [m] (robot base frame) ---
    # Right-side reach volume: the arm can grasp only inside this box.
    reach: Box3D = field(
        default_factory=lambda: Box3D(0.10, 0.50, 0.30, 0.60, 0.40, 1.10)
    )
    # Camera field of view: what detection can SEE (wider than reach).
    fov: Box3D = field(default_factory=lambda: Box3D(-0.80, 0.80, 0.10, 1.50, 0.00, 1.60))

    # --- movement [m] / counts ---
    # Harvest progresses RIGHT→LEFT: the discovery sweep steps left by this much.
    # (Reach stays on the right — the okra-ACT arm/Dex1 is the right one.)
    advance_step: float = 0.30  # lateral step magnitude when sweeping left to discover fruit
    standoff_min: float = 0.25  # never let an okra come closer than this (ridge safety)
    max_empty_advances: int = 2  # consecutive empty left-sweeps before done (§8 N)
    max_reposition_attempts: int = 3  # base moves toward one okra before skipping it


class HarvestState(TypedDict, total=False):
    """LangGraph state = the handbook's §3 blackboard.

    ``total=False`` so each node may return only the keys it changed; LangGraph
    merges the partial dict into the running state.
    """

    okra_visible: list[Okra]  # latest detection result for the current view
    target_id: str | None  # in-reach okra to grasp now
    approach_id: str | None  # visible-but-out-of-reach okra to move toward
    # Ids already picked, given-up-on, or unreachable this run. Detection
    # re-reports the same fruit every cycle, so the decision not to re-target it
    # must persist HERE (handbook §8 `update_field_map`), not on okra_visible.
    excluded_ids: list[str]
    basket_count: int
    basket_full: bool
    picks: int  # successfully harvested count (verified)
    iterations: int  # main-loop counter, guarded by max_harvest_iterations
    grasp_attempts: int  # re-grasp counter for the current target (reset on select)
    reposition_attempts: int  # base moves toward the current approach target
    empty_advances: int  # consecutive right-sweeps that found nothing (§8)
    mode: Mode
    last_verify_ok: bool  # result of the most recent verify_harvest()
    log: list[str]  # human-readable phase trace (demo / tests / slide figure)
    records: list[dict[str, Any]]  # §9 harvest records, one per picked/skipped fruit


def initial_state() -> HarvestState:
    """Return a fresh blackboard for the start of a harvest run."""
    return HarvestState(
        okra_visible=[],
        target_id=None,
        approach_id=None,
        excluded_ids=[],
        basket_count=0,
        basket_full=False,
        picks=0,
        iterations=0,
        grasp_attempts=0,
        reposition_attempts=0,
        empty_advances=0,
        mode="harvest",
        last_verify_ok=False,
        log=[],
        records=[],
    )


def find_okra(state: HarvestState, okra_id: str | None) -> Okra | None:
    """Look up an okra by id in the current detection list."""
    if okra_id is None:
        return None
    for okra in state.get("okra_visible", []):
        if okra.id == okra_id:
            return okra
    return None


__all__ = [
    "ImgRegion",
    "OkraStatus",
    "Mode",
    "Box3D",
    "Okra",
    "HarvestConfig",
    "HarvestState",
    "initial_state",
    "find_okra",
]
