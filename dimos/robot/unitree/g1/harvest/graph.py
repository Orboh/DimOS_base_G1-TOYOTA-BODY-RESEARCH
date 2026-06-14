# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""LangGraph orchestrator for the okra-harvest workflow.

This is the *control* half of the workflow: a LangGraph ``StateGraph`` whose
nodes are the handbook phases and whose edges encode the FIXED sequence. The
sequence cannot be skipped or reordered by the model — the model only supplies
*judgment* inside skill calls (detection, ripeness, target choice, harvest
verification). That is the whole point of using a graph instead of a free
ReAct loop.

Flow (handbook Phases 2→8 + §5 movement)::

    detect ─▶ select ─┬─ in-reach target ──────▶ grasp ─▶ verify ─┬ ok ─▶ record ─┐
                      │                                            │ retry≤N ─▶ grasp
                      │                                            └ exhausted ─▶ give_up ─▶ detect
                      ├─ visible but out of reach ─▶ reposition ───────────────────────────▶ detect
                      │      (forward/back fixes depth, left/right fixes lateral)
                      └─ nothing in view ─────────▶ advance_left (sweep) ──────────────────▶ detect
                                                       │ empty ≥ N and nothing pending
                                                       ▼
                                                      END (done)
    record ─(basket full / cap)─▶ END   ;   record ─(else)─▶ detect

Harvest progresses RIGHT→LEFT across the row, so the discovery sweep
(``advance_left``) steps in -x. The reach box stays on the RIGHT — the okra-ACT
arm and Dex1 gripper are the right ones — so moving left brings as-yet-unpicked
left-side okra into the right-side reach.

Movement is "compute once, then verify": ``reposition`` moves the base so the
target lands at the reach-box centre (handbook §5 / "横移動量を pos_3d から算出"),
then re-detects to correct any pose-estimate error. Too-far → forward,
too-close → back off the ridge (clamped by ``standoff_min``), off to the
left/right → strafe. An okra whose height is out of reach is skipped (the G1
cannot squat in this build).

Deferred to later milestones: navigation between stations, basket
transport/swap, pedicel cutting (no cutter on the robot yet), and the §6
background safety interrupt.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from dimos.robot.unitree.g1.harvest import announce
from dimos.robot.unitree.g1.harvest.announce import Announcer, NullAnnouncer
from dimos.robot.unitree.g1.harvest.blackboard import (
    HarvestConfig,
    HarvestState,
    Okra,
    find_okra,
)
from dimos.robot.unitree.g1.harvest.skills import HarvestSkills

# Node names (also used as routing targets) — kept as constants to avoid typos.
DETECT = "detect"
SELECT = "select"
GRASP = "grasp"
VERIFY = "verify"
RECORD = "record"
GIVE_UP = "give_up"
REPOSITION = "reposition"
ADVANCE_LEFT = "advance_left"
REVISIT = "revisit"
FINISH = "finish"


def build_harvest_graph(
    skills: HarvestSkills,
    config: HarvestConfig | None = None,
    announcer: Announcer | None = None,
):
    """Build and compile the harvest ``StateGraph``.

    Args:
        skills: the robot/perception backend (real or :class:`MockHarvestSkills`).
        config: tunable thresholds + geometry; defaults to :class:`HarvestConfig`.
        announcer: speaks Japanese status to the human at decision / info-update
            points; defaults to :class:`NullAnnouncer` (silent). Pass a
            ``RecordingAnnouncer`` in tests, or a ``CallableAnnouncer`` wired to
            the G1 speaker on the real robot.

    Returns:
        A compiled LangGraph app. Invoke with an :func:`initial_state` blackboard,
        e.g. ``app.invoke(initial_state(), {"recursion_limit": 400})``.
    """
    cfg = config or HarvestConfig()
    voice: Announcer = announcer or NullAnnouncer()

    def _ripe_visible(state: HarvestState) -> list[Okra]:
        """Visible, ripe, not-yet-excluded okra."""
        excluded = set(state.get("excluded_ids", []))
        return [
            o
            for o in state.get("okra_visible", [])
            if o.id not in excluded and o.ripeness >= cfg.ripeness_threshold
        ]

    def _offset(state: HarvestState) -> dict[str, float]:
        return state.get("robot_offset", {"x": 0.0, "y": 0.0})

    def _moved(state: HarvestState, lateral: float, forward: float) -> dict[str, float]:
        """Odometry after a base move of (lateral, forward) [m]."""
        o = _offset(state)
        return {"x": o["x"] + lateral, "y": o["y"] + forward}

    def _drop_pending(state: HarvestState, *ids: str) -> dict[str, dict[str, float]]:
        """Pending memory with the given ids removed (picked or excluded)."""
        pending = dict(state.get("pending", {}))
        for okra_id in ids:
            pending.pop(okra_id, None)
        return pending

    # ---- Nodes ---------------------------------------------------------------

    def detect(state: HarvestState) -> HarvestState:
        """Phase 2: observe the current view; list every okra in it."""
        okra = skills.detect_okra()
        iterations = state.get("iterations", 0) + 1
        if iterations == 1:
            voice.say(announce.start())
        # Remember every ripe, not-yet-excluded okra in the odometry frame, so we
        # can return for ones we pass. Refreshes the estimate on each sighting.
        offset = _offset(state)
        excluded = set(state.get("excluded_ids", []))
        pending = dict(state.get("pending", {}))
        for o in okra:
            if o.id in excluded or o.ripeness < cfg.ripeness_threshold:
                continue
            pending[o.id] = {
                "x": offset["x"] + o.pos_3d.get("x", 0.0),
                "y": offset["y"] + o.pos_3d.get("y", 0.0),
                "z": o.pos_3d.get("z", 0.0),
            }
        return HarvestState(
            okra_visible=okra,
            iterations=iterations,
            pending=pending,
            log=state.get("log", []) + [f"detect: saw {len(okra)} okra (iter {iterations})"],
        )

    def select(state: HarvestState) -> HarvestState:
        """Phase 3: decide the next move = grasp / approach / (nothing here).

        Priority: a ripe okra already in the reach box → grasp it. Otherwise the
        nearest ripe okra that is only out of reach in x/y (height OK) → approach
        it. Ripe okra out of HEIGHT reach are skipped (the G1 cannot squat).
        """
        ripe = _ripe_visible(state)
        log = state.get("log", [])

        in_box = [o for o in ripe if cfg.reach.contains(o.pos_3d)]
        if in_box:
            in_box.sort(key=lambda o: o.ripeness, reverse=True)
            target = in_box[0]
            voice.say(announce.grasping())
            return HarvestState(
                target_id=target.id,
                approach_id=None,
                grasp_attempts=0,
                reposition_attempts=0,
                empty_advances=0,
                mode="harvest",
                log=log + [f"select: grasp target={target.id}"],
            )

        # Height-unreachable ripe okra can't be fixed by base motion → skip them.
        excluded = list(state.get("excluded_ids", []))
        records = list(state.get("records", []))
        pending = dict(state.get("pending", {}))
        skipped_height = False
        for o in ripe:
            if not cfg.reach.z_contains(o.pos_3d):
                excluded.append(o.id)
                pending.pop(o.id, None)  # can't reach by base motion -> forget it
                records.append({"okra_id": o.id, "result": "skipped_height"})
                log = log + [f"select: {o.id} out of height reach -> skip"]
                skipped_height = True
        if skipped_height:
            voice.say(announce.skip_height())

        approachable = [
            o
            for o in ripe
            if o.id not in excluded
            and cfg.reach.z_contains(o.pos_3d)
            and not cfg.reach.contains(o.pos_3d)
        ]
        if approachable:
            def move_magnitude(o: Okra) -> float:
                lat, fwd = cfg.reach.move_to_center(o.pos_3d)
                return abs(lat) + abs(fwd)

            approachable.sort(key=move_magnitude)
            approach = approachable[0]
            out = HarvestState(
                target_id=None,
                approach_id=approach.id,
                excluded_ids=excluded,
                records=records,
                pending=pending,
                empty_advances=0,
                mode="reposition",
                log=log + [f"select: no in-reach okra; approach {approach.id}"],
            )
            # Reset the reposition counter only when switching to a new target.
            if approach.id != state.get("approach_id"):
                out["reposition_attempts"] = 0
            return out

        return HarvestState(
            target_id=None,
            approach_id=None,
            excluded_ids=excluded,
            records=records,
            pending=pending,
            log=log + ["select: nothing reachable/approachable here"],
        )

    def grasp(state: HarvestState) -> HarvestState:
        """Phase 5: reach + grasp the target (okra-ACT on the real robot)."""
        target = find_okra(state, state.get("target_id"))
        attempts = state.get("grasp_attempts", 0) + 1
        if attempts > 1:
            voice.say(announce.regrasp())
        if target is not None:
            skills.grasp_okra(target, cfg.grasp_force)
        return HarvestState(
            grasp_attempts=attempts,
            log=state.get("log", [])
            + [f"grasp: {state.get('target_id')} (attempt {attempts}, force {cfg.grasp_force})"],
        )

    def verify(state: HarvestState) -> HarvestState:
        """Phase 6: re-observe and check the fruit is actually held + separated."""
        ok = skills.verify_harvest()
        return HarvestState(
            last_verify_ok=ok,
            log=state.get("log", []) + [f"verify: {'OK' if ok else 'FAILED'}"],
        )

    def record(state: HarvestState) -> HarvestState:
        """Phase 7-8: count the fruit, persist a record, exclude it from re-targeting."""
        target_id = state.get("target_id")
        basket_count = state.get("basket_count", 0) + 1
        picks = state.get("picks", 0) + 1
        rec = {
            "okra_id": target_id,
            "result": "picked",
            "grasp_force": cfg.grasp_force,
            "retries": state.get("grasp_attempts", 1) - 1,
        }
        skills.record_harvest(rec)
        voice.say(announce.picked(basket_count))
        return HarvestState(
            basket_count=basket_count,
            basket_full=basket_count >= cfg.basket_capacity,
            picks=picks,
            excluded_ids=state.get("excluded_ids", []) + ([target_id] if target_id else []),
            pending=_drop_pending(state, target_id) if target_id else state.get("pending", {}),
            revisit_attempts=0,  # a successful pick is progress
            records=state.get("records", []) + [rec],
            log=state.get("log", []) + [f"record: picked, basket={basket_count}"],
        )

    def give_up(state: HarvestState) -> HarvestState:
        """§7: re-grasp retries exhausted — mark this okra failed and exclude it."""
        target_id = state.get("target_id")
        voice.say(announce.give_up())
        return HarvestState(
            target_id=None,
            excluded_ids=state.get("excluded_ids", []) + ([target_id] if target_id else []),
            pending=_drop_pending(state, target_id) if target_id else state.get("pending", {}),
            log=state.get("log", []) + [f"give_up: {target_id} marked failed"],
        )

    def reposition(state: HarvestState) -> HarvestState:
        """§5: move the base to bring the approach target into the reach box.

        Computes the lateral/forward move that lands the fruit at the box centre,
        clamped so the base never drives closer than ``standoff_min`` (ridge
        safety). Gives up on a fruit after ``max_reposition_attempts`` and skips it.
        """
        approach = find_okra(state, state.get("approach_id"))
        attempts = state.get("reposition_attempts", 0) + 1
        if approach is None:
            return HarvestState(log=state.get("log", []) + ["reposition: target lost"])
        if attempts > cfg.max_reposition_attempts:
            return HarvestState(
                approach_id=None,
                reposition_attempts=0,
                excluded_ids=state.get("excluded_ids", []) + [approach.id],
                pending=_drop_pending(state, approach.id),
                log=state.get("log", [])
                + [f"reposition: gave up on {approach.id} (attempt cap) -> skip"],
            )
        lateral, forward = cfg.reach.move_to_center(approach.pos_3d)
        # Ridge safety: never command a forward move that would bring the target
        # closer than the standoff minimum.
        forward = min(forward, approach.pos_3d.get("y", 0.0) - cfg.standoff_min)
        skills.relative_move(lateral, forward)
        new_offset = _moved(state, lateral, forward)
        # Announce the dominant direction of the move (depth wins ties).
        if abs(forward) >= abs(lateral) and abs(forward) > 1e-9:
            direction = "forward" if forward > 0 else "back"
        elif abs(lateral) > 1e-9:
            direction = "left" if lateral < 0 else "right"
        else:
            direction = "forward"
        voice.say(announce.approaching(direction))
        return HarvestState(
            reposition_attempts=attempts,
            robot_offset=new_offset,
            mode="reposition",
            log=state.get("log", [])
            + [f"reposition: move (lat {lateral:+.2f}, fwd {forward:+.2f}) toward {approach.id}"],
        )

    def advance_left(state: HarvestState) -> HarvestState:
        """§5 sweep: no fruit in view here — step LEFT to discover more.

        Harvest progresses right→left across the row, so the discovery sweep
        moves in the -x (left) direction by ``advance_step``.
        """
        skills.relative_move(-cfg.advance_step, 0.0)
        empty = state.get("empty_advances", 0) + 1
        if empty == 1:  # announce once per dry spell, not every sweep step
            voice.say(announce.searching())
        return HarvestState(
            empty_advances=empty,
            reposition_attempts=0,
            robot_offset=_moved(state, -cfg.advance_step, 0.0),
            mode="advance_left",
            log=state.get("log", []) + [f"advance_left: -{cfg.advance_step}m (empty {empty})"],
        )

    def revisit(state: HarvestState) -> HarvestState:
        """§5: go back for a left-behind okra remembered in `pending`.

        Picks the nearest pending okra (by its position relative to where we are
        now, from odometry) and moves the base to bring it to the reach centre,
        then re-detects. Bounded by ``max_revisits``; if exceeded, the remaining
        pending okra are skipped so the run can finish.
        """
        pending = dict(state.get("pending", {}))
        offset = _offset(state)
        attempts = state.get("revisit_attempts", 0) + 1
        if not pending or attempts > cfg.max_revisits:
            return HarvestState(
                pending={},
                revisit_attempts=0,
                excluded_ids=state.get("excluded_ids", []) + list(pending),
                log=state.get("log", []) + ["revisit: giving up on remaining pending"],
            )

        def rel_dist(item: tuple[str, dict[str, float]]) -> float:
            p = item[1]
            return abs(p["x"] - offset["x"]) + abs(p["y"] - offset["y"])

        pid, ppos = min(pending.items(), key=rel_dist)
        rel_x = ppos["x"] - offset["x"]
        rel_y = ppos["y"] - offset["y"]
        lateral = rel_x - cfg.reach.x_center
        forward = rel_y - cfg.reach.y_center
        forward = min(forward, rel_y - cfg.standoff_min)  # ridge safety
        skills.relative_move(lateral, forward)
        voice.say(announce.revisiting())
        return HarvestState(
            revisit_attempts=attempts,
            robot_offset=_moved(state, lateral, forward),
            empty_advances=0,
            mode="reposition",
            log=state.get("log", [])
            + [f"revisit: return to {pid} (lat {lateral:+.2f}, fwd {forward:+.2f})"],
        )

    def finish(state: HarvestState) -> HarvestState:
        """Terminal node: announce the harvest summary, then end."""
        voice.say(announce.done(state.get("picks", 0)))
        return HarvestState(
            mode="done",
            log=state.get("log", []) + [f"done: {state.get('picks', 0)} picked"],
        )

    # ---- Routers: the conditional edges --------------------------------------

    def route_after_select(state: HarvestState) -> str:
        if state.get("iterations", 0) >= cfg.max_harvest_iterations:
            return FINISH
        if state.get("basket_full"):
            return FINISH
        if state.get("target_id"):
            return GRASP
        if state.get("approach_id"):
            return REPOSITION
        if state.get("empty_advances", 0) < cfg.max_empty_advances:
            return ADVANCE_LEFT  # §5: sweep left (right→left harvest) to look for more
        if state.get("pending"):
            return REVISIT  # §5: swept enough — go back for okra we passed
        return FINISH  # §8: swept enough and nothing left behind → done

    def route_after_verify(state: HarvestState) -> str:
        if state.get("last_verify_ok"):
            return RECORD
        if state.get("grasp_attempts", 0) < cfg.max_grasp_retries:
            return GRASP  # §7: re-grasp (bounded)
        return GIVE_UP

    def route_after_record(state: HarvestState) -> str:
        if state.get("basket_full"):
            return FINISH
        if state.get("iterations", 0) >= cfg.max_harvest_iterations:
            return FINISH
        return DETECT  # loop: re-detect (occlusion/pose changes after a pick)

    # ---- Wire the graph -------------------------------------------------------

    g = StateGraph(HarvestState)
    for name, fn in (
        (DETECT, detect),
        (SELECT, select),
        (GRASP, grasp),
        (VERIFY, verify),
        (RECORD, record),
        (GIVE_UP, give_up),
        (REPOSITION, reposition),
        (ADVANCE_LEFT, advance_left),
        (REVISIT, revisit),
        (FINISH, finish),
    ):
        g.add_node(name, fn)

    g.add_edge(START, DETECT)
    g.add_edge(DETECT, SELECT)
    g.add_conditional_edges(
        SELECT,
        route_after_select,
        {
            GRASP: GRASP,
            REPOSITION: REPOSITION,
            ADVANCE_LEFT: ADVANCE_LEFT,
            REVISIT: REVISIT,
            FINISH: FINISH,
        },
    )
    g.add_edge(GRASP, VERIFY)
    g.add_conditional_edges(
        VERIFY, route_after_verify, {RECORD: RECORD, GRASP: GRASP, GIVE_UP: GIVE_UP}
    )
    g.add_conditional_edges(RECORD, route_after_record, {DETECT: DETECT, FINISH: FINISH})
    g.add_edge(GIVE_UP, DETECT)
    g.add_edge(REPOSITION, DETECT)
    g.add_edge(ADVANCE_LEFT, DETECT)
    g.add_edge(REVISIT, DETECT)
    g.add_edge(FINISH, END)

    return g.compile()


__all__ = [
    "build_harvest_graph",
    "DETECT",
    "SELECT",
    "GRASP",
    "VERIFY",
    "RECORD",
    "GIVE_UP",
    "REPOSITION",
    "ADVANCE_LEFT",
    "REVISIT",
    "FINISH",
]
