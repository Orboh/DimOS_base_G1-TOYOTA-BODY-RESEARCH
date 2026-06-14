# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Dry-run the okra-harvest LangGraph against a mock 3D field (no robot).

Run from the repo root::

    .venv/bin/python -m dimos.robot.unitree.g1.harvest.run_demo

Prints the phase-by-phase trace (including base moves) and the final blackboard
summary. The field below exercises: forward approach (too far), height skip
(too high), left strafe, and a right sweep to discover an out-of-view fruit.
"""

from __future__ import annotations

from dimos.robot.unitree.g1.harvest.announce import RecordingAnnouncer
from dimos.robot.unitree.g1.harvest.blackboard import HarvestConfig, initial_state
from dimos.robot.unitree.g1.harvest.graph import build_harvest_graph
from dimos.robot.unitree.g1.harvest.skills import FieldOkra, MockHarvestSkills

# The harvest loop revisits nodes; the default recursion limit (25) is too small.
_RECURSION_LIMIT = 400


def _demo_stations() -> list[list[FieldOkra]]:
    """Two work stations. Reach box centre is (x=0.30, y=0.45); robot starts at
    each station's origin. Harvest progresses RIGHT→LEFT.

    Station 0 showcases the recovery path: approach the nearest fruit (a LEFT
    strafe), which pushes the right fruit out of view; the leftward sweep moves
    away from it, so the robot REVISITS its remembered position. One fruit is too
    high (skipped), one unripe (ignored). Station 1 is a simple pair in reach.
    With basket_capacity=3 the basket fills mid-run and is swapped. (Depth
    moves — too-far→forward, too-close→back — are exercised in the tests.)
    """
    station0 = [
        FieldOkra("s0_high", x=0.30, y=0.45, z=1.50, ripeness=0.95),  # too high -> skip
        FieldOkra("s0_left", x=-0.05, y=0.45, z=0.80, ripeness=0.90),  # nearest -> approach left
        FieldOkra("s0_right", x=0.70, y=0.45, z=0.80, ripeness=0.88),  # pushed out -> revisited
        FieldOkra("s0_unripe", x=0.30, y=0.45, z=0.80, ripeness=0.20),  # unripe -> ignore
    ]
    station1 = [
        FieldOkra("s1_a", x=0.25, y=0.45, z=0.80, ripeness=0.92),  # in reach -> pick
        FieldOkra("s1_b", x=0.40, y=0.45, z=0.80, ripeness=0.90),  # in reach -> pick
    ]
    return [station0, station1]


def main() -> None:
    cfg = HarvestConfig(basket_capacity=3)  # small basket so a swap happens in the demo
    # s0_left fails its first verify once, to show the §7 re-grasp recovery.
    skills = MockHarvestSkills(
        reach=cfg.reach, fov=cfg.fov, stations=_demo_stations(), flaky_verifies={"s0_left": 1}
    )
    # Record what the robot WOULD say (no audio hardware needed for the dry run).
    voice = RecordingAnnouncer()
    app = build_harvest_graph(skills, cfg, announcer=voice)

    final = app.invoke(initial_state(), {"recursion_limit": _RECURSION_LIMIT})

    print("=== phase trace ===")
    for line in final["log"]:
        print(" ", line)

    print("\n=== spoken announcements (Japanese, to G1 speaker on the real robot) ===")
    for line in voice.said:
        print(" 🔊", line)

    print("\n=== result ===")
    print(f"  picks         : {final['picks']}")
    print(f"  stations moved: {skills.station_moves}  (final station_id={final['station_id']})")
    print(f"  basket swaps  : {skills.basket_swaps}")
    print(f"  grasp calls   : {skills.grasp_calls}")
    print(f"  records       : {len(final['records'])}")


if __name__ == "__main__":
    main()
