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


def _demo_field() -> list[FieldOkra]:
    """Reach box centre is (x=0.30, y=0.45). Robot starts at the origin.

    Harvest progresses RIGHT→LEFT. This field showcases the recovery path: the
    robot approaches the nearest fruit (a LEFT strafe), which pushes the right
    fruit out of view; the leftward discovery sweep moves away from it, so the
    robot REVISITS its remembered position to come back for it — nothing is
    abandoned. One fruit is above arm height (skipped) and one is unripe
    (ignored). (Depth moves — too-far→forward, too-close→back — are exercised in
    the tests.)
    """
    return [
        FieldOkra("okra_high", x=0.30, y=0.45, z=1.50, ripeness=0.95),  # too high -> skip
        FieldOkra("okra_left", x=-0.05, y=0.45, z=0.80, ripeness=0.90),  # nearest -> approach left
        FieldOkra("okra_right", x=0.70, y=0.45, z=0.80, ripeness=0.88),  # pushed out -> revisited
        FieldOkra("okra_unripe", x=0.30, y=0.45, z=0.80, ripeness=0.20),  # unripe -> ignore
    ]


def main() -> None:
    cfg = HarvestConfig()
    # okra_a fails its first verify once, to show the §7 re-grasp recovery.
    skills = MockHarvestSkills(
        _demo_field(), reach=cfg.reach, fov=cfg.fov, flaky_verifies={"okra_left": 1}
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
    print(f"  basket_count  : {final['basket_count']}")
    print(f"  base moves    : {skills.move_calls}")
    print(f"  grasp calls   : {skills.grasp_calls}")
    print(f"  records       : {final['records']}")


if __name__ == "__main__":
    main()
