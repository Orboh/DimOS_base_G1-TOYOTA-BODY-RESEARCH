#!/usr/bin/env python3
"""Standalone bounded-walk check for the harvest base motion (the robot WALKS).

⚠️⚠️ THIS MOVES THE ROBOT (legs / locomotion). ⚠️⚠️ It issues ONE small, bounded
relative move (forward / lateral / yaw) via the Unitree G1 LocoClient — the same
SetVelocity that the harvest's relative_move -> cmd_vel path ultimately calls —
then STOPS. Use it to confirm walking works on hardware before driving it from
the harvest flow.

Operator MUST have the robot already STANDING / balanced (main operation control,
e.g. via the controller/app) and e-stop in hand, with clear space around it.
This script does NOT change FSM/stand state on its own (no StandUp/Sit).

    ROBOT_INTERFACE=<nic> .venv/bin/python scripts/verify_g1_walk.py --forward 0.2
    ... --lateral -0.2        # step left (sign per robot; +y is left for SetVelocity)
    ... --yaw 0.3             # turn
    ... --speed 0.12 --dry    # print the plan only, send nothing

Defaults are intentionally tiny. Multiple axes run sequentially with a stop
between. Ctrl-C sends StopMove and exits.
"""

from __future__ import annotations

import argparse
import os
import time

_MIN = 1e-3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nic", default=os.getenv("ROBOT_INTERFACE", ""))
    ap.add_argument("--forward", type=float, default=0.0, help="relative forward move [m] (+fwd/-back)")
    ap.add_argument("--lateral", type=float, default=0.0, help="relative lateral move [m] (+left/-right)")
    ap.add_argument("--yaw", type=float, default=0.0, help="relative turn [rad] (+CCW)")
    ap.add_argument("--speed", type=float, default=0.12, help="translation speed [m/s] (kept small)")
    ap.add_argument("--yaw-speed", type=float, default=0.3, help="turn speed [rad/s]")
    ap.add_argument("--max", type=float, default=0.5, help="safety cap on any single move [m]")
    ap.add_argument("--hz", type=float, default=10.0, help="cmd resend rate [Hz] (G1 needs a continuous stream)")
    ap.add_argument("--dry", action="store_true", help="print the plan, send nothing")
    args = ap.parse_args()

    if not args.nic:
        raise SystemExit("Set --nic or ROBOT_INTERFACE to the wired NIC to the G1.")
    for axis, val, cap in (("forward", args.forward, args.max), ("lateral", args.lateral, args.max)):
        if abs(val) > cap:
            raise SystemExit(f"{axis}={val} exceeds safety cap {cap} m; lower it or raise --max deliberately")

    # (vx, vy, vyaw, duration) legs: vx=forward, vy=left(+), vyaw=turn.
    moves: list[tuple[str, float, float, float, float]] = []
    if abs(args.forward) >= _MIN:
        d = abs(args.forward) / args.speed
        moves.append(("forward", args.speed * (1 if args.forward > 0 else -1), 0.0, 0.0, d))
    if abs(args.lateral) >= _MIN:
        d = abs(args.lateral) / args.speed
        moves.append(("lateral", 0.0, args.speed * (1 if args.lateral > 0 else -1), 0.0, d))
    if abs(args.yaw) >= _MIN:
        d = abs(args.yaw) / args.yaw_speed
        moves.append(("yaw", 0.0, 0.0, args.yaw_speed * (1 if args.yaw > 0 else -1), d))
    if not moves:
        raise SystemExit("nothing to do; pass --forward / --lateral / --yaw")

    print("[walk] plan:")
    for name, vx, vy, vyaw, dur in moves:
        print(f"  {name}: vx={vx:+.2f} vy={vy:+.2f} vyaw={vyaw:+.2f} for {dur:.2f}s")
    if args.dry:
        print("[walk] --dry: nothing sent.")
        return 0

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

    print(f"[walk] ChannelFactoryInitialize nic={args.nic!r}")
    ChannelFactoryInitialize(0, args.nic)
    client = LocoClient()
    try:
        client.SetTimeout(10.0)
    except Exception as exc:  # noqa: BLE001
        print(f"[walk] SetTimeout n/a ({exc})")
    client.Init()

    # G1 needs a CONTINUOUS velocity stream — a single SetVelocity is silently
    # dropped/ignored (FleetSeek exp_01KTN1QQQ98REX20F8PFNKG63C). So resend at
    # --hz for the whole duration, then send zeros to stop.
    period = 1.0 / max(1.0, args.hz)
    try:
        for name, vx, vy, vyaw, dur in moves:
            n = max(1, int(dur / period))
            print(f"[walk] {name}: streaming SetVelocity({vx:+.2f},{vy:+.2f},{vyaw:+.2f}) "
                  f"@ {args.hz:.0f}Hz for {dur:.2f}s ({n} sends)")
            first_code = None
            for _ in range(n):
                code = client.SetVelocity(vx, vy, vyaw, max(period * 2, 0.5))
                if first_code is None:
                    first_code = code
                time.sleep(period)
            client.StopMove()
            print(f"[walk] {name}: first SetVelocity code={first_code} (0 usually = accepted); stopped")
            time.sleep(0.4)  # settle before the next axis
        print("[walk] DONE — robot performed the bounded move and stopped.")
    except KeyboardInterrupt:
        client.StopMove()
        print("\n[walk] interrupted -> StopMove sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
