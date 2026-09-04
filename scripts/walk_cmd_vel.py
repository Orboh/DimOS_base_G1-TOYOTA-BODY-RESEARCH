#!/usr/bin/env python3
"""Drive the G1 base by publishing Twist on /cmd_vel (the WebRTC walking path).

⚠️⚠️ THIS MOVES THE ROBOT (legs). ⚠️⚠️ This is the FleetSeek-proven locomotion
path (exp_01KTN1QQQ98REX20F8PFNKG63C): direct DDS LocoClient is silently ignored
on this G1 — walking only works through ``unitree-g1-basic`` (WebRTC → sport
commands) consuming /cmd_vel. So:

  1) start the walking stack (separate terminal, targets the control board .161):
       dimos --viewer none --robot-ip 192.168.123.161 run unitree-g1-basic
  2) with the robot STANDING / in running mode and an e-stop in hand, run this to
     publish a small bounded move on /cmd_vel at ~10 Hz, then zeros to stop:
       .venv/bin/python scripts/walk_cmd_vel.py --forward 0.2

This publishes raw LCM (same channel G1Connection subscribes to) — no DDS, no
dimos deploy here. Defaults are tiny. Ctrl-C publishes a stop.
"""

from __future__ import annotations

import argparse
import time

import lcm

from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3

CMD_VEL_CHANNEL = "/cmd_vel#geometry_msgs.Twist"


def _twist(vx: float, vy: float, vyaw: float) -> Twist:
    return Twist(linear=Vector3(vx, vy, 0.0), angular=Vector3(0.0, 0.0, vyaw))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forward", type=float, default=0.0, help="relative forward move [m] (+fwd/-back)")
    ap.add_argument("--lateral", type=float, default=0.0, help="relative lateral move [m] (+left/-right)")
    ap.add_argument("--yaw", type=float, default=0.0, help="relative turn [rad] (+CCW)")
    ap.add_argument("--speed", type=float, default=0.12, help="translation speed [m/s] (kept small)")
    ap.add_argument("--yaw-speed", type=float, default=0.3, help="turn speed [rad/s]")
    ap.add_argument("--hz", type=float, default=10.0, help="publish rate [Hz]")
    ap.add_argument("--max", type=float, default=0.5, help="safety cap on a single translation [m]")
    ap.add_argument("--dry", action="store_true", help="print the plan, publish nothing")
    args = ap.parse_args()

    for axis, val in (("forward", args.forward), ("lateral", args.lateral)):
        if abs(val) > args.max:
            raise SystemExit(f"{axis}={val} exceeds safety cap {args.max} m (raise --max deliberately)")

    moves: list[tuple[str, float, float, float, float]] = []
    if abs(args.forward) >= 1e-3:
        moves.append(("forward", args.speed * (1 if args.forward > 0 else -1), 0.0, 0.0,
                      abs(args.forward) / args.speed))
    if abs(args.lateral) >= 1e-3:
        moves.append(("lateral", 0.0, args.speed * (1 if args.lateral > 0 else -1), 0.0,
                      abs(args.lateral) / args.speed))
    if abs(args.yaw) >= 1e-3:
        moves.append(("yaw", 0.0, 0.0, args.yaw_speed * (1 if args.yaw > 0 else -1),
                      abs(args.yaw) / args.yaw_speed))
    if not moves:
        raise SystemExit("nothing to do; pass --forward / --lateral / --yaw")

    print("[walk] plan (publish on %s):" % CMD_VEL_CHANNEL)
    for name, vx, vy, vyaw, dur in moves:
        print(f"  {name}: vx={vx:+.2f} vy={vy:+.2f} vyaw={vyaw:+.2f} for {dur:.2f}s @ {args.hz:.0f}Hz")
    if args.dry:
        print("[walk] --dry: nothing published.")
        return 0

    lc = lcm.LCM()
    period = 1.0 / max(1.0, args.hz)

    def publish(vx: float, vy: float, vyaw: float) -> None:
        lc.publish(CMD_VEL_CHANNEL, _twist(vx, vy, vyaw).lcm_encode())

    try:
        for name, vx, vy, vyaw, dur in moves:
            n = max(1, int(dur / period))
            print(f"[walk] {name}: streaming {n} msgs @ {args.hz:.0f}Hz ({dur:.2f}s)")
            for _ in range(n):
                publish(vx, vy, vyaw)
                time.sleep(period)
            # stop: a few zero messages to be sure it halts
            for _ in range(3):
                publish(0.0, 0.0, 0.0)
                time.sleep(period)
            print(f"[walk] {name}: done, stopped")
            time.sleep(0.4)
        print("[walk] DONE — bounded move published on /cmd_vel.")
    except KeyboardInterrupt:
        for _ in range(3):
            publish(0.0, 0.0, 0.0)
            time.sleep(period)
        print("\n[walk] interrupted -> stop published.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
