#!/usr/bin/env python3
# Copyright 2025-2026 Dimensional Inc.
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

"""Operator stop helper for the okra-harvest pipeline (single key: 'q').

Runs ALONGSIDE the unitree-g1-okra-harvest blueprint and owns the terminal keyboard
(the clicking happens in the Rerun viewer GUI, so the terminal is free). LCM-only, no
arm_sdk — pressing 'q' does the safe stop in order:

  q -> CUT G1 transmission, THEN quit. Publishes True on /g1/arm_sdk_disconnect;
       G1ArmSdkConnection (enable_disconnect) ramps weight->0 (hands the upper body
       back to the onboard controller). This helper WAITS for that ramp to finish,
       then exits — the launcher then tears the blueprint down. So G1 transmission is
       always cut before the program stops.

Single keypress (cbreak) when stdin is a TTY; line-mode ("q" + Enter) otherwise.

Run (laptop, same LCM bus as the harvest blueprint):
  LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1' \
    .venv/bin/python scripts/okra_harvest_keys.py
"""

from __future__ import annotations

import os
import select
import sys
import termios
import threading
import time
import tty

from dimos.msgs.std_msgs.Bool import Bool
from dimos.protocol.pubsub.impl.lcmpubsub import LCM, Topic

LCM_URL = os.getenv("LCM_DEFAULT_URL", "udpm://239.255.76.67:7667?ttl=1")
DISCONNECT_TOPIC = "/g1/arm_sdk_disconnect"
# Wait for the weight->0 ramp to finish before quitting (arm_sdk weight_ramp_s=2.0s
# + margin), so G1 transmission is fully cut before the program is torn down.
DISCONNECT_WAIT_S = float(os.getenv("OKRA_DISCONNECT_WAIT_S", "2.5"))


def main() -> int:
    lc = LCM(url=LCM_URL)
    lc.start()
    disc = Topic(DISCONNECT_TOPIC, Bool)
    print(f"[harvest-keys] LCM {LCM_URL} -> {DISCONNECT_TOPIC}")
    print("[harvest-keys] press 'q' to CUT G1 transmission (weight->0, arm back to the "
          "onboard controller) and then quit.")

    cmd = {"quit": False}

    def key_thread() -> None:
        is_tty = sys.stdin.isatty()
        if not is_tty:
            for line in sys.stdin:  # fallback: "q" + Enter
                if line.strip().lower() == "q":
                    cmd["quit"] = True
                    return
            return
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not cmd["quit"]:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    if sys.stdin.read(1).lower() == "q":
                        cmd["quit"] = True
                        return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    threading.Thread(target=key_thread, daemon=True).start()

    try:
        while not cmd["quit"]:
            time.sleep(0.05)
    finally:
        # Cut G1 transmission and wait for the weight ramp to complete BEFORE returning
        # (the launcher tears the blueprint down once this exits).
        try:
            lc.publish(disc, Bool(data=True))
            print(f"[q] -> CUT G1 transmission (weight->0); waiting {DISCONNECT_WAIT_S:.1f}s "
                  "for the handover, then quitting ...")
            time.sleep(DISCONNECT_WAIT_S)
        except Exception:
            pass
        try:
            lc.stop()
        except Exception:
            pass
    print("[harvest-keys] G1 transmission cut; quitting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
