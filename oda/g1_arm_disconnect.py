#!/usr/bin/env python3
"""Publish the operator 2-stage stop to a running okra blueprint.

Sends ``/g1/arm_sdk_disconnect`` = True, which makes ``G1ArmSdkConnection`` ramp
the arm_sdk weight 1->0 over ``weight_ramp_s`` and hold the arm at its measured
pose (see g1_arm_sdk_connection.py ``_on_disconnect``). The blueprint process
stays alive; quit it only AFTER the "transmission CUT (weight=0)" line appears.

Why this exists: Ctrl-C alone does NOT ramp the weight down. The coordinator
dispatches module stop() with ``call_nowait`` (rpc_client.py:70-74, to dodge a
deadlock), so it never waits for the ~2s ramp in G1ArmSdkConnection.stop()
before tearing the workers down. Verified 2026-07-30: SIGINT -> full shutdown in
335 ms, last commanded weight still 1.00.

Usage (needs the same LCM bus, i.e. same host):

    .venv/bin/python oda/g1_arm_disconnect.py

Then watch the blueprint log for:

    G1ArmSdkConnection: G1 transmission CUT (weight=0; ...). Safe to quit ...
"""

from __future__ import annotations

import time

from dimos.core.transport import LCMTransport
from dimos.msgs.std_msgs.Bool import Bool

# Same topic the blueprints wire ``disconnect`` to.
_TOPIC = "/g1/arm_sdk_disconnect"

# The subscriber ignores repeats (``_on_disconnect`` returns early once latched),
# so resending only covers a dropped datagram on the multicast bus.
_REPEATS = 5
_INTERVAL_S = 0.2


def main() -> None:
    transport = LCMTransport(_TOPIC, Bool)
    transport.start()

    msg = Bool()
    msg.data = True
    for _ in range(_REPEATS):
        transport.broadcast(None, msg)
        time.sleep(_INTERVAL_S)

    print(f"published {_TOPIC}=True ({_REPEATS}x)")
    print("wait for 'transmission CUT (weight=0)' in the blueprint log before quitting it")


if __name__ == "__main__":
    main()
