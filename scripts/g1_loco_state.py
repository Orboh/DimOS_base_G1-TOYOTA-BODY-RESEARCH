#!/usr/bin/env python3
"""Read the G1 locomotion FSM state (READ-ONLY — no motion).

Per the G1 spec (動作サービス / sport_services_interface), Move/SetVelocity only
walks when the built-in motion control is in a WALK-capable FSM:
  0 ZeroTorque | 1 Damping | 2 Squat | 3 Sit | 4 Lock Standing  -> NO walking
  500/501 Walk Motion | 801/802 Run                              -> walks
GetFsmMode: 0 = standing, 1 = moving.

If Move returns code 0 but the robot doesn't move, the FSM is almost certainly
not walk-capable. This tool just reads GetFsmId / GetFsmMode so we can see it.

    ROBOT_INTERFACE=<nic> .venv/bin/python scripts/g1_loco_state.py
"""

from __future__ import annotations

import argparse
import os

_FSM = {
    0: "Zero Torque (no balance)", 1: "Damping (no balance)", 2: "Squat",
    3: "Sit", 4: "Lock Standing (no walking)", 500: "Walk Motion",
    501: "Walk Motion 3Dof-waist", 702: "Lie/Stand", 706: "Balance Squat",
    801: "Run", 802: "Run (ai_sport)",
}
_WALK_OK = {500, 501, 801, 802}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nic", default=os.getenv("ROBOT_INTERFACE", ""))
    args = ap.parse_args()
    if not args.nic:
        raise SystemExit("Set --nic or ROBOT_INTERFACE to the wired NIC to the G1.")

    import json

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.loco.g1_loco_api import (
        ROBOT_API_ID_LOCO_GET_FSM_ID,
        ROBOT_API_ID_LOCO_GET_FSM_MODE,
    )
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

    ChannelFactoryInitialize(0, args.nic)
    c = LocoClient()
    try:
        c.SetTimeout(10.0)
    except Exception:  # noqa: BLE001
        pass
    c.Init()

    def _get(api_id: int, label: str):
        # The SDK exposes the GET_* api ids but no helper method — call directly.
        code, data = c._Call(api_id, json.dumps({}))
        val = None
        try:
            d = json.loads(data) if isinstance(data, str) and data else data
            val = d.get("data", d) if isinstance(d, dict) else d
        except Exception:  # noqa: BLE001
            val = data
        print(f"[loco] {label} -> code={code} data={data!r} value={val}")
        return val

    fsm_id = _get(ROBOT_API_ID_LOCO_GET_FSM_ID, "GetFsmId(7001)")
    fsm_mode = _get(ROBOT_API_ID_LOCO_GET_FSM_MODE, "GetFsmMode(7002)")
    try:
        fsm_id = int(fsm_id)
    except (TypeError, ValueError):
        fsm_id = -1
    print(f"[loco] FSM id={fsm_id} ({_FSM.get(fsm_id, '?')}), mode={fsm_mode}")

    if fsm_id in _WALK_OK:
        print("[loco] -> FSM is WALK-CAPABLE. If it still won't move, the issue is the "
              "command path (cmd_vel not reaching the robot), not the mode.")
    else:
        print(f"[loco] -> FSM {fsm_id} is NOT walk-capable. The robot must be put into "
              "Walk/Run (e.g. operator: stand up + balance-stand via controller; or "
              "Start()/BalanceStand()). That's why Move was ignored despite code=0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
