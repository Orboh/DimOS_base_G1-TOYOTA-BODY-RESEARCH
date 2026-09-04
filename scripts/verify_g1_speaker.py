#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
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

# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Verify the G1 onboard speaker for the okra-harvest Japanese announcements.

⚠️ Connects to the real G1 (DDS over the wired NIC) and makes it SPEAK. No motion
— audio only — so it is safe, but it needs the robot powered and reachable. Run
it on your laptop with the robot connected (the operator runs robot-facing
commands).

It sweeps a few ``speaker_id`` values, speaking a Japanese test phrase (and an
English one) for each, so you can hear which id produces intelligible Japanese.
Unitree's onboard TTS may only support some languages — if NONE speaks Japanese,
we'll switch to synthesising Japanese audio off-board and pushing it with
``AudioClient.PlayStream`` instead of ``TtsMaker``.

Run:
    ROBOT_INTERFACE=<nic> .venv/bin/python scripts/verify_g1_speaker.py
    # or:  .venv/bin/python scripts/verify_g1_speaker.py --nic <nic> --speakers 0,1,2,3 --volume 80

Note: the SDK's TtsMaker has a quirky index counter; this script sets a unique
index before each call so repeated phrases are not de-duplicated by the robot.
"""

from __future__ import annotations

import argparse
import os
import time

_JA = "こんにちは。オクラ収穫ロボットです。日本語が聞こえますか？"
_EN = "Hello. This is the okra harvesting robot. Testing English."


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify the G1 onboard speaker (TTS).")
    ap.add_argument(
        "--nic",
        default=os.getenv("ROBOT_INTERFACE", ""),
        help="wired network interface to the G1 (or set ROBOT_INTERFACE)",
    )
    ap.add_argument(
        "--speakers", default="0,1,2,3", help="comma-separated speaker_id values to try"
    )
    ap.add_argument("--volume", type=int, default=None, help="0-100; leave unset to keep current")
    ap.add_argument("--ja", default=_JA, help="Japanese test phrase")
    ap.add_argument("--en", default=_EN, help="English test phrase")
    ap.add_argument("--pause", type=float, default=4.0, help="seconds to wait after each phrase")
    args = ap.parse_args()

    if not args.nic:
        raise SystemExit("Set --nic or ROBOT_INTERFACE to the wired NIC to the G1.")

    # Lazy import so the file is importable without the robot SDK present.
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

    print(f"[verify] ChannelFactoryInitialize on nic={args.nic!r}")
    ChannelFactoryInitialize(0, args.nic)

    client = AudioClient()
    try:
        client.SetTimeout(10.0)
    except Exception as exc:
        print(f"[verify] SetTimeout not available ({exc}); continuing")
    client.Init()

    if args.volume is not None:
        print(f"[verify] SetVolume({args.volume}) -> code={client.SetVolume(args.volume)}")
    try:
        print(f"[verify] GetVolume -> {client.GetVolume()}")
    except Exception as exc:
        print(f"[verify] GetVolume failed: {exc}")

    speakers = [int(s) for s in args.speakers.split(",") if s.strip() != ""]
    idx = 0
    for sid in speakers:
        for label, text in (("JA", args.ja), ("EN", args.en)):
            idx += 1
            client.tts_index = idx  # unique index (work around the SDK's += bug / de-dup)
            print(f"\n[verify] speaker_id={sid} [{label}] speaking: {text!r}")
            code = client.TtsMaker(text, sid)
            print(f"[verify] TtsMaker returned code={code} (0 usually = accepted)")
            print(f"[verify] ...listen... (waiting {args.pause}s)")
            time.sleep(args.pause)

    print("\n[verify] DONE. Note which speaker_id spoke intelligible JAPANESE.")
    print("[verify] If none did, the onboard TTS likely lacks Japanese -> we'll")
    print("[verify] synthesise Japanese audio off-board and use PlayStream instead.")


if __name__ == "__main__":
    main()
