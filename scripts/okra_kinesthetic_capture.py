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

"""Standalone kinesthetic data recorder (dimos venv, LCM read-only).

Runs ALONGSIDE the unitree-g1-okra-collect blueprint while a human hand-guides the
right arm. Subscribes the robot state + the wrist camera over LCM (NO arm_sdk — it
only reads, so no contention with the blueprint that owns rt/arm_sdk):
  /g1/motor_states        (JointState)  -> right-arm q = motors 22-28 (7)
  /camera/right_wrist_color (Image)     -> cam_right_wrist frame (BGR)

Operator loop (single keypress, like xr_teleoperate):
  s -> START: make the RIGHT arm compliant AND begin recording (one key = compliant +
       record). Press AFTER the click-driven IK reach has placed the arm at the pre-grasp.
       Press 's' again to STOP+save. (The next okra click re-stiffens & slews to the new pre-grasp.)
  c -> (optional) make the arm compliant WITHOUT recording — to pre-position by hand.
  q -> quit
Per episode: click okra (viewer) -> arm reaches+holds -> 's' (compliant+rec) -> hand-guide -> 's' stop.

Each episode is dumped raw (NOT lerobot — that conversion is offline in the ACT venv
via scripts/okra_lerobot_writer.py, to keep venvs clean):
  <out>/episode_NNN/q.npy        float32 [T,7]  right-arm q per recorded frame
  <out>/episode_NNN/NNNN.jpg     wrist frames (BGR), one per row of q.npy
  <out>/episode_NNN/meta.json    {"fps":..., "task":...}

All collection artifacts live under ONE base dir (default ~/okra_collect): raw/ here,
lerobot/ from the offline converter.

Run (laptop, same LCM bus as the collect blueprint):
  CYCLONEDDS_HOME=... LCM_DEFAULT_URL='udpm://239.255.76.67:7667?ttl=1' \
    .venv/bin/python scripts/okra_kinesthetic_capture.py --out ~/okra_collect/raw
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import shutil
import sys
import termios
import threading
import time
import tty

import cv2
import numpy as np

from dimos.msgs.sensor_msgs.Image import Image
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.msgs.std_msgs.Bool import Bool
from dimos.protocol.pubsub.impl.lcmpubsub import LCM, Topic

LCM_URL = os.getenv("LCM_DEFAULT_URL", "udpm://239.255.76.67:7667?ttl=1")
_RIGHT_SLICE = slice(22, 29)  # right arm in the 29-DOF motor vector
_ARM_END = 29


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default=str(Path.home() / "okra_collect" / "raw"))
    ap.add_argument("--fps", type=float, default=15.0, help="record rate [Hz] (match wrist cam)")
    ap.add_argument("--task", type=str, default="pick the okra")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lock = threading.Lock()
    latest = {"q": None, "img": None}

    def on_state(msg, _t):  # type: ignore[no-untyped-def]
        pos = list(msg.position)
        if len(pos) >= _ARM_END:
            with lock:
                latest["q"] = np.array([float(x) for x in pos[_RIGHT_SLICE]], dtype=np.float32)

    def on_wrist(msg, _t):  # type: ignore[no-untyped-def]
        try:
            bgr = msg.to_opencv()
        except Exception:
            return
        with lock:
            latest["img"] = bgr

    lc = LCM(url=LCM_URL)
    lc.start()
    lc.subscribe(Topic("/g1/motor_states", JointState), on_state)
    lc.subscribe(Topic("/camera/right_wrist_color", Image), on_wrist)
    compliant_topic = Topic("/g1/reach_done", Bool)  # 'c' -> right arm compliant
    print(f"[capture] LCM {LCM_URL}  out={out}  fps={args.fps}")
    print("[capture] waiting for motor_states + wrist frames ...")
    t0 = time.time()
    while time.time() - t0 < 5:
        with lock:
            ok = latest["q"] is not None and latest["img"] is not None
        if ok:
            break
        time.sleep(0.1)
    with lock:
        if latest["q"] is None:
            print("[capture] no /g1/motor_states — is the collect blueprint running?")
            return 2
        if latest["img"] is None:
            print("[capture] no /camera/right_wrist_color — is the wrist publisher running?")
            return 2
    print(
        "[capture] ready. Per episode: click okra -> arm reaches & holds -> 's' "
        "(arm goes compliant + record starts) -> hand-guide -> 's' stop. 'q' quit. "
        "('c' = compliant only, optional.)"
    )

    # Single-keypress control (like xr_teleoperate): 'c' compliant, 's' record toggle, 'q' quit.
    # Uses cbreak (no Enter needed) when stdin is a TTY; falls back to line-mode otherwise.
    cmd = {"toggle": False, "compliant": False, "quit": False}

    def key_thread() -> None:
        is_tty = sys.stdin.isatty()
        if not is_tty:
            for line in sys.stdin:  # fallback: "c"/"s"/"q" + Enter
                c = line.strip().lower()
                if c == "q":
                    cmd["quit"] = True
                    return
                if c == "s":
                    cmd["toggle"] = True
                if c == "c":
                    cmd["compliant"] = True
            return
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not cmd["quit"]:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    c = sys.stdin.read(1).lower()
                    if c == "q":
                        cmd["quit"] = True
                        return
                    if c == "s":
                        cmd["toggle"] = True
                    if c == "c":
                        cmd["compliant"] = True
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    threading.Thread(target=key_thread, daemon=True).start()

    # Continue numbering AFTER any existing episodes (don't reset to 0 / overwrite
    # earlier runs). Each run appends new episode_NNN dirs.
    existing = [
        int(p.name.split("_")[1]) for p in out.glob("episode_*") if p.name.split("_")[1].isdigit()
    ]
    ep = (max(existing) + 1) if existing else 0
    if existing:
        print(f"[capture] {len(existing)} existing episodes; new ones start at episode_{ep:03d}.")
    recording = False
    qbuf: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    period = 1.0 / max(1.0, args.fps)
    next_t = time.perf_counter()

    def save_episode() -> None:
        nonlocal ep
        if not qbuf:
            print("  [skip] empty episode")
            return
        d = out / f"episode_{ep:03d}"
        if d.exists():
            shutil.rmtree(d)  # never mix stale frames with a fresh episode (count mismatch)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "q.npy", np.asarray(qbuf, dtype=np.float32))
        for i, fr in enumerate(frames):
            cv2.imwrite(str(d / f"{i:04d}.jpg"), fr)
        (d / "meta.json").write_text(json.dumps({"fps": args.fps, "task": args.task}))
        print(f"  [saved] {d.name}: {len(qbuf)} frames")
        ep += 1

    try:
        while not cmd["quit"]:
            if cmd["compliant"]:
                cmd["compliant"] = False
                lc.publish(compliant_topic, Bool(data=True))
                print(
                    "[c] -> RIGHT arm compliant (hand-guide; support it). Next okra click re-stiffens."
                )
            if cmd["toggle"]:
                cmd["toggle"] = False
                if not recording:
                    # 's' START also makes the arm compliant (one key = compliant + record).
                    lc.publish(compliant_topic, Bool(data=True))
                    qbuf, frames = [], []
                    recording = True
                    print(
                        f"[REC] episode_{ep:03d} START (arm compliant) — hand-guide the grasp. 's' to stop."
                    )
                else:
                    recording = False
                    save_episode()
                    print(
                        "[capture] stopped. NEXT episode: click okra -> wait until the arm reaches "
                        "the pre-grasp -> 's' (compliant + record) -> hand-guide -> 's' stop.  ('q' quit)"
                    )
            if recording:
                with lock:
                    q = None if latest["q"] is None else latest["q"].copy()
                    img = None if latest["img"] is None else latest["img"].copy()
                if q is not None and img is not None:
                    qbuf.append(q)
                    frames.append(img)
            next_t += period
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()
    finally:
        if recording:
            save_episode()
        try:
            lc.stop()
        except Exception:
            pass
    print(f"[capture] done. {ep} episodes -> {out}")
    print(
        f"  convert (ACT venv): ~/act-okura/.venv_act/bin/python scripts/okra_lerobot_writer.py --raw {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
