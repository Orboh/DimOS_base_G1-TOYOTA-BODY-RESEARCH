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

"""Export human-VIEWABLE H.264 videos from the raw kinesthetic captures.

The LeRobot dataset stores wrist video as AV1 (lerobot v3 default), which many
players can't open. This makes per-episode H.264 mp4s from the RAW jpg frames
(~/okra_collect/raw/episode_*/), the source of truth. It does NOT touch the
LeRobot dataset (training stays on AV1). Output: ~/okra_collect/view/episode_NNN.mp4.

Run (ffmpeg required):
  python scripts/okra_export_view.py                       # all episodes in ~/okra_collect/raw
  python scripts/okra_export_view.py --raw <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=str, default=str(Path.home() / "okra_collect" / "raw"))
    ap.add_argument("--out", type=str, default=str(Path.home() / "okra_collect" / "view"))
    ap.add_argument("--fps", type=float, default=0.0, help="override fps (0 = read each episode's meta.json)")
    ap.add_argument("--gif", action="store_true", help="also write a downscaled .gif (plays in any image viewer/browser)")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("[export] ffmpeg not found — install it (sudo apt-get install -y ffmpeg).")
        return 2
    raw = Path(args.raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    eps = sorted(p for p in raw.iterdir() if p.is_dir() and p.name.startswith("episode_"))
    if not eps:
        print(f"[export] no episodes under {raw}")
        return 1
    n_ok = 0
    for ep in eps:
        jpgs = sorted(ep.glob("[0-9]*.jpg"))
        if not jpgs:
            print(f"  [skip] {ep.name}: no jpgs")
            continue
        fps = args.fps
        if fps <= 0:
            meta = ep / "meta.json"
            fps = float(json.loads(meta.read_text()).get("fps", 15.0)) if meta.exists() else 15.0
        dst = out / f"{ep.name}.mp4"
        # image2 needs a sequential pattern; raw frames are 0000.jpg, 0001.jpg, ...
        # +faststart moves the moov atom to the front so browsers/totem can play it.
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", str(ep / "%04d.jpg"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(dst),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  [FAIL] {ep.name}: ffmpeg rc={r.returncode}\n{r.stderr[-300:]}")
            continue
        msg = f"  [ok] {ep.name}: {len(jpgs)} frames @ {fps:g}fps -> {dst}"
        if args.gif:
            gif = out / f"{ep.name}.gif"
            rg = subprocess.run(
                ["ffmpeg", "-y", "-i", str(dst), "-vf", "fps=10,scale=320:-1:flags=lanczos", str(gif)],
                capture_output=True, text=True,
            )
            msg += "  +gif" if rg.returncode == 0 else "  (gif failed)"
        print(msg)
        n_ok += 1
    print(f"[export] {n_ok}/{len(eps)} episodes -> {out}  (H.264, viewable)")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
