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

"""Offline converter: raw kinesthetic capture -> LeRobot 0.4.1 dataset (wrist-only, 7-DoF).

Runs in the ACT venv (lerobot 0.4.1). The kinesthetic collection is two-stage to
keep venvs clean (LCM/cyclonedds live in the dimos venv; lerobot lives here):
  1. LIVE (dimos venv, in the harvest pipeline): dump per-frame raw data to disk —
     right-arm measured q (7) + the wrist UVC frame (cam_right_wrist) — one dir/episode.
  2. OFFLINE (this script, ACT venv): read the raw dirs and build a LeRobot dataset.

Dataset schema (matches the deployed tree-right model MINUS head cam & gripper):
  observation.state            float32 (7,)   right arm q (motors 22-28)
  action                       float32 (7,)   next-frame right arm q (kinesthetic)
  observation.images.cam_right_wrist  video (480,640,3)  wrist UVC only
(no cam_high: the head sees the teacher's arm during kinesthetic -> OOD; head is
used only for the IK click, not as an ACT input. no gripper dim: close is handled
outside ACT.)

Raw episode layout (written by the live capturer):
  <raw>/episode_000/
      q.npy            float32 [T,7]  measured right-arm q per frame
      0000.jpg ...     wrist frames (BGR jpg), one per row of q.npy
      meta.json        {"fps":30, "task":"..."}   (optional; defaults applied)

Run:
  selftest (no robot, fake data -> write -> reload -> verify):
    ~/act-okura/.venv_act/bin/python scripts/okra_lerobot_writer.py --selftest
  convert real captures (all under one base dir ~/okra_collect):
    ~/act-okura/.venv_act/bin/python scripts/okra_lerobot_writer.py \
        --raw ~/okra_collect/raw --repo-id sotata/okura-kinesthetic-wrist-7d --fps 30
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np

STATE_DIM = 7
IMG_H, IMG_W = 480, 640
WRIST_KEY = "observation.images.cam_right_wrist"
DEFAULT_TASK = "pick the okra"
ROBOT_TYPE = "Unitree_G1_Dex1"


def features(fps_use_video: bool = True) -> dict:
    img_dtype = "video" if fps_use_video else "image"
    return {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,),
                              "names": [f"r_arm_{i}" for i in range(STATE_DIM)]},
        "action": {"dtype": "float32", "shape": (STATE_DIM,),
                   "names": [f"r_arm_{i}" for i in range(STATE_DIM)]},
        WRIST_KEY: {"dtype": img_dtype, "shape": (IMG_H, IMG_W, 3),
                    "names": ["height", "width", "channel"]},
    }


def _build(raw_episodes, repo_id, root, fps, use_videos, task):
    """raw_episodes: list of (q[T,7] float32, frames[T] HxWx3 uint8 RGB)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if Path(root).exists():
        shutil.rmtree(root)
    ds = LeRobotDataset.create(
        repo_id=repo_id, fps=fps, features=features(use_videos), root=root,
        robot_type=ROBOT_TYPE, use_videos=use_videos,
    )
    for q, frames in raw_episodes:
        q = np.asarray(q, dtype=np.float32)
        T = len(q)
        for t in range(T):
            # action = next-frame measured q (last frame repeats); kinesthetic label.
            a = q[t + 1] if t + 1 < T else q[t]
            ds.add_frame({
                "observation.state": q[t],
                "action": a.astype(np.float32),
                WRIST_KEY: frames[t],
                "task": task,
            })
        ds.save_episode()
    return ds


def _read_raw(raw_dir: Path):
    import cv2
    eps = []
    for ep in sorted(p for p in raw_dir.iterdir() if p.is_dir() and p.name.startswith("episode_")):
        q = np.load(ep / "q.npy").astype(np.float32)
        jpgs = sorted(ep.glob("[0-9]*.jpg"))
        if len(jpgs) != len(q):
            print(f"  [skip] {ep.name}: {len(jpgs)} jpgs != {len(q)} q rows")
            continue
        frames = [cv2.cvtColor(cv2.imread(str(j)), cv2.COLOR_BGR2RGB) for j in jpgs]
        eps.append((q, frames))
        print(f"  {ep.name}: {len(q)} frames")
    return eps


def _selftest() -> int:
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="okra_lerobot_selftest_"))
    rng_q = np.cumsum(np.full((20, STATE_DIM), 0.01, dtype=np.float32), axis=0)  # smooth fake traj
    frames = [np.full((IMG_H, IMG_W, 3), (t * 10) % 255, dtype=np.uint8) for t in range(20)]
    print(f"[selftest] writing 1 fake episode (20 frames, 7-dim, wrist-only) to {root}")
    _build([(rng_q, frames)], repo_id="local/okra_selftest", root=str(root / "ds"),
           fps=30, use_videos=True, task=DEFAULT_TASK)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id="local/okra_selftest", root=str(root / "ds"))
    f = ds.meta.features
    ok = (
        tuple(f["observation.state"]["shape"]) == (STATE_DIM,)
        and tuple(f["action"]["shape"]) == (STATE_DIM,)
        and WRIST_KEY in f
        and "observation.images.cam_high" not in f
        and ds.num_frames == 20
        and ds.num_episodes == 1
    )
    frame0 = ds[0]
    print(f"[selftest] reloaded: num_frames={ds.num_frames} num_episodes={ds.num_episodes}")
    print(f"[selftest] state shape={tuple(frame0['observation.state'].shape)} "
          f"action shape={tuple(frame0['action'].shape)} "
          f"wrist shape={tuple(frame0[WRIST_KEY].shape)}")
    print(f"[selftest] features: {list(f.keys())}")
    # stats present (normalization depends on these)
    has_stats = bool(ds.meta.stats) and "observation.state" in ds.meta.stats
    print(f"[selftest] meta.stats has observation.state: {has_stats}")
    shutil.rmtree(root, ignore_errors=True)
    ok = ok and has_stats
    print(f"[selftest] {'OK ✅ — wrist-only 7-dim lerobot 0.4.1 dataset writes + reloads' if ok else 'FAIL ❌'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--raw", type=str, help="raw capture dir (episode_* subdirs)")
    ap.add_argument("--repo-id", type=str, default="sotata/okura-kinesthetic-wrist-7d")
    ap.add_argument("--root", type=str, default=None, help="output dataset dir (default ~/.cache/...)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--task", type=str, default=DEFAULT_TASK)
    ap.add_argument("--no-videos", action="store_true", help="store frames as images not video")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    if not args.raw:
        ap.error("give --raw <dir> or --selftest")
    eps = _read_raw(Path(args.raw))
    if not eps:
        print("no episodes found")
        return 1
    root = args.root or str(Path.home() / "okra_collect" / "lerobot" / args.repo_id.replace("/", "_"))
    _build(eps, repo_id=args.repo_id, root=root, fps=args.fps,
           use_videos=not args.no_videos, task=args.task)
    print(f"[done] wrote {len(eps)} episodes -> {root}")
    print(f"       upload: huggingface-cli upload {args.repo_id} {root} --repo-type dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
