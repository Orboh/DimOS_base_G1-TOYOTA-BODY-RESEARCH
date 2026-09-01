"""Headless wrist-camera framing check: is the live view anything like the training set?

`preview_gopro.py` needs a focused GUI window, which is awkward while the operator is at
the robot moving the arm. This writes the same comparison to a PNG on a timer instead, so
you can reposition the hand and just re-open the file (or have someone else read it).

The framing question is not cosmetic. On 2026-08-25 the live wrist view was a grey wall
while every training frame was a white table with okra on it, and the policy's output on
the real image was indistinguishable from its output on a pure black image (cos 0.992) --
it was extracting nothing. Matching the training framing is a precondition for any
meaningful evaluation of the policy.

    conda run -n umi --no-capture-output python watch_gopro.py
    conda run -n umi --no-capture-output python watch_gopro.py --seconds 300 --every 0.5
"""

import os
import sys
import time

import click
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smoke_gopro import GOPRO_DEV, HERE, preprocess  # noqa: E402

_TRAIN = ["train_frame_00000.png", "train_frame_05705.png", "train_frame_11409.png"]


def _load_train():
    out = []
    for name in _TRAIN:
        img = cv2.imread(os.path.join(HERE, name))
        if img is not None:
            out.append((name, cv2.resize(img, (224, 224))))
    return out


def _similarity(live_bgr, train_bgr):
    """Zero-mean normalised correlation on greyscale, in [-1, 1].

    Crude on purpose: it answers "is this roughly the same kind of scene", not "is this the
    same frame". The black border the fisheye preprocessing leaves is identical in both, so
    it inflates the score -- read the number as a relative gauge while you reposition, and
    trust the picture for the absolute judgement.
    """
    a = cv2.cvtColor(live_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64).ravel()
    b = cv2.cvtColor(train_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if d == 0 else float(a @ b / d)


@click.command()
@click.option("--cam-device", default=GOPRO_DEV)
@click.option("--every", default=1.0, type=float, help="seconds between saves")
@click.option("--seconds", default=600.0, type=float, help="stop after this long")
@click.option("--out", default=os.path.join(HERE, "gopro_vs_train.png"))
def main(cam_device, every, seconds, out):
    cap = cv2.VideoCapture(cam_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        print(f"ERROR: cannot open {cam_device}. Is the policy server still holding it?")
        sys.exit(1)

    train = _load_train()
    if not train:
        print("ERROR: no train_frame_*.png next to this script.")
        sys.exit(1)

    print(f"writing {out} every {every:.1f}s for {seconds:.0f}s. Ctrl-C to stop.")
    print("move the hand until the live tile looks like the training tiles.\n")
    gap = np.full((224, 10, 3), 255, np.uint8)
    t_end = time.time() + seconds
    try:
        while time.time() < t_end:
            ok, bgr = cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            live = preprocess(bgr, None, False)[..., ::-1]  # -> BGR for cv2
            black = (live.max(axis=2) <= 5).mean() * 100
            sims = [(n, _similarity(live, t)) for n, t in train]
            best = max(sims, key=lambda kv: kv[1])

            tiles = [live]
            for _, t in train:
                tiles += [gap, t]
            row = np.concatenate(tiles, axis=1)
            row = cv2.resize(row, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
            label = (f"live | training x{len(train)}   black={black:.1f}% (train ~21%)   "
                     f"best match {best[0].replace('train_frame_', '')} r={best[1]:+.3f}")
            colour = (0, 0, 255) if black > 99.0 else (0, 180, 0)
            cv2.putText(row, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            cv2.imwrite(out, row)
            print(f"  black={black:5.1f}%  " + "  ".join(
                f"{n.replace('train_frame_', '').replace('.png', '')}:{s:+.3f}" for n, s in sims),
                flush=True)
            time.sleep(every)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print(f"\nstopped. last comparison is in {out}")


if __name__ == "__main__":
    main()
