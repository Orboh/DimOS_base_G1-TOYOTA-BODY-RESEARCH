#!/usr/bin/env python
"""Live version of smoke_gopro.py: continuously show [ raw capture | live 224 preprocessed |
training frame ] in one window so you can AIM the GoPro / check the HDMI signal in real time.

Why this exists: an HDMI capture card (Elgato HD60 X = /dev/video6 on this laptop) is a
one-way UVC video source -- there is NO control channel, so nothing on the PC can drive the
GoPro. Operate the GoPro on its own touchscreen and use this window as the viewfinder.
A black window with black%=100 means no HDMI signal (GoPro off / cable / HDMI-out disabled).

Run (umi env):
  conda run -n umi python oda/umi_diffusion/preview_gopro.py    # --cam-device defaults to the by-id Elgato path
Keys: q or ESC = quit, s = save current pair to gopro_vs_train.png + gopro_live_224.png
"""
import os
import sys

import click
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# reuse the exact training-matched preprocessing (also does the umi sys.path setup)
from smoke_gopro import GOPRO_DEV, HERE, preprocess  # noqa: E402


@click.command()
@click.option("--cam-device", default=GOPRO_DEV,
              help="default = Elgato HD60 X by-id path (/dev/videoN numbers shuffle on replug)")
@click.option("--cap-w", default=1920, type=int)
@click.option("--cap-h", default=1080, type=int)
@click.option("--train-frame", default=os.path.join(HERE, "train_frame_00000.png"))
@click.option("--fisheye/--no-fisheye", default=False)
@click.option("--camera-intrinsics",
              default=os.path.expanduser(
                  "~/umi/universal_manipulation_interface/example/calibration/gopro_intrinsics_2_7k.json"))
@click.option("--sim-fov", default=None, type=float)
@click.option("--no-mirror", is_flag=True, default=False)
def main(cam_device, cap_w, cap_h, train_frame, fisheye, camera_intrinsics, sim_fov, no_mirror):
    fisheye_converter = None
    if fisheye:
        import json
        from umi.common.cv_util import parse_fisheye_intrinsics, FisheyeRectConverter
        assert sim_fov is not None, "--fisheye requires --sim-fov"
        intr = parse_fisheye_intrinsics(json.load(open(camera_intrinsics)))
        fisheye_converter = FisheyeRectConverter(**intr, out_size=(224, 224), out_fov=sim_fov)

    cap = cv2.VideoCapture(cam_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_h)
    if not cap.isOpened():
        print(f"ERROR: cannot open {cam_device}. Check `v4l2-ctl --list-devices`.")
        sys.exit(1)

    train_bgr = cv2.imread(train_frame)
    if train_bgr is None:
        print(f"WARN: {train_frame} not found; run inspect_dataset_frames.py first.")
        train_bgr = np.zeros((224, 224, 3), np.uint8)
    train_bgr = cv2.resize(train_bgr, (224, 224))

    print("q/ESC=quit  s=save.  black%=100 -> no HDMI signal (training frames were ~21%).")
    win = "GoPro live | preprocessed 224 | training"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        ok, bgr = cap.read()
        if not ok:
            print("WARN: frame read failed; retrying...")
            continue

        live_rgb = preprocess(bgr, fisheye_converter, no_mirror)
        live_bgr = live_rgb[..., ::-1]
        black = (live_rgb.sum(-1) == 0).mean() * 100

        raw_small = cv2.resize(bgr, (398, 224))  # keep 16:9, same height as the 224 tiles
        gap = np.full((224, 8, 3), 255, np.uint8)
        combo = np.concatenate([raw_small, gap, live_bgr, gap, train_bgr], axis=1)
        combo = cv2.resize(combo, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_NEAREST)

        signal = "NO HDMI SIGNAL" if black > 99.0 else f"black={black:.1f}% (train ~21%)"
        color = (0, 0, 255) if black > 99.0 else (0, 220, 0)
        cv2.putText(combo, signal, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow(win, combo)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            pair = np.concatenate([live_bgr, gap, train_bgr], axis=1)
            pair = cv2.resize(pair, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(HERE, "gopro_vs_train.png"), pair)
            cv2.imwrite(os.path.join(HERE, "gopro_live_224.png"), live_bgr)
            print(f"saved gopro_vs_train.png / gopro_live_224.png  ({signal})")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
