#!/usr/bin/env python
"""Step 2: capture a live GoPro frame, apply the training preprocessing, and lay it
beside a real training frame so you can eyeball the visual-domain match.

Ground truth (extracted by inspect_dataset_frames.py from the training replay buffer):
the policy trained on RAW GoPro FISHEYE images (barrel distortion, NOT rectified),
resized to 224x224 RGB, with draw_predefined_mask(mirror=False, gripper=True, finger=False)
painting ~21% of the frame black. So the DEFAULTS here (no --fisheye, no --no-mirror)
already match training — only add flags if the overlay says otherwise.

Run (umi env):
  conda run -n umi python oda/umi_diffusion/smoke_gopro.py      # --cam-device defaults to the by-id Elgato path
Writes gopro_vs_train.png = [ live preprocessed | training frame ].  Compare:
  * fisheye curvature (straight office/plant edges should bow the SAME way),
  * the black mask region should sit over the ACTUAL gripper in the live frame
    (if not, the GoPro mount differs from data-collection -> visual-domain shift),
  * overall brightness / colour / RGB-vs-BGR (skin/plant should look natural).
"""
import os
import sys

sys.path.append(os.path.expanduser("~/umi/universal_manipulation_interface"))

import click
import cv2
import numpy as np
from diffusion_policy.common.cv2_util import get_image_transform
from umi.common.cv_util import draw_predefined_mask

HERE = os.path.dirname(os.path.abspath(__file__))

# GoPro HERO9 -> Media Mod micro-HDMI -> Elgato HD60 X capture card.  Use the by-id symlink,
# NOT /dev/videoN: the numbers get reassigned on replug (ZED-M and the Elgato swapped 4<->6 on
# 2026-07-29, which silently fed ZED-M frames into this smoke test).
GOPRO_DEV = "/dev/v4l/by-id/usb-Elgato_Elgato_HD60_X_A00XB3442072PE-video-index0"


def preprocess(bgr, fisheye_converter, no_mirror, out_res=(224, 224)):
    """Identical to umi_policy_server.build_preproc / umi_env.py tf(). Returns RGB uint8."""
    if fisheye_converter is None:
        tf = get_image_transform(input_res=(bgr.shape[1], bgr.shape[0]),
                                 output_res=out_res, bgr_to_rgb=True)
        img = np.ascontiguousarray(tf(bgr))
        img = draw_predefined_mask(img, color=(0, 0, 0),
                                   mirror=no_mirror, gripper=True, finger=False, use_aa=True)
    else:
        img = fisheye_converter.forward(bgr)[..., ::-1]
    return img.astype(np.uint8)


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
@click.option("--out", default=os.path.join(HERE, "gopro_vs_train.png"))
def main(cam_device, cap_w, cap_h, train_frame, fisheye, camera_intrinsics, sim_fov, no_mirror, out):
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
        print(f"ERROR: cannot open {cam_device}. Check `ls -l /dev/video*` / `v4l2-ctl --list-devices`.")
        sys.exit(1)
    for _ in range(10):  # let auto-exposure settle
        ok, bgr = cap.read()
    cap.release()
    if not ok:
        print("ERROR: no frame captured.")
        sys.exit(1)
    print(f"captured raw frame: {bgr.shape} from {cam_device}")

    live_rgb = preprocess(bgr, fisheye_converter, no_mirror)     # (224,224,3) RGB
    black = (live_rgb.sum(-1) == 0).mean() * 100
    print(f"live preprocessed: 224x224 RGB, black-pixels={black:.1f}% "
          f"(training frames were ~21%)")

    train_bgr = cv2.imread(train_frame)  # saved BGR by inspect_dataset_frames
    if train_bgr is None:
        print(f"WARN: training frame {train_frame} not found; run inspect_dataset_frames.py first.")
        train_bgr = np.zeros((224, 224, 3), np.uint8)
    train_bgr = cv2.resize(train_bgr, (224, 224))

    # side-by-side (write BGR): [ live | train ]
    live_bgr = live_rgb[..., ::-1]
    gap = np.full((224, 8, 3), 255, np.uint8)
    combo = np.concatenate([live_bgr, gap, train_bgr], axis=1)
    combo = cv2.resize(combo, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out, combo)
    cv2.imwrite(os.path.join(HERE, "gopro_live_224.png"), live_bgr)
    print(f"wrote {out}  (LEFT=live preprocessed, RIGHT=training frame). Open it and compare.")


if __name__ == "__main__":
    main()
