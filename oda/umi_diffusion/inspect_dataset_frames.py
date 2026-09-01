#!/usr/bin/env python
"""Dump a few camera0_rgb frames from the training replay buffer (umi env).

The zarr's camera0_rgb IS exactly the preprocessed 224x224 image the policy trained on
— so it is the ground-truth target the live GoPro preprocessing must reproduce (Step 2).
Saves PNGs we can eyeball to determine: fisheye-rectified vs raw fisheye, black
gripper/mirror mask present?, mirrored?, RGB vs BGR.

Run: conda run -n umi python oda/umi_diffusion/inspect_dataset_frames.py
"""
import os
import sys

sys.path.append(os.path.expanduser("~/umi/universal_manipulation_interface"))

import numpy as np
import zarr
# UMI stores camera0_rgb JPEG-XL compressed -> register the numcodecs codec first.
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

register_codecs()

ZARR = os.path.expanduser("~/umi/okra_20260723_ishimaru/dataset.zarr.zip")
OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    z = zarr.open(ZARR, "r")

    def walk(g, prefix=""):
        keys = list(g.array_keys()) if hasattr(g, "array_keys") else []
        for k in keys:
            a = g[k]
            print(f"  {prefix}{k}: shape={a.shape} dtype={a.dtype}")
        if hasattr(g, "group_keys"):
            for k in g.group_keys():
                print(f"  [{prefix}{k}/]")
                walk(g[k], prefix + k + "/")

    print("=== zarr tree ===")
    walk(z)

    data = z["data"] if "data" in z else z
    cam = data["camera0_rgb"]
    print(f"\ncamera0_rgb: shape={cam.shape} dtype={cam.dtype}")
    n = cam.shape[0]
    idxs = [0, n // 2, n - 1]
    import cv2

    for i in idxs:
        img = np.asarray(cam[i])
        if img.dtype != np.uint8:  # float [0,1] -> uint8
            img = (img.clip(0, 1) * 255).astype(np.uint8)
        # zarr stores RGB (UMI obs). Save as-is (RGB) and note cv2 expects BGR:
        bgr = img[..., ::-1]
        p = os.path.join(OUT, f"train_frame_{i:05d}.png")
        cv2.imwrite(p, bgr)
        print(f"  saved {p}  (min={img.min()} max={img.max()} "
              f"mean={img.mean():.1f}  black-pixels={(img.sum(-1)==0).mean()*100:.1f}%)")

    # gripper_width sanity (should be the dead 1e-4 constant)
    if "robot0_gripper_width" in data.array_keys():
        gw = np.asarray(data["robot0_gripper_width"][:])
        print(f"\nrobot0_gripper_width: min={gw.min():.5f} max={gw.max():.5f} "
              f"std={gw.std():.6f}  (confirms dead channel if ~0)")


if __name__ == "__main__":
    main()
