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

"""GraspGenコンテナ内で実行: object点群(npy)→6-DoF把持姿勢を推論しJSON保存する。

DimOS `GraspGenModule._run_inference` と同じ呼び出し（gripper_type=franka_panda を既定近似採用、
[[SS-04-粗アプローチIK]] 差し替え検証用）。visualize_grasps.py の VISUALIZATION_FILE 形式で保存する。

実行（コンテナ内, ホストからは docker run 経由）:
  python graspgen_infer.py --in /data/points_cam.npy --out /data/grasp_visualization.json \
      --gripper franka_panda --num-grasps 400 --topk 100
"""

import argparse
import json
import os
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--gripper", default="franka_panda")
    ap.add_argument("--num-grasps", type=int, default=400)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--grasp-threshold", type=float, default=-1.0)
    args = ap.parse_args()

    graspgen_path = os.environ.get("GRASPGEN_PATH", "/app/GraspGen")
    if graspgen_path not in sys.path:
        sys.path.insert(0, graspgen_path)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
    from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal
    import torch

    points = np.load(args.in_path).astype(np.float64)
    print(f"[infer] input points: {points.shape}", flush=True)
    if len(points) > 100:
        pc_filtered, _ = point_cloud_outlier_removal(torch.from_numpy(points))
        points = pc_filtered.numpy()
        print(f"[infer] after outlier removal: {points.shape}", flush=True)

    config_name = f"graspgen_{args.gripper}.yml"
    cfg_path = None
    for subdir in ("GraspGenModels/checkpoints", "checkpoints"):
        p = os.path.join(graspgen_path, subdir, config_name)
        if os.path.exists(p):
            cfg_path = p
            break
    if cfg_path is None:
        raise FileNotFoundError(f"{config_name} not found under {graspgen_path}")

    cfg = load_grasp_cfg(cfg_path)
    sampler = GraspGenSampler(cfg)

    grasps, scores = GraspGenSampler.run_inference(
        points,
        sampler,
        grasp_threshold=args.grasp_threshold,
        num_grasps=args.num_grasps,
        topk_num_grasps=args.topk,
        remove_outliers=False,
    )
    print(f"[infer] grasps generated: {len(grasps)}", flush=True)
    if len(grasps) == 0:
        print("[infer] WARN: no grasps generated", flush=True)

    grasps_np = grasps.cpu().numpy() if hasattr(grasps, "cpu") else np.asarray(grasps)
    scores_np = scores.cpu().numpy() if hasattr(scores, "cpu") else np.asarray(scores)
    order = np.argsort(scores_np)[::-1]

    data = {
        "point_cloud": points.tolist(),
        "grasps": [grasps_np[i].tolist() for i in order],
        "scores": scores_np[order].tolist(),
        "gripper": args.gripper,
    }
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(data, f)
    print(f"[infer] saved -> {args.out_path}", flush=True)
    print("INFER_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
