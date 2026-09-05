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

"""visualize_grasps.py のヘッドレス版: Open3D対話ウィンドウの代わりに matplotlib で PNG保存する。

grasp_visualization.json（point_cloud + grasps + scores）を読み、上位グリッパ姿勢を
線分で描画した3視点の静止画を保存する。GUIが無い環境（このsim検証）用。
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# GraspGen の実際の規約（NVlabs/GraspGen config/grippers/franka_panda.yaml）:
# 原点=グリッパ root/base（手首寄りの取付フレーム）、接近方向=+Z、指先は root から
# +Z 方向に約 depth 先。DimOS 同梱 visualize_grasps.py の GRIPPER_WIDTH/FINGER_LENGTH/
# PALM_DEPTH は原点が指先近傍にある想定の汎用イラスト値で、franka_panda 実寸とズレて
# おり、グリッパー形状が接触点から depth ぶん(約10cm)手前にズレて描画される。
GRIPPER_WIDTH = 0.10537486  # franka_panda width
GRIPPER_DEPTH = 0.10527314  # franka_panda depth: root→指先


def gripper_lines(transform: np.ndarray):
    w = GRIPPER_WIDTH / 2.0
    d = GRIPPER_DEPTH
    root = np.array([0.0, 0.0, 0.0])
    hand = np.array([0.0, 0.0, d])  # root→hand(+Z, 指の付け根)
    l_base = np.array([-w, 0.0, d])
    r_base = np.array([w, 0.0, d])
    l_tip = np.array([-w, 0.0, d / 2.0])  # 指先（物体との接触点付近）
    r_tip = np.array([w, 0.0, d / 2.0])
    pts = np.vstack([root, hand, l_base, r_base, l_tip, r_tip])
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    world = (transform @ pts_h.T).T[:, :3]
    segs = [
        (world[0], world[1]),
        (world[1], world[2]),
        (world[1], world[3]),
        (world[2], world[4]),
        (world[3], world[5]),
    ]
    return segs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    with open(args.in_path) as f:
        data = json.load(f)
    pts = np.array(data["point_cloud"])
    grasps = [np.array(g).reshape(4, 4) for g in data["grasps"][: args.topk]]
    scores = data.get("scores", [])[: args.topk]

    fig = plt.figure(figsize=(15, 5))
    views = [("front (X-Z)", 0, -90), ("side (Y-Z)", 0, 0), ("top (X-Y)", 90, -90)]
    for i, (name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3, c="teal", alpha=0.5)
        for gi, g in enumerate(grasps):
            segs = gripper_lines(g)
            t = gi / max(len(grasps) - 1, 1)
            color = (min(1.0, 2 * t), max(0.0, 1.0 - t), 0.0)
            for p0, p1 in segs:
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=color, linewidth=1.2)
        ax.set_title(name)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=elev, azim=azim)
        # 等縮尺（軸ごとに範囲がバラバラだと歪んで見えるため）
        allp = np.vstack(
            [pts] + [np.array(s).reshape(-1, 3) for g in grasps for s in gripper_lines(g)]
        )
        c = allp.mean(axis=0)
        r = (allp.max(axis=0) - allp.min(axis=0)).max() / 2.0 + 1e-3
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
    fig.suptitle(
        f"{args.title}  (top{len(grasps)}/{len(data['grasps'])}, "
        f"score {min(scores):.2f}-{max(scores):.2f})"
        if scores
        else args.title
    )
    fig.tight_layout()
    fig.savefig(args.out_path, dpi=130)
    print(f"saved -> {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
