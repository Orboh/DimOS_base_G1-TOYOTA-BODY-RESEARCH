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

"""CAD の実測値から ``OKRA_CAM_TO_TORSO`` の7値文字列を作る（[[SS-04-粗アプローチIK]]）。

``harvest_module._parse_cam_to_torso`` が受け取る "x,y,z,qx,qy,qz,qw" は
**torso <- 光学フレーム** の剛体変換。光学フレームは REP-103（x=右, y=下, z=前）。

CAD からは以下の2つを読み取って与える:
  1. torso_link 原点 → ZED **左目の光学中心** のベクトル（torso 座標 x=前,y=左,z=上）
     ※ 筐体の取り付け面ではなく左目光学中心。ZED-M データシートのオフセットを足すこと。
  2. カメラの光学3軸（右/下/前）を torso 座標で表したベクトル

使い方（--right/--down/--forward は torso 座標での単位ベクトル）:

    python cam_to_torso_from_cad.py \
        --xyz 0.1110 0.0250 0.2585 \
        --right 0 -1 0 --down 0 0 -1 --forward 1 0 0

正しく取り付いていれば forward≈+X(前), right≈-Y(右), down≈-Z(下) になり、
出力は現行既定 (-0.49475, 0.49475, -0.50520, 0.50520) に近い値になるはず。
"""

from __future__ import annotations

import argparse

import numpy as np

# 比較の基準＝現行の本番既定値（unitree_g1_okra_honban.py の OKRA_CAM_TO_TORSO）。
# 本番側を変えたらここも更新すること（でないと差分が古い値との比較になる）。
CURRENT = "0.1110,0.0250,0.2585,-0.49475,0.49475,-0.50520,0.50520"


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 回転行列 -> (qx,qy,qz,qw)。Shepperd の分岐法（数値的に安定）。"""
    m = np.asarray(R, dtype=float)
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw, qx, qy, qz = (
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw, qx, qy, qz = (
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        )
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw, qx, qy, qz = (
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        )
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw, qx, qy, qz = (
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        )
    q = np.array([qx, qy, qz, qw])
    return q / np.linalg.norm(q)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--xyz",
        nargs=3,
        type=float,
        required=True,
        metavar=("X", "Y", "Z"),
        help="torso原点→左目光学中心 [m] (x=前,y=左,z=上)",
    )
    p.add_argument(
        "--right", nargs=3, type=float, required=True, help="光学+x(画像の右) を torso 座標で"
    )
    p.add_argument(
        "--down", nargs=3, type=float, required=True, help="光学+y(画像の下) を torso 座標で"
    )
    p.add_argument(
        "--forward", nargs=3, type=float, required=True, help="光学+z(レンズ正面) を torso 座標で"
    )
    a = p.parse_args()

    cols = [np.array(v, dtype=float) for v in (a.right, a.down, a.forward)]
    for nm, v in zip(("right", "down", "forward"), cols, strict=True):
        n = np.linalg.norm(v)
        if n < 1e-9:
            raise SystemExit(f"--{nm} がゼロベクトルです")
        v /= n
    R = np.column_stack(cols)

    # 直交性チェック: CAD から読んだ軸が直交していないと回転行列にならない。
    err = np.abs(R.T @ R - np.eye(3)).max()
    print(
        f"直交性チェック: 最大誤差 {err:.2e}  {'OK' if err < 1e-3 else '← 3軸が直交していません。要確認'}"
    )
    if np.linalg.det(R) < 0:
        print("⚠️ 行列式が負です（左手系）。軸の向きを1つ取り違えている可能性があります")
    # 最近傍の正規直交行列へ丸める（CAD 実測の微小誤差を吸収）
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt

    q = rot_to_quat(R)
    t = np.array(a.xyz, dtype=float)
    new = ",".join(f"{v:.5f}" for v in [*t, *q])

    print("\n回転行列 R (torso <- optical):")
    for r in R:
        print("   [" + " ".join(f"{x:8.4f}" for x in r) + "]")
    print("\n各軸の行き先:")
    for nm, v in (
        ("optical x (右)", [1, 0, 0]),
        ("optical y (下)", [0, 1, 0]),
        ("optical z (前)", [0, 0, 1]),
    ):
        print(f"  {nm:16s} -> torso {np.round(R @ np.array(v, dtype=float), 4)}")

    print(f'\nOKRA_CAM_TO_TORSO="{new}"')

    # 現行値との差
    cur = np.array([float(x) for x in CURRENT.split(",")])
    dt_ = t - cur[:3]
    dR = quat_to_rot(cur[3:]).T @ R
    ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    print("\n--- 現行既定値との差 ---")
    print(
        f"  平行移動: {np.round(dt_ * 1000, 1)} mm   (大きさ {np.linalg.norm(dt_) * 1000:.1f} mm)"
    )
    print(f"  回転    : {ang:.3f} 度")
    for dist in (0.30, 0.45, 0.60):
        print(
            f"    → {dist:.2f} m 先での位置ずれ換算: "
            f"{(np.linalg.norm(dt_) + dist * np.tan(np.radians(ang))) * 1000:.1f} mm"
        )


if __name__ == "__main__":
    main()
