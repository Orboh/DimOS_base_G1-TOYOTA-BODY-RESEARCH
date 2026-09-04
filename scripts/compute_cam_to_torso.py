#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""定規実測値 → OKRA_CAM_TO_TORSO 用の "x,y,z,qx,qy,qz,qw" 文字列に変換する。

torso 原点(G1 waist_pitch 出力、背骨付け根あたり)から ZED-M レンズ中心までを
定規で測った並進[mm]と、カメラの傾きを角度計/目視で測った回転[度]を入力する。

⚠️ 軸の向きは ZED SDK 生の光学フレーム(x=右,y=下,z=前)でも標準ROSボディ座標
(x=前,y=左,z=上)でもなく、収穫パイプライン(detect_yolo.py の default_pixel_to_base)
が使う独自の座標系:
    x = 左右（+はカメラから見て右）
    y = 奥行き（+はカメラの視線方向 = 前方）
    z = 高さ（+は上）
回転(roll/pitch/yaw)もこの x,y,z 軸まわり(XYZ順、euler_to_quaternion と同じ規約)。
例えばレンズが下を向いている(見下ろす)なら pitch は正の値。

使い方:
    python scripts/compute_cam_to_torso.py \
        --x_mm 60 --y_mm 20 --z_mm 430 \
        --roll_deg 0 --pitch_deg 45 --yaw_deg 0

出力された文字列をそのまま使う:
    OKRA_CAM_TO_TORSO="$(python scripts/compute_cam_to_torso.py --x_mm ... )" \
        dimos run unitree-g1-okra-harvest-ik
"""

from __future__ import annotations

import argparse

from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.utils.transform_utils import euler_to_quaternion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--x_mm", type=float, required=True, help="torso原点→カメラの左右方向オフセット [mm]（+右）")
    parser.add_argument("--y_mm", type=float, required=True, help="torso原点→カメラの奥行き方向オフセット [mm]（+前方）")
    parser.add_argument("--z_mm", type=float, required=True, help="torso原点→カメラの高さ方向オフセット [mm]（+上）")
    parser.add_argument("--roll_deg", type=float, default=0.0, help="カメラのroll [度]（x軸まわり）")
    parser.add_argument("--pitch_deg", type=float, default=0.0, help="カメラのpitch [度]（y軸まわり、下向きが正）")
    parser.add_argument("--yaw_deg", type=float, default=0.0, help="カメラのyaw [度]（z軸まわり）")
    args = parser.parse_args()

    x_m = args.x_mm / 1000.0
    y_m = args.y_mm / 1000.0
    z_m = args.z_mm / 1000.0

    quat = euler_to_quaternion(
        Vector3(args.roll_deg, args.pitch_deg, args.yaw_deg), degrees=True
    )

    result = f"{x_m},{y_m},{z_m},{quat.x},{quat.y},{quat.z},{quat.w}"

    print(f"# translation [m]: x={x_m:.4f} y={y_m:.4f} z={z_m:.4f}", flush=True)
    print(
        f"# rotation: roll={args.roll_deg}° pitch={args.pitch_deg}° yaw={args.yaw_deg}°"
        f" -> quat(x,y,z,w)=({quat.x:.4f},{quat.y:.4f},{quat.z:.4f},{quat.w:.4f})",
        flush=True,
    )
    print(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
