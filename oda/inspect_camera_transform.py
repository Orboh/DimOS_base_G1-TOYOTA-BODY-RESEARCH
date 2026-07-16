#!/usr/bin/env python3
"""カメラ座標 -> ロボット(torso)座標 変換の中身をプログラムで調べるスクリプト。

IkReachBridge が実際に使っている変換 (ik_reach_bridge.py の
_default_torso_from_optical()) を本物のコードから import して、
- 変換の構成要素(URDF取付値 + 光学フレーム回転)
- 合成された 4x4 同次変換行列
- サンプル点がどう写るか
を数値で表示する。ZED 版で「何を差し替えれば良いか」を最後に示す。

Run:
    cd ~/Toyota-auto-body-PoC/DimOS_oda
    CYCLONEDDS_HOME=~/cyclonedds-noshm LD_LIBRARY_PATH=~/cyclonedds-noshm/lib \
        .venv/bin/python oda/inspect_camera_transform.py
"""

from __future__ import annotations

import numpy as np
import pinocchio

# 本物の実装から import(コピーではない = 真値)
from dimos.robot.unitree.g1.act.ik_reach_bridge import (
    _D435_RPY,
    _D435_XYZ,
    _OPTICAL_WXYZ,
    _default_torso_from_optical,
)

np.set_printoptions(precision=4, suppress=True)


def se3_to_homogeneous(T: pinocchio.SE3) -> np.ndarray:
    H = np.eye(4)
    H[:3, :3] = T.rotation
    H[:3, 3] = T.translation
    return H


def main() -> None:
    print("=" * 72)
    print("[1] 構成要素")
    print("=" * 72)
    print(f"URDF d435_joint (torso_link -> d435_link):")
    print(f"  xyz = {_D435_XYZ}   (前+X, 左+Y, 上+Z [m])")
    print(f"  rpy = {_D435_RPY}   (pitch {np.degrees(_D435_RPY[1]):.1f}° 下向き)")
    print(f"REP-103 光学フレーム回転 (d435_link -> color_optical) wxyz = {_OPTICAL_WXYZ}")
    print()
    print("光学(optical)フレームの軸の意味 [ROS REP-103]:")
    print("  +X = 画像の右方向, +Y = 画像の下方向, +Z = カメラの視線方向(奥)")
    print("ロボット(torso_link)フレームの軸の意味:")
    print("  +X = ロボットの前方, +Y = 左, +Z = 上")

    # --- 個別の SE3 ---
    T_torso_d435 = pinocchio.SE3(
        pinocchio.rpy.rpyToMatrix(*_D435_RPY), _D435_XYZ.copy()
    )
    r_opt = pinocchio.Quaternion(*_OPTICAL_WXYZ).toRotationMatrix()
    T_d435_optical = pinocchio.SE3(r_opt, np.zeros(3))

    print()
    print("=" * 72)
    print("[2] 合成: T_torso<-optical = T_torso<-d435 * T_d435<-optical")
    print("=" * 72)
    T = _default_torso_from_optical()  # ik_reach_bridge が実際に使う SE3
    # 検算: 手で合成したものと一致するか
    T_manual = T_torso_d435 * T_d435_optical
    assert np.allclose(se3_to_homogeneous(T), se3_to_homogeneous(T_manual))
    print("4x4 同次変換行列 (p_torso = R @ p_optical + t):")
    print(se3_to_homogeneous(T))
    print()
    print("変換式(成分表示):")
    R, t = T.rotation, T.translation
    for i, ax in enumerate("xyz"):
        terms = " + ".join(
            f"({R[i, j]:+.4f})*{c}_cam" for j, c in enumerate("xyz")
        )
        print(f"  {ax}_torso = {terms} {t[i]:+.4f}")

    print()
    print("=" * 72)
    print("[3] サンプル点で意味を確認")
    print("=" * 72)
    samples = {
        "カメラ原点 (0,0,0)": np.array([0.0, 0.0, 0.0]),
        "カメラ正面 0.5m (0,0,0.5)": np.array([0.0, 0.0, 0.5]),
        "カメラ正面 1.0m (0,0,1.0)": np.array([0.0, 0.0, 1.0]),
        "画像右に 0.2m (0.2,0,0.5)": np.array([0.2, 0.0, 0.5]),
        "画像下に 0.2m (0,0.2,0.5)": np.array([0.0, 0.2, 0.5]),
    }
    for name, p_cam in samples.items():
        p_torso = np.asarray(T.act(p_cam))
        print(f"  {name:32s} -> torso {p_torso}")
    print()
    print("読み方: カメラ原点の写り先 = カメラの取付位置そのもの。")
    print("正面の点が前方かつ下に写る = カメラが約47.6°下を向いている効果。")

    print()
    print("=" * 72)
    print("[4] ZED でやる場合に差し替えるもの")
    print("=" * 72)
    print("同じ構造:  T_torso<-zed_optical = T_torso<-zed_mount * T_mount<-optical")
    print(" - T_mount<-optical : REP-103 の光学回転。カメラ機種に依らず同じ(流用可)")
    print(" - T_torso<-zed_mount : ZED の取付位置・角度。URDF に存在しない = 自分で")
    print("   決める必要がある(実測 or ハンドアイキャリブレーション)。")
    print()
    print("例: 「胸に取付、前 8cm・上 25cm、下向き 30°」と仮置きした場合:")
    zed_xyz = np.array([0.08, 0.0, 0.25])
    zed_rpy = np.array([0.0, np.radians(30.0), 0.0])
    T_torso_zed = pinocchio.SE3(pinocchio.rpy.rpyToMatrix(*zed_rpy), zed_xyz)
    T_zed = T_torso_zed * T_d435_optical  # 光学回転は同じものを流用
    print(se3_to_homogeneous(T_zed))
    p = np.array([0.0, 0.0, 0.5])
    print(f"  ZED正面0.5mの点 -> torso {np.asarray(T_zed.act(p))}")
    print()
    print("この T_torso<-zed_mount の数値を正しく求めることが、ZED 版 IK の")
    print("残された唯一の未知数(あとの経路は全てカメラ非依存)。")


if __name__ == "__main__":
    main()
