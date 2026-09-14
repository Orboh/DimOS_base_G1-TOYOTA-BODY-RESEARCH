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

"""``OKRA_CAM_TO_TORSO`` が指すカメラ位置を G1 モデル上に可視化する。

CAD から出した cam_to_torso が「胸のあたりを向いているか」を目視で確認するための
道具（[[SS-04-粗アプローチIK]]）。torso_link のフレーム軸と、そこから cam_to_torso で
置いたカメラの光学3軸（右=赤 / 下=緑 / 前=青）を描く。

    python oda/mujoco_sim/view_cam_frame.py                    # 既定値で PNG 出力
    python oda/mujoco_sim/view_cam_frame.py --cam "x,y,z,qx,qy,qz,qw"
    python oda/mujoco_sim/view_cam_frame.py --interactive      # ビューア（macOSは mjpython）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import PIL.Image

SCENE = Path(__file__).resolve().parent / "g1_okra_scene.xml"
DEFAULT_CAM = "0.1090,0.0300,0.2480,-0.49475,0.49475,-0.50520,0.50520"


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = np.asarray(q, float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def _arrow(scn, frm, to, rgba, width=0.006):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        np.asarray(rgba, np.float32),
    )
    mujoco.mjv_connector(
        g, mujoco.mjtGeom.mjGEOM_ARROW, width, np.asarray(frm, float), np.asarray(to, float)
    )
    scn.ngeom += 1


def _sphere(scn, pos, rgba, r=0.015):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([r, r, r]),
        np.asarray(pos, float),
        np.eye(3).flatten(),
        np.asarray(rgba, np.float32),
    )
    scn.ngeom += 1


def decorate(scn, d, bid, t, R, axis_len=0.12):
    """torso フレーム軸と、カメラ位置＋光学3軸を scn に描き足す。"""
    o = d.xpos[bid].copy()  # torso_link 原点（world）
    Rt = d.xmat[bid].reshape(3, 3)  # torso -> world
    # torso_link の座標軸（細め・淡色）: x=前 y=左 z=上
    for i, c in enumerate(((1, 0.4, 0.4, 0.6), (0.4, 1, 0.4, 0.6), (0.4, 0.4, 1, 0.6))):
        _arrow(scn, o, o + Rt[:, i] * axis_len, c, width=0.004)
    _sphere(scn, o, (1, 1, 0, 1), r=0.018)  # torso 原点 = 黄

    cam_w = o + Rt @ t  # カメラ位置（world）
    _sphere(scn, cam_w, (1, 0, 1, 1), r=0.020)  # カメラ = マゼンタ
    _arrow(scn, o, cam_w, (1, 1, 1, 0.8), width=0.004)  # torso->カメラ の白線
    Rc = Rt @ R  # optical -> world
    for i, c in enumerate(((1, 0, 0, 1), (0, 1, 0, 1), (0, 0.4, 1, 1))):  # 右/下/前
        _arrow(scn, cam_w, cam_w + Rc[:, i] * axis_len, c)
    return o, cam_w


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cam", default=DEFAULT_CAM, help='"x,y,z,qx,qy,qz,qw"')
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--out", default="cam_frame.png")
    a = p.parse_args()

    vals = [float(v) for v in a.cam.replace(" ", "").split(",")]
    if len(vals) != 7:
        raise SystemExit(f"--cam は7値必要です（{len(vals)}個でした）")
    t, R = np.array(vals[:3]), quat_to_rot(np.array(vals[3:]))
    W, H = 1200, 900

    # オフスクリーン既定(640x480)より大きく描くため offwidth/offheight を注入して読む。
    xml = SCENE.read_text()
    inject = f'<visual><global offwidth="{W}" offheight="{H}"/></visual>'
    xml = xml.replace("<worldbody>", inject + "\n  <worldbody>", 1)
    m = mujoco.MjModel.from_xml_string(xml, {})
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if bid < 0:
        raise SystemExit("torso_link が見つかりません")

    o_w = d.xpos[bid]
    cam_w = o_w + d.xmat[bid].reshape(3, 3) @ t
    print(f"torso_link 原点 (world) = {np.round(o_w, 4)}")
    print(f"カメラ光学中心 (world)   = {np.round(cam_w, 4)}")
    print(
        f"  → 床からの高さ {cam_w[2] * 100:.1f} cm / torso 原点より "
        f"前 {t[0] * 100:+.1f} cm, 左 {t[1] * 100:+.1f} cm, 上 {t[2] * 100:+.1f} cm"
    )
    fwd = d.xmat[bid].reshape(3, 3) @ R[:, 2]
    print(
        f"  レンズの向き (world)   = {np.round(fwd, 4)}  "
        f"（水平から {np.degrees(np.arcsin(np.clip(fwd[2], -1, 1))):+.2f} 度）"
    )

    if a.interactive:
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(m, d) as v:
            v.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
            while v.is_running():
                mujoco.mj_forward(m, d)
                v.user_scn.ngeom = 0
                decorate(v.user_scn, d, bid, t, R)
                v.sync()
        return

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = cam_w
    cam.distance, cam.elevation = 1.1, -12
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    scn = mujoco.MjvScene(m, maxgeom=20000)
    with mujoco.Renderer(m, H, W) as r:
        for az, tag in ((135, "front"), (200, "side")):
            cam.azimuth = az
            mujoco.mjv_updateScene(m, d, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
            decorate(scn, d, bid, t, R)
            r._scene = scn
            img = r.render()
            out = Path(a.out).with_name(Path(a.out).stem + f"_{tag}.png")
            PIL.Image.fromarray(img).save(out)
            print(f"保存: {out}")


if __name__ == "__main__":
    main()
