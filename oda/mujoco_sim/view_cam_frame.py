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

"""``OKRA_CAM_TO_TORSO`` が指すカメラ位置を、G1 の**実物の胴体形状**の上に描く。

CAD から出した cam_to_torso が「胸のどこに付いているか」を目視で確認するための道具
（[[SS-04-粗アプローチIK]]）。``torso_link`` の原点・座標軸と、そこから cam_to_torso で
置いたカメラの光学3軸（右=赤 / 下=緑 / 前=青）を重ねて描く。

胴体メッシュ ``torso_link_rev_1_0.STL`` が見つかればそれを表示する。**URDF の
torso_link は visual origin が ``xyz="0 0 0"`` なので、この STL の原点がそのまま
torso_link 原点**＝ CAD で位置決めするときの基準になる。見つからない場合は
``g1_okra_scene.xml``（カプセル骨格）へフォールバックするが、胸の面が分からないので
位置の判断には向かない。

    .venv/bin/python oda/mujoco_sim/view_cam_frame.py
    .venv/bin/python oda/mujoco_sim/view_cam_frame.py --cam "x,y,z,qx,qy,qz,qw"
    .venv/bin/python oda/mujoco_sim/view_cam_frame.py --mesh /path/to/torso_link_rev_1_0.STL
    .venv/bin/python oda/mujoco_sim/view_cam_frame.py --interactive   # macOS は mjpython
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import PIL.Image

SCENE = Path(__file__).resolve().parent / "g1_okra_scene.xml"
DEFAULT_CAM = "0.1090,0.0300,0.2480,-0.49475,0.49475,-0.50520,0.50520"

# 胴体メッシュの探索先（Unitree 公式メッシュ一式を展開した場所）。
MESH_CANDIDATES = (
    Path.home() / "Downloads/unitree_g1_meshes/torso_link_rev_1_0.STL",
    Path(__file__).resolve().parents[3] / "dimos/robot/unitree/g1/meshes/torso_link_rev_1_0.STL",
)
W, H = 1400, 1000


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = np.asarray(q, float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def _find_mesh(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.exists() else None
    return next((p for p in MESH_CANDIDATES if p.exists()), None)


def _torso_scene(mesh: Path) -> mujoco.MjModel:
    """胴体メッシュだけの最小シーン。原点 = torso_link 原点。

    MuJoCo はメッシュを慣性フレームへ再センタリングして読むため、``mesh_pos``/
    ``mesh_quat`` を geom 側に打ち消しで与えて **STL 本来の座標**に戻す。これを
    忘れると胴体が約 15cm 下にずれ、カメラ位置の判断を誤る。
    """
    probe = mujoco.MjModel.from_xml_string(
        f'<mujoco><asset><mesh name="t" file="{mesh}"/></asset>'
        '<worldbody><geom type="mesh" mesh="t"/></worldbody></mujoco>'
    )
    p, q = probe.mesh_pos[0], probe.mesh_quat[0]
    xml = f"""<mujoco model="torso_cam_frame">
  <compiler angle="radian"/>
  <visual>
    <global offwidth="{W}" offheight="{H}"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.55 0.55 0.55"/>
  </visual>
  <asset><mesh name="torso" file="{mesh}"/></asset>
  <worldbody>
    <body name="torso_link" pos="0 0 0">
      <geom type="mesh" mesh="torso" rgba="0.70 0.74 0.80 1"
            pos="{p[0]} {p[1]} {p[2]}" quat="{q[0]} {q[1]} {q[2]} {q[3]}"/>
    </body>
  </worldbody>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def _arrow(scn, frm, to, rgba, width=0.005):
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


def _sphere(scn, pos, rgba, r=0.012):
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


def decorate(scn, d, bid, t, R, axis_len=0.10):
    """torso フレーム軸と、カメラ位置＋光学3軸を scn に描き足す。"""
    o = d.xpos[bid].copy()  # torso_link 原点（world）
    Rt = d.xmat[bid].reshape(3, 3)  # torso -> world
    for i, c in enumerate(((1, 0.35, 0.35, 0.5), (0.35, 1, 0.35, 0.5), (0.35, 0.35, 1, 0.5))):
        _arrow(scn, o, o + Rt[:, i] * axis_len, c, width=0.003)
    _sphere(scn, o, (1, 1, 0, 1), r=0.014)  # torso 原点 = 黄

    cam_w = o + Rt @ t
    _arrow(scn, o, cam_w, (1, 1, 1, 0.9), width=0.003)  # torso -> カメラ
    _sphere(scn, cam_w, (1, 0, 1, 1), r=0.016)  # カメラ = マゼンタ
    Rc = Rt @ R  # optical -> world
    for i, c in enumerate(((1, 0, 0, 1), (0, 1, 0, 1), (0, 0.45, 1, 1))):  # 右/下/前
        _arrow(scn, cam_w, cam_w + Rc[:, i] * axis_len, c)
    return o, cam_w


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cam", default=DEFAULT_CAM, help='"x,y,z,qx,qy,qz,qw"')
    p.add_argument("--mesh", default=None, help="torso_link_rev_1_0.STL のパス（既定は自動探索）")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--out", default="cam_frame.png")
    a = p.parse_args()

    vals = [float(v) for v in a.cam.replace(" ", "").split(",")]
    if len(vals) != 7:
        raise SystemExit(f"--cam は7値必要です（{len(vals)}個でした）")
    t, R = np.array(vals[:3]), quat_to_rot(np.array(vals[3:]))

    mesh = _find_mesh(a.mesh)
    if mesh is not None:
        print(f"胴体メッシュ: {mesh}")
        m = _torso_scene(mesh)
    else:
        print("胴体メッシュが見つかりません → カプセル骨格で描画（胸の面は分かりません）")
        print(f"  探索先: {', '.join(str(c) for c in MESH_CANDIDATES)}")
        xml = SCENE.read_text().replace(
            "<worldbody>",
            f'<visual><global offwidth="{W}" offheight="{H}"/></visual>\n  <worldbody>',
            1,
        )
        m = mujoco.MjModel.from_xml_string(xml, {})

    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if bid < 0:
        raise SystemExit("torso_link が見つかりません")

    o_w = d.xpos[bid]
    cam_w = o_w + d.xmat[bid].reshape(3, 3) @ t
    fwd = d.xmat[bid].reshape(3, 3) @ R[:, 2]
    print(f"torso_link 原点      = {np.round(o_w, 4)}")
    print(f"カメラ光学中心       = {np.round(cam_w, 4)}")
    print(
        f"  → torso 原点より 前 {t[0] * 100:+.1f} cm, 左 {t[1] * 100:+.1f} cm, "
        f"上 {t[2] * 100:+.1f} cm"
    )
    print(
        f"  レンズの向き        = {np.round(fwd, 4)}  "
        f"（水平から {np.degrees(np.arcsin(np.clip(fwd[2], -1, 1))):+.2f} 度）"
    )
    if mesh is not None:
        v = m.mesh_vert[:] + m.mesh_pos[0]
        lo, hi = v.min(0), v.max(0)
        print(f"  胴体メッシュ bbox   = {np.round(lo, 4)} .. {np.round(hi, 4)}")
        print(f"  → カメラは胸の前面(X={hi[0]:.4f})より {(t[0] - hi[0]) * 1000:+.1f} mm 前方")

    if a.interactive:
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(m, d) as v_:
            v_.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
            while v_.is_running():
                mujoco.mj_forward(m, d)
                v_.user_scn.ngeom = 0
                decorate(v_.user_scn, d, bid, t, R)
                v_.sync()
        return

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    # 胴体メッシュ＋カメラ位置＋座標軸の全部が入るように自動フレーミングする。
    if mesh is not None:
        v = m.mesh_vert[:] + m.mesh_pos[0]
        pts = np.vstack([v, cam_w, o_w, cam_w + 0.10, cam_w - 0.10])
    else:
        pts = np.vstack([d.xpos, cam_w])
    lo, hi = pts.min(0), pts.max(0)
    cam.lookat[:] = (lo + hi) / 2.0
    cam.distance = float(np.linalg.norm(hi - lo)) * 1.9
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    scn = mujoco.MjvScene(m, maxgeom=20000)
    views = (("front", 180, -8), ("side", 90, -8), ("top", 180, -70))
    with mujoco.Renderer(m, H, W) as r:
        for tag, az, el in views:
            cam.azimuth, cam.elevation = az, el
            mujoco.mjv_updateScene(m, d, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
            decorate(scn, d, bid, t, R)
            r._scene = scn
            out = Path(a.out).with_name(Path(a.out).stem + f"_{tag}.png")
            PIL.Image.fromarray(r.render()).save(out)
            print(f"保存: {out.resolve()}")


if __name__ == "__main__":
    main()
