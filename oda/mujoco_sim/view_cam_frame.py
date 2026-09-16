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

"""``OKRA_CAM_TO_TORSO`` が指すカメラ位置を、G1 の**実物の形状**の上に描く。

CAD から出した cam_to_torso が「胸のどこに付いているか」を目視で確認するための道具
（[[SS-04-粗アプローチIK]]）。``torso_link`` の原点・座標軸と、そこから cam_to_torso で
置いたカメラの光学3軸（右=赤 / 下=緑 / 前=青）を重ねて描く。

表示は3段階でフォールバックする:

* ``--view full``（既定）: ``g1.urdf`` をそのまま MuJoCo に読ませた**全身**。頭・腕・脚
  との位置関係が分かるので「胸のどのあたりか」を掴みやすい。
* ``--view torso``: ``torso_link_rev_1_0.STL`` だけの寄り。**URDF の torso_link は
  visual origin が ``xyz="0 0 0"`` なので、この STL の原点がそのまま torso_link 原点**
  ＝ CAD で位置決めするときの基準そのもの。mm 単位で詰めるときはこちら。
* メッシュが無い場合: ``g1_okra_scene.xml``（カプセル骨格）。胸の面が分からないので
  位置の判断には向かない。

メッシュは Unitree 公式の STL 一式（``~/Downloads/unitree_g1_meshes`` 等）を自動探索
する。``--meshdir`` で明示も可。

    .venv/bin/python oda/mujoco_sim/view_cam_frame.py
    .venv/bin/python oda/mujoco_sim/view_cam_frame.py --cam "x,y,z,qx,qy,qz,qw"
    .venv/bin/python oda/mujoco_sim/view_cam_frame.py --view torso
    .venv/bin/python oda/mujoco_sim/view_cam_frame.py --interactive   # macOS は mjpython
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import PIL.Image

SCENE = Path(__file__).resolve().parent / "g1_okra_scene.xml"
URDF = Path(__file__).resolve().parents[2] / "dimos/robot/unitree/g1/g1.urdf"
# 現行の本番既定値（unitree_g1_okra_honban.py の OKRA_CAM_TO_TORSO）と揃える。
DEFAULT_CAM = "0.1110,0.0250,0.2585,-0.49475,0.49475,-0.50520,0.50520"

# Unitree 公式メッシュ一式を展開した場所（URDF が参照する *.STL が直下にあるもの）。
MESHDIR_CANDIDATES = (
    Path.home() / "Downloads/unitree_g1_meshes",
    Path(__file__).resolve().parents[2] / "dimos/robot/unitree/g1/meshes",
)
TORSO_STL = "torso_link_rev_1_0.STL"
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


def _find_meshdir(explicit: str | None) -> Path | None:
    """URDF が参照する STL が入ったディレクトリ。見つからなければ None。"""
    cands = (Path(explicit).expanduser(),) if explicit else MESHDIR_CANDIDATES
    return next((p for p in cands if (p / TORSO_STL).exists()), None)


def _full_scene(meshdir: Path) -> mujoco.MjModel:
    """g1.urdf をそのまま MuJoCo に読ませて全身を実メッシュで表示する。

    URDF 内の ``<mujoco><compiler meshdir="meshes"/>`` は repo 相対で、この repo には
    メッシュが同梱されていない。実体のあるディレクトリへ差し替え、``filename`` から
    ``meshes/`` 接頭辞を外して読む。原点は pelvis（floating base の基準）。
    """
    xml = URDF.read_text()
    xml = xml.replace(
        '<compiler meshdir="meshes" discardvisual="false"/>',
        f'<compiler meshdir="{meshdir}" discardvisual="false" balanceinertia="true"/>',
    )
    xml = xml.replace('filename="meshes/', 'filename="')
    return mujoco.MjModel.from_xml_string(xml)


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
  <visual><headlight ambient="0.45 0.45 0.45" diffuse="0.55 0.55 0.55"/></visual>
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


def decorate(scn, d, bid, t, R, axis_len=0.10, r_mark=0.015):
    """torso フレーム軸と、カメラ位置＋光学3軸を scn に描き足す。

    ``axis_len``/``r_mark`` は画角に合わせて呼び出し側が渡す（全身表示で胴体基準の
    大きさのままだとマーカーが点にしか見えない）。
    """
    o = d.xpos[bid].copy()  # torso_link 原点（world）
    Rt = d.xmat[bid].reshape(3, 3)  # torso -> world
    w = r_mark * 0.28
    for i, c in enumerate(((1, 0.35, 0.35, 0.5), (0.35, 1, 0.35, 0.5), (0.35, 0.35, 1, 0.5))):
        _arrow(scn, o, o + Rt[:, i] * axis_len * 0.8, c, width=w * 0.7)
    _sphere(scn, o, (1, 1, 0, 1), r=r_mark * 0.9)  # torso 原点 = 黄

    cam_w = o + Rt @ t
    _arrow(scn, o, cam_w, (1, 1, 1, 0.9), width=w * 0.7)  # torso -> カメラ
    _sphere(scn, cam_w, (1, 0, 1, 1), r=r_mark)  # カメラ = マゼンタ
    Rc = Rt @ R  # optical -> world
    for i, c in enumerate(((1, 0, 0, 1), (0, 1, 0, 1), (0, 0.45, 1, 1))):  # 右/下/前
        _arrow(scn, cam_w, cam_w + Rc[:, i] * axis_len, c, width=w)
    return o, cam_w


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cam", default=DEFAULT_CAM, help='"x,y,z,qx,qy,qz,qw"')
    p.add_argument("--meshdir", default=None, help="Unitree の STL 一式があるディレクトリ")
    p.add_argument(
        "--view",
        choices=("full", "torso"),
        default="full",
        help="full=全身URDF（既定） / torso=胴体だけの寄り（CADの位置決め確認向き）",
    )
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--out", default="cam_frame.png")
    a = p.parse_args()

    vals = [float(v) for v in a.cam.replace(" ", "").split(",")]
    if len(vals) != 7:
        raise SystemExit(f"--cam は7値必要です（{len(vals)}個でした）")
    t, R = np.array(vals[:3]), quat_to_rot(np.array(vals[3:]))

    meshdir = _find_meshdir(a.meshdir)
    mesh = None
    if meshdir is not None and a.view == "full":
        print(f"全身 URDF: {URDF}  (メッシュ {meshdir})")
        m = _full_scene(meshdir)
    elif meshdir is not None:
        mesh = meshdir / TORSO_STL
        print(f"胴体メッシュ: {mesh}")
        m = _torso_scene(mesh)
    else:
        print("メッシュが見つかりません → カプセル骨格で描画（胸の面は分かりません）")
        print(f"  探索先: {', '.join(str(c) for c in MESHDIR_CANDIDATES)}")
        m = mujoco.MjModel.from_xml_string(SCENE.read_text(), {})

    # オフスクリーン既定は 640x480。XML に書くより後から代入する方が、URDF 経路でも
    # 確実に効く。
    m.vis.global_.offwidth, m.vis.global_.offheight = W, H
    # G1 のメッシュは暗いグレーで、既定のライトだと形が読み取りにくい。
    m.vis.headlight.ambient[:] = 0.55
    m.vis.headlight.diffuse[:] = 0.65
    m.vis.headlight.specular[:] = 0.1
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
    # 描くもの全部が入るように自動フレーミングする。マーカーの大きさも画角に合わせて
    # 変える（全身表示で胴体基準のままだと点にしか見えない）。
    if mesh is not None:
        v = m.mesh_vert[:] + m.mesh_pos[0]
        pts = np.vstack([v, cam_w, o_w, cam_w + 0.10, cam_w - 0.10])
    else:
        # geom の中心だけだと頭や足先が画面外に切れる。各 geom の外接半径を
        # 足し引きして実際の広がりを取る。
        rb = m.geom_rbound.reshape(-1, 1)
        pts = np.vstack([d.geom_xpos - rb, d.geom_xpos + rb, cam_w, o_w])
    lo, hi = pts.min(0), pts.max(0)
    span = float(np.linalg.norm(hi - lo))
    cam.lookat[:] = (lo + hi) / 2.0
    cam.distance = span * (1.05 if mesh is None else 1.9)
    axis_len, r_mark = span * 0.10, span * 0.016
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    scn = mujoco.MjvScene(m, maxgeom=20000)
    views = (("front", 180, -8), ("side", 90, -8), ("top", 180, -70))
    with mujoco.Renderer(m, H, W) as r:
        for tag, az, el in views:
            cam.azimuth, cam.elevation = az, el
            mujoco.mjv_updateScene(m, d, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
            decorate(scn, d, bid, t, R, axis_len=axis_len, r_mark=r_mark)
            r._scene = scn
            out = Path(a.out).with_name(Path(a.out).stem + f"_{tag}.png")
            PIL.Image.fromarray(r.render()).save(out)
            print(f"保存: {out.resolve()}")


if __name__ == "__main__":
    main()
