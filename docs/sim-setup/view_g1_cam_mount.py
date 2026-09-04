#!/usr/bin/env python
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

"""G1 の URDF を rerun で 3D 表示し、胸カメラ用フレーム（d435_link 等）を可視化する。

目的: 実機で ZED をボルト止めした「胸の穴」が URDF のどのフレームに当たるかを目視確認する。
- 全リンクの visual メッシュを中立姿勢（全関節 0）で描画
- torso_link / d435_link / mid360_link / head_link を座標軸(triad)で強調
- d435_link には視線方向（+X＝前やや下）の矢印を追加
出力: .rrd ファイル（rerun ビューワーで開く）
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import rerun as rr
import trimesh

# URDF / メッシュ / 出力先は環境変数で上書き可（既定はこのマシンの実在パス）
URDF = os.getenv(
    "G1_URDF",
    "/home/kota-ueda/Desktop/dimos-hackathon/dimos/robot/unitree/g1/g1.urdf",
)
# meshes/NAME.STL -> MESH_DIR/NAME.STL
MESH_DIR = os.getenv("G1_MESH_DIR", "/home/kota-ueda/mujoco_menagerie/unitree_g1/assets")
OUT_RRD = os.getenv(
    "G1_OUT_RRD",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "g1_cam_mount.rrd"),
)

# 強調したいフレーム（torso_link 相対のカメラ/センサ取り付け点）
HIGHLIGHT = {
    "torso_link": (0.20, (200, 200, 200)),  # 基準フレーム（大きめの軸）
    "d435_link": (0.18, (255, 60, 60)),  # ★純正 D435 前面カメラ穴（ZED を挿す想定）
    "mid360_link": (0.10, (60, 160, 255)),  # LiDAR（参考）
    "head_link": (0.10, (120, 255, 120)),  # 頭（参考）
}


def rpy_to_mat(r: float, p: float, y: float) -> np.ndarray:
    """URDF の rpy（固定軸 X->Y->Z）を回転行列に。R = Rz(y) @ Ry(p) @ Rx(r)."""
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def origin_to_T(elem) -> np.ndarray:
    """<origin xyz rpy> を 4x4 同次変換に。無ければ単位行列。"""
    T = np.eye(4)
    if elem is None:
        return T
    xyz = [float(v) for v in elem.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in elem.get("rpy", "0 0 0").split()]
    T[:3, :3] = rpy_to_mat(*rpy)
    T[:3, 3] = xyz
    return T


def main() -> None:
    tree = ET.parse(URDF)
    root = tree.getroot()

    # link -> visual (mesh filename, visual origin T)
    link_visual = {}
    for link in root.findall("link"):
        name = link.get("name")
        vis = link.find("visual")
        if vis is None:
            link_visual[name] = None
            continue
        mesh = vis.find("geometry/mesh")
        if mesh is None:
            link_visual[name] = None
            continue
        fname = os.path.basename(mesh.get("filename"))
        link_visual[name] = (fname, origin_to_T(vis.find("origin")))

    # joints: child -> (parent, T)
    child_joint = {}
    children = set()
    for j in root.findall("joint"):
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        child_joint[child] = (parent, origin_to_T(j.find("origin")))
        children.add(child)

    all_links = [l.get("name") for l in root.findall("link")]
    roots = [l for l in all_links if l not in children]

    # 中立姿勢の world 変換を再帰計算（メモ化）
    world_T = {}

    def fk(link: str) -> np.ndarray:
        if link in world_T:
            return world_T[link]
        if link not in child_joint:
            world_T[link] = np.eye(4)
        else:
            parent, T = child_joint[link]
            world_T[link] = fk(parent) @ T
        return world_T[link]

    for l in all_links:
        fk(l)

    # rerun 出力
    rr.init("g1_cam_mount")
    rr.save(OUT_RRD)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # メッシュ（全身）
    for name, vis in link_visual.items():
        if vis is None:
            continue
        fname, vT = vis
        path = os.path.join(MESH_DIR, fname)
        if not os.path.exists(path):
            print(f"  [skip] mesh not found: {path}", file=sys.stderr)
            continue
        m = trimesh.load(path, force="mesh")
        T = world_T[name] @ vT
        v = (T[:3, :3] @ m.vertices.T).T + T[:3, 3]
        # torso/head は淡色ハイライト、他は薄いグレー
        if name == "torso_link":
            color = (255, 210, 140)
        elif name == "head_link":
            color = (180, 255, 180)
        else:
            color = (190, 190, 195)
        rr.log(
            f"g1/{name}",
            rr.Mesh3D(
                vertex_positions=v.astype(np.float32),
                triangle_indices=m.faces.astype(np.uint32),
                albedo_factor=color,
            ),
        )

    # 強調フレーム（座標軸 triad を Arrows3D で手描き: X=赤, Y=緑, Z=青）
    for name, (axis_len, color) in HIGHLIGHT.items():
        if name not in world_T:
            print(f"  [warn] frame not in urdf: {name}", file=sys.stderr)
            continue
        T = world_T[name]
        o = T[:3, 3].astype(np.float32)
        rr.log(
            f"frames/{name}/axes",
            rr.Arrows3D(
                origins=[o, o, o],
                vectors=[
                    (T[:3, 0] * axis_len).astype(np.float32),  # X
                    (T[:3, 1] * axis_len).astype(np.float32),  # Y
                    (T[:3, 2] * axis_len).astype(np.float32),  # Z
                ],
                colors=[(255, 0, 0), (0, 200, 0), (0, 80, 255)],
            ),
        )
        # 取り付け点に球＋ラベル
        rr.log(
            f"frames/{name}/origin",
            rr.Points3D(
                positions=[o],
                radii=[0.012],
                colors=[color],
                labels=[name],
            ),
        )

    # d435_link の視線方向（+X 軸＝前やや下）を長い矢印で
    if "d435_link" in world_T:
        T = world_T["d435_link"]
        origin = T[:3, 3]
        view_dir = T[:3, 0]  # link +X
        rr.log(
            "frames/d435_view_direction",
            rr.Arrows3D(
                origins=[origin.astype(np.float32)],
                vectors=[(view_dir * 0.4).astype(np.float32)],
                colors=[(255, 0, 0)],
                labels=["d435 view (+X)"],
            ),
        )
        # OKRA_CAM_TO_TORSO に使うのは torso_link 相対（world ではない）
        T_rel = np.linalg.inv(world_T["torso_link"]) @ T
        rp = T_rel[:3, 3]
        pitch_deg = np.degrees(np.arctan2(-view_dir[2], view_dir[0]))
        print(f"d435_link  world     pos = ({origin[0]:.4f}, {origin[1]:.4f}, {origin[2]:.4f}) m")
        print(
            f"d435_link  torso相対 pos = ({rp[0]:.4f}, {rp[1]:.4f}, {rp[2]:.4f}) m  ← OKRA_CAM_TO_TORSO の x,y,z"
        )
        print(
            f"d435_link  視線(+X) = ({view_dir[0]:.3f}, {view_dir[1]:.3f}, {view_dir[2]:.3f})  → 下向き約 {pitch_deg:.1f}°"
        )

    # ★ 実機の ZED 取り付け位置 = 胸の UNITREE ロゴ（logo_link メッシュの前面中心）
    #   URDF の logo_link 原点は腰位置なので使えない。メッシュ実体から算出する。
    if link_visual.get("logo_link") is not None:
        fn, vT = link_visual["logo_link"]
        lm = trimesh.load(os.path.join(MESH_DIR, fn), force="mesh")
        Tw = world_T["logo_link"] @ vT
        lv = (Tw[:3, :3] @ lm.vertices.T).T + Tw[:3, 3]  # world
        front = lv[lv[:, 0] > lv[:, 0].max() - 0.01]  # 前面(最大X)付近
        mount_w = front.mean(0)  # 取り付け面中心 (world)
        Tt = world_T["torso_link"]
        mount_rel = Tt[:3, :3].T @ (mount_w - Tt[:3, 3])  # torso 相対
        rr.log(
            "frames/ZED_mount",
            rr.Points3D(
                positions=[mount_w.astype(np.float32)],
                radii=[0.02],
                colors=[(255, 0, 255)],
                labels=["ZED mount (UNITREE logo)"],
            ),
        )
        # 既定の視線＝水平前向き（実機の傾きが分かれば差し替え）
        rr.log(
            "frames/ZED_view_level",
            rr.Arrows3D(
                origins=[mount_w.astype(np.float32)],
                vectors=[np.array([0.4, 0.0, 0.0], np.float32)],
                colors=[(255, 0, 255)],
                labels=["ZED view (level, 仮)"],
            ),
        )
        print(
            f"ZED mount (UNITREEロゴ前面)  world=({mount_w[0]:.3f},{mount_w[1]:.3f},{mount_w[2]:.3f})"
        )
        print(
            f"ZED mount  torso相対=({mount_rel[0]:.4f},{mount_rel[1]:.4f},{mount_rel[2]:.4f})  ← OKRA_CAM_TO_TORSO の x,y,z 土台"
        )

    print(f"\nwrote: {OUT_RRD}")
    print(f"roots: {roots}")


if __name__ == "__main__":
    main()
