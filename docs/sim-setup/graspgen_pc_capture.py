#!/usr/bin/env python3
"""GraspGen検証用: Isaac Sim上のオクラ1本を撮って対象の点群(npy)を書き出す（isaac-sim env, headless）。

目的: 「直立(upright)」と「横倒し(fallen)」の2姿勢で同じオクラを撮り、GraspGenが姿勢に
関わらず妥当な6-DoF把持姿勢を出せるかを比較検証するための入力データを作る。

instance_id_segmentation（leaf prim単位、semanticラベル不要）× depth で対象オクラの
画素だけを抜き、カメラ内部パラメータで逆投影して camera 光学系のオクラ点群(Nx3, [m])を得る。

実行（別プロセス, 現在稼働中の摩擦把持sim(SIM_TABLE=1 SIM_OKRA=10)とは独立）:
  SIM_OKRA_POSE=upright OUT_DIR=/tmp/graspgen_pc/upright \
    ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/graspgen_pc_capture.py
  SIM_OKRA_POSE=fallen  OUT_DIR=/tmp/graspgen_pc/fallen \
    ~/miniconda3/envs/isaac-sim/bin/python docs/sim-setup/graspgen_pc_capture.py
"""
import math
import os
import sys

import numpy as np

REPO = "/home/kota-ueda/Desktop/dimos-hackathon"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/docs/sim-setup")

POSE = os.getenv("SIM_OKRA_POSE", "upright")
OUT_DIR = os.getenv("OUT_DIR", f"/tmp/graspgen_pc/{POSE}")
CAM_W = int(os.getenv("SIM_CAM_W", "640"))
CAM_H = int(os.getenv("SIM_CAM_H", "480"))
HFOV_DEG = float(os.getenv("SIM_CAM_HFOV", "55"))
os.makedirs(OUT_DIR, exist_ok=True)


def main() -> int:
    from isaacsim import SimulationApp

    sim_app = SimulationApp({"headless": True})

    import cv2
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.sensors.camera import Camera

    import sim_scene

    world = World(stage_units_in_meters=1.0)
    stage = world.stage
    table_h = 0.72
    table_cx = sim_scene.TABLE_CX

    if POSE == "upright":
        okra_paths = sim_scene.build_table_okra(stage, table_h=table_h, n_okra=1)
        target_path = okra_paths[0]
        world.reset()
        for _ in range(5):
            world.step(render=False)
    else:
        # 机のみ（n_okra=0）。対象は別途、崩れた姿勢から自由落下させ横倒しに定常させる。
        sim_scene.build_table_okra(stage, table_h=table_h, n_okra=0)
        target_path = "/Okra_test"
        add_reference_to_stage(usd_path=sim_scene.OKRA_USD, prim_path=target_path)
        prim = stage.GetPrimAtPath(target_path)
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(prim)
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
        # 直立(qd)から大きく倒した姿勢で机の少し上に置き、重力で落下→横倒しに定常させる。
        qd = (Gf.Rotation(Gf.Vec3d(1, 0, 0), 80.0) * Gf.Rotation(Gf.Vec3d(0, 1, 0), 20.0)).GetQuat()
        xi, yi, zc = table_cx, 0.0, table_h + 0.15
        op = UsdGeom.Xformable(prim)
        op.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(xi, yi, zc))
        op.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(qd)
        world.reset()
        for _ in range(180):  # 60Hz*3s: 落下→机上で定常するまで待つ
            world.step(render=False)

    print(f"[capture] target={target_path} pose={POSE}", flush=True)
    _t = np.array(UsdGeom.Xformable(stage.GetPrimAtPath(target_path)).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()))
    print(f"[capture] target world transform=\n{_t}", flush=True)

    # --- カメラ: 対象の斜め上から見下ろす固定 look-at ---
    eye = np.array([table_cx - 0.35, 0.25, table_h + 0.45])
    tgt = np.array([table_cx, 0.0, table_h + 0.08])
    m = Gf.Matrix4d()
    m.SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*tgt), Gf.Vec3d(0, 0, 1))
    q = m.GetInverse().ExtractRotationQuat()
    ori = np.array([q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]])
    cam = Camera(prim_path="/pc_cam", resolution=(CAM_W, CAM_H))
    cam.initialize()
    cam.set_world_pose(eye, ori, camera_axes="usd")

    A = 20.955
    hfov = math.radians(HFOV_DEG)
    F = (A / 2.0) / math.tan(hfov / 2.0)
    cam.set_horizontal_aperture(A)
    cam.set_vertical_aperture(A * CAM_H / CAM_W)
    cam.set_focal_length(F)
    cam.set_clipping_range(0.03, 100.0)
    fx = fy = F * CAM_W / A
    cx, cy = CAM_W / 2.0, CAM_H / 2.0
    print(f"[capture] intrinsics fx=fy={fx:.2f} cx={cx} cy={cy}", flush=True)

    cam.add_distance_to_image_plane_to_frame()
    cam.add_instance_id_segmentation_to_frame()

    for _ in range(15):  # annotator が値を持つまで数フレーム要る
        world.step(render=True)

    frame = cam.get_current_frame()
    print(f"[capture] frame keys={list(frame.keys())}", flush=True)

    rgba = cam.get_rgba()
    depth = frame.get("distance_to_image_plane")
    seg = frame.get("instance_id_segmentation")
    if seg is None:
        print("[capture] ERROR: instance_id_segmentation が取得できませんでした", flush=True)
        return 1
    seg_data = seg["data"] if isinstance(seg, dict) else seg
    seg_info = seg.get("info", {}) if isinstance(seg, dict) else {}
    id_to_labels = seg_info.get("idToLabels", {})
    print(f"[capture] idToLabels={id_to_labels}", flush=True)

    target_id = None
    for sid, label in id_to_labels.items():
        lp = label if isinstance(label, str) else str(label)
        if target_path in lp:
            target_id = int(sid)
            break
    if target_id is None:
        uniq = np.unique(seg_data)
        print(f"[capture] WARN: {target_path} が idToLabels に見つからず。unique ids={uniq}", flush=True)
        return 1

    mask = (seg_data.squeeze() == target_id)
    n_px = int(mask.sum())
    print(f"[capture] target_id={target_id} mask画素数={n_px}", flush=True)
    if n_px < 20:
        print("[capture] ERROR: マスク画素が少なすぎる（カメラが対象を捉えていない可能性）", flush=True)

    d = np.asarray(depth).squeeze()
    ys, xs = np.where(mask & np.isfinite(d) & (d > 0.03) & (d < 10.0))
    zs = d[ys, xs].astype(np.float64)
    xs_3d = (xs - cx) / fx * zs
    ys_3d = (ys - cy) / fy * zs
    points_cam = np.stack([xs_3d, ys_3d, zs], axis=1)  # optical frame [X右,Y下,Z前]
    print(f"[capture] 点群点数={len(points_cam)}", flush=True)

    np.save(os.path.join(OUT_DIR, "points_cam.npy"), points_cam)
    if rgba is not None:
        cv2.imwrite(os.path.join(OUT_DIR, "rgb.png"), cv2.cvtColor(np.asarray(rgba)[..., :3], cv2.COLOR_RGB2BGR))
    mask_vis = (mask.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(OUT_DIR, "mask.png"), mask_vis)
    print(f"[capture] 保存先={OUT_DIR} (points_cam.npy, rgb.png, mask.png)", flush=True)
    print("CAPTURE_OK", flush=True)

    sim_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
