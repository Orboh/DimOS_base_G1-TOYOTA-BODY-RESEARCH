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

import os

import cv2
from dotenv import load_dotenv
import numpy as np
import open3d as o3d
import pyzed.sl as sl
from ultralytics import YOLO

load_dotenv()

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/home/sota/Videos/ZED_M_Depth_check/finetune_V5/output/okra_finetune_v5/weights/best.pt",
)
CONF = 0.7


def enhance_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


model = YOLO(MODEL_PATH)

zed = sl.Camera()
init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.HD1080
init.depth_mode = sl.DEPTH_MODE.NEURAL
init.coordinate_units = sl.UNIT.METER
init.depth_minimum_distance = 0.2

status = zed.open(init)
print("open() status:", status)
if status != sl.ERROR_CODE.SUCCESS:
    exit(1)

runtime = sl.RuntimeParameters()
image = sl.Mat()
point_cloud = sl.Mat()

cv2.namedWindow("Okra Detection + Depth (2D)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Okra Detection + Depth (2D)", 1280, 720)

vis = o3d.visualization.Visualizer()
vis.create_window("Okra 3D Point Cloud - detections in red (close to quit)", width=1280, height=800)
pcd = o3d.geometry.PointCloud()
first_frame = True

print("=" * 60)
print("Okra Detection + Neural Depth + 3D Point Cloud")
print("Press Q in the 2D window, or close the 3D window, to quit.")
print("=" * 60)

try:
    while True:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        frame = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)
        pc_np = point_cloud.get_data()  # (H, W, 4): X, Y, Z, packed RGBA
        H, W = pc_np.shape[:2]

        enhanced = enhance_frame(frame)
        results = model(enhanced, conf=CONF, device=0, verbose=False)
        r = results[0]
        annotated = r.plot()

        highlight = np.zeros((H, W), dtype=bool)

        for idx, box in enumerate(r.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cx = min(max(cx, 0), W - 1)
            cy = min(max(cy, 0), H - 1)

            X, Y, Z = pc_np[cy, cx, :3]
            if np.isfinite([X, Y, Z]).all() and Z > 0:
                dist = float(np.sqrt(X * X + Y * Y + Z * Z))
                coord_label = f"X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f} d:{dist:.2f}m"
                print(f"Okra #{idx}: X={X:.3f}m Y={Y:.3f}m Z={Z:.3f}m dist={dist:.3f}m")
            else:
                coord_label = "coord: N/A"

            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(
                annotated,
                coord_label,
                (x1, max(y1 - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

            if r.masks is not None and idx < len(r.masks.data):
                m = r.masks.data[idx].cpu().numpy()
                m_resized = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                highlight |= m_resized > 0.5
            else:
                bx1, by1 = max(x1, 0), max(y1, 0)
                bx2, by2 = min(x2, W - 1), min(y2, H - 1)
                highlight[by1:by2, bx1:bx2] = True

        cv2.putText(
            annotated,
            f"Okra detected: {len(r.boxes)}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Okra Detection + Depth (2D)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        xyz = pc_np[:, :, :3].reshape(-1, 3)
        rgba_packed = pc_np[:, :, 3].reshape(-1)
        rgba_bytes = rgba_packed.copy().view(np.uint8).reshape(-1, 4)
        colors = rgba_bytes[:, :3].astype(np.float64) / 255.0
        colors[highlight.reshape(-1)] = [1.0, 0.0, 0.0]

        valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0)
        xyz = xyz[valid]
        colors = colors[valid]

        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors)

        if first_frame:
            vis.add_geometry(pcd)
            first_frame = False
        else:
            vis.update_geometry(pcd)

        if not vis.poll_events():
            break
        vis.update_renderer()
finally:
    vis.destroy_window()
    cv2.destroyAllWindows()
    zed.close()
    print("Closed, camera released.")
