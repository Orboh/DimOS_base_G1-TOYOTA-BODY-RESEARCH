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
import matplotlib
import numpy as np

matplotlib.use("Agg")  # pure CPU rasterization - avoids OpenGL/CUDA context conflicts
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pyzed.sl as sl
from ultralytics import YOLO

load_dotenv()

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/home/sota/Videos/ZED_M_Depth_check/finetune_V5/output/okra_finetune_v5/weights/best.pt",
)
CONF = 0.7

PANEL_H = 720
PANEL_2D_W = 1280
PANEL_3D_W = 960
MAX_POINTS_3D = 4000  # subsampled for render speed


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

cv2.namedWindow("Okra Detection - 2D + 3D Depth", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Okra Detection - 2D + 3D Depth", PANEL_2D_W + PANEL_3D_W, PANEL_H)

fig = plt.figure(figsize=(PANEL_3D_W / 100, PANEL_H / 100), dpi=100)
ax = fig.add_subplot(111, projection="3d")

print("=" * 60)
print("Okra Detection + Neural Depth + 3D Point Cloud (combined view)")
print("Press Q to quit.")
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
        okra_xyz = []

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
                okra_xyz.append((X, Y, Z))
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

        # build the 3D point cloud panel (matplotlib, CPU-rendered)
        xyz = pc_np[:, :, :3].reshape(-1, 3)
        valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0)
        xyz_v = xyz[valid]
        highlight_v = highlight.reshape(-1)[valid]

        n = xyz_v.shape[0]
        if n > MAX_POINTS_3D:
            sel = np.random.choice(n, size=MAX_POINTS_3D, replace=False)
            xyz_v = xyz_v[sel]
            highlight_v = highlight_v[sel]

        ax.clear()
        bg = ~highlight_v
        if bg.any():
            ax.scatter(xyz_v[bg, 0], xyz_v[bg, 2], -xyz_v[bg, 1], s=1, c="dodgerblue", alpha=0.5)
        if highlight_v.any():
            ax.scatter(
                xyz_v[highlight_v, 0], xyz_v[highlight_v, 2], -xyz_v[highlight_v, 1], s=6, c="red"
            )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_zlabel("Y (m)")
        ax.set_title(f"Point cloud - {len(okra_xyz)} okra")

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        panel_3d = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        panel_3d = cv2.resize(panel_3d, (PANEL_3D_W, PANEL_H))

        panel_2d = cv2.resize(annotated, (PANEL_2D_W, PANEL_H))
        combined = np.hstack([panel_2d, panel_3d])

        cv2.imshow("Okra Detection - 2D + 3D Depth", combined)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    plt.close(fig)
    cv2.destroyAllWindows()
    zed.close()
    print("Closed, camera released.")
