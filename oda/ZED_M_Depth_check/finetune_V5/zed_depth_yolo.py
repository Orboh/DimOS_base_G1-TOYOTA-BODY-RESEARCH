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
import pyzed.sl as sl
from ultralytics import YOLO

load_dotenv()

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/home/techshare/user/Okra_detaction/ZED_M_Depth_check/finetune_V5/output/okra_finetune_v5/weights/best.pt",
)
CONF = 0.7


def enhance_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def open_zed():
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.depth_minimum_distance = 0.2

    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening ZED camera: {status}")
        exit(1)
    return zed


model = YOLO(MODEL_PATH)
zed = open_zed()

runtime = sl.RuntimeParameters()
image = sl.Mat()
depth = sl.Mat()

print("=" * 50)
print("ZED Okra Detection + Neural Depth")
print("Press Q to quit")
print("=" * 50)

cv2.namedWindow("Okra Detection + Depth", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Okra Detection + Depth", 1280, 720)

try:
    while True:
        if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue

        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        frame = cv2.cvtColor(image.get_data(), cv2.COLOR_BGRA2BGR)
        depth_np = depth.get_data()

        enhanced = enhance_frame(frame)
        results = model(enhanced, conf=CONF, device=0, verbose=False)
        annotated = results[0].plot()

        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cx = min(max(cx, 0), depth_np.shape[1] - 1)
            cy = min(max(cy, 0), depth_np.shape[0] - 1)

            d = depth_np[cy, cx]
            label = f"{d:.2f}m" if np.isfinite(d) and d > 0 else "N/A"

            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.putText(
            annotated,
            f"Okra detected: {len(boxes)}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Okra Detection + Depth", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    zed.close()
    cv2.destroyAllWindows()
    print("Camera closed.")
