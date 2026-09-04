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

import glob
import os

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "/home/techshare/user/Okra_detaction/finetune_V5/output/okra_finetune_v5/weights/best.pt",
)
CONF = 0.7


def enhance_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def find_camera():
    devices = sorted(glob.glob("/dev/video*"))
    print(f"Found video devices: {devices}")
    for dev in devices:
        idx = int(dev.replace("/dev/video", ""))
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Using camera: {dev} (index {idx}) — {w}x{h}")
                return cap
            cap.release()
    return None


model = YOLO(MODEL_PATH)
cap = find_camera()

if cap is None:
    print("Error: No working camera found. Please check connection.")
    exit()

print("=" * 50)
print("Real World Camera Testing (v5 — merged model)")
print("Press Q to quit")
print("=" * 50)

cv2.namedWindow("Okra Detection v5", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Okra Detection v5", 1280, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    enhanced = enhance_frame(frame)
    results = model(enhanced, conf=CONF, device=0, verbose=False)
    annotated = results[0].plot()

    count = len(results[0].boxes)
    cv2.putText(
        annotated,
        f"Okra detected: {count}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Okra Detection v5", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera closed.")
