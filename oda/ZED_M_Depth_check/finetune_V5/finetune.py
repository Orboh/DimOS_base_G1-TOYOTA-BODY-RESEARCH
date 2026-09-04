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

from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

MODEL_PATH = os.getenv(
    "MODEL_PATH", "/home/techshare/user/Okra_detaction/finetune_V5/model/okra11n-seg.pt"
)
DATA_YAML = os.getenv(
    "DATA_YAML", "/home/techshare/user/Okra_detaction/finetune_V5/dataset/data.yaml"
)
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/home/techshare/user/Okra_detaction/finetune_V5/output")

model = YOLO(MODEL_PATH)

results = model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.0001,
    lrf=0.01,
    weight_decay=0.0005,
    optimizer="AdamW",
    patience=20,
    dropout=0.1,
    warmup_epochs=3,
    mosaic=1.0,
    close_mosaic=10,
    degrees=0,
    flipud=0.0,
    fliplr=0.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    project=OUTPUT_DIR,
    name="okra_finetune_v5",
    device=0,
    plots=True,
    save=True,
    verbose=True,
)

print("\n" + "=" * 50)
print("Fine-tuning Complete! (v5 — merged outdoor + indoor)")
print(f"Best model: {OUTPUT_DIR}/okra_finetune_v5/weights/best.pt")
print(f"mAP50   (mask) : {results.results_dict.get('metrics/mAP50(M)', 'N/A')}")
print(f"mAP50-95(mask) : {results.results_dict.get('metrics/mAP50-95(M)', 'N/A')}")
print(f"mAP50   (box)  : {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print(f"mAP50-95(box)  : {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
print("=" * 50)
