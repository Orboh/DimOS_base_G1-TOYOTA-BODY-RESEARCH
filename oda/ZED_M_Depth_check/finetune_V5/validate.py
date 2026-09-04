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
    "MODEL_PATH",
    "/home/techshare/user/Okra_detaction/finetune_V5/output/okra_finetune_v5/weights/best.pt",
)
DATA_YAML = os.getenv(
    "DATA_YAML", "/home/techshare/user/Okra_detaction/finetune_V5/dataset/data.yaml"
)

model = YOLO(MODEL_PATH)
metrics = model.val(data=DATA_YAML, device=0)

print("\n" + "=" * 50)
print("Validation Results (v5 — merged outdoor + indoor)")
print("=" * 50)
print(f"mAP50       : {metrics.results_dict.get('metrics/mAP50(M)', 'N/A'):.4f}")
print(f"mAP50-95    : {metrics.results_dict.get('metrics/mAP50-95(M)', 'N/A'):.4f}")
print(f"Precision   : {metrics.results_dict.get('metrics/precision(M)', 'N/A'):.4f}")
print(f"Recall      : {metrics.results_dict.get('metrics/recall(M)', 'N/A'):.4f}")
print("=" * 50)
