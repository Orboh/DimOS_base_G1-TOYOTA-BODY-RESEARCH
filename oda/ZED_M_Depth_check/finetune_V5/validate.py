from ultralytics import YOLO
from dotenv import load_dotenv
import os

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "/home/techshare/user/Okra_detaction/finetune_V5/output/okra_finetune_v5/weights/best.pt")
DATA_YAML  = os.getenv("DATA_YAML",  "/home/techshare/user/Okra_detaction/finetune_V5/dataset/data.yaml")

model   = YOLO(MODEL_PATH)
metrics = model.val(data=DATA_YAML, device=0)

print("\n" + "="*50)
print("Validation Results (v5 — merged outdoor + indoor)")
print("="*50)
print(f"mAP50       : {metrics.results_dict.get('metrics/mAP50(M)', 'N/A'):.4f}")
print(f"mAP50-95    : {metrics.results_dict.get('metrics/mAP50-95(M)', 'N/A'):.4f}")
print(f"Precision   : {metrics.results_dict.get('metrics/precision(M)', 'N/A'):.4f}")
print(f"Recall      : {metrics.results_dict.get('metrics/recall(M)', 'N/A'):.4f}")
print("="*50)
