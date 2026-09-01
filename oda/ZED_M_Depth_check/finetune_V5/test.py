from ultralytics import YOLO
from dotenv import load_dotenv
import cv2
import os

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "/home/techshare/user/Okra_detaction/finetune_V5/output/okra_finetune_v5/weights/best.pt")
TEST_DIR   = "/home/techshare/user/Okra_detaction/finetune_V5/dataset/test/images"
OUTPUT_DIR = "/home/techshare/user/Okra_detaction/finetune_V5/output/test_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 8
model      = YOLO(MODEL_PATH)
img_paths  = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

print(f"Testing on {len(img_paths)} images (batch={BATCH_SIZE})...\n")

all_results = []
for i in range(0, len(img_paths), BATCH_SIZE):
    batch         = img_paths[i:i + BATCH_SIZE]
    batch_results = model(batch, device=0)
    for r, path in zip(batch_results, batch):
        cv2.imwrite(os.path.join(OUTPUT_DIR, os.path.basename(path)), r.plot())
    all_results.extend(zip(batch_results, batch))
    print(f"  [{i + len(batch)}/{len(img_paths)}] processed")

print(f"\nAll results saved → {OUTPUT_DIR}")
print("\n" + "="*50)
print("Test Results (v5 — merged outdoor + indoor)")
print("="*50)
for r, path in all_results:
    print(f"\n[{os.path.basename(path)}]")
    print(f"  Detected : {len(r.boxes)} okra instance(s)")
    for box in r.boxes:
        print(f"  Confidence : {float(box.conf):.2f}")
print("="*50)
