import os
import shutil
import random

SOURCES = [
    "/home/techshare/user/Okra_detaction/finetune_v2/dataset",
    "/home/techshare/user/Okra_detaction/finetune_V4/dataset",
]

OUT_DIR   = "/home/techshare/user/Okra_detaction/finetune_V5/dataset"
SPLITS    = {"train": 0.8, "valid": 0.1, "test": 0.1}
SEED      = 42

random.seed(SEED)

# Collect all unique image paths with their label counterparts
pairs = {}  # filename → (img_path, lbl_path)

for src in SOURCES:
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(src, split, "images")
        lbl_dir = os.path.join(src, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem  = os.path.splitext(fname)[0]
            lbl   = os.path.join(lbl_dir, stem + ".txt")
            if not os.path.exists(lbl):
                continue
            if fname not in pairs:
                pairs[fname] = (os.path.join(img_dir, fname), lbl)

all_pairs = list(pairs.values())
random.shuffle(all_pairs)

n      = len(all_pairs)
n_tr   = int(n * SPLITS["train"])
n_val  = int(n * SPLITS["valid"])

split_map = {
    "train": all_pairs[:n_tr],
    "valid": all_pairs[n_tr:n_tr + n_val],
    "test":  all_pairs[n_tr + n_val:],
}

print(f"Total unique pairs : {n}")

for split, items in split_map.items():
    img_out = os.path.join(OUT_DIR, split, "images")
    lbl_out = os.path.join(OUT_DIR, split, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    for img_path, lbl_path in items:
        shutil.copy2(img_path, os.path.join(img_out, os.path.basename(img_path)))
        shutil.copy2(lbl_path, os.path.join(lbl_out, os.path.basename(lbl_path)))
    print(f"  {split:5s}: {len(items)} images")

yaml_path = os.path.join(OUT_DIR, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(f"train: {OUT_DIR}/train/images\n")
    f.write(f"val:   {OUT_DIR}/valid/images\n")
    f.write(f"test:  {OUT_DIR}/test/images\n")
    f.write(f"nc: 1\n")
    f.write(f"names: ['okra']\n")

print(f"\ndata.yaml → {yaml_path}")
print("Merge complete!")
