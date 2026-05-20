import os
import csv
from glob import glob
from sklearn.model_selection import train_test_split

DATA_ROOT = "data/Kidney/data"   # top-level folder containing case_XXXXX/ dirs
SPLIT_DIR = "data/splits"
os.makedirs(SPLIT_DIR, exist_ok=True)

# Collect all case IDs from case_XXXXX/ subdirectories
# Only include folders that have both imaging.nii.gz and segmentation.nii.gz
case_dirs = sorted(glob(os.path.join(DATA_ROOT, "case_*")))
case_ids  = []

for case_dir in case_dirs:
    case_name  = os.path.basename(case_dir)
    img_path   = os.path.join(case_dir, "imaging.nii.gz")
    label_path = os.path.join(case_dir, "segmentation.nii.gz")

    if os.path.exists(img_path) and os.path.exists(label_path):
        case_ids.append(case_name)
    else:
        print(f"[WARN] Skipping {case_name} — missing imaging or segmentation file.")

print(f"Total valid cases found: {len(case_ids)}")

train_cases, val_cases = train_test_split(
    case_ids,
    test_size=0.2,
    random_state=42,
)

def write_csv(path, cases):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id"])
        for c in sorted(cases):
            writer.writerow([c])

write_csv(os.path.join(SPLIT_DIR, "train_kidney.csv"), train_cases)
write_csv(os.path.join(SPLIT_DIR, "val_kidney.csv"),   val_cases)

print(f"✅ Splits created | Train: {len(train_cases)} | Val: {len(val_cases)}")