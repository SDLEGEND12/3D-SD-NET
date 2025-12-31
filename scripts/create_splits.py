import os
import csv
from glob import glob
from sklearn.model_selection import train_test_split

DATA_ROOT = "data/BraTS2021_Training_Data"
SPLIT_DIR = "data/splits"

os.makedirs(SPLIT_DIR, exist_ok=True)

# --------------------------------------------------
# Collect all case IDs
# --------------------------------------------------
case_dirs = sorted(
    [os.path.basename(d)
     for d in glob(os.path.join(DATA_ROOT, "BraTS2021_*"))]
)

print(f"Total cases found: {len(case_dirs)}")

# --------------------------------------------------
# Train / Val split (80 / 20)
# --------------------------------------------------
train_cases, val_cases = train_test_split(
    case_dirs,
    test_size=0.2,
    random_state=42   # ✅ reproducible
)

# --------------------------------------------------
# Write CSVs
# --------------------------------------------------
def write_csv(path, cases):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id"])
        for c in cases:
            writer.writerow([c])

write_csv(os.path.join(SPLIT_DIR, "train.csv"), train_cases)
write_csv(os.path.join(SPLIT_DIR, "val.csv"), val_cases)

print("✅ train.csv and val.csv created")
print(f"Train cases: {len(train_cases)}")
print(f"Val cases: {len(val_cases)}")
