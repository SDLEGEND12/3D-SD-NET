import os
import csv
from glob import glob
from sklearn.model_selection import train_test_split

DATA_ROOT  = "data/Task03_Liver"
SPLIT_DIR = "data/splits"

os.makedirs(SPLIT_DIR, exist_ok=True)

# Collect case IDs from imagesTr (strip leading underscore)
# e.g. _lung_001.nii.gz → lung_001
case_files = sorted(glob(os.path.join(DATA_ROOT, "imagesTr", "liver_*.nii.gz")))
case_ids   = [os.path.basename(f).replace(".nii.gz", "") for f in case_files]

print(f"Total cases found: {len(case_ids)}")

train_cases, val_cases = train_test_split(case_ids, test_size=0.2, random_state=42)

def write_csv(path, cases):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id"])
        for c in cases:
            writer.writerow([c])

write_csv(os.path.join(SPLIT_DIR, "train_liver.csv"), train_cases)
write_csv(os.path.join(SPLIT_DIR, "val_liver.csv"),   val_cases)

print(f"✅ Splits created | Train: {len(train_cases)} | Val: {len(val_cases)}")