import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_loaders.dataset_brats import get_brats_dataloader_from_csv


DATA_ROOT = "data/BraTS2021_Training_Data"
TRAIN_CSV = "data/splits/train.csv"
VAL_CSV   = "data/splits/val.csv"


def main():
    # --------------------------------------------------
    # BUILD TRAIN CACHE
    # --------------------------------------------------
    print("🔄 Building TRAIN cache...")
    train_loader = get_brats_dataloader_from_csv(
        csv_path=TRAIN_CSV,
        data_root=DATA_ROOT,
        batch_size=1,
        train=False,          # ❗ NO AUGMENTATION during caching
        num_workers=0,
        cache_dir="cache/brats2021/train"
    )

    for i, _ in enumerate(train_loader):
        if i % 25 == 0:
            print(f"Train cached {i} / {len(train_loader.dataset)} cases")

    print("✅ Train cache complete\n")

    # --------------------------------------------------
    # BUILD VAL CACHE
    # --------------------------------------------------
    print("🔄 Building VAL cache...")
    val_loader = get_brats_dataloader_from_csv(
        csv_path=VAL_CSV,
        data_root=DATA_ROOT,
        batch_size=1,
        train=False,
        num_workers=0,
        cache_dir="cache/brats2021/val"
    )

    for i, _ in enumerate(val_loader):
        if i % 25 == 0:
            print(f"Val cached {i} / {len(val_loader.dataset)} cases")

    print("✅ Val cache complete")


if __name__ == "__main__":
    main()
