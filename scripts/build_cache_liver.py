
import sys, os
import gc
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_loaders.dataset_liver import get_lung_dataloader_from_csv

DATA_ROOT = "data/Task03_Liver"
TRAIN_CSV = "data/splits/train_liver.csv"
VAL_CSV   = "data/splits/val_liver.csv"

def main():
    print("🔄 Building TRAIN cache...")
    train_loader = get_lung_dataloader_from_csv(
        csv_path=TRAIN_CSV, data_root=DATA_ROOT,
        batch_size=1, train=False,
        num_workers=0, cache_dir="cache/liver/train"
    )
    for i, batch in enumerate(train_loader):
        try:
            print(f"  Train cached {i+1} / {len(train_loader.dataset)}")
            del batch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ❌ Failed at {i}: {e}")
            import traceback
            traceback.print_exc()
            break
    print("✅ Train cache complete\n")

    print("🔄 Building VAL cache...")
    val_loader = get_lung_dataloader_from_csv(
        csv_path=VAL_CSV, data_root=DATA_ROOT,
        batch_size=1, train=False,
        num_workers=0, cache_dir="cache/liver/val"
    )
    for i, batch in enumerate(val_loader):
        try:
            print(f"  Val cached {i+1} / {len(val_loader.dataset)}")
            del batch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ❌ Failed at {i}: {e}")
            import traceback
            traceback.print_exc()
            break
    print("✅ Val cache complete")

if __name__ == "__main__":
    main()
