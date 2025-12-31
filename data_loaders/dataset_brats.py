import os
import csv
from monai.data import Dataset, DataLoader, PersistentDataset
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    CropForegroundd,
    Resized,
    ScaleIntensityRanged,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    EnsureTyped,
    Compose,
    Rand3DElasticd,        
    RandAdjustContrastd,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
)
from monai.transforms import MapTransform
class RemapBraTSLabels(MapTransform):
    """
    BraTS labels: {0,1,2,4} → {0,1,2,3}
    """
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = d[key].clone()
            d[key][d[key] == 4] = 3
        return d

# ------------------------------------------------------------------
# STEP 1: LOAD CASES FROM CSV
# ------------------------------------------------------------------

def get_brats_from_csv(csv_path, data_root):
    """
    CSV format:
        case_id
        BraTS2021_00000
        BraTS2021_00001
        ...

    Returns MONAI-style data dicts.
    """

    data_dicts = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row["case_id"]
            case_dir = os.path.join(data_root, case_id)

            data_dicts.append({
                "image": [
                    os.path.join(case_dir, f"{case_id}_flair.nii.gz"),
                    os.path.join(case_dir, f"{case_id}_t1.nii.gz"),
                    os.path.join(case_dir, f"{case_id}_t1ce.nii.gz"),
                    os.path.join(case_dir, f"{case_id}_t2.nii.gz"),
                ],
                "label": os.path.join(case_dir, f"{case_id}_seg.nii.gz"),
            })

    print(f"[INFO] Loaded {len(data_dicts)} cases from {csv_path}")
    return data_dicts


# ------------------------------------------------------------------
# STEP 2: DETERMINISTIC (CACHED) PREPROCESSING
# ------------------------------------------------------------------

def get_persistent_transforms():
    """
    Runs ONCE per case and is cached to disk.
    """
    return Compose([
        LoadImaged(keys=["image", "label"], image_only=True),
        EnsureChannelFirstd(keys=["image", "label"]),

        Orientationd(keys=["image", "label"], axcodes="RAS"),

        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),


        Resized(
            keys=["image", "label"],
            spatial_size=(128, 128, 128),
            mode=("trilinear", "nearest"),
        ),
        RemapBraTSLabels(keys="label"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=3000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),

        EnsureTyped(keys=["image", "label"]),
    ])


# ------------------------------------------------------------------
# STEP 3: RANDOM AUGMENTATION (TRAIN ONLY, NOT CACHED)
# ------------------------------------------------------------------

def get_train_aug_transforms():
    return Compose([
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),

        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),

        RandAffined(
            keys=["image", "label"],
            rotate_range=(0.2, 0.2, 0.2),
            scale_range=(0.1, 0.1, 0.1),
            translate_range=(10, 10, 10),
            mode=("bilinear", "nearest"),
            prob=0.5,
        ),
        # ===================================
        # INTENSITY AUGMENTATIONS
        # ===================================
        # ✅ NEW: Gamma intensity shift (contrast variations)
        RandAdjustContrastd(
            keys=["image"],
            prob=0.2,
            gamma=(0.7, 1.5),  # Range for gamma correction
        ),

        # Gaussian noise
        RandGaussianNoised(
            keys=["image"],
            prob=0.2,
            mean=0.0,
            std=0.1,
        ),

        # Gaussian smoothing (blur)
        RandGaussianSmoothd(
            keys=["image"],
            prob=0.2,
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0),
        ),

        # Intensity scaling
        RandScaleIntensityd(
            keys=["image"],
            factors=0.1,
            prob=0.3,
        ),
    ])


# ------------------------------------------------------------------
# STEP 4: DATALOADER (CSV + CACHE + AUGMENT-SAFE)
# ------------------------------------------------------------------

def get_brats_dataloader_from_csv(
    csv_path,
    data_root,
    batch_size=1,
    train=True,
    num_workers=0,
    cache_dir="cache/brats2021",
):
    """
    Returns a DataLoader that:
      - uses CSV-defined split
      - caches deterministic preprocessing
      - applies augmentation only for training
    """

    data_dicts = get_brats_from_csv(csv_path, data_root)

    # Persistent (cached) dataset
    persistent_ds = PersistentDataset(
        data=data_dicts,
        transform=get_persistent_transforms(),
        cache_dir=cache_dir,
    )

    # Apply random augmentation for training only
    if train:
        final_ds = Dataset(
            data=persistent_ds,
            transform=get_train_aug_transforms(),
        )
    else:
        final_ds = persistent_ds

    loader = DataLoader(
        final_ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
