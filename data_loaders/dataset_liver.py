import os
import csv
from monai.data import Dataset, DataLoader, PersistentDataset
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    Resized, ScaleIntensityRanged, RandFlipd, RandRotate90d,
    RandAffined, EnsureTyped, Compose,
    RandAdjustContrastd, RandGaussianNoised,
    RandGaussianSmoothd, RandScaleIntensityd,
    Rand3DElasticd, CropForegroundd, RandCropByPosNegLabeld,
    RandCropByLabelClassesd,
)


def get_lung_from_csv(csv_path, data_root):
    data_dicts = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row["case_id"]
            data_dicts.append({
                "image": os.path.join(data_root, "imagesTr", f"{case_id}.nii.gz"),
                "label": os.path.join(data_root, "labelsTr", f"{case_id}.nii.gz"),
            })
    print(f"[INFO] Loaded {len(data_dicts)} cases from {csv_path}")
    return data_dicts


def get_persistent_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"], image_only=True),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear", "nearest"),
        ),
        CropForegroundd(                          # ✅ crops empty air around liver
            keys=["image", "label"],
            source_key="image",
        ),
        Resized(                                  # ✅ normalizes to fixed 128x128x128
            keys=["image", "label"],
            spatial_size=(128, 128, 128),
            mode=("trilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-17,
            a_max=201,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_train_aug_transforms():
    return Compose([
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(128, 128, 128),
            pos=5,          # ← higher tumour focus
            neg=1,
            num_samples=2,
            image_key="image",
        ),
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
            prob=0.3,            # ← reduce from 0.5
        ),
        # ← REMOVE Rand3DElasticd entirely — very slow
        RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.7, 1.5)),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
        RandGaussianSmoothd(
            keys=["image"], prob=0.2,
            sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0),
        ),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
    ])


def get_lung_dataloader_from_csv(
    csv_path,
    data_root,
    batch_size=1,
    train=True,
    num_workers=0,
    cache_dir="cache/lung",
):
    data_dicts = get_lung_from_csv(csv_path, data_root)

    persistent_ds = PersistentDataset(
        data=data_dicts,
        transform=get_persistent_transforms(),
        cache_dir=cache_dir,
    )

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