import os
import torch
from torch.utils.data import Dataset, DataLoader

# ── PyTorch 2.6 compatibility ──
from monai.data.meta_tensor import MetaTensor
torch.serialization.add_safe_globals([MetaTensor])

from monai.data import PersistentDataset, list_data_collate
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    SpatialPadd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    EnsureTyped,
    ToTensord,
    Rand3DElasticd,
    RandCropByLabelClassesd,
    NormalizeIntensityd,
    AsDiscreted,
)


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
KIDNEY_SPACING = (1.5, 1.5, 1.5)   # mm isotropic — safe for 4GB VRAM
ROI_SIZE       = (128, 128, 128)       # fits 4GB with base_ch=24; use (64,64,64) if OOM
NUM_SAMPLES    = 2                  # patches per volume per step
NUM_CLASSES    = 3                  # 0=background, 1=kidney, 2=tumour

# CT HU window for kidney/tumour — standard abdomen soft tissue window
HU_MIN = -200
HU_MAX =  300


# ─────────────────────────────────────────────
#  DATA LIST BUILDER
# ─────────────────────────────────────────────
def _build_datalist(data_root: str) -> list:
    """
    Walks data/Kidney/data/case_XXXXX/ folders.
    Expects imaging.nii.gz and segmentation.nii.gz in each case folder.

    Directory structure expected:
        data/
          Kidney/
            data/
              case_00000/
                imaging.nii.gz
                segmentation.nii.gz
              case_00001/
                ...
    """
    kidney_dir = os.path.join(data_root, "Kidney", "data")

    if not os.path.isdir(kidney_dir):
        raise RuntimeError(
            f"Expected Kidney/data directory not found at: {kidney_dir}"
        )

    data = []
    for case_name in sorted(os.listdir(kidney_dir)):
        case_dir = os.path.join(kidney_dir, case_name)

        if not os.path.isdir(case_dir) or not case_name.startswith("case_"):
            continue

        img_path   = os.path.join(case_dir, "imaging.nii.gz")
        label_path = os.path.join(case_dir, "segmentation.nii.gz")

        if not os.path.exists(img_path):
            print(f"[WARN] Missing imaging.nii.gz for {case_name}, skipping.")
            continue
        if not os.path.exists(label_path):
            print(f"[WARN] Missing segmentation.nii.gz for {case_name}, skipping.")
            continue

        data.append({"image": img_path, "label": label_path})

    return data


# ─────────────────────────────────────────────
#  TRANSFORMS
# ─────────────────────────────────────────────
def get_cache_transform() -> Compose:
    """
    Deterministic heavy transforms — result cached to disk via PersistentDataset.
    Runs once per volume; never re-runs after cache is warm.

    Key decisions:
    - HU windowing [-200, 300]: covers kidney parenchyma and tumour well;
      clips irrelevant bone/air extremes that hurt normalisation.
    - No CropForegroundd — peripheral tumours can sit near the image boundary
      and aggressive cropping risks removing them entirely.
    - No random ops here — cache must be deterministic.
    """
    return Compose([
        LoadImaged(keys=["image", "label"], image_only=True),
        EnsureChannelFirstd(keys=["image", "label"]),

        # Canonical orientation first — before resampling
        Orientationd(keys=["image", "label"], axcodes="RAS"),

        # Resample to isotropic 1.5mm — handles the wide voxel spacing
        # variation in your dataset (0.32mm³ to 3.81mm³ voxel volumes)
        Spacingd(
            keys=["image", "label"],
            pixdim=KIDNEY_SPACING,
            mode=("bilinear", "nearest"),   # bilinear for image, nearest for labels
        ),

        # HU windowing — standard abdomen soft tissue
        # Clips to [-200, 300] then scales to [0.0, 1.0]
        ScaleIntensityRanged(
            keys=["image"],
            a_min=HU_MIN,
            a_max=HU_MAX,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),

        # Guarantee minimum spatial size for patch sampling
        SpatialPadd(keys=["image", "label"], spatial_size=ROI_SIZE),

        EnsureTyped(keys=["image", "label"]),
    ])


def get_train_post_transform() -> Compose:
    """
    Random augmentations — applied live after cache load (not cached).

    Patch sampling strategy:
        ratios=[0, 1, 8] → background:kidney:tumour = 0:1:8
        Tumour is heavily upsampled because it's a tiny fraction of voxels
        (as low as 1.85 cm³ in your dataset). Without this, the model
        sees almost no tumour patches and converges to predicting kidney only.

    Augmentation choices:
        - Flips + Rotate90: free augmentation, always beneficial for 3D CT
        - Rand3DElastic: simulates organ deformation — important for kidney
          shape variability across patients
        - Intensity shifts/scale: simulate scanner variability
        - Gaussian noise/smooth: simulate reconstruction kernel differences
        - No gamma augmentation — HU-windowed CT is already well-normalised
    """
    return Compose([
        # Tumour-focused patch sampling
        # ratios: [background, kidney, tumour]
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=ROI_SIZE,
            ratios=[0, 1, 16],       # never sample background-only patches
            num_classes=NUM_CLASSES,
            num_samples=NUM_SAMPLES,
            image_key="image",
            image_threshold=0,
        ),

        # ── Spatial augmentations ──
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),

        # Elastic deformation — simulates soft tissue variability
        Rand3DElasticd(
            keys=["image", "label"],
            sigma_range=(5, 8),
            magnitude_range=(50, 150),  # lighter than pancreas — kidney is stiffer
            prob=0.15,
            mode=("bilinear", "nearest"),
        ),

        # ── Intensity augmentations ──
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandGaussianNoised(keys=["image"],  std=0.01,    prob=0.3),
        RandGaussianSmoothd(
            keys=["image"],
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0),
            prob=0.2,
        ),

        ToTensord(keys=["image", "label"]),
    ])


def get_val_post_transform() -> Compose:
    """No augmentation — deterministic only."""
    return Compose([
        ToTensord(keys=["image", "label"]),
    ])


# ─────────────────────────────────────────────
#  WRAPPER DATASETS
# ─────────────────────────────────────────────
class AugmentedDataset(Dataset):
    """
    Wraps PersistentDataset with live augmentation.
    Flattens the patch list from RandCropByLabelClassesd into individual samples.

    index_map: list of (base_idx, patch_idx) pairs
        total length = len(base_dataset) * NUM_SAMPLES
    """

    def __init__(self, base_dataset, post_transform):
        self.base      = base_dataset
        self.post_tf   = post_transform
        self.index_map = [
            (bi, pi)
            for bi in range(len(base_dataset))
            for pi in range(NUM_SAMPLES)
        ]

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        base_idx, patch_idx = self.index_map[idx]
        item    = self.base[base_idx]
        patches = self.post_tf(item)
        if isinstance(patches, list):
            return patches[patch_idx % len(patches)]
        return patches


class ValDataset(Dataset):
    """Full-volume validation — no patch sampling, no augmentation."""

    def __init__(self, base_dataset, post_transform):
        self.base    = base_dataset
        self.post_tf = post_transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.post_tf(self.base[idx])


# ─────────────────────────────────────────────
#  DATASET + LOADER FACTORY
# ─────────────────────────────────────────────
def get_kidney_datasets(
    data_root: str,
    cache_dir: str,
    val_split: float = 0.2,
) -> tuple:
    """
    Args:
        data_root:  path to your top-level data/ folder
                    (the one containing the Kidney/ subfolder)
        cache_dir:  path to persistent cache directory (needs ~50–100GB free)
        val_split:  fraction of cases held out for validation (default 20%)

    Returns:
        (train_dataset, val_dataset)

    Split is deterministic (no shuffle) so cache keys stay stable across runs.
    With 300 cases and val_split=0.2: 240 train, 60 val.
    """
    all_files = _build_datalist(data_root)

    if not all_files:
        raise RuntimeError(f"No valid image/label pairs found under {data_root}")

    n_val       = max(1, int(len(all_files) * val_split))
    val_files   = all_files[:n_val]
    train_files = all_files[n_val:]

    print(f"[Kidney] Total cases      : {len(all_files)}")
    print(f"[Kidney] Train cases      : {len(train_files)}")
    print(f"[Kidney] Val cases        : {len(val_files)}")
    print(f"[Kidney] Spacing          : {KIDNEY_SPACING} mm")
    print(f"[Kidney] ROI size         : {ROI_SIZE}")
    print(f"[Kidney] Patches/vol      : {NUM_SAMPLES}")
    print(f"[Kidney] Classes          : {NUM_CLASSES}  (0=bg, 1=kidney, 2=tumour)")
    print(f"[Kidney] Tumour sampling  : 8x upsampled")

    train_cache = os.path.join(cache_dir, "kidney_train")
    val_cache   = os.path.join(cache_dir, "kidney_val")
    os.makedirs(train_cache, exist_ok=True)
    os.makedirs(val_cache,   exist_ok=True)

    train_base = PersistentDataset(
        data=train_files,
        transform=get_cache_transform(),
        cache_dir=train_cache,
    )
    val_base = PersistentDataset(
        data=val_files,
        transform=get_cache_transform(),
        cache_dir=val_cache,
    )

    return (
        AugmentedDataset(train_base, get_train_post_transform()),
        ValDataset(val_base, get_val_post_transform()),
    )


def get_kidney_loaders(
    data_root:   str,
    cache_dir:   str,
    batch_size:  int   = 1,     # 1 is safest for 4GB VRAM with 96³ patches
    num_workers: int   = 2,
    val_split:   float = 0.2,
) -> tuple:
    """
    Convenience wrapper — returns (train_loader, val_loader).

    batch_size=1 is intentional for 4GB VRAM.
    If you have more VRAM headroom, try batch_size=2 with ROI_SIZE=(64,64,64).
    """
    train_ds, val_ds = get_kidney_datasets(data_root, cache_dir, val_split)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,           # always 1 for full-volume val inference
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader