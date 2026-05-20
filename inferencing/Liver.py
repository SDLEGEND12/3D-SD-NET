"""
evaluate_liver_dice.py
─────────────────────────────────────────────────────────────────────────────
Batch Dice Score Evaluator for Liver Segmentation

Runs inference on ALL cases in imagesTr/ and computes Dice scores against
the corresponding masks in labelsTr/. Saves a CSV summary.

Usage:
    python evaluate_liver_dice.py

Edit the paths in the CONFIG block below before running.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import csv
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from scipy import ndimage

# ─────────────────────────────────────────────────────────────────────────────
# Project path  (so model.model is importable)
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    from model.model import Custom3DSegModel
    _MODEL_AVAILABLE = True
except ImportError:
    _MODEL_AVAILABLE = False
    print("⚠️  Could not import Custom3DSegModel – check PROJECT_ROOT.")


# ══════════════════════════════════════════════════════════════════════════════
# ★  EDIT THESE PATHS  ★
# ══════════════════════════════════════════════════════════════════════════════
IMAGES_DIR  = r"D:\MajorProject\3D SD-NET\data\Task03_Liver\imagesTr"
LABELS_DIR  = r"D:\MajorProject\3D SD-NET\data\Task03_Liver\labelsTr"
CHECKPOINT  = r"D:\MajorProject\3D SD-NET\outputs\checkpoints_liver\best_model.pt"

# Where to save the CSV results
OUTPUT_CSV  = r"D:\MajorProject\3D SD-NET\outputs\liver_dice_scores.csv"

SKIP_POSTPROC = False   # set True to skip CC-filtering / hole-fill
# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NUM_CLASSES  = 3
    EMBED_DIM    = 64
    IN_CHANNELS  = 1
    SPATIAL_SIZE = (128, 128, 128)

    CT_WIN_MIN = -17.0
    CT_WIN_MAX = 201.0

    SEGMENT_SETTINGS: Dict[int, dict] = {
        1: {"name": "Liver Parenchyma"},
        2: {"name": "Tumor / Lesion"},
    }

    # Post-processing
    CC_KEEP: Dict[int, int]     = {1: 1, 2: 10}
    MIN_VOXELS: Dict[int, int]  = {1: 1000, 2: 30}
    FILL_LIVER_HOLES            = True


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_ct(ct_path: Path, cfg: Config) -> torch.Tensor:
    from monai.transforms import (
        Compose, CropForegroundd, EnsureChannelFirstd, EnsureTyped,
        LoadImaged, Orientationd, Resized,
        ScaleIntensityRanged, Spacingd,
    )

    transforms = Compose([
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
        CropForegroundd(keys=["image"], source_key="image"),
        Resized(keys=["image"], spatial_size=cfg.SPATIAL_SIZE, mode="trilinear"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=cfg.CT_WIN_MIN, a_max=cfg.CT_WIN_MAX,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=["image"]),
    ])

    vol = transforms({"image": str(ct_path)})["image"]
    return vol.unsqueeze(0)   # (1, 1, D, H, W)


def load_ground_truth(gt_path: Path, cfg: Config) -> Optional[np.ndarray]:
    from monai.transforms import (
        Compose, EnsureChannelFirst, EnsureType,
        LoadImage, Orientation, Resize, Spacing,
    )

    transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        Spacing(pixdim=(1.5, 1.5, 1.5), mode="nearest"),
        Resize(spatial_size=cfg.SPATIAL_SIZE, mode="nearest"),
        EnsureType(),
    ])

    try:
        gt = transforms(str(gt_path)).squeeze().numpy().astype(np.int32)
        return gt
    except Exception as exc:
        warnings.warn(f"Could not load ground truth {gt_path.name}: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint: str, cfg: Config) -> "Custom3DSegModel":
    if not _MODEL_AVAILABLE:
        raise RuntimeError("Custom3DSegModel import failed – check PROJECT_ROOT.")

    model = Custom3DSegModel(
        in_channels=cfg.IN_CHANNELS,
        embed_dim=cfg.EMBED_DIM,
        n_classes=cfg.NUM_CLASSES,
        final_activation=None,
    ).to(cfg.DEVICE)

    state = torch.load(checkpoint, map_location=cfg.DEVICE, weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def run_inference(model, volume: torch.Tensor, cfg: Config) -> np.ndarray:
    volume = volume.to(cfg.DEVICE)
    logits = model(volume)
    logits = torch.clamp(logits, -10, 10)
    preds  = torch.argmax(torch.softmax(logits, dim=1), dim=1)
    if cfg.DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return preds[0].cpu().numpy().astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def postprocess(pred: np.ndarray, cfg: Config) -> np.ndarray:
    cleaned = np.zeros_like(pred)
    struct  = ndimage.generate_binary_structure(3, 2)

    for label, max_keep in cfg.CC_KEEP.items():
        mask = (pred == label)
        if mask.sum() == 0:
            continue
        cc_map, n_cc = ndimage.label(mask, structure=struct)
        sizes = ndimage.sum(mask, cc_map, range(1, n_cc + 1))
        order = np.argsort(sizes)[::-1]
        kept = 0
        for idx in order:
            if kept >= max_keep:
                break
            if int(sizes[idx]) < cfg.MIN_VOXELS.get(label, 0):
                break
            cleaned[cc_map == (idx + 1)] = label
            kept += 1

    if cfg.FILL_LIVER_HOLES:
        liver_mask = (cleaned == 1)
        if liver_mask.sum() > 0:
            filled    = ndimage.binary_fill_holes(liver_mask)
            new_liver = filled & (cleaned == 0)
            cleaned[new_liver] = 1

    tumor_mask = (cleaned == 2)
    if tumor_mask.sum() > 0:
        liver_dilated = ndimage.binary_dilation(
            (cleaned == 1), structure=struct, iterations=3
        )
        outside = tumor_mask & ~liver_dilated
        cleaned[outside] = 0

    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def _dice(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    p = (pred == label)
    g = (gt   == label)
    inter = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    return 1.0 if denom == 0 else float(2 * inter / denom)


def compute_dice_scores(pred: np.ndarray, gt: np.ndarray, cfg: Config) -> Dict:
    scores = {label: _dice(pred, gt, label) for label in cfg.SEGMENT_SETTINGS}
    p_wl   = np.isin(pred, [1, 2])
    g_wl   = np.isin(gt,   [1, 2])
    inter  = np.logical_and(p_wl, g_wl).sum()
    denom  = p_wl.sum() + g_wl.sum()
    scores["whole_liver"] = float(2 * inter / denom) if denom > 0 else 1.0
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def find_pairs(images_dir: Path, labels_dir: Path):
    """
    Matches imagesTr/liver_XX.nii.gz  ->  labelsTr/liver_XX.nii.gz
    Returns sorted list of (image_path, label_path) tuples.
    Warns (and skips) if a label is missing.
    """
    pairs = []
    image_files = sorted(images_dir.glob("liver_*.nii.gz"))

    if not image_files:
        # Try without the 'liver_' prefix (some datasets use plain numbering)
        image_files = sorted(images_dir.glob("*.nii.gz"))

    for img_path in image_files:
        lbl_path = labels_dir / img_path.name
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
        else:
            print(f"  ⚠️  No label found for {img_path.name} – skipping")

    return pairs


def print_summary(results: list) -> None:
    if not results:
        print("No results to summarise.")
        return

    liver_scores = [r["dice_liver"] for r in results if r["dice_liver"] is not None]
    tumor_scores = [r["dice_tumor"] for r in results if r["dice_tumor"] is not None]
    whole_scores = [r["dice_whole"] for r in results if r["dice_whole"] is not None]

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Cases evaluated      : {len(results)}")

    def _stats(vals, label):
        if not vals:
            print(f"  {label:<30}: N/A")
            return
        arr = np.array(vals)
        print(f"  {label:<30}: "
              f"mean={arr.mean():.4f}  "
              f"std={arr.std():.4f}  "
              f"min={arr.min():.4f}  "
              f"max={arr.max():.4f}")

    _stats(liver_scores, "Liver Parenchyma Dice")
    _stats(tumor_scores, "Tumor / Lesion Dice")
    _stats(whole_scores, "Whole Liver Dice")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    cfg = Config()

    images_dir = Path(IMAGES_DIR)
    labels_dir = Path(LABELS_DIR)
    output_csv = Path(OUTPUT_CSV)

    # Validate directories
    for d, name in [(images_dir, "IMAGES_DIR"), (labels_dir, "LABELS_DIR")]:
        if not d.exists():
            print(f"❌ {name} not found: {d}")
            sys.exit(1)

    print("=" * 65)
    print("LIVER BATCH DICE EVALUATOR")
    print("=" * 65)
    print(f"  Device     : {cfg.DEVICE}")
    print(f"  Images dir : {images_dir}")
    print(f"  Labels dir : {labels_dir}")
    print(f"  Checkpoint : {CHECKPOINT}")
    print(f"  Output CSV : {output_csv}")
    print("=" * 65)

    # Load model once
    try:
        model = load_model(CHECKPOINT, cfg)
        print(f"✅ Model loaded\n")
    except Exception as exc:
        print(f"❌ {exc}")
        sys.exit(1)

    # Discover all image/label pairs
    pairs = find_pairs(images_dir, labels_dir)
    if not pairs:
        print("❌ No matching image/label pairs found. Check your directory paths.")
        sys.exit(1)
    print(f"Found {len(pairs)} case(s) to evaluate\n")

    results = []

    for i, (img_path, lbl_path) in enumerate(pairs, 1):
        case_name = img_path.stem.replace(".nii", "")
        print(f"[{i:>3}/{len(pairs)}]  {case_name}")

        row = {
            "case":        case_name,
            "dice_liver":  None,
            "dice_tumor":  None,
            "dice_whole":  None,
            "status":      "ok",
        }

        try:
            # Preprocess
            volume = preprocess_ct(img_path, cfg)

            # Inference
            pred_raw = run_inference(model, volume, cfg)

            # Post-process
            pred = postprocess(pred_raw, cfg) if not SKIP_POSTPROC else pred_raw

            # Load GT and compute Dice
            gt = load_ground_truth(lbl_path, cfg)
            if gt is not None:
                scores = compute_dice_scores(pred, gt, cfg)
                row["dice_liver"] = scores[1]
                row["dice_tumor"] = scores[2]
                row["dice_whole"] = scores["whole_liver"]
                print(f"         Liver={scores[1]:.4f}  "
                      f"Tumor={scores[2]:.4f}  "
                      f"Whole={scores['whole_liver']:.4f}")
            else:
                row["status"] = "gt_load_failed"
                print(f"         ⚠️  Ground truth load failed")

        except Exception as exc:
            row["status"] = f"error: {exc}"
            print(f"         ❌ {exc}")

        results.append(row)

    # Print summary statistics
    print_summary(results)

    # Save CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case", "dice_liver", "dice_tumor", "dice_whole", "status"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n💾 Results saved → {output_csv}")
    print("✅ Done!")


if __name__ == "__main__":
    main()