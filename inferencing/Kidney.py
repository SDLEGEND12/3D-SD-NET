"""
evaluate_kidney_dice.py
─────────────────────────────────────────────────────────────────────────────
Batch Dice Score Evaluator  –  KiTS (Kidney Tumor Segmentation) structure

Expected layout:
    Kidney/
        data/
            case_00000/
                imaging.nii.gz        ← CT scan
                segmentation.nii.gz   ← ground truth
            case_00001/
                ...

KiTS label convention:
    0 → Background
    1 → Kidney parenchyma
    2 → Kidney tumor

Reported Dice metrics:
    Kidney        label 1
    Tumor         label 2
    Composite     mean of (Kidney Dice, Tumor Dice)  ← official KiTS metric

Usage:
    python inferencing/evaluate_kidney_dice.py
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import csv
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import ndimage

# ─────────────────────────────────────────────────────────────────────────────
# Project root  (script lives in inferencing/, model/ is one level up)
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
DATA_DIR   = r"D:\MajorProject\3D SD-NET\data\Kidney\data"
CHECKPOINT = r"D:\MajorProject\3D SD-NET\outputs\kidney_transformer\best_model(0.2641).pth"
OUTPUT_CSV = r"D:\MajorProject\3D SD-NET\outputs\kidney_dice_scores.csv"

SKIP_POSTPROC = False   # set True to skip connected-component filtering
# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NUM_CLASSES  = 3        # 0=bg, 1=kidney, 2=tumor
    EMBED_DIM    = 32
    IN_CHANNELS  = 1        # single-channel CT

    SPATIAL_SIZE = (128, 128, 128)

    # KiTS CT window  (kidney protocol)  W=400 L=50  →  [-150, 250] HU
    CT_WIN_MIN = -150.0
    CT_WIN_MAX =  250.0

    LABEL_NAMES: Dict[int, str] = {
        1: "Kidney",
        2: "Tumor",
    }

    # Post-processing: connected-component filtering per label
    CC_KEEP: Dict[int, int]    = {1: 2, 2: 10}   # up to 2 kidneys, 10 tumor blobs
    MIN_VOXELS: Dict[int, int] = {1: 500, 2: 20}

    FILL_KIDNEY_HOLES = True   # binary fill-holes on kidney mask


# ─────────────────────────────────────────────────────────────────────────────
# CASE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────
def find_cases(data_dir: Path) -> List[Tuple[str, Path, Path]]:
    """
    Walks data_dir for subdirs named case_XXXXX.
    Returns sorted list of (case_name, imaging_path, seg_path).
    Skips cases with missing files.
    """
    cases = []
    case_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    for case_dir in case_dirs:
        case_name = case_dir.name

        img_path = case_dir / "imaging.nii.gz"
        seg_path = case_dir / "segmentation.nii.gz"

        missing = []
        if not img_path.exists():
            missing.append("imaging.nii.gz")
        if not seg_path.exists():
            missing.append("segmentation.nii.gz")

        if missing:
            print(f"  ⚠️  {case_name}: missing {', '.join(missing)} – skipping")
            continue

        cases.append((case_name, img_path, seg_path))

    return cases


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_ct(img_path: Path, cfg: Config) -> torch.Tensor:
    """
    MONAI pipeline – mirrors KiTS training:
      Load -> EnsureChannelFirst -> Orientation(RAS) -> Spacing(1.5mm iso)
      -> CropForeground -> Resize(128^3)
      -> ScaleIntensityRange([CT_WIN_MIN, CT_WIN_MAX] -> [0, 1])

    Returns tensor shape (1, 1, D, H, W)
    """
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

    vol = transforms({"image": str(img_path)})["image"]   # (1, D, H, W)
    return vol.unsqueeze(0)                                # (1, 1, D, H, W)


def load_ground_truth(seg_path: Path, cfg: Config) -> Optional[np.ndarray]:
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
        gt = transforms(str(seg_path)).squeeze().numpy().astype(np.int32)
        return gt
    except Exception as exc:
        warnings.warn(f"Could not load seg {seg_path.name}: {exc}")
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

    ckpt = torch.load(checkpoint, map_location=cfg.DEVICE, weights_only=False)

    # Unwrap whichever training-dict key was used to save the weights
    if isinstance(ckpt, dict):
        for key in ("model", "model_state", "state_dict", "net", "network"):
            if key in ckpt:
                ckpt = ckpt[key]
                print(f"  \u21b3 Loaded weights from checkpoint key: '{key}'")
                break

    model.load_state_dict(ckpt, strict=True)
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
    return preds[0].cpu().numpy().astype(np.int32)   # (D, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def postprocess(pred: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Step 1 – Keep N largest connected components per label.
              Kidney : up to 2  (left + right kidney)
              Tumor  : up to 10 (multifocal)
    Step 2 – Fill holes inside kidney mask.
    Step 3 – Clip tumor voxels to kidney+dilation region.
    """
    cleaned = np.zeros_like(pred)
    struct  = ndimage.generate_binary_structure(3, 2)   # 18-connectivity

    # Step 1
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
        print(f"    Label {label} ({cfg.LABEL_NAMES[label]}): "
              f"{n_cc} component(s) → kept {kept}")

    # Step 2 – hole-fill kidney
    if cfg.FILL_KIDNEY_HOLES:
        kidney_mask = (cleaned == 1)
        if kidney_mask.sum() > 0:
            filled     = ndimage.binary_fill_holes(kidney_mask)
            new_kidney = filled & (cleaned == 0)
            cleaned[new_kidney] = 1

    # Step 3 – clip tumor to kidney neighbourhood
    tumor_mask = (cleaned == 2)
    if tumor_mask.sum() > 0:
        kidney_dilated = ndimage.binary_dilation(
            (cleaned == 1), structure=struct, iterations=3
        )
        outside = tumor_mask & ~kidney_dilated
        if outside.sum() > 0:
            print(f"    Removed {int(outside.sum()):,} tumor voxels outside kidney")
            cleaned[outside] = 0

    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def _dice(pred: np.ndarray, gt: np.ndarray, labels: List[int]) -> float:
    p = np.isin(pred, labels)
    g = np.isin(gt,   labels)
    inter = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    return 1.0 if denom == 0 else float(2 * inter / denom)


def compute_dice_scores(pred: np.ndarray, gt: np.ndarray) -> Dict:
    dice_kidney = _dice(pred, gt, [1])
    dice_tumor  = _dice(pred, gt, [2])
    # Official KiTS composite = mean of the two
    composite   = (dice_kidney + dice_tumor) / 2.0
    return {
        "dice_kidney":    dice_kidney,
        "dice_tumor":     dice_tumor,
        "dice_composite": composite,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(results: list) -> None:
    if not results:
        print("No results to summarise.")
        return

    metrics = {
        "dice_kidney":    "Kidney (label 1)",
        "dice_tumor":     "Tumor  (label 2)",
        "dice_composite": "Composite (mean)",
    }

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Cases evaluated : {len(results)}")
    print()

    for key, label in metrics.items():
        vals = [r[key] for r in results if r.get(key) is not None]
        if not vals:
            print(f"  {label:<28}: N/A")
            continue
        arr = np.array(vals)
        print(f"  {label:<28}: "
              f"mean={arr.mean():.4f}  "
              f"std={arr.std():.4f}  "
              f"min={arr.min():.4f}  "
              f"max={arr.max():.4f}")

    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    cfg      = Config()
    data_dir = Path(DATA_DIR)
    out_csv  = Path(OUTPUT_CSV)

    if not data_dir.exists():
        print(f"❌ DATA_DIR not found: {data_dir}")
        sys.exit(1)

    print("=" * 65)
    print("KiTS KIDNEY BATCH DICE EVALUATOR")
    print("=" * 65)
    print(f"  Device     : {cfg.DEVICE}")
    print(f"  Data dir   : {data_dir}")
    print(f"  Checkpoint : {CHECKPOINT}")
    print(f"  Output CSV : {out_csv}")
    print("=" * 65)

    # Load model once
    try:
        model = load_model(CHECKPOINT, cfg)
        print("✅ Model loaded\n")
    except Exception as exc:
        print(f"❌ {exc}")
        sys.exit(1)

    # Discover all case_XXXXX folders
    cases = find_cases(data_dir)
    if not cases:
        print("❌ No valid cases found. Check DATA_DIR and folder structure.")
        sys.exit(1)
    print(f"Found {len(cases)} valid case(s)\n")

    results    = []
    fieldnames = ["case", "dice_kidney", "dice_tumor", "dice_composite", "status"]

    for i, (case_name, img_path, seg_path) in enumerate(cases, 1):
        print(f"[{i:>3}/{len(cases)}]  {case_name}")

        row = {k: None for k in fieldnames}
        row["case"]   = case_name
        row["status"] = "ok"

        try:
            # Preprocess
            volume = preprocess_ct(img_path, cfg)

            # Inference
            pred_raw = run_inference(model, volume, cfg)

            # Post-process
            pred = postprocess(pred_raw, cfg) if not SKIP_POSTPROC else pred_raw

            # Load GT and compute Dice
            gt = load_ground_truth(seg_path, cfg)
            if gt is None:
                row["status"] = "gt_load_failed"
                print("         ⚠️  Ground truth load failed")
            else:
                scores = compute_dice_scores(pred, gt)
                row.update(scores)
                print(f"         Kidney={scores['dice_kidney']:.4f}  "
                      f"Tumor={scores['dice_tumor']:.4f}  "
                      f"Composite={scores['dice_composite']:.4f}")

        except Exception as exc:
            row["status"] = f"error: {exc}"
            print(f"         ❌ {exc}")

        results.append(row)

    # Summary
    print_summary(results)

    # Save CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n💾 Results saved → {out_csv}")
    print("✅ Done!")


if __name__ == "__main__":
    main()