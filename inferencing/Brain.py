"""
evaluate_brats_dice.py
─────────────────────────────────────────────────────────────────────────────
Batch Dice Score Evaluator  –  BraTS 2021 folder structure

Expected layout:
    BraTS2021_Training_Data/
        BraTS2021_00000/
            BraTS2021_00000_flair.nii.gz
            BraTS2021_00000_t1.nii.gz
            BraTS2021_00000_t1ce.nii.gz
            BraTS2021_00000_t2.nii.gz
            BraTS2021_00000_seg.nii.gz   ← ground truth
        BraTS2021_00001/
            ...

BraTS label convention:
    0 → Background
    1 → Necrotic Core  (NCR)
    2 → Edema          (ED)
    3 → Enhancing Tumor (ET)

Reported Dice metrics (standard BraTS):
    ET  – Enhancing Tumor         label 3
    TC  – Tumor Core              labels {1, 3}
    WT  – Whole Tumor             labels {1, 2, 3}

Usage:
    python inferencing/evaluate_brats_dice.py
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
DATA_DIR   = r"D:\MajorProject\3D SD-NET\data\BraTS2021_Training_Data"
CHECKPOINT = r"D:\MajorProject\3D SD-NET\outputs\checkpoints\best_model(65.03)(StrongAug+SEB).pt"
OUTPUT_CSV = r"D:\MajorProject\3D SD-NET\outputs\brats_dice_scores.csv"

SKIP_POSTPROC = False   # set True to skip connected-component filtering
# ══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NUM_CLASSES  = 4        # 0=bg, 1=NCR, 2=ED, 3=ET
    EMBED_DIM    = 64
    IN_CHANNELS  = 4        # flair, t1, t1ce, t2  (stacked as channels)
    SPATIAL_SIZE = (128, 128, 128)

    # MRI modality suffixes in the order they are stacked as channels
    MODALITY_SUFFIXES: List[str] = ["flair", "t1", "t1ce", "t2"]

    # Segmentation file suffix
    SEG_SUFFIX: str = "seg"

    # Per-label names (for logging)
    LABEL_NAMES: Dict[int, str] = {
        1: "Necrotic Core (NCR)",
        2: "Edema (ED)",
        3: "Enhancing Tumor (ET)",
    }

    # Post-processing: keep N largest components per label
    CC_KEEP: Dict[int, int]    = {1: 5, 2: 5, 3: 5}
    MIN_VOXELS: Dict[int, int] = {1: 20, 2: 20, 3: 20}


# ─────────────────────────────────────────────────────────────────────────────
# CASE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────
def find_cases(data_dir: Path, cfg: Config) -> List[Tuple[str, List[Path], Path]]:
    """
    Walks data_dir and returns a list of:
        (case_name, [flair, t1, t1ce, t2 paths], seg_path)

    Skips any case where a modality or the seg file is missing.
    """
    cases = []
    case_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    for case_dir in case_dirs:
        case_name = case_dir.name

        modality_paths = []
        ok = True
        for suffix in cfg.MODALITY_SUFFIXES:
            # Try both .nii.gz and .nii
            p = case_dir / f"{case_name}_{suffix}.nii.gz"
            if not p.exists():
                p = case_dir / f"{case_name}_{suffix}.nii"
            if not p.exists():
                print(f"  ⚠️  Missing modality '{suffix}' for {case_name} – skipping")
                ok = False
                break
            modality_paths.append(p)

        seg_path = case_dir / f"{case_name}_{cfg.SEG_SUFFIX}.nii.gz"
        if not seg_path.exists():
            seg_path = case_dir / f"{case_name}_{cfg.SEG_SUFFIX}.nii"
        if not seg_path.exists():
            print(f"  ⚠️  Missing seg file for {case_name} – skipping")
            ok = False

        if ok:
            cases.append((case_name, modality_paths, seg_path))

    return cases


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_brats(modality_paths: List[Path], cfg: Config) -> torch.Tensor:
    """
    For each modality:
        Load -> EnsureChannelFirst -> Orientation(RAS) -> Spacing(1mm iso)
        -> Resize(128^3) -> NormalizeIntensity (zero-mean, unit-std, non-zero)
    Stack all 4 modalities along channel dim -> (1, 4, D, H, W)
    """
    from monai.transforms import (
        Compose, EnsureChannelFirst, EnsureType,
        LoadImage, NormalizeIntensity, Orientation, Resize, Spacing,
    )

    single_tfm = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        Resize(spatial_size=cfg.SPATIAL_SIZE, mode="trilinear"),
        NormalizeIntensity(nonzero=True, channel_wise=True),
        EnsureType(),
    ])

    channels = []
    for p in modality_paths:
        vol = single_tfm(str(p))   # (1, D, H, W)
        channels.append(vol)

    stacked = torch.cat(channels, dim=0)   # (4, D, H, W)
    return stacked.unsqueeze(0)            # (1, 4, D, H, W)


def load_ground_truth(seg_path: Path, cfg: Config) -> Optional[np.ndarray]:
    from monai.transforms import (
        Compose, EnsureChannelFirst, EnsureType,
        LoadImage, Orientation, Resize, Spacing,
    )

    transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        Spacing(pixdim=(1.0, 1.0, 1.0), mode="nearest"),
        Resize(spatial_size=cfg.SPATIAL_SIZE, mode="nearest"),
        EnsureType(),
    ])

    try:
        gt = transforms(str(seg_path)).squeeze().numpy().astype(np.int32)
        # BraTS 2021 uses label 4 for ET in some versions – remap to 3
        gt[gt == 4] = 3
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
    return preds[0].cpu().numpy().astype(np.int32)   # (D, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def postprocess(pred: np.ndarray, cfg: Config) -> np.ndarray:
    """Keep the N largest connected components per label; discard tiny blobs."""
    cleaned = np.zeros_like(pred)
    struct  = ndimage.generate_binary_structure(3, 2)   # 18-connectivity

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

    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# METRICS  –  BraTS standard: ET, TC, WT
# ─────────────────────────────────────────────────────────────────────────────
def _dice_region(pred: np.ndarray, gt: np.ndarray,
                 labels: List[int]) -> float:
    """Binary Dice for a union of label indices."""
    p = np.isin(pred, labels)
    g = np.isin(gt,   labels)
    inter = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    return 1.0 if denom == 0 else float(2 * inter / denom)


def compute_dice_scores(pred: np.ndarray, gt: np.ndarray) -> Dict:
    return {
        # Individual labels
        "dice_ncr": _dice_region(pred, gt, [1]),   # Necrotic Core
        "dice_ed":  _dice_region(pred, gt, [2]),   # Edema
        "dice_et":  _dice_region(pred, gt, [3]),   # Enhancing Tumor
        # BraTS composite regions
        "dice_tc":  _dice_region(pred, gt, [1, 3]),       # Tumor Core
        "dice_wt":  _dice_region(pred, gt, [1, 2, 3]),    # Whole Tumor
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(results: list) -> None:
    if not results:
        print("No results to summarise.")
        return

    metrics = ["dice_ncr", "dice_ed", "dice_et", "dice_tc", "dice_wt"]
    labels  = {
        "dice_ncr": "NCR (label 1)",
        "dice_ed":  "ED  (label 2)",
        "dice_et":  "ET  (label 3)",
        "dice_tc":  "Tumor Core  (TC)",
        "dice_wt":  "Whole Tumor (WT)",
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Cases evaluated : {len(results)}")
    print()

    for key in metrics:
        vals = [r[key] for r in results if r.get(key) is not None]
        if not vals:
            print(f"  {labels[key]:<28}: N/A")
            continue
        arr = np.array(vals)
        print(f"  {labels[key]:<28}: "
              f"mean={arr.mean():.4f}  "
              f"std={arr.std():.4f}  "
              f"min={arr.min():.4f}  "
              f"max={arr.max():.4f}")

    print("=" * 70)


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

    print("=" * 70)
    print("BraTS BATCH DICE EVALUATOR")
    print("=" * 70)
    print(f"  Device     : {cfg.DEVICE}")
    print(f"  Data dir   : {data_dir}")
    print(f"  Checkpoint : {CHECKPOINT}")
    print(f"  Output CSV : {out_csv}")
    print(f"  Modalities : {cfg.MODALITY_SUFFIXES}  (IN_CHANNELS={cfg.IN_CHANNELS})")
    print("=" * 70)

    # Load model once
    try:
        model = load_model(CHECKPOINT, cfg)
        print("✅ Model loaded\n")
    except Exception as exc:
        print(f"❌ {exc}")
        sys.exit(1)

    # Discover cases
    cases = find_cases(data_dir, cfg)
    if not cases:
        print("❌ No valid cases found. Check DATA_DIR and folder structure.")
        sys.exit(1)
    print(f"Found {len(cases)} valid case(s)\n")

    results = []
    fieldnames = ["case", "dice_ncr", "dice_ed", "dice_et",
                  "dice_tc", "dice_wt", "status"]

    for i, (case_name, modality_paths, seg_path) in enumerate(cases, 1):
        print(f"[{i:>3}/{len(cases)}]  {case_name}")

        row = {k: None for k in fieldnames}
        row["case"]   = case_name
        row["status"] = "ok"

        try:
            # Preprocess  (stack 4 modalities)
            volume = preprocess_brats(modality_paths, cfg)

            # Inference
            pred_raw = run_inference(model, volume, cfg)

            # Post-process
            pred = postprocess(pred_raw, cfg) if not SKIP_POSTPROC else pred_raw

            # Load GT
            gt = load_ground_truth(seg_path, cfg)
            if gt is None:
                row["status"] = "gt_load_failed"
                print("         ⚠️  Ground truth load failed")
            else:
                scores = compute_dice_scores(pred, gt)
                row.update(scores)
                print(f"         NCR={scores['dice_ncr']:.4f}  "
                      f"ED={scores['dice_ed']:.4f}  "
                      f"ET={scores['dice_et']:.4f}  │  "
                      f"TC={scores['dice_tc']:.4f}  "
                      f"WT={scores['dice_wt']:.4f}")

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