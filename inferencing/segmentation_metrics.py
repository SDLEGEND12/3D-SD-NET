"""
segmentation_metrics.py
═══════════════════════════════════════════════════════════════════════════════
Comprehensive Segmentation Metrics for Medical Image Segmentation Projects.

Computes per-class and aggregate:
  • Dice Similarity Coefficient (DSC)
  • Hausdorff Distance 95th percentile (HD95)
  • Intersection over Union / Jaccard Index (IoU)
  • Sensitivity (Recall)
  • Specificity
  • Precision
  • F1-score  (= DSC for binary, same formula)
  • Predicted & GT tumor volume (cm³)
  • Volume error (absolute + percentage)

Works for all three organs in your project:
  ─ Brain   : labels 0-3   (0=bg, 1=necrotic, 2=edema, 3=enhancing)
  ─ Kidney  : labels 0-2   (0=bg, 1=kidney,   2=tumour)
  ─ Liver   : labels 0-2   (0=bg, 1=liver,    2=tumor/lesion)

Usage (standalone):
    from segmentation_metrics import SegmentationMetrics

    # voxel_spacing_mm: (dz, dy, dx) in mm — read from NIfTI header
    metrics = SegmentationMetrics(
        pred=pred_np,
        gt=gt_np,
        labels={1: "Kidney", 2: "Tumour"},
        voxel_spacing_mm=(1.5, 1.5, 1.5),
    )
    metrics.compute_all()
    metrics.print_report()
    df = metrics.to_dataframe()   # pandas DataFrame for tables/CSV

Requirements:
    pip install numpy scipy pandas scikit-image
    (surface_distance is optional — falls back to scipy if unavailable)
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage

# ── Optional: Google's surface_distance library (more accurate HD95) ──────────
try:
    import surface_distance as surf_dist_lib
    _HAS_SURFACE_DIST = True
except ImportError:
    _HAS_SURFACE_DIST = False
    warnings.warn(
        "surface_distance library not found. HD95 will use scipy-based fallback.\n"
        "For exact results: pip install surface-distance",
        stacklevel=2,
    )


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASS — holds all metrics for ONE class
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClassMetrics:
    label:       int
    name:        str
    dice:        float = float("nan")
    iou:         float = float("nan")
    sensitivity: float = float("nan")   # recall / true positive rate
    specificity: float = float("nan")   # true negative rate
    precision:   float = float("nan")   # positive predictive value
    f1:          float = float("nan")   # same as dice for binary
    hd95:        float = float("nan")   # mm
    hd_max:      float = float("nan")   # max Hausdorff distance (mm)
    pred_volume_cm3: float = float("nan")
    gt_volume_cm3:   float = float("nan")
    vol_error_cm3:   float = float("nan")
    vol_error_pct:   float = float("nan")

    # Confusion matrix elements (voxel-level)
    TP: int = 0
    FP: int = 0
    FN: int = 0
    TN: int = 0


# ══════════════════════════════════════════════════════════════════════════════
# CORE METRIC FUNCTIONS  (all accept binary numpy arrays)
# ══════════════════════════════════════════════════════════════════════════════

def _confusion(pred_bin: np.ndarray, gt_bin: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (TP, FP, FN, TN) for two binary masks."""
    TP = int(np.sum(pred_bin & gt_bin))
    FP = int(np.sum(pred_bin & ~gt_bin))
    FN = int(np.sum(~pred_bin & gt_bin))
    TN = int(np.sum(~pred_bin & ~gt_bin))
    return TP, FP, FN, TN


def compute_dice(TP: int, FP: int, FN: int) -> float:
    denom = 2 * TP + FP + FN
    return (2 * TP) / denom if denom > 0 else 1.0


def compute_iou(TP: int, FP: int, FN: int) -> float:
    denom = TP + FP + FN
    return TP / denom if denom > 0 else 1.0


def compute_sensitivity(TP: int, FN: int) -> float:
    """Sensitivity = TP / (TP + FN)  — how much of GT is captured."""
    denom = TP + FN
    return TP / denom if denom > 0 else 1.0


def compute_specificity(TN: int, FP: int) -> float:
    """Specificity = TN / (TN + FP)  — how much background is correctly rejected."""
    denom = TN + FP
    return TN / denom if denom > 0 else 1.0


def compute_precision(TP: int, FP: int) -> float:
    """Precision = TP / (TP + FP)  — of predicted positives, how many are real."""
    denom = TP + FP
    return TP / denom if denom > 0 else 1.0


def compute_f1(precision: float, sensitivity: float) -> float:
    """F1 = harmonic mean of precision and recall (= Dice for binary)."""
    denom = precision + sensitivity
    return 2 * precision * sensitivity / denom if denom > 0 else 1.0


# ──────────────────────────────────────────────────────────────────────────────
# HAUSDORFF DISTANCE  (scipy fallback — no extra install needed)
# ──────────────────────────────────────────────────────────────────────────────

def _surface_voxels(binary_mask: np.ndarray) -> np.ndarray:
    """Extract surface voxels using morphological erosion."""
    if binary_mask.sum() == 0:
        return np.empty((0, binary_mask.ndim), dtype=np.int64)
    eroded  = ndimage.binary_erosion(binary_mask)
    surface = binary_mask & ~eroded
    return np.argwhere(surface)


def compute_hd95_scipy(
    pred_bin: np.ndarray,
    gt_bin:   np.ndarray,
    spacing:  Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[float, float]:
    """
    Returns (HD95, HD_max) in mm using scipy (no surface_distance lib needed).
    Uses surface voxel extraction + distance transform.
    """
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float("nan"), float("nan")

    # Distance transform: distance of every voxel from the GT surface
    gt_surf   = _surface_voxels(gt_bin)
    pred_surf = _surface_voxels(pred_bin)

    if len(gt_surf) == 0 or len(pred_surf) == 0:
        return float("nan"), float("nan")

    # Scale voxel coords by spacing to get mm distances
    spacing = np.array(spacing)
    gt_surf_mm   = gt_surf   * spacing
    pred_surf_mm = pred_surf * spacing

    # Compute directed distances via distance transform
    # pred → GT direction
    dt_gt   = ndimage.distance_transform_edt(~gt_bin,   sampling=spacing)
    dt_pred = ndimage.distance_transform_edt(~pred_bin, sampling=spacing)

    pred_to_gt = dt_gt  [tuple(pred_surf.T)]   # distance from each pred surface pt to GT
    gt_to_pred = dt_pred[tuple(gt_surf.T)]     # distance from each GT surface pt to pred

    all_distances = np.concatenate([pred_to_gt, gt_to_pred])

    hd95 = float(np.percentile(all_distances, 95))
    hd_max = float(np.max(all_distances))
    return hd95, hd_max


def compute_hd95_surface_dist(
    pred_bin: np.ndarray,
    gt_bin:   np.ndarray,
    spacing:  Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[float, float]:
    """
    HD95 using Google's surface_distance library (more accurate).
    Falls back to scipy version if library unavailable.
    """
    if not _HAS_SURFACE_DIST:
        return compute_hd95_scipy(pred_bin, gt_bin, spacing)

    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float("nan"), float("nan")

    sd = surf_dist_lib.compute_surface_distances(
        gt_bin.astype(bool), pred_bin.astype(bool), spacing
    )
    hd95  = surf_dist_lib.compute_robust_hausdorff(sd, 95)
    hd_max = surf_dist_lib.compute_robust_hausdorff(sd, 100)
    return float(hd95), float(hd_max)


# ──────────────────────────────────────────────────────────────────────────────
# VOLUME ESTIMATION
# ──────────────────────────────────────────────────────────────────────────────

def compute_volume_cm3(
    binary_mask: np.ndarray,
    voxel_spacing_mm: Tuple[float, float, float],
) -> float:
    """Convert voxel count to cm³ using voxel spacing."""
    voxel_volume_mm3 = float(np.prod(voxel_spacing_mm))
    volume_mm3       = float(binary_mask.sum()) * voxel_volume_mm3
    return volume_mm3 / 1000.0   # mm³ → cm³


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ══════════════════════════════════════════════════════════════════════════════

class SegmentationMetrics:
    """
    Compute and report comprehensive segmentation metrics.

    Parameters
    ----------
    pred : np.ndarray
        Predicted segmentation (integer labels), shape (D, H, W).
    gt : np.ndarray
        Ground truth segmentation (integer labels), same shape.
    labels : dict
        Mapping {label_int: "Name"} for classes to evaluate.
        e.g. {1: "Kidney", 2: "Tumour"}
    voxel_spacing_mm : tuple of 3 floats
        (dz, dy, dx) voxel spacing in mm. Read from NIfTI header:
            import nibabel as nib
            img = nib.load("imaging.nii.gz")
            spacing = img.header.get_zooms()[:3]
    compute_hd : bool
        Set False to skip HD95 (slow for large volumes).
    """

    def __init__(
        self,
        pred:             np.ndarray,
        gt:               np.ndarray,
        labels:           Dict[int, str],
        voxel_spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        compute_hd:       bool = True,
    ):
        assert pred.shape == gt.shape, "pred and gt must have the same shape"
        self.pred             = pred.astype(np.int32)
        self.gt               = gt.astype(np.int32)
        self.labels           = labels
        self.voxel_spacing_mm = voxel_spacing_mm
        self.compute_hd       = compute_hd
        self.results: Dict[int, ClassMetrics] = {}

    # ── per-class computation ─────────────────────────────────────────────────

    def _compute_class(self, label: int, name: str) -> ClassMetrics:
        pred_bin = (self.pred == label)
        gt_bin   = (self.gt   == label)

        cm = ClassMetrics(label=label, name=name)

        # Confusion matrix
        TP, FP, FN, TN = _confusion(pred_bin, gt_bin)
        cm.TP, cm.FP, cm.FN, cm.TN = TP, FP, FN, TN

        # Core metrics
        cm.dice        = compute_dice(TP, FP, FN)
        cm.iou         = compute_iou(TP, FP, FN)
        cm.sensitivity = compute_sensitivity(TP, FN)
        cm.specificity = compute_specificity(TN, FP)
        cm.precision   = compute_precision(TP, FP)
        cm.f1          = compute_f1(cm.precision, cm.sensitivity)

        # Hausdorff Distance
        if self.compute_hd:
            try:
                hd_fn = (
                    compute_hd95_surface_dist
                    if _HAS_SURFACE_DIST
                    else compute_hd95_scipy
                )
                cm.hd95, cm.hd_max = hd_fn(pred_bin, gt_bin, self.voxel_spacing_mm)
            except Exception as e:
                warnings.warn(f"HD95 failed for label {label}: {e}")

        # Volume estimation
        cm.pred_volume_cm3 = compute_volume_cm3(pred_bin, self.voxel_spacing_mm)
        cm.gt_volume_cm3   = compute_volume_cm3(gt_bin,   self.voxel_spacing_mm)
        cm.vol_error_cm3   = cm.pred_volume_cm3 - cm.gt_volume_cm3
        if cm.gt_volume_cm3 > 0:
            cm.vol_error_pct = (cm.vol_error_cm3 / cm.gt_volume_cm3) * 100.0
        else:
            cm.vol_error_pct = float("nan")

        return cm

    # ── public API ────────────────────────────────────────────────────────────

    def compute_all(self) -> Dict[int, ClassMetrics]:
        """Compute metrics for every label. Returns dict of ClassMetrics."""
        print("\n⏳ Computing segmentation metrics...")
        for label, name in self.labels.items():
            print(f"   → {name} (label {label})")
            self.results[label] = self._compute_class(label, name)
        print("✅ Done.\n")
        return self.results

    # ── reporting ─────────────────────────────────────────────────────────────

    def print_report(self) -> None:
        """Pretty-print a full metrics report to stdout."""
        if not self.results:
            print("⚠️  Call compute_all() first.")
            return

        W = 72
        print("=" * W)
        print("  SEGMENTATION METRICS REPORT")
        print(f"  Voxel spacing : {self.voxel_spacing_mm} mm")
        print("=" * W)

        for label, cm in self.results.items():
            print(f"\n  ▶  {cm.name}  (label {label})")
            print("  " + "─" * (W - 2))

            print(f"  {'Metric':<35} {'Value':>12}")
            print("  " + "─" * (W - 2))

            def row(name, val, unit=""):
                if isinstance(val, float):
                    s = f"{val:.4f}{unit}" if not np.isnan(val) else "  N/A"
                else:
                    s = str(val)
                print(f"  {name:<35} {s:>12}")

            row("Dice Similarity Coefficient",  cm.dice)
            row("IoU / Jaccard Index",          cm.iou)
            row("Sensitivity (Recall)",         cm.sensitivity)
            row("Specificity",                  cm.specificity)
            row("Precision",                    cm.precision)
            row("F1-Score",                     cm.f1)
            row("HD95",                         cm.hd95,   " mm")
            row("HD Max",                       cm.hd_max, " mm")
            print("  " + "─" * (W - 2))
            row("Predicted Volume",             cm.pred_volume_cm3, " cm³")
            row("GT Volume",                    cm.gt_volume_cm3,   " cm³")
            row("Volume Error (absolute)",      cm.vol_error_cm3,   " cm³")
            row("Volume Error (%)",             cm.vol_error_pct,   " %")
            print("  " + "─" * (W - 2))
            print(f"  TP={cm.TP:,}  FP={cm.FP:,}  FN={cm.FN:,}  TN={cm.TN:,}")

        # Mean summary across all classes
        print("\n" + "=" * W)
        print("  SUMMARY  (mean across all evaluated classes)")
        print("=" * W)
        metrics_to_avg = ["dice", "iou", "sensitivity", "specificity",
                          "precision", "f1", "hd95"]
        for attr in metrics_to_avg:
            vals = [
                getattr(cm, attr)
                for cm in self.results.values()
                if not np.isnan(getattr(cm, attr))
            ]
            mean_val = np.mean(vals) if vals else float("nan")
            unit = " mm" if attr in ("hd95",) else ""
            label = attr.upper() if len(attr) <= 4 else attr.replace("_", " ").title()
            print(f"  {'Mean ' + label:<35} {mean_val:>10.4f}{unit}")
        print("=" * W + "\n")

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export results as a pandas DataFrame — easy to save to CSV/Excel.

        Returns
        -------
        pd.DataFrame with one row per class and columns for every metric.
        """
        rows = []
        for label, cm in self.results.items():
            rows.append({
                "Label":           label,
                "Class":           cm.name,
                "Dice":            round(cm.dice,        4),
                "IoU":             round(cm.iou,         4),
                "Sensitivity":     round(cm.sensitivity, 4),
                "Specificity":     round(cm.specificity, 4),
                "Precision":       round(cm.precision,   4),
                "F1":              round(cm.f1,          4),
                "HD95 (mm)":       round(cm.hd95,        2) if not np.isnan(cm.hd95) else None,
                "HD_Max (mm)":     round(cm.hd_max,      2) if not np.isnan(cm.hd_max) else None,
                "Pred_Vol (cm3)":  round(cm.pred_volume_cm3, 3),
                "GT_Vol (cm3)":    round(cm.gt_volume_cm3,   3),
                "VolErr (cm3)":    round(cm.vol_error_cm3,   3),
                "VolErr (%)":      round(cm.vol_error_pct,   2) if not np.isnan(cm.vol_error_pct) else None,
                "TP": cm.TP, "FP": cm.FP, "FN": cm.FN, "TN": cm.TN,
            })
        return pd.DataFrame(rows)

    def save_csv(self, path: str) -> None:
        """Save metrics table to CSV."""
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"💾 Metrics saved to {path}")


# ══════════════════════════════════════════════════════════════════════════════
# ORGAN-SPECIFIC CONVENIENCE WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════

class BrainMetrics(SegmentationMetrics):
    """
    BraTS brain tumor labels:
        1 = Necrotic / Non-Enhancing Tumor Core
        2 = Peritumoral Edema
        3 = Enhancing Tumor

    Also computes the three standard BraTS composite regions:
        WT = Whole Tumor     (labels 1 + 2 + 3)
        TC = Tumor Core      (labels 1 + 3)
        ET = Enhancing Tumor (label  3)
    """

    LABELS = {
        1: "Necrotic/Non-Enhancing",
        2: "Edema",
        3: "Enhancing Tumor",
    }

    def __init__(self, pred, gt, voxel_spacing_mm=(1.0, 1.0, 1.0), compute_hd=True):
        # Remap BraTS label 4 → 3 if needed
        gt = gt.copy()
        gt[gt == 4] = 3
        super().__init__(pred, gt, self.LABELS, voxel_spacing_mm, compute_hd)
        self.brats_results: Dict[str, ClassMetrics] = {}

    def compute_all(self):
        super().compute_all()
        self._compute_brats_composite()
        return self.results

    def _compute_brats_composite(self):
        """Compute WT, TC, ET composite Dice scores."""
        composites = {
            "WT": (np.isin(self.pred, [1, 2, 3]), np.isin(self.gt, [1, 2, 3])),
            "TC": (np.isin(self.pred, [1, 3]),     np.isin(self.gt, [1, 3])),
            "ET": (self.pred == 3,                  self.gt   == 3),
        }
        names = {"WT": "Whole Tumor", "TC": "Tumor Core", "ET": "Enhancing Tumor"}
        print("⏳ Computing BraTS composite metrics (WT / TC / ET)...")
        for key, (pb, gb) in composites.items():
            TP, FP, FN, TN = _confusion(pb, gb)
            cm = ClassMetrics(label=-1, name=names[key])
            cm.TP, cm.FP, cm.FN, cm.TN = TP, FP, FN, TN
            cm.dice        = compute_dice(TP, FP, FN)
            cm.iou         = compute_iou(TP, FP, FN)
            cm.sensitivity = compute_sensitivity(TP, FN)
            cm.specificity = compute_specificity(TN, FP)
            cm.precision   = compute_precision(TP, FP)
            cm.f1          = compute_f1(cm.precision, cm.sensitivity)
            cm.pred_volume_cm3 = compute_volume_cm3(pb, self.voxel_spacing_mm)
            cm.gt_volume_cm3   = compute_volume_cm3(gb, self.voxel_spacing_mm)
            cm.vol_error_cm3   = cm.pred_volume_cm3 - cm.gt_volume_cm3
            cm.vol_error_pct   = (
                (cm.vol_error_cm3 / cm.gt_volume_cm3) * 100.0
                if cm.gt_volume_cm3 > 0 else float("nan")
            )
            if self.compute_hd:
                try:
                    hd_fn = compute_hd95_surface_dist if _HAS_SURFACE_DIST else compute_hd95_scipy
                    cm.hd95, cm.hd_max = hd_fn(pb, gb, self.voxel_spacing_mm)
                except Exception:
                    pass
            self.brats_results[key] = cm
        print("✅ BraTS composite metrics done.\n")

    def print_report(self):
        super().print_report()
        if not self.brats_results:
            return
        W = 72
        print("=" * W)
        print("  BraTS COMPOSITE METRICS (WT / TC / ET)")
        print("=" * W)
        print(f"  {'Region':<20} {'Dice':>7}  {'IoU':>7}  {'Sens':>7}  {'HD95':>8}  {'Vol-Pred':>9}  {'Vol-GT':>9}")
        print("  " + "─" * (W - 2))
        for key, cm in self.brats_results.items():
            hd  = f"{cm.hd95:.2f}" if not np.isnan(cm.hd95)  else "  N/A"
            print(
                f"  {cm.name:<20} {cm.dice:>7.4f}  {cm.iou:>7.4f}  "
                f"{cm.sensitivity:>7.4f}  {hd:>8}  "
                f"{cm.pred_volume_cm3:>8.2f}  {cm.gt_volume_cm3:>8.2f}"
            )
        mean_dice = np.mean([cm.dice for cm in self.brats_results.values()])
        print("  " + "─" * (W - 2))
        print(f"  {'Mean Dice (WT/TC/ET)':<20} {mean_dice:>7.4f}")
        print("=" * W + "\n")

    def to_dataframe(self) -> pd.DataFrame:
        df_base = super().to_dataframe()
        rows = []
        for key, cm in self.brats_results.items():
            rows.append({
                "Label": key, "Class": cm.name,
                "Dice": round(cm.dice, 4), "IoU": round(cm.iou, 4),
                "Sensitivity": round(cm.sensitivity, 4),
                "Specificity": round(cm.specificity, 4),
                "Precision": round(cm.precision, 4), "F1": round(cm.f1, 4),
                "HD95 (mm)": round(cm.hd95, 2) if not np.isnan(cm.hd95) else None,
                "HD_Max (mm)": round(cm.hd_max, 2) if not np.isnan(cm.hd_max) else None,
                "Pred_Vol (cm3)": round(cm.pred_volume_cm3, 3),
                "GT_Vol (cm3)": round(cm.gt_volume_cm3, 3),
                "VolErr (cm3)": round(cm.vol_error_cm3, 3),
                "VolErr (%)": round(cm.vol_error_pct, 2) if not np.isnan(cm.vol_error_pct) else None,
            })
        return pd.concat([df_base, pd.DataFrame(rows)], ignore_index=True)


class KidneyMetrics(SegmentationMetrics):
    """KiTS kidney tumor labels: 1=Kidney, 2=Tumour"""
    LABELS = {1: "Kidney", 2: "Tumour"}

    def __init__(self, pred, gt, voxel_spacing_mm=(1.5, 1.5, 1.5), compute_hd=True):
        super().__init__(pred, gt, self.LABELS, voxel_spacing_mm, compute_hd)


class LiverMetrics(SegmentationMetrics):
    """LiTS / Task03 liver labels: 1=Liver Parenchyma, 2=Tumor/Lesion"""
    LABELS = {1: "Liver Parenchyma", 2: "Tumor / Lesion"}

    def __init__(self, pred, gt, voxel_spacing_mm=(1.5, 1.5, 1.5), compute_hd=True):
        super().__init__(pred, gt, self.LABELS, voxel_spacing_mm, compute_hd)

    def compute_all(self):
        super().compute_all()
        self._compute_whole_liver()
        return self.results

    def _compute_whole_liver(self):
        """Whole liver = liver parenchyma + tumor combined."""
        pb = np.isin(self.pred, [1, 2])
        gb = np.isin(self.gt,   [1, 2])
        TP, FP, FN, TN = _confusion(pb, gb)
        cm = ClassMetrics(label=-1, name="Whole Liver (liver+tumor)")
        cm.TP, cm.FP, cm.FN, cm.TN = TP, FP, FN, TN
        cm.dice        = compute_dice(TP, FP, FN)
        cm.iou         = compute_iou(TP, FP, FN)
        cm.sensitivity = compute_sensitivity(TP, FN)
        cm.specificity = compute_specificity(TN, FP)
        cm.precision   = compute_precision(TP, FP)
        cm.f1          = compute_f1(cm.precision, cm.sensitivity)
        cm.pred_volume_cm3 = compute_volume_cm3(pb, self.voxel_spacing_mm)
        cm.gt_volume_cm3   = compute_volume_cm3(gb, self.voxel_spacing_mm)
        cm.vol_error_cm3   = cm.pred_volume_cm3 - cm.gt_volume_cm3
        cm.vol_error_pct   = (
            (cm.vol_error_cm3 / cm.gt_volume_cm3) * 100.0
            if cm.gt_volume_cm3 > 0 else float("nan")
        )
        self.results[-1] = cm   # store under key -1


# ══════════════════════════════════════════════════════════════════════════════
# MULTI-CASE AGGREGATOR  — for dataset-level statistics
# ══════════════════════════════════════════════════════════════════════════════

class DatasetMetrics:
    """
    Aggregate metrics across multiple cases for dataset-level statistics.

    Example
    -------
        agg = DatasetMetrics(labels={1: "Kidney", 2: "Tumour"})
        for pred, gt in zip(pred_list, gt_list):
            m = KidneyMetrics(pred, gt, voxel_spacing_mm=(1.5, 1.5, 1.5))
            m.compute_all()
            agg.add(m)
        agg.print_summary()
        df = agg.to_dataframe()
    """

    def __init__(self, labels: Dict[int, str]):
        self.labels = labels
        self._records: List[Dict] = []

    def add(self, metrics: SegmentationMetrics) -> None:
        for label, cm in metrics.results.items():
            self._records.append({
                "label": label,
                "name":  cm.name,
                "dice":  cm.dice,
                "iou":   cm.iou,
                "sensitivity": cm.sensitivity,
                "specificity": cm.specificity,
                "precision":   cm.precision,
                "f1":          cm.f1,
                "hd95":        cm.hd95,
                "pred_vol":    cm.pred_volume_cm3,
                "gt_vol":      cm.gt_volume_cm3,
                "vol_err_pct": cm.vol_error_pct,
            })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)

    def print_summary(self) -> None:
        df = self.to_dataframe()
        print("\n" + "=" * 72)
        print("  DATASET-LEVEL SUMMARY")
        print("=" * 72)
        cols = ["dice", "iou", "sensitivity", "specificity", "precision", "hd95"]
        for label, name in self.labels.items():
            sub = df[df["label"] == label]
            if sub.empty:
                continue
            print(f"\n  ▶ {name}")
            for col in cols:
                valid = sub[col].dropna()
                if valid.empty:
                    continue
                print(
                    f"    {col:<15} mean={valid.mean():.4f}  "
                    f"std={valid.std():.4f}  "
                    f"min={valid.min():.4f}  "
                    f"max={valid.max():.4f}"
                )
        print("=" * 72 + "\n")

    def save_csv(self, path: str) -> None:
        self.to_dataframe().to_csv(path, index=False)
        print(f"💾 Dataset metrics saved to {path}")


# ══════════════════════════════════════════════════════════════════════════════
# HOW TO PLUG INTO YOUR EXISTING RENDER SCRIPTS
# ══════════════════════════════════════════════════════════════════════════════
"""
──────────────────────────────────────────────────────────────────────────────
BRAIN  (render_brain_3d.py)
──────────────────────────────────────────────────────────────────────────────
Replace your existing calculate_all_dice_scores() call with:

    from segmentation_metrics import BrainMetrics

    ground_truth = load_ground_truth(args.case_dir)
    if ground_truth is not None:
        import nibabel as nib
        spacing = nib.load(
            str(find_nii_file(Path(args.case_dir), f"{Path(args.case_dir).name}_flair"))
        ).header.get_zooms()[:3]

        m = BrainMetrics(preds, ground_truth, voxel_spacing_mm=tuple(spacing))
        m.compute_all()
        m.print_report()
        m.save_csv("brain_metrics.csv")

──────────────────────────────────────────────────────────────────────────────
KIDNEY  (render_kidney_3d.py)
──────────────────────────────────────────────────────────────────────────────
Replace compute_dice() with:

    from segmentation_metrics import KidneyMetrics

    gt = load_ground_truth(args.case_dir)
    if gt is not None:
        m = KidneyMetrics(pred_seg, gt, voxel_spacing_mm=(1.5, 1.5, 1.5))
        m.compute_all()
        m.print_report()
        m.save_csv("kidney_metrics.csv")
        dice = {cm.name: cm.dice for cm in m.results.values()}  # for renderer

──────────────────────────────────────────────────────────────────────────────
LIVER  (RenderLiver.py)
──────────────────────────────────────────────────────────────────────────────
Replace compute_dice_scores() with:

    from segmentation_metrics import LiverMetrics

    gt = load_ground_truth(gt_path, cfg)
    if gt is not None:
        m = LiverMetrics(pred, gt, voxel_spacing_mm=(1.5, 1.5, 1.5))
        m.compute_all()
        m.print_report()
        m.save_csv("liver_metrics.csv")

──────────────────────────────────────────────────────────────────────────────
"""


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST  (runs when executed directly with dummy data)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    shape = (64, 64, 64)

    # Simulate a ground truth with two tumor regions
    gt = np.zeros(shape, dtype=np.int32)
    gt[20:45, 20:45, 20:45] = 1   # kidney
    gt[28:38, 28:38, 28:38] = 2   # tumor inside kidney

    # Simulate slightly-off prediction
    pred = np.zeros(shape, dtype=np.int32)
    pred[22:47, 18:43, 21:46] = 1
    pred[29:39, 27:37, 29:37] = 2

    print("\n" + "━" * 72)
    print("  KIDNEY TEST")
    print("━" * 72)
    m = KidneyMetrics(pred, gt, voxel_spacing_mm=(1.5, 1.5, 1.5))
    m.compute_all()
    m.print_report()
    df = m.to_dataframe()
    print(df.to_string(index=False))

    # ── Brain test ────────────────────────────────────────────────────────────
    gt_brain = np.zeros(shape, dtype=np.int32)
    gt_brain[15:50, 15:50, 15:50] = 2   # edema
    gt_brain[25:45, 25:45, 25:45] = 1   # necrotic core
    gt_brain[30:40, 30:40, 30:40] = 3   # enhancing

    pred_brain = np.zeros(shape, dtype=np.int32)
    pred_brain[17:52, 13:48, 16:51] = 2
    pred_brain[26:46, 24:44, 26:44] = 1
    pred_brain[31:41, 29:39, 31:39] = 3

    print("\n" + "━" * 72)
    print("  BRAIN TEST")
    print("━" * 72)
    m_brain = BrainMetrics(pred_brain, gt_brain, voxel_spacing_mm=(1.0, 1.0, 1.0))
    m_brain.compute_all()
    m_brain.print_report()