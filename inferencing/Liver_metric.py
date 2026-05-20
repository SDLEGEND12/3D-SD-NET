"""
visualize_liver_3d.py
─────────────────────────────────────────────────────────────────────────────
3-D Liver Segmentation Visualizer  –  single-file mode

Just point CT_FILE and CHECKPOINT at your files and run:
    python inferencing/RenderLiver.py

Optionally set GT_FILE to a ground-truth mask to get Dice scores.

Label convention (Task03_Liver / LiTS):
    0 → Background
    1 → Liver parenchyma
    2 → Tumor / lesion
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pyvista as pv
import torch
from scipy import ndimage
from segmentation_metrics import LiverMetrics
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
CT_FILE    = r"D:\MajorProject\3D SD-NET\data\Task03_Liver\imagesTr\liver_102.nii.gz"
GT_FILE    = r"D:\MajorProject\3D SD-NET\data\Task03_Liver\labelsTr\liver_102.nii.gz"
# Set GT_FILE = None if you have no ground-truth mask:
# GT_FILE  = None

CHECKPOINT = r"D:\MajorProject\3D SD-NET\outputs\checkpoints_liver\best_model.pt"

SAVE_PATH     = None    # e.g. r"D:\MajorProject\outputs\liver_0.png"  or None
INTERACTIVE   = True    # set False if you only want to save a screenshot
SKIP_POSTPROC = False   # set True to skip CC-filtering / hole-fill
# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  (no need to touch these unless your model differs)
# ──────────────────────────────────────────────────────────────────────────────
class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NUM_CLASSES  = 3        # 0=bg, 1=liver, 2=tumor
    EMBED_DIM    = 64
    IN_CHANNELS  = 1        # 1 for CT (liver), 4 for multi-modal MRI (BraTS)
    SPATIAL_SIZE = (128, 128, 128)

    # Liver CT window  W=150 L=60  →  [-15, 135] HU
    CT_WIN_MIN = -17.0
    CT_WIN_MAX = 201.0

    SEGMENT_SETTINGS: Dict[int, dict] = {
        1: {
            "name":     "Liver Parenchyma",
            "color":    "#C87137",
            "opacity":  0.22,
            "specular": 0.35,
        },
        2: {
            "name":     "Tumor / Lesion",
            "color":    "#FF2020",
            "opacity":  0.92,
            "specular": 0.65,
        },
    }

    # Post-processing
    CC_KEEP: Dict[int, int] = {1: 1, 2: 10}
    MIN_VOXELS: Dict[int, int] = {1: 1000, 2: 30}
    FILL_LIVER_HOLES = True

    # Rendering
    WINDOW_SIZE       = (1280, 960)
    BACKGROUND_COLOR  = "white"
    SMOOTH_ITERATIONS = 35
    SHOW_AXES         = True
    SHOW_EDGES        = True


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def preprocess_ct(ct_path: Path, cfg: Config) -> torch.Tensor:
    """
    MONAI pipeline – mirrors training exactly:
      Load -> EnsureChannelFirst -> Orientation(RAS) -> Spacing(1mm iso)
      -> Resize(128^3) -> ClipNormalise([WIN_MIN, WIN_MAX] -> [0,1])

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

    vol = transforms({"image": str(ct_path)})["image"]   # (1, D, H, W)
    return vol.unsqueeze(0)                               # (1, 1, D, H, W)


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
        print(f"✅ Ground truth loaded   unique labels: {np.unique(gt).tolist()}")
        return gt
    except Exception as exc:
        warnings.warn(f"Could not load ground truth: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════
def load_model(checkpoint: str, cfg: Config) -> "Custom3DSegModel":
    if not _MODEL_AVAILABLE:
        raise RuntimeError("Custom3DSegModel import failed.")

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
    print(f"✅ Model loaded  ->  {checkpoint}")
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


# ══════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def postprocess(pred: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Step 1 – Connected-component filtering per label.
              Liver  : keep only the 1 largest component  (single organ)
              Tumor  : keep up to 10 largest components   (multifocal)
              Discard anything below MIN_VOXELS.
    Step 2 – Fill holes inside liver mask (watertight surface).
    Step 3 – Remove tumor voxels that lie outside the liver boundary.
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
        name = cfg.SEGMENT_SETTINGS[label]["name"]
        print(f"  Label {label} ({name}): {n_cc} component(s) -> kept {kept}")

    # Step 2 – hole-fill liver
    if cfg.FILL_LIVER_HOLES:
        liver_mask = (cleaned == 1)
        if liver_mask.sum() > 0:
            filled    = ndimage.binary_fill_holes(liver_mask)
            new_liver = filled & (cleaned == 0)
            cleaned[new_liver] = 1
            print(f"  Liver hole-fill: +{int(new_liver.sum()):,} voxels")

    # Step 3 – clip tumors to liver region
    tumor_mask = (cleaned == 2)
    if tumor_mask.sum() > 0:
        liver_dilated = ndimage.binary_dilation(
            (cleaned == 1), structure=struct, iterations=3
        )
        outside = tumor_mask & ~liver_dilated
        if outside.sum() > 0:
            print(f"  Removed {int(outside.sum()):,} tumor voxels outside liver")
            cleaned[outside] = 0

    return cleaned


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════
def _dice(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    p = (pred == label)
    g = (gt   == label)
    inter = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    return 1.0 if denom == 0 else float(2 * inter / denom)


def compute_dice_scores(pred: np.ndarray, gt: np.ndarray,
                        cfg: Config) -> Dict:
    scores = {label: _dice(pred, gt, label) for label in cfg.SEGMENT_SETTINGS}
    p_wl   = np.isin(pred, [1, 2])
    g_wl   = np.isin(gt,   [1, 2])
    inter  = np.logical_and(p_wl, g_wl).sum()
    denom  = p_wl.sum() + g_wl.sum()
    scores["whole_liver"] = float(2 * inter / denom) if denom > 0 else 1.0
    return scores


def print_statistics(pred: np.ndarray,
                     dice_scores: Optional[Dict],
                     cfg: Config) -> None:
    total = pred.size
    print("\n📊 Voxel Statistics")
    print("=" * 60)
    print(f"  {'Background':<30}: {np.sum(pred == 0):>9,} "
          f"({100 * np.sum(pred == 0) / total:.2f}%)")
    for label, s in cfg.SEGMENT_SETTINGS.items():
        cnt = int(np.sum(pred == label))
        print(f"  {s['name']:<30}: {cnt:>9,} ({100 * cnt / total:.2f}%)")
    print(f"  {'Total voxels':<30}: {total:>9,}")
    print("=" * 60)
    if dice_scores:
        print("\n🎯 Dice Scores")
        print("=" * 60)
        for label, s in cfg.SEGMENT_SETTINGS.items():
            print(f"  {s['name']:<30}: {dice_scores.get(label, 0.0):.4f}")
        print(f"  {'Whole Liver (liver+tumor)':<30}: "
              f"{dice_scores.get('whole_liver', 0.0):.4f}")
        print("=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# 3-D RENDERING
# ══════════════════════════════════════════════════════════════════════════════
def _build_mesh(mask: np.ndarray, smooth: int) -> Optional[pv.PolyData]:
    mesh = pv.wrap(mask.astype(np.uint8)).contour([0.5])
    if mesh.n_points == 0:
        return None
    mesh = mesh.smooth(n_iter=smooth)
    mesh = mesh.compute_normals(
        cell_normals=False, point_normals=True, auto_orient_normals=True
    )
    return mesh


def render_3d(
    ct_volume:   np.ndarray,
    pred:        np.ndarray,
    cfg:         Config,
    case_name:   str,
    dice_scores: Optional[Dict] = None,
    save_path:   Optional[str]  = None,
    interactive: bool           = True,
) -> None:
    plotter = pv.Plotter(window_size=cfg.WINDOW_SIZE)
    plotter.set_background(cfg.BACKGROUND_COLOR)

    # ── Liver outer shell (liver + tumor union = complete, non-hollow shell) ──
    liver_shell = (pred == 1) | (pred == 2)
    if liver_shell.sum() > 0:
        try:
            mesh = _build_mesh(liver_shell, cfg.SMOOTH_ITERATIONS)
            if mesh is not None:
                mesh = mesh.connectivity(largest=True)
                s = cfg.SEGMENT_SETTINGS[1]
                plotter.add_mesh(
                    mesh,
                    color=s["color"], opacity=s["opacity"],
                    smooth_shading=True,
                    specular=s["specular"], specular_power=20,
                    label="Liver Surface",
                )
                if cfg.SHOW_EDGES:
                    edges = mesh.extract_feature_edges(
                        boundary_edges=True, feature_edges=False,
                        manifold_edges=False, non_manifold_edges=False,
                    )
                    if edges and edges.n_points > 0:
                        plotter.add_mesh(edges, color="saddlebrown",
                                         line_width=0.7, opacity=0.25)
                print("✅ Liver surface rendered")
        except Exception as exc:
            warnings.warn(f"Liver surface error: {exc}")

    # ── Tumor blobs ──────────────────────────────────────────────────────────
    tumor_mask = (pred == 2)
    n_tumor    = int(tumor_mask.sum())
    if n_tumor > 0:
        try:
            mesh = _build_mesh(tumor_mask, cfg.SMOOTH_ITERATIONS)
            if mesh is not None:
                s   = cfg.SEGMENT_SETTINGS[2]
                leg = (f"{s['name']}  Dice={dice_scores[2]:.3f}"
                       if dice_scores and 2 in dice_scores
                       else f"{s['name']}  ({n_tumor:,} vox)")
                plotter.add_mesh(
                    mesh,
                    color=s["color"], opacity=s["opacity"],
                    smooth_shading=True,
                    specular=s["specular"], specular_power=30,
                    label=leg,
                )
                print(f"✅ Tumor rendered  ({n_tumor:,} voxels)")
        except Exception as exc:
            warnings.warn(f"Tumor rendering error: {exc}")
    else:
        print("⚠️  No tumor voxels predicted for this case.")

    # ── Axes, legend, title, Dice overlay ────────────────────────────────────
    if cfg.SHOW_AXES:
        plotter.add_axes(xlabel="X (L->R)", ylabel="Y (P->A)",
                         zlabel="Z (I->S)", line_width=3)

    plotter.add_legend(size=(0.30, 0.18), face="circle")
    plotter.add_text(f"Liver Segmentation  –  {case_name}",
                     position="upper_left", font_size=11, color="black")

    if dice_scores:
        overlay = "\n".join([
            "Dice Scores",
            f"  Liver : {dice_scores.get(1, 0.0):.4f}",
            f"  Tumor : {dice_scores.get(2, 0.0):.4f}",
            f"  Whole : {dice_scores.get('whole_liver', 0.0):.4f}",
        ])
        plotter.add_text(overlay, position="upper_right",
                         font_size=10, color="black")

    plotter.camera_position = "iso"
    plotter.reset_camera()
    plotter.enable_anti_aliasing("fxaa")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(save_path, transparent_background=False)
        print(f"💾 Screenshot saved  ->  {save_path}")

    if interactive:
        plotter.show()
    else:
        plotter.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    cfg = Config()

    # ── Validate input files ──────────────────────────────────────────────────
    ct_path = Path(CT_FILE)
    if not ct_path.exists():
        print(f"❌ CT file not found: {ct_path}")
        sys.exit(1)

    gt_path = Path(GT_FILE) if GT_FILE is not None else None
    if gt_path is not None and not gt_path.exists():
        print(f"⚠️  GT file not found (skipping Dice): {gt_path}")
        gt_path = None

    case_name = ct_path.stem.replace(".nii", "")   # friendly display name

    print("=" * 65)
    print("LIVER 3D SEGMENTATION VISUALIZER")
    print("=" * 65)
    print(f"Device     : {cfg.DEVICE}")
    print(f"CT file    : {ct_path.name}")
    print(f"GT file    : {gt_path.name if gt_path else 'None (no Dice scores)'}")
    print(f"Checkpoint : {CHECKPOINT}")
    print(f"In channels: {cfg.IN_CHANNELS}  "
          f"({'CT single-channel' if cfg.IN_CHANNELS == 1 else 'multi-modal MRI'})")
    print("=" * 65)

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        model = load_model(CHECKPOINT, cfg)
    except Exception as exc:
        print(f"❌ {exc}")
        sys.exit(1)

    # ── Preprocess CT ─────────────────────────────────────────────────────────
    print("\nPreprocessing CT...")
    volume = preprocess_ct(ct_path, cfg)
    print(f"✅ Volume tensor shape: {tuple(volume.shape)}")

    # ── Inference ─────────────────────────────────────────────────────────────
    print("\nRunning inference...")
    pred_raw = run_inference(model, volume, cfg)
    print(f"✅ Raw prediction  unique labels: {np.unique(pred_raw).tolist()}")

    # ── Post-processing ───────────────────────────────────────────────────────
    if SKIP_POSTPROC:
        pred = pred_raw
        print("ℹ️  Post-processing skipped.")
    else:
        print("\nPost-processing...")
        pred = postprocess(pred_raw, cfg)
        print(f"✅ Cleaned  unique labels: {np.unique(pred).tolist()}")

    # ── Ground truth + Dice ───────────────────────────────────────────────────
    dice_scores = None
    if gt_path is not None:
        print("\nLoading ground truth...")
        gt = load_ground_truth(gt_path, cfg)
        if gt is not None:
            m = LiverMetrics(pred, gt, voxel_spacing_mm=(1.5, 1.5, 1.5))
            m.compute_all()
            m.print_report()
            m.save_csv("liver_metrics.csv")

    # ── Statistics ────────────────────────────────────────────────────────────
    print_statistics(pred, dice_scores, cfg)

    # ── Render ───────────────────────────────────────────────────────────────
    ct_np = volume[0, 0].cpu().numpy()
    print("\nRendering 3D visualisation...")
    render_3d(
        ct_volume=ct_np,
        pred=pred,
        cfg=cfg,
        case_name=case_name,
        dice_scores=dice_scores,
        save_path=SAVE_PATH,
        interactive=INTERACTIVE,
    )
    print("\n✅ Done!")


if __name__ == "__main__":
    main()