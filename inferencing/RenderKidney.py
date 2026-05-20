"""
3D Kidney Tumour Segmentation Visualization

Renders:
  - Full kidney surface (semi-transparent light blue)
  - Tumour region (solid red/orange, coloured on top)
  - Optional surrounding CT body surface for anatomical context

Dice scores computed against ground truth segmentation.nii.gz if available.

Usage:
    python scripts/render_kidney_3d.py
    python scripts/render_kidney_3d.py --case-dir data/Kidney/data/case_00000
    python scripts/render_kidney_3d.py --case-dir data/Kidney/data/case_00000 --save output.png
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Optional, Dict
import scipy.ndimage as ndi
import numpy as np
import torch

# =======================================================
# PROJECT PATH
# =======================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.model import Custom3DSegModel

# =======================================================
# CONFIG
# =======================================================
class Config:
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 3   # 0=background, 1=kidney, 2=tumour

    # CT HU window — must match training preprocessing
    HU_MIN = -200
    HU_MAX  =  300

    # Kidney surface
    KIDNEY_COLOR     = "#5B9BD5"   # steel blue
    KIDNEY_OPACITY   = 0.18        # semi-transparent so tumour is visible inside
    KIDNEY_THRESHOLD = 0.5         # marching cubes threshold

    # Tumour surface
    TUMOUR_COLOR   = "#FF4500"   # orange-red
    TUMOUR_OPACITY = 0.75

    # Body surface (optional CT surface for anatomical context)
    BODY_COLOR      = "#D4C5A9"
    BODY_OPACITY    = 0.06
    BODY_THRESHOLD  = 0.15       # fraction of normalised CT above which body surface shown

    # Window
    WINDOW_SIZE      = (1280, 960)
    BACKGROUND_COLOR = "#FFFFFF"

    # Smoothing
    SMOOTH_ITERATIONS = 30

    # FIX: how many largest kidney components to keep (2 = bilateral kidneys)
    MAX_KIDNEY_COMPONENTS = 2


# =======================================================
# PATH HELPERS
# =======================================================
def find_nii(case_path: Path, stem: str) -> Path:
    for ext in [".nii.gz", ".nii"]:
        p = case_path / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing: {stem}.nii or {stem}.nii.gz in {case_path}")


def validate_paths(checkpoint: str, case_dir: str) -> None:
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not os.path.exists(case_dir):
        raise FileNotFoundError(f"Case directory not found: {case_dir}")
    find_nii(Path(case_dir), "imaging")   # will raise if missing


# =======================================================
# PREPROCESSING  (matches training get_cache_transform)
# =======================================================
def preprocess_volume(case_dir: str) -> torch.Tensor:
    """
    Exactly replicates get_cache_transform() from kidney_dataset.py:
        1. Load imaging.nii.gz
        2. EnsureChannelFirst
        3. Orientationd → RAS
        4. Spacingd → 1.5mm isotropic
        5. ScaleIntensityRanged HU [-200, 300] → [0, 1]
        6. SpatialPadd to 128³ (pads if smaller, no crop if larger)

    Volumes larger than 128³ after resampling stay at their full size —
    sliding window inference handles them.

    Returns:
        Tensor of shape (1, 1, D, H, W) where D,H,W >= 128
    """
    from monai.transforms import (
        LoadImaged,
        EnsureChannelFirstd,
        Orientationd,
        Spacingd,
        ScaleIntensityRanged,
        SpatialPadd,
        EnsureTyped,
        Compose,
    )

    data = {"image": str(find_nii(Path(case_dir), "imaging"))}

    transforms = Compose([
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.5, 1.5, 1.5),
            mode="bilinear",
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=Config.HU_MIN,
            a_max=Config.HU_MAX,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        SpatialPadd(keys=["image"], spatial_size=(128, 128, 128)),
        EnsureTyped(keys=["image"]),
    ])

    vol = transforms(data)["image"]
    return vol.unsqueeze(0)   # (1, 1, D, H, W)


# =======================================================
# GROUND TRUTH LOADER
# =======================================================
def load_ground_truth(case_dir: str) -> Optional[np.ndarray]:
    """Load segmentation.nii.gz and resample to match prediction resolution."""
    from monai.transforms import (
        LoadImage,
        EnsureChannelFirst,
        Orientation,
        Spacing,
        SpatialPad,
        EnsureType,
        Compose,
    )

    case_path = Path(case_dir)
    try:
        seg_path = find_nii(case_path, "segmentation")
    except FileNotFoundError:
        print("⚠️  No ground truth segmentation found — Dice scores will not be shown.")
        return None

    try:
        transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=(1.5, 1.5, 1.5), mode="nearest"),
            SpatialPad(spatial_size=(128, 128, 128)),
            EnsureType(),
        ])
        gt = transforms(str(seg_path))
        print(f"✅ Ground truth loaded: {seg_path.name}")
        return gt.squeeze().numpy().astype(np.int32)
    except Exception as e:
        warnings.warn(f"Failed to load ground truth: {e}")
        return None


# =======================================================
# MODEL LOADER
# =======================================================
def load_model(checkpoint_path: str) -> Custom3DSegModel:
    model = Custom3DSegModel(
        in_channels=1,
        embed_dim=32,
        n_classes=Config.NUM_CLASSES,
        final_activation=None,
    ).to(Config.DEVICE)

    ckpt = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=False)

    # Checkpoint may be full training dict or bare state_dict
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"✅ Model loaded from {checkpoint_path}")
    return model


# =======================================================
# POSTPROCESSING HELPERS
# =======================================================
def _keep_top_n_components(binary_mask: np.ndarray, n: int) -> np.ndarray:
    """
    Keep the N largest connected components in a binary mask.
    Used to support bilateral kidneys (n=2) instead of dropping one kidney
    by keeping only the single largest component.
    """
    labeled, num = ndi.label(binary_mask)
    if num == 0:
        return binary_mask

    sizes = ndi.sum(binary_mask, labeled, range(1, num + 1))
    # Sort by size descending, take top n
    top_labels = np.argsort(sizes)[::-1][:n] + 1  # +1 because labels start at 1
    kept = np.isin(labeled, top_labels)
    return kept.astype(np.uint8)


# =======================================================
# INFERENCE
# =======================================================
@torch.no_grad()
def run_inference(model: Custom3DSegModel, volume: torch.Tensor) -> np.ndarray:
    """
    Full-volume inference using sliding window to handle memory on 4GB VRAM.
    Returns predicted segmentation array with values 0/1/2.

    Postprocessing steps:
        1. Argmax over softmax logits
        2. Expand kidney mask to include tumour voxels (prevents split components)
        3. Keep top MAX_KIDNEY_COMPONENTS kidney blobs (supports bilateral kidneys)
        4. Restrict tumour to within dilated kidney region
        5. Keep largest tumour component only
        6. Rebuild final segmentation
    """
    from monai.inferers import sliding_window_inference

    volume = volume.to(Config.DEVICE)
    outputs = sliding_window_inference(
        inputs=volume,
        roi_size=(128, 128, 128),
        sw_batch_size=1,
        predictor=model,
        overlap=0.25,
        mode="gaussian",
    )

    probs = torch.softmax(outputs, dim=1)
    preds = torch.argmax(probs, dim=1)
    pred_np = preds[0].cpu().numpy()

    # ── Step 1: Raw masks ──
    kidney_mask = (pred_np == 1)
    tumour_mask = (pred_np == 2)

    # ── Step 2: Expand kidney to include tumour before component analysis ──
    # Prevents tumours from being counted as separate disconnected kidney blobs
    kidney_mask = kidney_mask | tumour_mask

    # ── Step 3: Keep top N kidney components (FIX: bilateral kidney support) ──
    # Previously kept only 1 — would silently drop the second kidney.
    # MAX_KIDNEY_COMPONENTS=2 handles single and bilateral cases.
    kidney_mask = _keep_top_n_components(kidney_mask, Config.MAX_KIDNEY_COMPONENTS)

    # ── Step 4: Restrict tumour to near kidney region ──
    dilated_kidney = ndi.binary_dilation(kidney_mask, iterations=10)
    tumour_mask = (pred_np == 2) & dilated_kidney

    # ── Step 5: Keep largest tumour component only ──
    # Single tumour assumption — reasonable for KiTS dataset
    tumour_mask = _keep_top_n_components(tumour_mask, 1).astype(bool)

    # ── Step 6: Rebuild final segmentation ──
    final_pred = np.zeros_like(pred_np, dtype=np.uint8)
    final_pred[kidney_mask.astype(bool)] = 1
    final_pred[tumour_mask] = 2

    if Config.DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return final_pred.astype(np.int32)


# =======================================================
# DICE SCORES
# =======================================================
def dice_score(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    p = pred == label
    g = gt  == label
    inter = np.sum(p & g)
    denom = np.sum(p) + np.sum(g)
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    return {
        "Kidney": dice_score(pred, gt, 1),
        "Tumour": dice_score(pred, gt, 2),
    }


# =======================================================
# STATISTICS
# =======================================================
def print_stats(pred: np.ndarray, dice: Optional[Dict] = None) -> None:
    total = pred.size
    labels = {0: "Background", 1: "Kidney", 2: "Tumour"}
    print("\n📊 Voxel Statistics:")
    print("=" * 55)
    for lbl, name in labels.items():
        count   = int(np.sum(pred == lbl))
        percent = count / total * 100
        print(f"  {name:<15}: {count:>10,}  ({percent:6.3f}%)")
    print("=" * 55)
    if dice:
        print("\n🎯 Dice Scores (vs ground truth):")
        print("=" * 55)
        for name, val in dice.items():
            print(f"  {name:<15}: {val:.4f}")
        mean = np.mean(list(dice.values()))
        print(f"  {'Mean':<15}: {mean:.4f}")
        print("=" * 55)


# =======================================================
# 3D RENDERING
# =======================================================
def render_3d(
    ct_volume:   np.ndarray,
    pred_seg:    np.ndarray,
    config:      Config,
    dice:        Optional[Dict] = None,
    save_path:   Optional[str]  = None,
    interactive: bool = True,
    show_body:   bool = False,   # FIX: synced default with argparse (was True here, False in argparse)
) -> None:
    """
    Renders:
      - (Optional) CT body surface — very faint, anatomical context
      - Kidney surface — semi-transparent steel blue
        NOTE: kidney mask is dilated by 3 voxels here for a smoother visual
        surface only. This does NOT affect Dice computation (which uses the
        raw pred_seg before dilation).
      - Tumour surface — solid orange-red
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("pyvista is required: pip install pyvista")

    plotter = pv.Plotter(window_size=config.WINDOW_SIZE)
    plotter.set_background(config.BACKGROUND_COLOR)

    # ── Body surface (faint CT isosurface for anatomical context) ──
    if show_body:
        try:
            body_bin = (ct_volume > config.BODY_THRESHOLD).astype(np.uint8)
            body_mesh = pv.wrap(body_bin).contour([0.5])
            if body_mesh.n_points > 0:
                body_mesh = body_mesh.connectivity(largest=True)
                body_mesh = body_mesh.smooth(n_iter=10)
                plotter.add_mesh(
                    body_mesh,
                    color=config.BODY_COLOR,
                    opacity=config.BODY_OPACITY,
                    smooth_shading=True,
                    label="Body Surface",
                )
        except Exception as e:
            warnings.warn(f"Body surface skipped: {e}")

    # ── Kidney surface ──
    kidney_mask = (pred_seg == 1).astype(np.uint8)

    # Dilate by 3 voxels for a smoother visual surface only.
    # Dice is computed on pred_seg before this dilation, so metrics are unaffected.
    kidney_mask_visual = ndi.binary_dilation(kidney_mask, iterations=3).astype(np.uint8)
    kidney_voxels = int(kidney_mask.sum())   # report actual (pre-dilation) voxel count

    if kidney_voxels == 0:
        print("⚠️  No kidney voxels predicted.")
    else:
        try:
            kidney_mesh = pv.wrap(kidney_mask_visual).contour([config.KIDNEY_THRESHOLD])
            if kidney_mesh.n_points > 0:
                # Use largest=True for rendering connectivity — both kidneys appear
                # as one mesh object since they're visually separate islands
                kidney_mesh = kidney_mesh.smooth(n_iter=config.SMOOTH_ITERATIONS)

                kidney_label = f"Kidney  ({kidney_voxels:,} vox)"
                if dice and "Kidney" in dice:
                    kidney_label += f"  |  Dice: {dice['Kidney']:.4f}"

                plotter.add_mesh(
                    kidney_mesh,
                    color=config.KIDNEY_COLOR,
                    opacity=config.KIDNEY_OPACITY,
                    smooth_shading=True,
                    specular=0.4,
                    specular_power=15,
                    label=kidney_label,
                )
                print(f"✅ Kidney rendered: {kidney_voxels:,} voxels")
        except Exception as e:
            warnings.warn(f"Kidney render failed: {e}")

    # ── Tumour surface ──
    tumour_mask   = (pred_seg == 2).astype(np.uint8)
    tumour_voxels = int(tumour_mask.sum())

    if tumour_voxels == 0:
        print("⚠️  No tumour voxels predicted.")
    else:
        try:
            tumour_mesh = pv.wrap(tumour_mask).contour([0.5])
            if tumour_mesh.n_points > 0:
                tumour_mesh = tumour_mesh.smooth(n_iter=config.SMOOTH_ITERATIONS)

                tumour_label = f"Tumour  ({tumour_voxels:,} vox)"
                if dice and "Tumour" in dice:
                    tumour_label += f"  |  Dice: {dice['Tumour']:.4f}"

                plotter.add_mesh(
                    tumour_mesh,
                    color=config.TUMOUR_COLOR,
                    opacity=config.TUMOUR_OPACITY,
                    smooth_shading=True,
                    specular=0.7,
                    specular_power=30,
                    label=tumour_label,
                )
                print(f"✅ Tumour rendered: {tumour_voxels:,} voxels")
        except Exception as e:
            warnings.warn(f"Tumour render failed: {e}")

    # ── Axes and legend ──
    plotter.add_axes(
        xlabel="X (L→R)",
        ylabel="Y (P→A)",
        zlabel="Z (I→S)",
        line_width=3,
    )

    # FIX: legend background changed from #1A1A1A (near-black) to #333333 with
    # white face text — readable on the white canvas background.
    plotter.add_legend(
        size=(0.30, 0.12),
        loc="lower right",          # move legend below dice text
        face="circle",
        bcolor="#333333",
        border=True,
    )

    # ── Dice score overlay text ──
    if dice:
        mean_dice = float(np.mean(list(dice.values())))
        text = (
            f"Dice Scores\n"
            f"Kidney : {dice.get('Kidney', 0):.4f}\n"
            f"Tumour : {dice.get('Tumour', 0):.4f}\n"
            f"Mean   : {mean_dice:.4f}"
        )
        # Normalised viewport coords: (0,0)=bottom-left, (1,1)=top-right
        # x=0.72 leaves ~28% right margin so the text block never gets clipped
        plotter.add_text(
            text,
            position=(0.72, 0.82),
            font_size=11,
            color="black",
            font="courier",
        )

    # ── Case name ──
    plotter.add_text(
        "Kidney Tumour Segmentation",
        position="upper_left",
        font_size=13,
        color="black",   # black text on white background
        font="courier",
    )

    plotter.camera_position = "iso"
    plotter.reset_camera()
    plotter.enable_anti_aliasing("fxaa")

    if save_path:
        plotter.screenshot(save_path, transparent_background=False)
        print(f"💾 Screenshot saved: {save_path}")

    if interactive:
        plotter.show()
    else:
        plotter.close()


# =======================================================
# MAIN
# =======================================================
def main():
    parser = argparse.ArgumentParser(description="3D Kidney Tumour Segmentation Visualization")
    parser.add_argument(
        "--checkpoint", type=str,
        default=r"D:\MajorProject\3D SD-NET\outputs\kidney_transformer\best_model(0.2641).pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--case-dir", type=str,
        default=r"data/Kidney/data/case_00183",
        help="Path to case directory containing imaging.nii.gz",
    )
    parser.add_argument("--save",           type=str,  default=None,
                        help="Path to save screenshot (e.g. render.png)")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Disable interactive window (requires --save)")
    parser.add_argument("--show-body",      action="store_true", default=False,
                        help="Show faint CT body surface")   # FIX: explicit default=False
    parser.add_argument("--smooth",         type=int,  default=30,
                        help="Smoothing iterations for mesh surfaces")
    args = parser.parse_args()

    Config.SMOOTH_ITERATIONS = args.smooth

    print("=" * 60)
    print("🫘 3D KIDNEY TUMOUR SEGMENTATION VISUALIZATION")
    print("=" * 60)
    print(f"Device     : {Config.DEVICE}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Case       : {args.case_dir}")
    print("=" * 60)

    # Validate
    try:
        validate_paths(args.checkpoint, args.case_dir)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # Load model
    print("\n🔧 Loading model...")
    try:
        model = load_model(args.checkpoint)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # Preprocess CT
    print("\n🔄 Preprocessing CT volume...")
    volume = preprocess_volume(args.case_dir)
    print(f"✅ Volume shape: {volume.shape}")

    # Inference + postprocessing
    print("\n🔮 Running inference...")
    pred_seg = run_inference(model, volume)
    print(f"✅ Prediction shape: {pred_seg.shape}")
    print(f"   Unique labels   : {np.unique(pred_seg).tolist()}")

    # Ground truth + Dice
    gt   = load_ground_truth(args.case_dir)
    dice = compute_dice(pred_seg, gt) if gt is not None else None

    # Stats
    print_stats(pred_seg, dice)

    # CT volume for body surface (squeeze to 3D)
    ct_np = volume[0, 0].cpu().numpy()

    # Render
    print("\n🎨 Rendering 3D visualization...")
    render_3d(
        ct_volume   = ct_np,
        pred_seg    = pred_seg,
        config      = Config,
        dice        = dice,
        save_path   = args.save,
        interactive = not args.no_interactive,
        show_body   = args.show_body,
    )

    print("\n✅ Done.")


if __name__ == "__main__":
    main()