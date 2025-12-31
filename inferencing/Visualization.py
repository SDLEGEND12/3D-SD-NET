import os
import sys
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Slider
from scipy.ndimage import label
import argparse
from typing import Optional, Dict

# -------------------------------------------------------
# PROJECT PATH
# -------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.model import Custom3DSegModel

# =======================================================
# CONFIG
# =======================================================
class Config:
    NUM_CLASSES = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MIN_SIZE_NCR = 30
    MIN_SIZE_ED = 100
    MIN_SIZE_ET = 20
    
    COLORS = ["black", "red", "yellow", "blue"]
    CLASS_NAMES = {
        0: "Background",
        1: "Necrotic/Non-Enhancing (NCR)",
        2: "Edema (ED)",
        3: "Enhancing Tumor (ET)"
    }

# =======================================================
# PREPROCESSING
# =======================================================
def preprocess_volume(case_dir: str) -> torch.Tensor:
    """Preprocess MRI volume"""
    from monai.transforms import (
        LoadImaged, EnsureChannelFirstd, Orientationd,
        Spacingd, Resized, ScaleIntensityRanged, EnsureTyped, Compose
    )

    case_path = Path(case_dir)
    data = {
        "image": [str(case_path / f"{case_path.name}_{mod}.nii.gz") 
                  for mod in ['flair', 't1', 't1ce', 't2']]
    }

    transforms = Compose([
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        Resized(keys=["image"], spatial_size=(128, 128, 128), mode="trilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=3000, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image"]),
    ])

    return transforms(data)["image"].unsqueeze(0)


def preprocess_gt(case_dir: str) -> Optional[np.ndarray]:
    """Preprocess ground truth"""
    from monai.transforms import (
        LoadImage, EnsureChannelFirst, Orientation,
        Spacing, Resize, EnsureType, Compose
    )

    seg_path = Path(case_dir) / f"{Path(case_dir).name}_seg.nii.gz"
    if not seg_path.exists():
        return None

    transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        Spacing(pixdim=(1.0, 1.0, 1.0), mode="nearest"),
        Resize(spatial_size=(128, 128, 128), mode="nearest"),
        EnsureType(),
    ])

    gt = transforms(str(seg_path)).squeeze().numpy()
    gt[gt == 4] = 3  # BraTS label remapping
    return gt

# =======================================================
# INFERENCE
# =======================================================
@torch.no_grad()
def predict(model: Custom3DSegModel, volume: torch.Tensor) -> np.ndarray:
    """Run inference"""
    volume = volume.to(Config.DEVICE)
    logits = model(volume)
    logits = torch.clamp(logits, min=-10, max=10)
    preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)[0].cpu().numpy()
    
    if Config.DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    
    return preds

# =======================================================
# POSTPROCESSING
# =======================================================
def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove small connected components"""
    labeled, num = label(mask)
    cleaned = np.zeros_like(mask)
    for i in range(1, num + 1):
        component = (labeled == i)
        if component.sum() >= min_size:
            cleaned[component] = 1
    return cleaned


def postprocess_brats(pred: np.ndarray) -> np.ndarray:
    """Apply morphological postprocessing"""
    pred = pred.copy()
    
    ncr = (pred == 1).astype(np.uint8)
    ed = (pred == 2).astype(np.uint8)
    et = (pred == 3).astype(np.uint8)
    
    ncr_clean = remove_small_components(ncr, Config.MIN_SIZE_NCR)
    et_clean = remove_small_components(et, Config.MIN_SIZE_ET)
    tumor_core = ncr_clean | et_clean
    ed_clean = ed & (tumor_core | remove_small_components(ed, Config.MIN_SIZE_ED))
    et_clean = et_clean & tumor_core
    
    cleaned = np.zeros_like(pred, dtype=np.uint8)
    cleaned[ed_clean.astype(bool)] = 2
    cleaned[ncr_clean.astype(bool)] = 1
    cleaned[et_clean.astype(bool)] = 3
    
    return cleaned

# =======================================================
# DICE METRICS
# =======================================================
def compute_dice(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Compute Dice scores"""
    pred, gt = pred.copy(), gt.copy()
    pred[pred == 4] = 3
    gt[gt == 4] = 3
    
    scores = {}
    
    # Individual classes
    for cls in [1, 2, 3]:
        p, g = (pred == cls), (gt == cls)
        if g.sum() == 0:
            scores[cls] = np.nan
        else:
            scores[cls] = (2.0 * (p & g).sum()) / (p.sum() + g.sum())
    
    # BraTS regions
    pred_wt, gt_wt = np.isin(pred, [1, 2, 3]), np.isin(gt, [1, 2, 3])
    scores['WT'] = (2.0 * (pred_wt & gt_wt).sum()) / (pred_wt.sum() + gt_wt.sum())
    
    pred_tc, gt_tc = np.isin(pred, [1, 3]), np.isin(gt, [1, 3])
    scores['TC'] = (2.0 * (pred_tc & gt_tc).sum()) / (pred_tc.sum() + gt_tc.sum())
    
    scores['ET'] = scores[3]
    
    return scores


def print_dice(scores: Dict[str, float]) -> None:
    """Print Dice scores"""
    print("\n" + "="*70)
    print("📊 DICE SCORES")
    print("="*70)
    
    print("\n🔹 Individual Classes:")
    for cls in [1, 2, 3]:
        if cls in scores and not np.isnan(scores[cls]):
            print(f"  {Config.CLASS_NAMES[cls]:<45} : {scores[cls]:>6.4f}")
    
    print("\n🔹 BraTS Regions:")
    for region in ['WT', 'TC', 'ET']:
        print(f"  {region:<45} : {scores[region]:>6.4f}")
    
    mean = np.mean([scores['WT'], scores['TC'], scores['ET']])
    print(f"\n  {'Mean Dice':<45} : {mean:>6.4f}")
    print("="*70 + "\n")

# =======================================================
# VISUALIZATION
# =======================================================
def visualize(volume: np.ndarray, pred: np.ndarray, gt: Optional[np.ndarray] = None,
              dice_scores: Optional[Dict] = None, save_path: Optional[str] = None) -> None:
    """Interactive slice viewer"""
    
    # Find initial slice with predictions
    pred_slices = np.where(pred.sum(axis=(0,1)) > 0)[0]
    z_init = pred_slices[len(pred_slices)//2] if len(pred_slices) > 0 else volume.shape[-1] // 2
    
    cmap = ListedColormap(Config.COLORS)
    num_cols = 3 if gt is not None else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(6*num_cols, 6))
    plt.subplots_adjust(bottom=0.12, top=0.92, left=0.05, right=0.95)
    
    if num_cols == 2:
        axes = [axes[0], axes[1]]
    
    # Title with better spacing
    if dice_scores:
        mean_dice = np.mean([dice_scores['WT'], dice_scores['TC'], dice_scores['ET']])
        title = f"BraTS Segmentation    |    Mean Dice: {mean_dice:.4f}"
    else:
        title = "BraTS Segmentation"
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    
    # Normalize for display
    def normalize(img):
        img = img.astype(np.float32)
        return (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    # Input
    img_ax = axes[0].imshow(normalize(volume[2, :, :, z_init]), cmap="gray")
    axes[0].set_title("Input (T1ce)", fontsize=12)
    axes[0].axis("off")
    
    # Prediction
    pred_bg = axes[1].imshow(normalize(volume[2, :, :, z_init]), cmap="gray")
    pred_ol = axes[1].imshow(
        np.ma.masked_where(pred[:, :, z_init] == 0, pred[:, :, z_init]),
        cmap=cmap, alpha=0.6, vmin=0, vmax=3
    )
    axes[1].set_title("Prediction", fontsize=12)
    axes[1].axis("off")
    
    # Ground truth
    if gt is not None:
        gt_bg = axes[2].imshow(normalize(volume[2, :, :, z_init]), cmap="gray")
        gt_ol = axes[2].imshow(
            np.ma.masked_where(gt[:, :, z_init] == 0, gt[:, :, z_init]),
            cmap=cmap, alpha=0.6, vmin=0, vmax=3
        )
        axes[2].set_title("Ground Truth", fontsize=12)
        axes[2].axis("off")
    
    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, volume.shape[-1] - 1, valinit=z_init, valstep=1)
    
    def update(val):
        z = int(slider.val)
        img_ax.set_data(normalize(volume[2, :, :, z]))
        pred_bg.set_data(normalize(volume[2, :, :, z]))
        pred_ol.set_data(np.ma.masked_where(pred[:, :, z] == 0, pred[:, :, z]))
        if gt is not None:
            gt_bg.set_data(normalize(volume[2, :, :, z]))
            gt_ol.set_data(np.ma.masked_where(gt[:, :, z] == 0, gt[:, :, z]))
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    # Legend
    from matplotlib.patches import Patch
    legend = [Patch(facecolor='red', label='NCR'), 
              Patch(facecolor='yellow', label='Edema'),
              Patch(facecolor='blue', label='ET')]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, 0.96), 
               ncol=3, frameon=False, fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
    
    plt.show()

# =======================================================
# MAIN
# =======================================================
def main():
    parser = argparse.ArgumentParser(description='BraTS Inference')
    parser.add_argument('--checkpoint', type=str, 
                       default=r"D:\MajorProject\3D SD-NET\outputs\checkpoints\best_model(65.03)(StrongAug+SEB).pt",
                       help='Model checkpoint path')
    parser.add_argument('--case-dir', type=str,
                       default=r"D:\MajorProject\3D SD-NET\data\BraTS2021_Training_Data\BraTS2021_01335",
                       help='Case directory')
    parser.add_argument('--output-dir', type=str, default="outputs/predictions", help='Output directory')
    parser.add_argument('--save-viz', type=str, default=None, help='Save visualization')
    parser.add_argument('--no-postprocess', action='store_true', help='Skip postprocessing')
    args = parser.parse_args()
    
    print("="*70)
    print("🧠 BRATS INFERENCE")
    print("="*70)
    print(f"Device: {Config.DEVICE}")
    print(f"Case: {args.case_dir}")
    print("="*70)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = Custom3DSegModel(embed_dim=64, n_classes=Config.NUM_CLASSES).to(Config.DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=Config.DEVICE))
    model.eval()
    print("✅ Model loaded")
    
    # Preprocess
    print("\n🔄 Preprocessing...")
    volume = preprocess_volume(args.case_dir)
    
    # Predict
    print("🔮 Running inference...")
    pred = predict(model, volume)
    
    # Postprocess
    if not args.no_postprocess:
        print("🔧 Postprocessing...")
        pred = postprocess_brats(pred)
    
    # Load GT and compute Dice
    gt = preprocess_gt(args.case_dir)
    dice_scores = None
    if gt is not None:
        print("\n🎯 Computing Dice scores...")
        dice_scores = compute_dice(pred, gt)
        print_dice(dice_scores)
    
    # Save prediction
    case_name = Path(args.case_dir).name
    ref_path = Path(args.case_dir) / f"{case_name}_t1ce.nii.gz"
    out_path = os.path.join(args.output_dir, f"{case_name}_prediction.nii.gz")
    
    ref = nib.load(str(ref_path))
    nib.save(nib.Nifti1Image(pred.astype(np.uint8), ref.affine, ref.header), out_path)
    print(f"💾 Saved: {out_path}")
    
    # Visualize
    print("\n🎨 Launching visualization...")
    visualize(volume[0].cpu().numpy(), pred, gt, dice_scores, args.save_viz)
    
    print("\n✅ Complete!")

if __name__ == "__main__":
    main()