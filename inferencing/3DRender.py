import os
import sys
import torch
import numpy as np
import nibabel as nib
import pyvista as pv
from pathlib import Path
import argparse
from typing import Optional, Tuple, Dict
import warnings

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
    """Configuration class for rendering parameters"""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 4
    
    # Visualization settings
    BRAIN_THRESHOLD = 0.15  # Adjustable brain surface threshold
    BRAIN_OPACITY = 0.15
    BRAIN_COLOR = "lightgray"
    
    # Tumor colors and opacities
    TUMOR_SETTINGS = {
        1: {"name": "Necrotic/Non-Enhancing", "color": "red", "opacity": 0.95},
        2: {"name": "Edema", "color": "yellow", "opacity": 0.35},
        3: {"name": "Enhancing", "color": "blue", "opacity": 0.95},
    }
    
    # Window settings
    WINDOW_SIZE = (1200, 900)
    BACKGROUND_COLOR = "white"

# =======================================================
# VALIDATION
# =======================================================
def validate_paths(checkpoint_path: str, case_dir: str) -> None:
    """Validate that required paths exist"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if not os.path.exists(case_dir):
        raise FileNotFoundError(f"Case directory not found: {case_dir}")
    
    case_path = Path(case_dir)
    name = case_path.name
    required_files = [
        f"{name}_flair.nii.gz",
        f"{name}_t1.nii.gz",
        f"{name}_t1ce.nii.gz",
        f"{name}_t2.nii.gz",
    ]
    
    missing_files = [f for f in required_files if not (case_path / f).exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing files in case directory: {missing_files}")

# =======================================================
# MONAI PREPROCESS (EXACT TRAINING MATCH)
# =======================================================
def preprocess_volume(case_dir: str) -> torch.Tensor:
    """Preprocess MRI volume using MONAI transforms"""
    from monai.transforms import (
        LoadImaged,
        EnsureChannelFirstd,
        Orientationd,
        Spacingd,
        Resized,
        ScaleIntensityRanged,
        EnsureTyped,
        Compose,
    )

    case_path = Path(case_dir)
    name = case_path.name

    data = {
        "image": [
            str(case_path / f"{name}_flair.nii.gz"),
            str(case_path / f"{name}_t1.nii.gz"),
            str(case_path / f"{name}_t1ce.nii.gz"),
            str(case_path / f"{name}_t2.nii.gz"),
        ]
    }

    transforms = Compose([
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.0, 1.0, 1.0),
            mode="bilinear",
        ),
        Resized(
            keys=["image"],
            spatial_size=(128, 128, 128),
            mode="trilinear",
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=3000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=["image"]),
    ])

    vol = transforms(data)["image"]
    return vol.unsqueeze(0)  # (1,4,128,128,128)

def load_ground_truth(case_dir: str) -> Optional[np.ndarray]:
    """
    Load ground truth segmentation if available
    
    Args:
        case_dir: Path to case directory
        
    Returns:
        Ground truth array or None if not found
    """
    from monai.transforms import (
        LoadImage,
        EnsureChannelFirst,
        Orientation,
        Spacing,
        Resize,
        EnsureType,
        Compose,
    )
    
    case_path = Path(case_dir)
    name = case_path.name
    seg_path = case_path / f"{name}_seg.nii.gz"
    
    if not seg_path.exists():
        print("⚠️  No ground truth segmentation found. Dice scores will not be calculated.")
        return None
    
    try:
        transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=(1.0, 1.0, 1.0), mode="nearest"),
            Resize(spatial_size=(128, 128, 128), mode="nearest"),
            EnsureType(),
        ])
        
        gt = transforms(str(seg_path))
        
        # Apply BraTS label remapping (4->3)
        gt_np = gt.squeeze().numpy()
        gt_np[gt_np == 4] = 3
        
        print(f"✅ Ground truth loaded from {seg_path.name}")
        return gt_np
        
    except Exception as e:
        warnings.warn(f"Failed to load ground truth: {e}")
        return None
def load_model(checkpoint_path: str, device: torch.device, num_classes: int = 4) -> Custom3DSegModel:
    """Load model from checkpoint with error handling"""
    try:
        model = Custom3DSegModel(embed_dim=64, n_classes=num_classes).to(device)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        model.eval()
        print(f"✅ Model loaded successfully from {checkpoint_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

# =======================================================
# INFERENCE
# =======================================================
@torch.no_grad()
def infer(model: Custom3DSegModel, volume: torch.Tensor, device: torch.device) -> np.ndarray:
    """Run inference on preprocessed volume"""
    volume = volume.to(device)
    logits = model(volume)
    logits = torch.clamp(logits, -10, 10)
    preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
    
    # Clear CUDA cache if using GPU
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return preds[0].cpu().numpy()  # (128,128,128)

# =======================================================
# DICE SCORE CALCULATION
# =======================================================
def calculate_dice_score(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    """Calculate Dice score for a specific class"""
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    
    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (2.0 * intersection) / union

def calculate_all_dice_scores(pred: np.ndarray, gt: np.ndarray) -> Dict[int, float]:
    """Calculate Dice scores for all tumor classes"""
    dice_scores = {}
    
    # Individual class scores
    for label in [1, 2, 3]:
        dice_scores[label] = calculate_dice_score(pred, gt, label)
    
    # Whole tumor (all tumor classes combined)
    pred_wt = np.isin(pred, [1, 2, 3])
    gt_wt = np.isin(gt, [1, 2, 3])
    intersection = np.sum(pred_wt & gt_wt)
    union = np.sum(pred_wt) + np.sum(gt_wt)
    dice_scores['WT'] = (2.0 * intersection) / union if union > 0 else 1.0
    
    # Tumor core (necrotic + enhancing)
    pred_tc = np.isin(pred, [1, 3])
    gt_tc = np.isin(gt, [1, 3])
    intersection = np.sum(pred_tc & gt_tc)
    union = np.sum(pred_tc) + np.sum(gt_tc)
    dice_scores['TC'] = (2.0 * intersection) / union if union > 0 else 1.0
    
    # Enhancing tumor only
    dice_scores['ET'] = dice_scores[3]
    
    return dice_scores

# =======================================================
# STATISTICS
# =======================================================
def print_voxel_statistics(tumor: np.ndarray, dice_scores: Optional[Dict] = None) -> Dict[int, int]:
    """Print and return voxel statistics for each class"""
    total_voxels = tumor.size

    class_map = {
        0: "Background",
        1: "Necrotic / Non-Enhancing Tumor (Red)",
        2: "Edema (Yellow)",
        3: "Enhancing Tumor (Blue)",
    }

    print("\n📊 Voxel Statistics:")
    print("=" * 60)

    stats = {}
    for label, name in class_map.items():
        count = np.sum(tumor == label)
        percent = (count / total_voxels) * 100
        stats[label] = count
        print(f"{name:<45}: {count:>8} ({percent:6.3f}%)")

    print("=" * 60)
    print(f"{'Total voxels':<45}: {total_voxels:>8}\n")
    
    # Print Dice scores if available
    if dice_scores is not None:
        print("🎯 Dice Scores:")
        print("=" * 60)
        print(f"{'Class 1 (Necrotic/Non-Enhancing)':<40}: {dice_scores[1]:>6.4f}")
        print(f"{'Class 2 (Edema)':<40}: {dice_scores[2]:>6.4f}")
        print(f"{'Class 3 (Enhancing)':<40}: {dice_scores[3]:>6.4f}")
        print("-" * 60)
        print(f"{'Whole Tumor (WT)':<40}: {dice_scores['WT']:>6.4f}")
        print(f"{'Tumor Core (TC)':<40}: {dice_scores['TC']:>6.4f}")
        print(f"{'Enhancing Tumor (ET)':<40}: {dice_scores['ET']:>6.4f}")
        print("=" * 60)
        print(f"{'Mean Dice':<40}: {np.mean([dice_scores['WT'], dice_scores['TC'], dice_scores['ET']]):>6.4f}\n")
    
    return stats

# =======================================================
# ENHANCED 3D RENDERING
# =======================================================
def render_3d(
    brain: np.ndarray,
    tumor: np.ndarray,
    config: Config,
    dice_scores: Optional[Dict] = None,
    save_path: Optional[str] = None,
    interactive: bool = True,
    camera_position: Optional[Tuple[float, float, float]] = None,
    show_axes: bool = True,
    show_edges: bool = True,
    smooth_iterations: int = 20
) -> None:
    """
    Enhanced 3D rendering with multiple options
    
    Args:
        brain: Brain intensity volume (128,128,128)
        tumor: Tumor segmentation (128,128,128)
        config: Configuration object
        dice_scores: Dictionary of dice scores per class (optional)
        save_path: Path to save screenshot (optional)
        interactive: Whether to show interactive window
        camera_position: Custom camera position (x, y, z)
        show_axes: Whether to show coordinate axes
        show_edges: Whether to show brain surface edges
        smooth_iterations: Number of smoothing iterations for meshes
    """
    brain = brain.astype(np.float32)

    # Robust normalization
    p2, p98 = np.percentile(brain[brain > 0], (2, 98))
    brain = np.clip(brain, p2, p98)
    brain = (brain - p2) / (p98 - p2 + 1e-8)

    # Initialize plotter
    plotter = pv.Plotter(window_size=config.WINDOW_SIZE)
    plotter.set_background(config.BACKGROUND_COLOR)

    # ============================================
    # 🧠 BRAIN SURFACE
    # ============================================
    brain_binary = (brain > config.BRAIN_THRESHOLD).astype(np.uint8)
    
    try:
        brain_mesh = pv.wrap(brain_binary).contour([0.5])
        
        # Keep largest connected component
        if brain_mesh.n_points > 0:
            brain_mesh = brain_mesh.connectivity(largest=True)
            
            # Apply smoothing
            if smooth_iterations > 0:
                brain_mesh = brain_mesh.smooth(n_iter=smooth_iterations)
            
            plotter.add_mesh(
                brain_mesh,
                color=config.BRAIN_COLOR,
                opacity=config.BRAIN_OPACITY,
                smooth_shading=True,
                specular=0.5,
                specular_power=20,
                label="Brain Surface"
            )
            
            # Add subtle edges
            if show_edges:
                edges = brain_mesh.extract_feature_edges(
                    boundary_edges=True,
                    feature_edges=False,
                    manifold_edges=False,
                    non_manifold_edges=False
                )
                
                if edges.n_points > 0:
                    plotter.add_mesh(
                        edges,
                        color="black",
                        line_width=0.8,
                        opacity=0.3
                    )
    except Exception as e:
        warnings.warn(f"Failed to render brain surface: {e}")

    # ============================================
    # 🎯 TUMOR REGIONS
    # ============================================
    tumor_stats = {}
    
    for label, settings in config.TUMOR_SETTINGS.items():
        mask = (tumor == label).astype(np.uint8)
        voxel_count = mask.sum()
        tumor_stats[label] = voxel_count
        
        if voxel_count == 0:
            print(f"⚠️  No voxels found for {settings['name']} (label {label})")
            continue
        
        try:
            mesh = pv.wrap(mask).contour([0.5])
            
            if mesh.n_points > 0:
                # Apply smoothing for better visualization
                if smooth_iterations > 0:
                    mesh = mesh.smooth(n_iter=smooth_iterations)
                
                # Create label with dice score if available
                if dice_scores and label in dice_scores:
                    mesh_label = f"{settings['name']} (Dice: {dice_scores[label]:.3f})"
                else:
                    mesh_label = f"{settings['name']} (n={voxel_count})"
                
                plotter.add_mesh(
                    mesh,
                    color=settings["color"],
                    opacity=settings["opacity"],
                    smooth_shading=True,
                    specular=0.6,
                    specular_power=25,
                    label=mesh_label
                )
                print(f"✅ Rendered {settings['name']}: {voxel_count} voxels")
        except Exception as e:
            warnings.warn(f"Failed to render {settings['name']}: {e}")

    # ============================================
    # CAMERA & VISUALIZATION
    # ============================================
    if show_axes:
        plotter.add_axes(
            xlabel='X (L-R)',
            ylabel='Y (P-A)', 
            zlabel='Z (I-S)',
            line_width=3
        )
    
    # Add legend with dice scores
    plotter.add_legend(size=(0.25, 0.25), face='circle')
    
    # Add text annotation with overall metrics
    if dice_scores:
        text = (
            f"Dice Scores:\n"
            f"WT: {dice_scores['WT']:.4f}\n"
            f"TC: {dice_scores['TC']:.4f}\n"
            f"ET: {dice_scores['ET']:.4f}\n"
            f"Mean: {np.mean([dice_scores['WT'], dice_scores['TC'], dice_scores['ET']]):.4f}"
        )
        plotter.add_text(
            text,
            position='upper_right',
            font_size=10,
            color='black',
            font='arial'
        )
    
    # Set camera position
    if camera_position:
        plotter.camera_position = camera_position
    else:
        plotter.camera_position = 'iso'
    
    plotter.reset_camera()
    
    # Enable anti-aliasing for better quality
    plotter.enable_anti_aliasing('fxaa')
    
    # Save screenshot if requested
    if save_path:
        print(f"💾 Saving screenshot to: {save_path}")
        plotter.screenshot(save_path, transparent_background=False)
    
    # Show interactive window
    if interactive:
        plotter.show()
    else:
        plotter.close()

# =======================================================
# MAIN PIPELINE
# =======================================================
def main():
    parser = argparse.ArgumentParser(description='3D Brain Tumor Visualization')
    parser.add_argument('--checkpoint', type=str, 
                       default=r"D:\MajorProject\3D SD-NET\outputs\checkpoints\best_model(65.03)(StrongAug+SEB).pt",
                       help='Path to model checkpoint')
    parser.add_argument('--case-dir', type=str,
                       default=r"D:\MajorProject\3D SD-NET\data\BraTS2021_Training_Data\BraTS2021_00310",
                       help='Path to case directory')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save screenshot')
    parser.add_argument('--no-interactive', action='store_true',
                       help='Disable interactive window (use with --save)')
    parser.add_argument('--brain-threshold', type=float, default=0.15,
                       help='Threshold for brain surface (0-1)')
    parser.add_argument('--smooth', type=int, default=20,
                       help='Number of smoothing iterations')
    parser.add_argument('--no-edges', action='store_true',
                       help='Disable brain surface edges')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧠 3D BRAIN TUMOR SEGMENTATION VISUALIZATION")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Case: {args.case_dir}")
    print("=" * 60)
    
    # Validate paths
    try:
        validate_paths(args.checkpoint, args.case_dir)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Load model
    try:
        model = load_model(args.checkpoint, Config.DEVICE, Config.NUM_CLASSES)
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    # Preprocess volume
    print("\n🔄 Preprocessing volume...")
    volume = preprocess_volume(args.case_dir)
    print(f"✅ Volume shape: {volume.shape}")
    
    # Run inference
    print("\n🔮 Running inference...")
    preds = infer(model, volume, Config.DEVICE)
    print(f"✅ Prediction shape: {preds.shape}")
    
    # Load ground truth and calculate dice scores
    dice_scores = None
    ground_truth = load_ground_truth(args.case_dir)
    
    if ground_truth is not None:
        print("\n🎯 Calculating Dice scores...")
        dice_scores = calculate_all_dice_scores(preds, ground_truth)
    
    # Print statistics
    stats = print_voxel_statistics(preds, dice_scores)
    
    # Check if any tumor regions exist
    has_tumor = any(stats.get(i, 0) > 0 for i in [1, 2, 3])
    if not has_tumor:
        print("⚠️  Warning: No tumor regions detected in this case!")
    
    # Update config with custom threshold
    Config.BRAIN_THRESHOLD = args.brain_threshold
    
    # Use T1ce for visualization (contrast-enhanced)
    brain = volume[0, 2].cpu().numpy()
    
    # Render
    print("\n🎨 Rendering 3D visualization...")
    render_3d(
        brain=brain,
        tumor=preds,
        config=Config,
        dice_scores=dice_scores,
        save_path=args.save,
        interactive=not args.no_interactive,
        show_edges=not args.no_edges,
        smooth_iterations=args.smooth
    )
    
    print("\n✅ Rendering complete!")

if __name__ == "__main__":
    main()