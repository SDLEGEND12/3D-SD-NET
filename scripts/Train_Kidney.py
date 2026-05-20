"""
Training script: Custom3DSegModel (Transformer) on Kidney Tumour Dataset
Optimised for RTX 3050 4GB VRAM / 16GB RAM

Architecture decisions:
  - Custom3DSegModel: PatchEmbedding + LocalSelfAttention encoder,
    ASPP + GlobalSelfAttention bottleneck, skip-connection decoder
  - 3-class output (background, kidney, tumour)
  - DiceLoss + FocalLoss combined manually for per-class tumour upweighting
  - Tumour class weighted 16x in FocalLoss — severely underrepresented
  - Kidney class weighted 2x — anchors tumour localisation

VRAM decisions:
  - batch_size=1  (128³ patches with embed_dim=32 fits ~3.8GB with AMP)
  - sw_batch_size=1 during val inference
  - AMP enabled — halves VRAM for activations
  - Attention computed in float32 to prevent NaN from float16 softmax
  - num_workers=0 on Windows (spawn overhead > single-thread for cached reads)

Scheduler:
  - ReduceLROnPlateau on (1 - tumour_dice), patience=10
  - Tumour Dice is a smooth, meaningful signal vs noisy patch-sampled val loss

Metric tracked:
  - Per-class Dice: kidney_dice, tumour_dice, mean_dice
  - Best model saved on tumour_dice
  - Tumour Dice is the primary clinical metric

Usage:
    python scripts/Train_kidney_cnn.py
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from monai.losses import DiceLoss, FocalLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Compose

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.dataset_kidney_transformer import get_kidney_loaders
from model.model import Custom3DSegModel


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
ROI_SIZE    = (128, 128, 128)
SW_OVERLAP  = 0.25
N_CLASSES   = 3             # 0=background, 1=kidney, 2=tumour
CLASS_NAMES = ["Kidney", "Tumour"]


# ─────────────────────────────────────────────
#  ARGS
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    type=str,   default=r"data")
    p.add_argument("--cache_dir",    type=str,   default=r"cache/kidney")
    p.add_argument("--output_dir",   type=str,   default="outputs/kidney_transformer")
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch_size",   type=int,   default=2,
                   help="1 is safest for 4GB VRAM with 128³ patches and AMP")
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_workers",  type=int,   default=0,
                   help="0 is fastest on Windows for PersistentDataset cache reads")
    p.add_argument("--val_every",    type=int,   default=3,
                   help="Validate every N epochs — full-volume SW inference is slow on 4GB")
    p.add_argument("--amp",          action="store_true", default=True)
    return p.parse_args()


# ─────────────────────────────────────────────
#  LOSS
# ─────────────────────────────────────────────
def build_criterion(device: torch.device):
    """
    DiceLoss + FocalLoss combined manually.

    Weight rationale [background=0.1, kidney=2.0, tumour=16.0]:
        - Background suppressed (0.1) — required by MONAI FocalLoss when to_onehot_y=True
        - Kidney (class 1) mild upweight — needed as spatial anchor for tumour
        - Tumour (class 2) 16x — severely underrepresented in 128³ patches
          (e.g. 2 cm³ tumour = ~593 voxels out of 2,097,152 total = 0.028%)

    lambda_focal=1.5: stronger focal contribution for extreme class imbalance.
    """
    dice_loss_fn = DiceLoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        reduction="mean",
    )

    focal_loss_fn = FocalLoss(
        to_onehot_y=True,
        gamma=3.0,
        weight=torch.tensor([0.1, 2.0, 16.0]).to(device),  # [background, kidney, tumour]
        reduction="mean",
    )

    lambda_focal = 1.5

    def criterion(pred, label):
        return dice_loss_fn(pred, label) + lambda_focal * focal_loss_fn(pred, label)

    return criterion


# ─────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────
def _sample_patch(tensor, roi):
    """
    Randomly crop a single roi-sized patch from a (C, D, H, W) or (B, C, D, H, W) tensor.
    Used to compute val loss on a patch rather than the full volume to avoid OOM.
    Note: for volumes barely larger than ROI_SIZE, the crop offset may be 0
    (fixed crop) — val loss is still a meaningful estimate, just less random.
    """
    t = tensor.squeeze(0) if tensor.dim() == 5 else tensor
    _, d, h, w = t.shape[-4], t.shape[-3], t.shape[-2], t.shape[-1]
    rd, rh, rw = roi
    d0 = torch.randint(0, max(d - rd, 1), (1,)).item()
    h0 = torch.randint(0, max(h - rh, 1), (1,)).item()
    w0 = torch.randint(0, max(w - rw, 1), (1,)).item()
    return t[..., d0:d0+rd, h0:h0+rh, w0:w0+rw].unsqueeze(0)


def validate(model, val_loader, criterion, dice_metric, device, amp_enabled):
    """
    Val loss: computed on a randomly sampled 128³ patch from each volume's
    SW inference output — avoids OOM from one_hot expanding the full CT volume
    (B, 3, D, H, W) at full resolution on 4GB VRAM.

    Val Dice: computed on the full volume via sliding window — accurate metric.
    """
    model.eval()

    post_pred  = Compose([AsDiscrete(argmax=True, to_onehot=N_CLASSES)])
    post_label = Compose([AsDiscrete(to_onehot=N_CLASSES)])

    val_loss = 0.0
    n_val    = len(val_loader)

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc="  Validating", ncols=80,
                       leave=False, unit="vol")
        for batch in val_bar:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            labels = torch.clamp(labels.long(), 0, N_CLASSES - 1)

            with autocast(device_type="cuda", enabled=amp_enabled):
                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=ROI_SIZE,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=SW_OVERLAP,
                    mode="gaussian",
                )

            # Convert to plain tensor — sliding_window_inference returns a
            # MetaTensor whose metadata causes decollate_batch to crash
            outputs_plain = outputs.as_tensor() if hasattr(outputs, 'as_tensor') else outputs
            labels_plain  = labels.as_tensor()  if hasattr(labels,  'as_tensor') else labels

            # ── Val loss on a single 128³ patch (memory-safe) ──
            with autocast(device_type="cuda", enabled=amp_enabled):
                out_patch  = _sample_patch(outputs_plain, ROI_SIZE)
                lbl_patch  = _sample_patch(labels_plain,  ROI_SIZE)
                patch_loss = criterion(out_patch, lbl_patch)
            val_loss += patch_loss.item()

            # ── Full-volume Dice ──
            outputs_list = [outputs_plain[i] for i in range(outputs_plain.shape[0])]
            labels_list  = [labels_plain[i]  for i in range(labels_plain.shape[0])]

            dice_metric(
                y_pred=[post_pred(o) for o in outputs_list],
                y=[post_label(l) for l in labels_list],
            )

    dice_scores = dice_metric.aggregate()   # shape: (2,) → [kidney_dice, tumour_dice]
    dice_metric.reset()

    kidney_dice = dice_scores[0].item() if dice_scores.numel() >= 1 else 0.0
    tumour_dice = dice_scores[1].item() if dice_scores.numel() >= 2 else 0.0
    mean_dice   = (kidney_dice + tumour_dice) / 2

    return val_loss / max(n_val, 1), mean_dice, kidney_dice, tumour_dice


# ─────────────────────────────────────────────
#  CACHE WARMUP CHECK
# ─────────────────────────────────────────────
def check_cache_speed(train_loader):
    print("  Checking cache speed (loading 1 batch)...", end=" ", flush=True)
    t       = time.time()
    batch   = next(iter(train_loader))
    elapsed = time.time() - t
    if elapsed > 5.0:
        print(f"{elapsed:.1f}s  ⚠ Cache cold — epoch 1 will be slow, epoch 2+ will be fast")
    else:
        print(f"{elapsed:.1f}s  ✓ Cache warm")
    del batch


# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────
def train(args):
    os.makedirs(args.output_dir, exist_ok=True)

    log_path = os.path.join(args.output_dir, "train_log.txt")
    log_file = open(log_path, "a", encoding="utf-8")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    # ── GPU settings ──
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device : {device}")
    if torch.cuda.is_available():
        log(f"GPU    : {torch.cuda.get_device_name(0)}")
        log(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data ──
    log("\nLoading datasets...")
    train_loader, val_loader = get_kidney_loaders(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    n_steps = len(train_loader)
    log(f"Train steps/epoch : {n_steps}  (batch_size={args.batch_size})")
    log(f"Val volumes       : {len(val_loader)}")

    check_cache_speed(train_loader)

    # ── Model ──
    # embed_dim=32: bottleneck has 32*16=512 channels at 2³ spatial resolution.
    # embed_dim=64 would give 1024 channels — OOM on 4GB with 128³ input.
    model = Custom3DSegModel(
        in_channels=1,          # single-channel CT
        embed_dim=32,           # 32*16=512 bottleneck channels — fits 4GB with AMP
        n_classes=N_CLASSES,    # 3: background, kidney, tumour
        final_activation=None,  # raw logits — loss handles softmax
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model parameters  : {total_params / 1e6:.2f}M")

    # ── Loss ──
    criterion = build_criterion(device)
    log("Loss: DiceLoss + 1.5 × FocalLoss(gamma=3.0, weight=[bg=0.1, kidney=2.0, tumour=16.0])")

    # ── Optimiser ──
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=True,
    )

    # ── Scheduler ──
    # Steps on (1 - tumour_dice) — smooth, meaningful signal.
    # patience=10: transformer needs more epochs to stabilise than pure CNN.
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
    )

    scaler = GradScaler(device="cuda", enabled=args.amp)

    # include_background=False → dice_scores shape: (2,) = [kidney, tumour]
    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean_batch",
        get_not_nans=False,
    )

    # ── Auto-resume ──
    start_epoch = 0
    best_dice   = 0.0

    RESUME_PATH = os.path.join(args.output_dir, "latest_checkpoint.pth")
    if os.path.exists(RESUME_PATH):
        ckpt = torch.load(RESUME_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_dice   = ckpt.get("best_dice", 0.0)
        log(f"🔄 Auto-resumed from epoch {start_epoch} | Best Tumour Dice: {best_dice:.4f}")

    log("=" * 70)
    log(f"Training for {args.epochs} epochs | ReduceLROnPlateau(patience=10) | "
        f"batch={args.batch_size} | AMP={args.amp}")
    log(f"ROI: {ROI_SIZE} | SW overlap: {SW_OVERLAP} | Classes: {N_CLASSES}")
    log(f"Model: Custom3DSegModel | embed_dim=32 | Scheduler: 1 - tumour_dice")
    log("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        model.train()

        epoch_loss = 0.0
        skipped    = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            total=n_steps,
            ncols=110,
            unit="it",
            dynamic_ncols=False,
        )

        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            labels = torch.clamp(labels.long(), 0, N_CLASSES - 1)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=args.amp):
                preds = model(images)
                loss  = criterion(preds, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                skipped += 1

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", skip=skipped)

        pbar.close()

        avg_train_loss = epoch_loss / max(n_steps, 1)
        current_lr     = optimizer.param_groups[0]["lr"]

        # ── Validation ──
        if (epoch + 1) % args.val_every == 0:
            avg_val_loss, mean_dice, kidney_dice, tumour_dice = validate(
                model, val_loader, criterion, dice_metric, device, args.amp
            )

            # Step on (1 - tumour_dice) — smooth signal, avoids noisy patch loss
            scheduler.step(1.0 - tumour_dice)

            is_best = tumour_dice > best_dice
            if is_best:
                best_dice = tumour_dice
                torch.save({
                    "epoch":       epoch,
                    "model":       model.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "scheduler":   scheduler.state_dict(),
                    "best_dice":   best_dice,
                    "kidney_dice": kidney_dice,
                    "tumour_dice": tumour_dice,
                }, os.path.join(args.output_dir, "best_model.pth"))

            summary = (
                f"📊 Epoch {epoch+1:03d} | "
                f"Train: {avg_train_loss:.4f} | "
                f"Val: {avg_val_loss:.4f} | "
                f"Mean: {mean_dice:.4f} | "
                f"Kidney: {kidney_dice:.4f} | "
                f"Tumour: {tumour_dice:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Skip: {skipped}"
            )
            if is_best:
                summary += "  ⭐ new best"

        else:
            summary = (
                f"📊 Epoch {epoch+1:03d} | "
                f"Train: {avg_train_loss:.4f} | "
                f"Val: ------ | "
                f"Mean: ------ | "
                f"Kidney: ------ | "
                f"Tumour: ------ | "
                f"LR: {current_lr:.2e} | "
                f"Skip: {skipped}"
            )

        log(summary)

        # ── Checkpoint every 5 epochs ──
        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_dice": best_dice,
            }, os.path.join(args.output_dir, "latest_checkpoint.pth"))
            log(f"💾 Checkpoint saved at epoch {epoch+1}")

    log("=" * 70)
    log("Training complete.")
    log(f"Best Tumour Dice : {best_dice:.4f}")
    log_file.close()


if __name__ == "__main__":
    args = parse_args()
    train(args)