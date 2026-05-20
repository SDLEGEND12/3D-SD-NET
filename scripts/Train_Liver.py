import os
import sys
import torch
from tqdm import tqdm
import monai
import numpy as np
import logging
logging.getLogger("monai").setLevel(logging.ERROR)
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot
from monai.losses import DiceLoss, FocalLoss
# =======================================================
# PyTorch 2.6 + MONAI Fix
# =======================================================
_original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

torch.serialization.add_safe_globals([
    monai.data.meta_tensor.MetaTensor,
    np.core.multiarray._reconstruct,
])

# =======================================================
# Imports
# =======================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data_loaders.dataset_liver import get_lung_dataloader_from_csv
from model.model import Custom3DSegModel

from torch.amp import autocast, GradScaler

# =======================================================
# CONFIG
# =======================================================
DATA_ROOT = "data/Task03_Liver"
TRAIN_CSV = "data/splits/train_liver.csv"
VAL_CSV   = "data/splits/val_liver.csv"

CACHE_TRAIN = "cache/liver/train"
CACHE_VAL   = "cache/liver/val"

CHECKPOINT_DIR = "outputs/checkpoints_liver"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

NUM_CLASSES = 3
CLASS_NAMES = ["Background", "Liver", "Liver Tumour"]

BATCH_SIZE = 1
TOTAL_EPOCHS = 200
SAVE_EVERY = 5

LR = 1e-4
MIN_LR = 1e-6

NUM_WORKERS = 0

RESUME = True
RESUME_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pt")

# Per-class dice metric
dice_metric = DiceMetric(
    include_background=False,
    reduction="mean_batch",
    get_not_nans=False
)

# =======================================================
# VALIDATION
# =======================================================
@torch.no_grad()
def validate(model, loader, device, dice_loss_fn, focal_loss_fn):
    model.eval()
    val_loss = 0.0
    dice_metric.reset()
    valid_batches = 0

    class_dice_scores = [[] for _ in range(NUM_CLASSES - 1)]
    class_pixel_counts = [0 for _ in range(NUM_CLASSES)]

    for batch in loader:
        try:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(images)

            if not torch.isfinite(logits).all():
                continue

            loss = dice_loss_fn(logits, labels) + 4.0 * focal_loss_fn(logits, labels)

            if not torch.isfinite(loss):
                continue

            val_loss += loss.item()
            valid_batches += 1

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1, keepdim=True)

            labels_onehot = one_hot(labels, num_classes=NUM_CLASSES)
            preds_onehot  = one_hot(preds,  num_classes=NUM_CLASSES)

            dice_metric(y_pred=preds_onehot, y=labels_onehot)

            for class_idx in range(1, NUM_CLASSES):
                y_true_class = labels_onehot[:, class_idx:class_idx+1]
                y_pred_class = preds_onehot[:, class_idx:class_idx+1]

                class_pixel_counts[class_idx] += y_true_class.sum().item()

                intersection = (y_true_class * y_pred_class).sum()
                union = y_true_class.sum() + y_pred_class.sum()

                if union > 0:
                    dice_score = (2.0 * intersection / (union + 1e-8)).item()
                    class_dice_scores[class_idx - 1].append(dice_score)

        except Exception as e:
            print(f"⚠️ Validation error: {e}")
            continue

    if valid_batches == 0:
        return float("inf"), 0.0, [0.0] * (NUM_CLASSES - 1)

    dice_scores = dice_metric.aggregate()
    if dice_scores.numel() > 1:
        mean_dice = dice_scores.mean().item()
    else:
        mean_dice = dice_scores.item()

    dice_metric.reset()
    avg_val_loss = val_loss / valid_batches

    per_class_dice = []
    for class_idx in range(NUM_CLASSES - 1):
        if len(class_dice_scores[class_idx]) > 0:
            per_class_dice.append(np.mean(class_dice_scores[class_idx]))
        else:
            per_class_dice.append(0.0)

    return avg_val_loss, mean_dice, per_class_dice


def save_checkpoint(epoch, model, optimizer, scheduler, scaler,
                    best_val_loss, best_val_dice):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "best_val_dice": best_val_dice,
    }, RESUME_CHECKPOINT)


def load_checkpoint(model, optimizer, scheduler, scaler, device):
    checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    scaler.load_state_dict(checkpoint["scaler_state"])
    best_val_dice = checkpoint.get("best_val_dice", 0.0)
    return checkpoint["epoch"], checkpoint["best_val_loss"], best_val_dice


# =======================================================
# TRAIN
# =======================================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ Using device:", device)

    train_loader = get_lung_dataloader_from_csv(
        csv_path=TRAIN_CSV,
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        train=True,
        num_workers=NUM_WORKERS,
        cache_dir=CACHE_TRAIN
    )

    val_loader = get_lung_dataloader_from_csv(
        csv_path=VAL_CSV,
        data_root=DATA_ROOT,
        batch_size=1,
        train=False,
        num_workers=NUM_WORKERS,
        cache_dir=CACHE_VAL
    )

    model = Custom3DSegModel(
        in_channels=1,
        embed_dim=64,
        n_classes=NUM_CLASSES,
        final_activation=None
    ).to(device)

    def init_weights(m):
        if isinstance(m, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, (torch.nn.InstanceNorm3d, torch.nn.LayerNorm)):
            if m.weight is not None:
                torch.nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    model.apply(init_weights)

    #class_weights = torch.tensor([0.1, 1.0, 4.0]).to(device)  # [BG, Liver, Tumour]

    dice_loss_fn = DiceLoss(
        to_onehot_y=True, softmax=True,
        include_background=False,
        smooth_nr=1e-3, smooth_dr=1e-3, squared_pred=True
    )

    class_weights = torch.tensor([0.1, 0.8, 8.0]).to(device)  # heavy tumour weight

    focal_loss_fn = FocalLoss(
        to_onehot_y=True, gamma=5.0,
        weight=class_weights
    )
    

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.7,
        patience=10,
        min_lr=MIN_LR,
        threshold=1e-4,
        cooldown=5
    )

    scaler = GradScaler('cuda')

    start_epoch = 1
    best_val_loss = float("inf")
    best_val_dice = 0.0

    if RESUME and os.path.exists(RESUME_CHECKPOINT):
        print(f"🔄 Resuming from {RESUME_CHECKPOINT}")
        last_epoch, best_val_loss, best_val_dice = load_checkpoint(
            model, optimizer, scheduler, scaler, device
        )
        start_epoch = last_epoch + 1
        print(f"✅ Resumed from epoch {last_epoch}")

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        valid_batches = 0
        skipped_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{TOTAL_EPOCHS}", ncols=100)

        for batch in pbar:
            try:
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)

                if not torch.isfinite(images).all():
                    skipped_batches += 1
                    continue

                optimizer.zero_grad(set_to_none=True)

                outputs = model(images)

                if not torch.isfinite(outputs).all():
                    skipped_batches += 1
                    continue

                outputs = torch.clamp(outputs, min=-10, max=10)

                loss  = dice_loss_fn(outputs, labels) + 5.0 * focal_loss_fn(outputs, labels)

                if not torch.isfinite(loss):
                    skipped_batches += 1
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                valid_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}", skip=skipped_batches)

            except Exception as e:
                print(f"⚠️ Training error: {e}")
                skipped_batches += 1
                continue

        if valid_batches == 0:
            print("⚠️ No valid batches in this epoch!")
            continue

        avg_train_loss = train_loss / valid_batches
        avg_val_loss, mean_dice, per_class_dice = validate(model, val_loader, device, dice_loss_fn, focal_loss_fn)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"📊 Epoch {epoch:03d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Mean Dice: {mean_dice:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Skipped: {skipped_batches}"
        )

        print("   Per-Class Dice Scores:")
        for class_idx in range(1, NUM_CLASSES):
            print(f"      {CLASS_NAMES[class_idx]}: {per_class_dice[class_idx-1]:.4f}")

        #if mean_dice > best_val_dice:
         #   best_val_dice = mean_dice
         #   torch.save(
         #       model.state_dict(),
          #      os.path.join(CHECKPOINT_DIR, "best_model.pt")
          #  )
          #  print(f"🏆 Best model saved (Dice = {best_val_dice:.4f})")

        tumour_dice = per_class_dice[1]
        if tumour_dice > best_val_dice:
            best_val_dice = tumour_dice
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            print(f"🏆 Best model saved (Tumour Dice = {best_val_dice:.4f})")

        if epoch % 25 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pt")
            )
            print(f"💾 Saved snapshot at epoch {epoch}")

        if epoch % SAVE_EVERY == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, scaler,
                          best_val_loss, best_val_dice)
            print(f"💾 Resumable checkpoint saved at epoch {epoch}")


# =======================================================
# MAIN
# =======================================================
if __name__ == "__main__":
    train()