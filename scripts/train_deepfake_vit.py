"""
Satya Drishti — ViT-B/16 Deepfake Detector Training
====================================================
Fine-tunes google/vit-base-patch16-224-in21k for binary deepfake detection
(real vs fake) on the image forensics dataset.

Features:
- Loads ViT-B/16 from local HuggingFace pretrained directory
- Layer freezing: first 8/12 blocks frozen, last 4 + classifier trainable
- HuggingFace ViT normalization: mean/std [0.5, 0.5, 0.5]
- Mixed precision training (AMP) for 4GB VRAM
- Cosine annealing LR with warmup, gradient clipping
- Label smoothing, JPEG compression augmentation, RandomErasing
- Early stopping on validation AUC

Usage:
    python scripts/train_deepfake_vit.py --config configs/train_deepfake_vit.yaml
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import ViTForImageClassification

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─── JPEG Compression Augmentation ───


class JPEGCompression:
    """Simulates JPEG compression artifacts at random quality levels."""

    def __init__(self, quality_range=(30, 95)):
        self.quality_range = quality_range

    def __call__(self, img):
        import io
        quality = np.random.randint(self.quality_range[0], self.quality_range[1] + 1)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


# ─── Dataset ───


class DeepfakeDataset(Dataset):
    """
    Loads images from a JSON manifest with format:
    [{"path": "path/to/image.jpg", "label": 0}, ...]
    where label 0 = real, 1 = fake.
    """

    def __init__(self, manifest_entries: list, transform=None):
        self.entries = manifest_entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img = Image.open(entry["path"]).convert("RGB")
        label = entry["label"]

        if self.transform:
            img = self.transform(img)

        return img, label


# ─── Build Transforms ───


# ViT-B/16 uses [0.5, 0.5, 0.5] normalization (NOT ImageNet stats)
VIT_MEAN = [0.5, 0.5, 0.5]
VIT_STD = [0.5, 0.5, 0.5]


def build_transforms(cfg):
    input_size = cfg["model"]["input_size"]
    aug_cfg = cfg["data"].get("augmentation", {})

    train_transforms = []

    # RandomResizedCrop doubles as resize + crop augmentation
    if aug_cfg.get("random_resized_crop"):
        scale = tuple(aug_cfg["random_resized_crop"]["scale"])
        train_transforms.append(transforms.RandomResizedCrop(input_size, scale=scale))
    else:
        train_transforms.append(transforms.Resize((input_size, input_size)))

    if aug_cfg.get("horizontal_flip", False):
        train_transforms.append(transforms.RandomHorizontalFlip())

    if aug_cfg.get("color_jitter"):
        cj = aug_cfg["color_jitter"]
        train_transforms.append(transforms.ColorJitter(
            brightness=cj.get("brightness", 0),
            contrast=cj.get("contrast", 0),
            saturation=cj.get("saturation", 0),
            hue=cj.get("hue", 0),
        ))

    if aug_cfg.get("jpeg_compression", {}).get("enabled", False):
        qr = aug_cfg["jpeg_compression"].get("quality_range", [30, 95])
        train_transforms.append(JPEGCompression(tuple(qr)))

    train_transforms += [
        transforms.ToTensor(),
        transforms.Normalize(mean=VIT_MEAN, std=VIT_STD),
    ]

    # RandomErasing applied after ToTensor
    if aug_cfg.get("random_erasing", {}).get("enabled", False):
        p = aug_cfg["random_erasing"].get("probability", 0.1)
        train_transforms.append(transforms.RandomErasing(p=p))

    val_transforms = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=VIT_MEAN, std=VIT_STD),
    ]

    return transforms.Compose(train_transforms), transforms.Compose(val_transforms)


# ─── Build Model with Layer Freezing ───


def build_model(cfg):
    """Load ViT-B/16 from local pretrained dir and freeze early layers."""
    pretrained_dir = cfg["model"]["pretrained_dir"]
    num_classes = cfg["model"]["num_classes"]
    freeze_blocks = cfg["model"].get("freeze_blocks", 8)

    # Load from local dir or HuggingFace hub
    model_source = pretrained_dir if os.path.isdir(pretrained_dir) else "google/vit-base-patch16-224-in21k"

    print(f"[Model] Loading ViT-B/16 from {model_source}")
    model = ViTForImageClassification.from_pretrained(
        model_source,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    # Freeze embeddings
    for param in model.vit.embeddings.parameters():
        param.requires_grad = False

    # Freeze first N encoder blocks
    for i in range(freeze_blocks):
        for param in model.vit.encoder.layer[i].parameters():
            param.requires_grad = False

    total_blocks = len(model.vit.encoder.layer)
    trainable_blocks = total_blocks - freeze_blocks

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"[Model] Total params: {total_params / 1e6:.1f}M")
    print(f"[Model] Trainable: {trainable_params / 1e6:.1f}M (blocks {freeze_blocks}-{total_blocks - 1} + classifier)")
    print(f"[Model] Frozen: {frozen_params / 1e6:.1f}M (embeddings + blocks 0-{freeze_blocks - 1})")

    return model


# ─── Training Loop ───


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, max_grad_norm):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(pixel_values=images)
            loss = criterion(outputs.logits, labels)

        scaler.scale(loss).backward()

        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # Progress every 500 batches
        if (batch_idx + 1) % 500 == 0:
            print(f"  [batch {batch_idx + 1}/{num_batches}] loss={running_loss / total:.4f} acc={correct / total:.4f}")

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(pixel_values=images)
            loss = criterion(outputs.logits, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        probs = torch.softmax(outputs.logits, dim=1)[:, 1]
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return running_loss / total, correct / total, auc


# ─── Main ───


def main():
    parser = argparse.ArgumentParser(description="Train ViT-B/16 Deepfake Detector")
    parser.add_argument("--config", type=str, default="configs/train_deepfake_vit.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    if device.type == "cuda":
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Train] VRAM: {vram_gb:.1f} GB")

    train_cfg = cfg["training"]
    use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"
    max_grad_norm = train_cfg.get("gradient_clip", 1.0)

    # Load manifest
    manifest_path = cfg["data"]["manifest_path"]
    if not os.path.exists(manifest_path):
        print(f"[Error] Manifest not found: {manifest_path}")
        print("[Error] Run: python scripts/prepare_image_forensics_manifest.py")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Split data
    np.random.seed(42)
    np.random.shuffle(manifest)
    n = len(manifest)
    train_end = int(n * cfg["data"]["train_split"])
    val_end = train_end + int(n * cfg["data"]["val_split"])

    train_entries = manifest[:train_end]
    val_entries = manifest[train_end:val_end]
    test_entries = manifest[val_end:]

    real_train = sum(1 for e in train_entries if e["label"] == 0)
    fake_train = sum(1 for e in train_entries if e["label"] == 1)

    print(f"[Train] Dataset: {n} total — {len(train_entries)} train ({real_train} real, {fake_train} fake), "
          f"{len(val_entries)} val, {len(test_entries)} test")

    # Transforms and dataloaders
    train_transform, val_transform = build_transforms(cfg)
    train_dataset = DeepfakeDataset(train_entries, transform=train_transform)
    val_dataset = DeepfakeDataset(val_entries, transform=val_transform)

    # num_workers=0 on Windows to avoid multiprocessing pickle/spawn issues
    num_workers = 0 if sys.platform == "win32" else 2
    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg["batch_size"],
        shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg["batch_size"],
        shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    # Clear VRAM before loading model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Model
    model = build_model(cfg).to(device)

    if device.type == "cuda":
        vram_used = torch.cuda.memory_allocated(0) / 1e9
        print(f"[Train] VRAM after model load: {vram_used:.2f} GB")

    # Loss with label smoothing
    label_smoothing = train_cfg.get("label_smoothing", 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Optimizer — only on trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    # Cosine annealing after warmup
    warmup_epochs = train_cfg.get("warmup_epochs", 2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"] - warmup_epochs,
    )
    scaler = torch.GradScaler(enabled=use_amp)

    # Output dirs
    checkpoint_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = cfg["output"].get("log_dir", "logs/deepfake_vit_training")
    os.makedirs(log_dir, exist_ok=True)

    best_auc = 0.0
    patience_counter = 0
    patience = train_cfg.get("early_stopping_patience", 5)
    best_model_path = os.path.join(checkpoint_dir, cfg["output"]["best_model_name"])
    start_epoch = 1

    # Resume from checkpoint
    if args.resume:
        resume_path = args.resume
        if not os.path.exists(resume_path):
            print(f"[Error] Resume checkpoint not found: {resume_path}")
            sys.exit(1)
        print(f"[Train] Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        # Try to restore optimizer state; skip if param groups changed
        # (e.g. different freeze_blocks between original and fine-tune config)
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_auc = checkpoint.get("val_auc", 0.0)
            print(f"[Train] Full resume from epoch {start_epoch}, best AUC so far: {best_auc:.4f}")
            # Advance scheduler to the correct position
            for _ in range(warmup_epochs, checkpoint["epoch"]):
                scheduler.step()
        except (ValueError, RuntimeError) as e:
            print(f"[Train] Optimizer state mismatch (different trainable params), loading model weights only")
            print(f"[Train] Fine-tuning from epoch 1 with fresh optimizer (prior AUC={checkpoint.get('val_auc', 0.0):.4f})")
            # Keep start_epoch=1, best_auc=0.0 — fresh training run with pretrained weights

    print(f"\n[Train] Starting ViT-B/16 fine-tuning for {train_cfg['epochs']} epochs (from epoch {start_epoch})")
    print(f"[Train] AMP={use_amp}, batch_size={train_cfg['batch_size']}, label_smoothing={label_smoothing}, grad_clip={max_grad_norm}")
    print(f"[Train] LR={train_cfg['learning_rate']}, warmup={warmup_epochs} epochs")
    print(f"[Train] DataLoader workers={num_workers}\n")
    sys.stdout.flush()

    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp, max_grad_norm,
        )
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device, use_amp)

        # Step scheduler after warmup
        if epoch > warmup_epochs:
            scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch:3d}/{train_cfg['epochs']}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f} | "
            f"lr={lr:.6f} time={elapsed:.1f}s"
        )
        sys.stdout.flush()

        # Early stopping on val AUC
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "val_acc": val_acc,
                "config": cfg,
            }, best_model_path)
            print(f"  -> Saved best model (AUC={val_auc:.4f}) to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Train] Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\n[Train] Done. Best validation AUC: {best_auc:.4f}")
    print(f"[Train] Best model saved to: {best_model_path}")

    # Final test evaluation
    if test_entries:
        print("\n[Train] Running final evaluation on test set...")
        test_dataset = DeepfakeDataset(test_entries, transform=val_transform)
        test_loader = DataLoader(
            test_dataset, batch_size=train_cfg["batch_size"],
            shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"),
        )

        # Load best checkpoint for test
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_loss, test_acc, test_auc = validate(model, test_loader, criterion, device, use_amp)
        print(f"[Test] loss={test_loss:.4f} acc={test_acc:.4f} auc={test_auc:.4f}")


if __name__ == "__main__":
    main()
