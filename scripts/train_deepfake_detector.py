"""
Satya Drishti — EfficientNet-B4 Deepfake Detector Training
==========================================================
Trains a binary classifier (real vs fake) using transfer learning
from EfficientNet-B4 pretrained on ImageNet.

Features:
- YAML config driven
- Mixed precision training (AMP) for 4GB VRAM budgets
- Cosine annealing LR schedule with warmup
- Early stopping on validation AUC
- JPEG compression augmentation to simulate real-world degradation
- Saves best checkpoint to models/deepfake_efficientnet_b4.pt

Usage:
    python scripts/train_deepfake_detector.py --config configs/train_deepfake.yaml
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
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import roc_auc_score

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


def build_transforms(cfg):
    input_size = cfg["model"]["input_size"]
    aug_cfg = cfg["data"].get("augmentation", {})

    train_transforms = [transforms.Resize((input_size, input_size))]

    if aug_cfg.get("random_resized_crop"):
        scale = tuple(aug_cfg["random_resized_crop"]["scale"])
        train_transforms.append(transforms.RandomResizedCrop(input_size, scale=scale))

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
        train_transforms.append(transforms.Lambda(lambda img: JPEGCompression(qr)(img)))

    train_transforms += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    val_transforms = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    return transforms.Compose(train_transforms), transforms.Compose(val_transforms)


# ─── Build Model ───


def build_model(cfg):
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=cfg["model"].get("dropout", 0.4), inplace=True),
        nn.Linear(in_features, cfg["model"]["num_classes"]),
    )
    return model


# ─── Training Loop ───


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

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

        with torch.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        probs = torch.softmax(outputs, dim=1)[:, 1]
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return running_loss / total, correct / total, auc


# ─── Main ───


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B4 Deepfake Detector")
    parser.add_argument("--config", type=str, default="configs/train_deepfake.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    if device.type == "cuda":
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Train] VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    train_cfg = cfg["training"]
    use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"

    # Load manifest
    manifest_path = cfg["data"]["manifest_path"]
    if not os.path.exists(manifest_path):
        print(f"[Error] Manifest not found: {manifest_path}")
        print("[Error] Run scripts/prepare_dataset.py first to create the dataset.")
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

    print(f"[Train] Dataset: {n} total — {len(train_entries)} train, {len(val_entries)} val")

    # Transforms and dataloaders
    train_transform, val_transform = build_transforms(cfg)
    train_dataset = DeepfakeDataset(train_entries, transform=train_transform)
    val_dataset = DeepfakeDataset(val_entries, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg["batch_size"],
        shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True,
    )

    # Model, criterion, optimizer, scheduler
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"] - train_cfg.get("warmup_epochs", 0),
    )
    scaler = torch.GradScaler(enabled=use_amp)

    # Output dirs
    checkpoint_dir = cfg["output"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = cfg["output"].get("log_dir", "logs/deepfake_training")
    os.makedirs(log_dir, exist_ok=True)

    best_auc = 0.0
    patience_counter = 0
    patience = train_cfg.get("early_stopping_patience", 5)
    best_model_path = os.path.join(checkpoint_dir, cfg["output"]["best_model_name"])

    print(f"[Train] Starting training for {train_cfg['epochs']} epochs (AMP={use_amp})")

    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp,
        )
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device, use_amp)

        # Step scheduler after warmup
        if epoch > train_cfg.get("warmup_epochs", 0):
            scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch:3d}/{train_cfg['epochs']}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f} | "
            f"lr={lr:.6f} time={elapsed:.1f}s"
        )

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
            }, best_model_path)
            print(f"  -> Saved best model (AUC={val_auc:.4f}) to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Train] Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\n[Train] Done. Best validation AUC: {best_auc:.4f}")
    print(f"[Train] Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
