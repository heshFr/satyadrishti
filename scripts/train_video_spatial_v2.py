"""
Satya Drishti — Video Spatial Stream Training v2
==================================================
Trains ViT-B/16 on multi-source face-cropped frames (FF++, Celeb-DF,
DeeperForensics) for robust spatial manipulation detection.

Key improvements over v1:
  - Multi-source dataset for generalization
  - Stronger augmentation pipeline
  - Warmup + cosine schedule
  - Mixup regularization

Usage:
    python scripts/train_video_spatial_v2.py
    python scripts/train_video_spatial_v2.py --epochs 25 --batch_size 8
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from tqdm import tqdm

from engine.video.spatial_vit import DeepfakeViT

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = PROJECT_ROOT / "datasets" / "deepfake" / "frames_v2"
MANIFEST_PATH = FRAMES_DIR / "manifest.json"
SAVE_DIR = PROJECT_ROOT / "models" / "video"


class MultiSourceSpatialDataset(Dataset):
    """Dataset of face-cropped frames from multiple deepfake datasets."""

    def __init__(self, entries: list, transform=None):
        self.samples = [
            (PROJECT_ROOT / e["path"], e["label"])
            for e in entries if e["type"] == "spatial"
        ]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            # Return a black image as fallback
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, label


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation for better generalization."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(args):
    print("=" * 60)
    print("Satya Drishti - Video Spatial Stream v2 (ViT-B/16)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")

    # 1. Load manifest
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    # 2. Transforms — aggressive augmentation for generalization
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = MultiSourceSpatialDataset(manifest["train"], train_transform)
    val_dataset = MultiSourceSpatialDataset(manifest["val"], eval_transform)
    test_dataset = MultiSourceSpatialDataset(manifest["test"], eval_transform)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # 3. Model — ViT-B/16, freeze first 8 blocks, fine-tune last 4 + classifier
    model = DeepfakeViT(num_classes=2, pretrained=True).to(device)

    for name, param in model.backbone.named_parameters():
        param.requires_grad = False
    for i in range(8, 12):
        for param in model.backbone.encoder.layers[i].parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # 4. Optimizer with warmup + cosine
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )

    # Warmup for first 2 epochs then cosine
    warmup_epochs = 2
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler() if use_amp else None

    # 5. Training loop
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for step, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)

            # Apply mixup with probability 0.5
            use_mixup = np.random.rand() < 0.5 and epoch > warmup_epochs
            if use_mixup:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.2)

            if use_amp:
                with autocast(device_type="cuda"):
                    logits = model(inputs)
                    if use_mixup:
                        loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam) / args.grad_accum
                    else:
                        loss = criterion(logits, labels) / args.grad_accum
                scaler.scale(loss).backward()
                if (step + 1) % args.grad_accum == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                logits = model(inputs)
                if use_mixup:
                    loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam) / args.grad_accum
                else:
                    loss = criterion(logits, labels) / args.grad_accum
                loss.backward()
                if (step + 1) % args.grad_accum == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            train_loss += loss.item() * args.grad_accum
            pbar.set_postfix({"loss": f"{loss.item() * args.grad_accum:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        all_labels, all_probs = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if use_amp:
                    with autocast(device_type="cuda"):
                        logits = model(inputs)
                        loss = criterion(logits, labels)
                else:
                    logits = model(inputs)
                    loss = criterion(logits, labels)

                val_loss += loss.item()
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0
        acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_probs])

        print(f"\nEpoch {epoch}: train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | val_auc={auc:.4f} | val_acc={acc:.4f}")

        if auc > best_auc:
            best_auc = auc
            patience_counter = 0
            save_path = SAVE_DIR / "vit_spatial_v2_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_auc": best_auc,
            }, save_path)
            print(f"  -> New best AUC! Saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # 6. Test evaluation
    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)

    ckpt = torch.load(SAVE_DIR / "vit_spatial_v2_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_labels, all_probs, all_preds = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if use_amp:
                with autocast(device_type="cuda"):
                    logits = model(inputs)
            else:
                logits = model(inputs)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs > 0.5).long()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    test_auc = roc_auc_score(all_labels, all_probs)
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=["real", "fake"], digits=4))

    metrics = {
        "best_epoch": ckpt["epoch"],
        "test_auc": test_auc,
        "test_accuracy": test_acc,
    }
    with open(SAVE_DIR / "spatial_v2_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[DONE] Best model at epoch {ckpt['epoch']}, test AUC={test_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT spatial deepfake detector v2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()

    train(args)
