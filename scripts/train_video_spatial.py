"""
Satya Drishti — Video Spatial Stream Training
===============================================
Trains ViT-B/16 on face-cropped frames from FaceForensics++ to detect
spatial manipulation artifacts (blending boundaries, GAN textures).

Usage:
    python scripts/train_video_spatial.py
    python scripts/train_video_spatial.py --epochs 20 --batch_size 8
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
FRAMES_DIR = PROJECT_ROOT / "datasets" / "deepfake" / "frames"
MANIFEST_PATH = FRAMES_DIR / "manifest.json"
SAVE_DIR = PROJECT_ROOT / "models" / "video"


class FFSpatialDataset(Dataset):
    """Dataset of face-cropped frames from FF++ videos."""

    def __init__(self, entries: list, transform=None):
        self.samples = [(PROJECT_ROOT / e["path"], e["label"]) for e in entries if e["type"] == "spatial"]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, label


def train(args):
    print("=" * 60)
    print("Satya Drishti - Video Spatial Stream (ViT-B/16)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")

    # 1. Load manifest
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    # 2. Transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = FFSpatialDataset(manifest["train"], train_transform)
    val_dataset = FFSpatialDataset(manifest["val"], eval_transform)
    test_dataset = FFSpatialDataset(manifest["test"], eval_transform)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # 3. Model — freeze first 8 ViT blocks, finetune last 4 + classifier
    model = DeepfakeViT(num_classes=2, pretrained=True).to(device)

    # Freeze backbone except last 4 encoder blocks
    for name, param in model.backbone.named_parameters():
        param.requires_grad = False
    for i in range(8, 12):
        for param in model.backbone.encoder.layers[i].parameters():
            param.requires_grad = True
    # Classifier is always trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # 4. Optimizer, scheduler, loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler() if use_amp else None

    # 5. Training loop
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for step, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)

            if use_amp:
                with autocast(device_type="cuda"):
                    logits = model(inputs)
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
                loss = criterion(logits, labels) / args.grad_accum
                loss.backward()
                if (step + 1) % args.grad_accum == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            train_loss += loss.item() * args.grad_accum
            pbar.set_postfix({"loss": f"{loss.item() * args.grad_accum:.4f}"})

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

        # Save best
        if auc > best_auc:
            best_auc = auc
            patience_counter = 0
            save_path = SAVE_DIR / "vit_spatial_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
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

    # Load best model
    ckpt = torch.load(SAVE_DIR / "vit_spatial_best.pt", map_location=device)
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

    # Save metrics
    metrics = {
        "best_epoch": ckpt["epoch"],
        "test_auc": test_auc,
        "test_accuracy": test_acc,
    }
    with open(SAVE_DIR / "spatial_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[DONE] Best model at epoch {ckpt['epoch']}, test AUC={test_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT spatial deepfake detector")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    train(args)
