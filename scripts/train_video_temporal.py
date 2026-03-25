"""
Satya Drishti — Video Temporal Stream Training
================================================
Trains R3D-18 (pretrained on Kinetics-400) on 16-frame face-cropped
clips from FaceForensics++ to detect temporal inconsistencies.

Usage:
    python scripts/train_video_temporal.py
    python scripts/train_video_temporal.py --epochs 20 --batch_size 2
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from tqdm import tqdm

from engine.video.temporal_r3d import TemporalR3D

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = PROJECT_ROOT / "datasets" / "deepfake" / "frames"
MANIFEST_PATH = FRAMES_DIR / "manifest.json"
SAVE_DIR = PROJECT_ROOT / "models" / "video"


class FFTemporalDataset(Dataset):
    """Dataset of 16-frame face-cropped clips from FF++ videos."""

    def __init__(self, entries: list, augment: bool = False):
        self.samples = [(PROJECT_ROOT / e["path"], e["label"]) for e in entries if e["type"] == "temporal"]
        self.augment = augment
        self.normalize_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.normalize_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # Load clip: (16, 224, 224, 3) uint8
        clip = np.load(str(path))

        # Augment
        if self.augment:
            if np.random.rand() > 0.5:
                clip = clip[:, :, ::-1, :]  # horizontal flip
            # Temporal jitter: randomly drop and duplicate a frame
            if np.random.rand() > 0.7:
                drop_idx = np.random.randint(0, clip.shape[0])
                dup_idx = np.random.randint(0, clip.shape[0])
                clip[drop_idx] = clip[dup_idx]
            # Color jitter (per-clip)
            if np.random.rand() > 0.5:
                brightness = np.random.uniform(0.8, 1.2)
                clip = np.clip(clip.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
            # Gaussian noise
            if np.random.rand() > 0.7:
                noise = np.random.normal(0, 5, clip.shape).astype(np.float32)
                clip = np.clip(clip.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Normalize: (16, H, W, 3) -> float32 [0,1] -> normalized
        clip = clip.astype(np.float32) / 255.0
        clip = (clip - self.normalize_mean) / self.normalize_std

        # Rearrange to (3, 16, 224, 224) for 3D CNN
        clip = clip.transpose(3, 0, 1, 2)  # (3, T, H, W)

        return torch.from_numpy(clip.copy()).float(), label


def train(args):
    print("=" * 60)
    print("Satya Drishti - Video Temporal Stream (R3D-18 Pretrained)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")

    # 1. Load manifest
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    train_dataset = FFTemporalDataset(manifest["train"], augment=True)
    val_dataset = FFTemporalDataset(manifest["val"], augment=False)
    test_dataset = FFTemporalDataset(manifest["test"], augment=False)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # 2. Model — pretrained R3D-18, freeze early layers
    model = TemporalR3D(num_classes=2, pretrained=True).to(device)

    # Freeze stem + layer1 + layer2, fine-tune layer3 + layer4 + classifier
    for name, param in model.backbone.named_parameters():
        param.requires_grad = False
    for param in model.backbone.layer3.parameters():
        param.requires_grad = True
    for param in model.backbone.layer4.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # 3. Optimizer, scheduler, loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-3,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler() if use_amp else None

    # 4. Training loop
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for step, (clips, labels) in enumerate(pbar):
            clips, labels = clips.to(device), labels.to(device)

            if use_amp:
                with autocast(device_type="cuda"):
                    logits = model(clips)
                    loss = criterion(logits, labels) / args.grad_accum
                scaler.scale(loss).backward()
                if (step + 1) % args.grad_accum == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                logits = model(clips)
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
            for clips, labels in val_loader:
                clips, labels = clips.to(device), labels.to(device)
                if use_amp:
                    with autocast(device_type="cuda"):
                        logits = model(clips)
                        loss = criterion(logits, labels)
                else:
                    logits = model(clips)
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
            save_path = SAVE_DIR / "r3d_temporal_best.pt"
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

    # 5. Test evaluation
    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)

    ckpt = torch.load(SAVE_DIR / "r3d_temporal_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_labels, all_probs, all_preds = [], [], []
    with torch.no_grad():
        for clips, labels in test_loader:
            clips, labels = clips.to(device), labels.to(device)
            if use_amp:
                with autocast(device_type="cuda"):
                    logits = model(clips)
            else:
                logits = model(clips)
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
    with open(SAVE_DIR / "temporal_r3d_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[DONE] Best model at epoch {ckpt['epoch']}, test AUC={test_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train R3D-18 temporal deepfake detector")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    train(args)
