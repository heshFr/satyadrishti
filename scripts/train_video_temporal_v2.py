"""
Satya Drishti — Video Temporal Stream Training v2
===================================================
Trains R3D-18 (pretrained Kinetics-400) on multi-source 16-frame
face-cropped clips for temporal inconsistency detection.

Key improvements over v1:
  - Multi-source dataset (FF++, Celeb-DF, DeeperForensics)
  - Pretrained R3D-18 instead of X3D-S from scratch
  - Stronger temporal augmentation
  - Warmup + cosine schedule

Usage:
    python scripts/train_video_temporal_v2.py
    python scripts/train_video_temporal_v2.py --epochs 25 --batch_size 2
    python scripts/train_video_temporal_v2.py --resume models/video/r3d_temporal_v2_best.pt
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
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from tqdm import tqdm

from engine.video.temporal_r3d import TemporalR3D

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = PROJECT_ROOT / "datasets" / "deepfake" / "frames_v2"
MANIFEST_PATH = FRAMES_DIR / "manifest.json"
SAVE_DIR = PROJECT_ROOT / "models" / "video"


class MultiSourceTemporalDataset(Dataset):
    """Dataset of 16-frame face-cropped clips from multiple deepfake datasets."""

    def __init__(self, entries: list, augment: bool = False):
        self.samples = [
            (PROJECT_ROOT / e["path"], e["label"])
            for e in entries if e["type"] == "temporal"
        ]
        self.augment = augment
        self.normalize_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.normalize_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        clip = np.load(str(path))  # (16, 224, 224, 3) uint8

        if self.augment:
            # Horizontal flip
            if np.random.rand() > 0.5:
                clip = clip[:, :, ::-1, :]

            # Temporal jitter: randomly drop and duplicate frames
            if np.random.rand() > 0.6:
                n_jitter = np.random.randint(1, 3)
                for _ in range(n_jitter):
                    drop_idx = np.random.randint(0, clip.shape[0])
                    dup_idx = np.random.randint(0, clip.shape[0])
                    clip[drop_idx] = clip[dup_idx]

            # Temporal reverse (play backwards)
            if np.random.rand() > 0.8:
                clip = clip[::-1].copy()

            # Color jitter (per-clip)
            if np.random.rand() > 0.5:
                brightness = np.random.uniform(0.7, 1.3)
                clip = np.clip(clip.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

            # Saturation jitter
            if np.random.rand() > 0.6:
                factor = np.random.uniform(0.7, 1.3)
                clip_float = clip.astype(np.float32)
                gray = np.mean(clip_float, axis=-1, keepdims=True)
                clip = np.clip(gray + factor * (clip_float - gray), 0, 255).astype(np.uint8)

            # Gaussian noise
            if np.random.rand() > 0.7:
                noise = np.random.normal(0, 5, clip.shape).astype(np.float32)
                clip = np.clip(clip.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            # Random crop (instead of center crop)
            if np.random.rand() > 0.5:
                h, w = clip.shape[1], clip.shape[2]
                crop_size = int(h * np.random.uniform(0.85, 1.0))
                y = np.random.randint(0, h - crop_size + 1)
                x = np.random.randint(0, w - crop_size + 1)
                clip = clip[:, y:y+crop_size, x:x+crop_size, :]
                # Resize back
                import cv2
                resized = []
                for frame in clip:
                    resized.append(cv2.resize(frame, (224, 224)))
                clip = np.stack(resized)

        # Normalize
        clip = clip.astype(np.float32) / 255.0
        clip = (clip - self.normalize_mean) / self.normalize_std
        clip = clip.transpose(3, 0, 1, 2)  # (3, T, H, W)

        return torch.from_numpy(clip.copy()).float(), label


def train(args):
    print("=" * 60)
    print("Satya Drishti - Video Temporal Stream v2 (R3D-18 Pretrained)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")

    # 1. Load manifest
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    train_dataset = MultiSourceTemporalDataset(manifest["train"], augment=True)
    val_dataset = MultiSourceTemporalDataset(manifest["val"], augment=False)
    test_dataset = MultiSourceTemporalDataset(manifest["test"], augment=False)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # 2. Model — pretrained R3D-18, freeze stem + layer1 + layer2
    model = TemporalR3D(num_classes=2, pretrained=True).to(device)

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

    # 3. Optimizer with warmup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-3,
    )

    warmup_epochs = 2
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler() if use_amp else None

    # 4. Resume from checkpoint if requested
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    best_auc = 0.0
    patience_counter = 0
    start_epoch = 1

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_auc = ckpt.get("best_auc", 0.0)
        start_epoch = ckpt.get("epoch", 0) + 1
        # Restore optimizer/scheduler/scaler if available
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        # Fast-forward scheduler if no saved state
        if "scheduler_state_dict" not in ckpt:
            for _ in range(start_epoch - 1):
                scheduler.step()
        print(f"  Resuming from epoch {start_epoch}, best_auc={best_auc:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
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
            pbar.set_postfix({"loss": f"{loss.item() * args.grad_accum:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

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
            save_path = SAVE_DIR / "r3d_temporal_v2_best.pt"
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_auc": best_auc,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            if scaler is not None:
                save_dict["scaler_state_dict"] = scaler.state_dict()
            torch.save(save_dict, save_path)
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

    ckpt = torch.load(SAVE_DIR / "r3d_temporal_v2_best.pt", map_location=device)
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
    with open(SAVE_DIR / "temporal_r3d_v2_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[DONE] Best model at epoch {ckpt['epoch']}, test AUC={test_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train R3D-18 temporal deepfake detector v2")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    train(args)
