"""
Satya Drishti — Two-Stream Video Combined Training
====================================================
Late-fusion of pretrained Spatial (ViT) + Temporal (X3D) models.
Only trains the fusion weight alpha on paired spatial+temporal data.

Usage:
    python scripts/train_video_combined.py
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from tqdm import tqdm

from engine.video.spatial_vit import DeepfakeViT
from engine.video.temporal_x3d import TemporalX3D

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "datasets" / "deepfake" / "frames" / "manifest.json"
SAVE_DIR = PROJECT_ROOT / "models" / "video"


class TwoStreamNetwork(nn.Module):
    """Late fusion of spatial ViT and temporal X3D with learned alpha."""

    def __init__(self):
        super().__init__()
        self.spatial = DeepfakeViT(num_classes=2, pretrained=False)
        self.temporal = TemporalX3D(clip_length=16, num_classes=2)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def load_pretrained(self, spatial_path: str, temporal_path: str, device):
        """Load pretrained weights from individual stream checkpoints."""
        spatial_ckpt = torch.load(spatial_path, map_location=device)
        self.spatial.load_state_dict(spatial_ckpt["model_state_dict"])

        temporal_ckpt = torch.load(temporal_path, map_location=device)
        self.temporal.load_state_dict(temporal_ckpt["model_state_dict"])

    def forward(self, frames, clips):
        spatial_logits = self.spatial(frames)
        temporal_logits = self.temporal(clips)
        fused = self.alpha * spatial_logits + (1 - self.alpha) * temporal_logits
        return fused


class CombinedVideoDataset(Dataset):
    """Pairs spatial frames with temporal clips from the same video."""

    def __init__(self, entries: list, augment: bool = False):
        # Group by video_id
        by_video = defaultdict(lambda: {"spatial": [], "temporal": [], "label": None})
        for e in entries:
            vid = e["video_id"]
            by_video[vid][e["type"]].append(PROJECT_ROOT / e["path"])
            by_video[vid]["label"] = e["label"]

        # Only keep videos that have both spatial AND temporal data
        self.pairs = []
        for vid, data in by_video.items():
            if data["spatial"] and data["temporal"]:
                for clip_path in data["temporal"]:
                    # Pick a random spatial frame from same video
                    frame_path = random.choice(data["spatial"])
                    self.pairs.append((frame_path, clip_path, data["label"]))

        self.augment = augment
        self.normalize_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.normalize_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        frame_path, clip_path, label = self.pairs[idx]

        # Load spatial frame
        img = cv2.imread(str(frame_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        if self.augment and np.random.rand() > 0.5:
            img = img[:, ::-1, :]  # horizontal flip
        img = (img - self.normalize_mean) / self.normalize_std
        img = img.transpose(2, 0, 1)  # (3, H, W)

        # Load temporal clip
        clip = np.load(str(clip_path))  # (16, 224, 224, 3)
        clip = clip.astype(np.float32) / 255.0
        if self.augment and np.random.rand() > 0.5:
            clip = clip[:, :, ::-1, :]
        clip = (clip - self.normalize_mean) / self.normalize_std
        clip = clip.transpose(3, 0, 1, 2)  # (3, T, H, W)

        return (
            torch.from_numpy(img.copy()).float(),
            torch.from_numpy(clip.copy()).float(),
            label,
        )


def train(args):
    print("=" * 60)
    print("Satya Drishti - Two-Stream Combined (ViT + X3D)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")

    # 1. Load manifest
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    train_dataset = CombinedVideoDataset(manifest["train"], augment=True)
    val_dataset = CombinedVideoDataset(manifest["val"], augment=False)
    test_dataset = CombinedVideoDataset(manifest["test"], augment=False)

    print(f"Paired samples — Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

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

    # 2. Model — load pretrained streams, freeze them, only train alpha
    model = TwoStreamNetwork().to(device)

    spatial_path = SAVE_DIR / "vit_spatial_best.pt"
    temporal_path = SAVE_DIR / "x3d_temporal_best.pt"
    model.load_pretrained(str(spatial_path), str(temporal_path), device)
    print(f"Loaded spatial from {spatial_path}")
    print(f"Loaded temporal from {temporal_path}")

    # Freeze both streams — only train fusion alpha
    for param in model.spatial.parameters():
        param.requires_grad = False
    for param in model.temporal.parameters():
        param.requires_grad = False
    model.alpha.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable} (fusion alpha only)")

    # 3. Optimizer, loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam([model.alpha], lr=args.lr)
    scaler = GradScaler() if use_amp else None

    # 4. Training loop
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    best_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for frames, clips, labels in pbar:
            frames = frames.to(device)
            clips = clips.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            if use_amp:
                with autocast(device_type="cuda"):
                    logits = model(frames, clips)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(frames, clips)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "alpha": f"{model.alpha.item():.3f}"})

        avg_train_loss = train_loss / len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        all_labels, all_probs = [], []

        with torch.no_grad():
            for frames, clips, labels in val_loader:
                frames, clips, labels = frames.to(device), clips.to(device), labels.to(device)
                if use_amp:
                    with autocast(device_type="cuda"):
                        logits = model(frames, clips)
                        loss = criterion(logits, labels)
                else:
                    logits = model(frames, clips)
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

        print(f"\nEpoch {epoch}: train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | val_auc={auc:.4f} | val_acc={acc:.4f} | alpha={model.alpha.item():.3f}")

        if auc > best_auc:
            best_auc = auc
            save_path = SAVE_DIR / "video_twostream_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_auc": best_auc,
                "alpha": model.alpha.item(),
            }, save_path)
            print(f"  -> New best AUC! Saved to {save_path}")

    # 5. Test evaluation
    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)

    ckpt = torch.load(SAVE_DIR / "video_twostream_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_labels, all_probs, all_preds = [], [], []
    with torch.no_grad():
        for frames, clips, labels in test_loader:
            frames, clips, labels = frames.to(device), clips.to(device), labels.to(device)
            if use_amp:
                with autocast(device_type="cuda"):
                    logits = model(frames, clips)
            else:
                logits = model(frames, clips)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs > 0.5).long()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    test_auc = roc_auc_score(all_labels, all_probs)
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print(f"Fusion Alpha: {ckpt['alpha']:.3f}")
    print(classification_report(all_labels, all_preds, target_names=["real", "fake"], digits=4))

    metrics = {
        "best_epoch": ckpt["epoch"],
        "test_auc": test_auc,
        "test_accuracy": test_acc,
        "fusion_alpha": ckpt["alpha"],
    }
    with open(SAVE_DIR / "combined_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[DONE] Best model at epoch {ckpt['epoch']}, test AUC={test_auc:.4f}, alpha={ckpt['alpha']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train two-stream combined video detector")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    train(args)
