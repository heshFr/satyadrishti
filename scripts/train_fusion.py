"""
Satya Drishti — Cross-Attention Fusion Transformer Training
============================================================
Trains the multimodal fusion network on pre-extracted embeddings
from the three frozen encoders (Audio AST, Video ViT, Text DeBERTaV3).

Phase 1: Trains on synthetic embedding combinations (bootstrapping)
Phase 2: Fine-tune on real multimodal paired data when available

Target Classes:
    0 = Safe/Authentic
    1 = Deepfake Only (synthetic voice/video, no coercion)
    2 = Coercion Only (real human, but scamming)
    3 = Combined Threat (deepfake + coercion)

Usage:
    python scripts/train_fusion.py
    python scripts/train_fusion.py --epochs 30 --batch_size 64
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
from sklearn.metrics import f1_score, classification_report, accuracy_score
from tqdm import tqdm

from engine.fusion.cross_attention import MultimodalFusionNetwork

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = PROJECT_ROOT / "models" / "fusion"


class SyntheticFusionDataset(Dataset):
    """
    Generates synthetic embedding triplets for fusion training.

    Strategy: Create class-conditional embeddings that mimic the statistical
    properties of real embeddings from each modality's encoder.

    Class distributions:
        0 (safe): ~40% — all modalities show authentic patterns
        1 (deepfake): ~20% — audio/video show synthetic, text is benign
        2 (coercion): ~20% — audio/video are authentic, text is coercive
        3 (combined): ~20% — synthetic voice/video + coercive text
    """

    def __init__(self, size: int = 5000, embed_dim: int = 768, seed: int = 42):
        self.size = size
        self.embed_dim = embed_dim
        rng = np.random.RandomState(seed)

        # Generate class-conditional embeddings
        self.audio_embs = []
        self.video_embs = []
        self.text_embs = []
        self.labels = []

        # Define class-conditional distribution parameters
        # Each class has distinct mean/variance patterns per modality
        for i in range(size):
            label = rng.choice([0, 1, 2, 3], p=[0.4, 0.2, 0.2, 0.2])

            if label == 0:  # Safe: all authentic
                audio = rng.normal(0.5, 0.3, embed_dim).astype(np.float32)
                video = rng.normal(0.5, 0.3, embed_dim).astype(np.float32)
                text = rng.normal(-0.2, 0.3, embed_dim).astype(np.float32)
            elif label == 1:  # Deepfake only: audio/video shift, text normal
                audio = rng.normal(-0.5, 0.4, embed_dim).astype(np.float32)
                video = rng.normal(-0.5, 0.4, embed_dim).astype(np.float32)
                text = rng.normal(-0.2, 0.3, embed_dim).astype(np.float32)
            elif label == 2:  # Coercion only: text shifts, audio/video normal
                audio = rng.normal(0.5, 0.3, embed_dim).astype(np.float32)
                video = rng.normal(0.5, 0.3, embed_dim).astype(np.float32)
                text = rng.normal(0.8, 0.4, embed_dim).astype(np.float32)
            else:  # Combined: all shifted
                audio = rng.normal(-0.5, 0.4, embed_dim).astype(np.float32)
                video = rng.normal(-0.5, 0.4, embed_dim).astype(np.float32)
                text = rng.normal(0.8, 0.4, embed_dim).astype(np.float32)

            # Add noise to prevent trivial separation
            audio += rng.normal(0, 0.15, embed_dim).astype(np.float32)
            video += rng.normal(0, 0.15, embed_dim).astype(np.float32)
            text += rng.normal(0, 0.15, embed_dim).astype(np.float32)

            self.audio_embs.append(torch.from_numpy(audio))
            self.video_embs.append(torch.from_numpy(video))
            self.text_embs.append(torch.from_numpy(text))
            self.labels.append(label)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.audio_embs[idx], self.video_embs[idx], self.text_embs[idx], self.labels[idx]


def train(args):
    print("=" * 60)
    print("Satya Drishti - Cross-Attention Fusion Transformer")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Model — all modalities produce 768d embeddings
    model = MultimodalFusionNetwork(
        audio_embed_dim=768,
        video_embed_dim=768,
        text_embed_dim=768,
        latent_dim=256,
        num_heads=8,
        num_layers=4,
        num_classes=4,
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # 2. Data
    train_dataset = SyntheticFusionDataset(size=args.train_size, seed=42)
    val_dataset = SyntheticFusionDataset(size=args.train_size // 5, seed=123)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # 3. Loss with class weights (minority upweighting)
    weights = torch.tensor([1.0, 2.5, 2.5, 4.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 4. Training loop
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for a_emb, v_emb, t_emb, labels in pbar:
            a_emb = a_emb.to(device)
            v_emb = v_emb.to(device)
            t_emb = t_emb.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(a_emb, v_emb, t_emb)
            logits = outputs["logits"]
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for a_emb, v_emb, t_emb, labels in val_loader:
                a_emb = a_emb.to(device)
                v_emb = v_emb.to(device)
                t_emb = t_emb.to(device)
                labels = labels.to(device)

                outputs = model(a_emb, v_emb, t_emb)
                logits = outputs["logits"]
                val_loss += criterion(logits, labels).item()

                preds = torch.argmax(logits, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        acc = accuracy_score(all_labels, all_preds)

        print(f"\nEpoch {epoch}: train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | val_f1={f1:.4f} | val_acc={acc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            save_path = SAVE_DIR / "fusion_network_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_f1": best_f1,
            }, save_path)
            print(f"  -> New best F1! Saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # 5. Final evaluation
    print("\n" + "=" * 60)
    print("Final Validation Report")
    print("=" * 60)

    ckpt = torch.load(SAVE_DIR / "fusion_network_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_labels, all_preds = [], []
    with torch.no_grad():
        for a_emb, v_emb, t_emb, labels in val_loader:
            a_emb = a_emb.to(device)
            v_emb = v_emb.to(device)
            t_emb = t_emb.to(device)
            labels = labels.to(device)

            outputs = model(a_emb, v_emb, t_emb)
            preds = torch.argmax(outputs["logits"], dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    print(classification_report(
        all_labels, all_preds,
        target_names=["safe", "deepfake", "coercion", "combined"],
        digits=4,
    ))

    metrics = {
        "best_epoch": ckpt["epoch"],
        "best_f1": best_f1,
        "val_accuracy": accuracy_score(all_labels, all_preds),
        "note": "Trained on synthetic embeddings. Fine-tune on real multimodal data for production.",
    }
    with open(SAVE_DIR / "fusion_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[DONE] Best model at epoch {ckpt['epoch']}, F1={best_f1:.4f}")
    print("NOTE: This was trained on synthetic embeddings. Production performance")
    print("will improve after fine-tuning on real multimodal paired data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cross-Attention Fusion Network")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--train_size", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()

    train(args)
