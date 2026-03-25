"""
Train a linear probe classifier on CLIP embeddings.
Much faster and more accurate than zero-shot CLIP.

Usage:
    python scripts/train_clip_probe.py --config configs/train_vit_v2_local.yaml
"""

import sys, os, io, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from sklearn.metrics import roc_auc_score, classification_report
import yaml

def extract_clip_embeddings(parquet_paths, max_samples=5000):
    """Extract CLIP embeddings from parquet images."""
    from transformers import CLIPProcessor, CLIPModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.to(device).eval()

    import pyarrow.parquet as pq
    import random

    embeddings = []
    labels = []
    count = 0

    for path in parquet_paths:
        try:
            pf = pq.ParquetFile(path)
        except Exception:
            continue

        for batch in pf.iter_batches(batch_size=64, columns=["image", "label"]):
            images = batch.column("image")
            batch_labels = batch.column("label")

            for i in range(len(batch)):
                if count >= max_samples:
                    break

                img_data = images[i].as_py()
                label_str = batch_labels[i].as_py()

                if isinstance(img_data, dict) and "bytes" in img_data:
                    img_bytes = img_data["bytes"]
                elif isinstance(img_data, bytes):
                    img_bytes = img_data
                else:
                    continue

                label = 1 if label_str == "fake" else 0

                try:
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    inputs = processor(images=img, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    with torch.no_grad():
                        features = model.get_image_features(**inputs)
                        features = features / features.norm(dim=-1, keepdim=True)
                        embeddings.append(features.cpu().squeeze(0).numpy())
                        labels.append(label)
                        count += 1

                except Exception:
                    continue

            if count >= max_samples:
                break

        if count >= max_samples:
            break

        print(f"  Extracted {count}/{max_samples} embeddings...")

    return np.stack(embeddings), np.array(labels)


def train_probe(embeddings, labels, embed_dim=768, epochs=50, lr=1e-3):
    """Train a linear probe classifier on CLIP embeddings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Split train/val
    n = len(labels)
    indices = np.random.RandomState(42).permutation(n)
    split = int(n * 0.85)
    train_idx, val_idx = indices[:split], indices[split:]

    train_emb = torch.from_numpy(embeddings[train_idx]).float()
    train_lab = torch.from_numpy(labels[train_idx]).long()
    val_emb = torch.from_numpy(embeddings[val_idx]).float()
    val_lab = torch.from_numpy(labels[val_idx]).long()

    train_loader = DataLoader(TensorDataset(train_emb, train_lab), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_emb, val_lab), batch_size=256)

    probe = nn.Sequential(
        nn.Linear(embed_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 2),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)

    best_auc = 0.0
    for epoch in range(1, epochs + 1):
        probe.train()
        for emb, lab in train_loader:
            emb, lab = emb.to(device), lab.to(device)
            optimizer.zero_grad()
            loss = criterion(probe(emb), lab)
            loss.backward()
            optimizer.step()

        # Validate
        probe.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for emb, lab in val_loader:
                emb = emb.to(device)
                probs = torch.softmax(probe(emb), dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(lab.numpy())

        auc = roc_auc_score(all_labels, all_probs)
        if epoch % 10 == 0 or auc > best_auc:
            print(f"  Epoch {epoch}: val_auc={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_state = probe.state_dict()

    probe.load_state_dict(best_state)
    print(f"\n  Best probe AUC: {best_auc:.4f}")

    return probe, best_auc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_vit_v2_local.yaml")
    parser.add_argument("--max_samples", type=int, default=5000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    parquet_paths = sorted(glob.glob(cfg["data"]["parquet_pattern"]))
    if not parquet_paths:
        print("No parquet files found!")
        return

    print(f"Found {len(parquet_paths)} shards")
    print(f"Extracting CLIP embeddings (max {args.max_samples})...")

    embeddings, labels = extract_clip_embeddings(parquet_paths, args.max_samples)
    print(f"Extracted {len(embeddings)} embeddings")
    print(f"Distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    embed_dim = embeddings.shape[1]
    print(f"\nTraining linear probe (dim={embed_dim})...")
    probe, best_auc = train_probe(embeddings, labels, embed_dim)

    save_path = "models/image_forensics/clip_probe.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "probe_state_dict": probe.state_dict(),
        "embed_dim": embed_dim,
        "best_auc": best_auc,
    }, save_path)
    print(f"\nSaved probe to {save_path}")


if __name__ == "__main__":
    main()
