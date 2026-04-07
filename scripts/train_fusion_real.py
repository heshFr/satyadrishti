"""
Satya Drishti — Fusion Retraining on Real Embeddings
======================================================
Extracts real embeddings from all three modality encoders using
actual dataset samples, creates cross-modal triplets, and retrains
the fusion network for production-quality predictions.

Strategy:
  1. Extract text embeddings from coercion dataset (DeBERTaV3 + LoRA)
  2. Extract video embeddings from deepfake frames (prithivMLmods ViT)
  3. For audio, use the Wav2Vec2 model on synthetic audio to generate
     embeddings from the model's actual distribution (since we lack
     naturally paired audio-video data)
  4. Create labeled triplets by combining real embeddings
  5. Train the fusion network

Usage:
    python -m scripts.train_fusion_real
    python -m scripts.train_fusion_real --epochs 40 --batch_size 32
    python -m scripts.train_fusion_real --extract_only  # Only extract embeddings
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
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import f1_score, classification_report, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = PROJECT_ROOT / "models" / "fusion"
EMBEDDINGS_DIR = SAVE_DIR / "real_embeddings"

# Dataset paths
COERCION_TRAIN = PROJECT_ROOT / "datasets" / "coercion" / "train.jsonl"
COERCION_TEST = PROJECT_ROOT / "datasets" / "coercion" / "test.jsonl"
DEEPFAKE_MANIFEST = PROJECT_ROOT / "datasets" / "deepfake" / "frames_v2" / "manifest.json"


def extract_text_embeddings(output_path: Path, data_path: Path, max_samples: int = 2000):
    """Extract real text embeddings from coercion dataset using DeBERTaV3 + LoRA."""
    print("\n[1/3] Extracting text embeddings...")

    from engine.text.coercion_detector import CoercionDetector

    checkpoint_dir = str(PROJECT_ROOT / "models" / "text" / "deberta_coercion_lora" / "best_model")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.isdir(checkpoint_dir):
        detector = CoercionDetector(checkpoint_dir=checkpoint_dir)
        print(f"  Loaded trained LoRA weights from {checkpoint_dir}")
    else:
        detector = CoercionDetector()
        print("  Warning: Using untrained model (no checkpoint found)")

    detector.model.eval()
    detector.model.to(device)

    # Read coercion data
    samples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            samples.append((item["text"], item["label"]))

    # Subsample if needed
    if len(samples) > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(samples), max_samples, replace=False)
        samples = [samples[i] for i in indices]

    embeddings = []
    labels = []
    batch_size = 16

    print(f"  Extracting from {len(samples)} samples...")

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        for text, label in batch:
            try:
                with torch.no_grad():
                    emb = detector.extract_embedding(text, device=device)
                    embeddings.append(emb.cpu().squeeze(0).numpy())
                    labels.append(label)
            except Exception as e:
                continue

        if (i // batch_size) % 20 == 0:
            print(f"  Processed {min(i + batch_size, len(samples))}/{len(samples)}")

    embeddings = np.stack(embeddings)
    labels = np.array(labels)

    os.makedirs(output_path.parent, exist_ok=True)
    np.savez(output_path, embeddings=embeddings, labels=labels)
    print(f"  Saved {len(embeddings)} text embeddings to {output_path}")
    print(f"  Shape: {embeddings.shape}, Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    return embeddings, labels


def extract_video_embeddings(output_path: Path, max_samples: int = 2000):
    """Extract real video embeddings from deepfake frames using HuggingFace ViT detector."""
    print("\n[2/3] Extracting video embeddings...")

    from engine.image_forensics.vit_detector import ViTDetector
    import cv2

    detector = ViTDetector()
    print(f"  Loaded {detector.MODEL_NAME} on {detector.device}")

    # Load manifest
    if not DEEPFAKE_MANIFEST.exists():
        print(f"  Error: Manifest not found at {DEEPFAKE_MANIFEST}")
        return None, None

    with open(DEEPFAKE_MANIFEST) as f:
        manifest = json.load(f)

    # Collect frame paths from train split
    frame_entries = []
    for entry in manifest.get("train", []):
        if entry.get("type") == "spatial":
            frame_entries.append((str(PROJECT_ROOT / entry["path"]), entry["label"]))

    if len(frame_entries) > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(frame_entries), max_samples, replace=False)
        frame_entries = [frame_entries[i] for i in indices]

    print(f"  Extracting from {len(frame_entries)} frames...")

    embeddings = []
    labels = []

    for idx, (path, label) in enumerate(frame_entries):
        try:
            img = cv2.imread(path)
            if img is None:
                continue

            emb = detector.extract_embedding(img)  # BGR input, (1, hidden_dim)
            embeddings.append(emb.cpu().squeeze(0).numpy())
            labels.append(label)
        except Exception:
            continue

        if idx % 200 == 0:
            print(f"  Processed {idx}/{len(frame_entries)}")

    if not embeddings:
        print("  Error: No frames could be processed")
        return None, None

    embeddings = np.stack(embeddings)
    labels = np.array(labels)

    np.savez(output_path, embeddings=embeddings, labels=labels)
    print(f"  Saved {len(embeddings)} video embeddings to {output_path}")
    print(f"  Shape: {embeddings.shape}, Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    return embeddings, labels


def extract_audio_embeddings(output_path: Path, n_samples: int = 2000):
    """
    Extract audio embeddings from the Wav2Vec2 deepfake detector.
    Uses synthetic audio since we lack paired audio data,
    but the embeddings come from the real Wav2Vec2 encoder's actual distribution.
    """
    print("\n[3/3] Extracting audio embeddings...")

    from engine.audio.ast_spoof import ASTSpoofDetector

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASTSpoofDetector()
    model.to(device).eval()

    rng = np.random.RandomState(42)
    embeddings = []
    labels = []

    print(f"  Generating {n_samples} audio embeddings from Wav2Vec2 model...")

    for i in range(n_samples):
        # Create diverse synthetic audio:
        # - Clean speech-like: smooth low-frequency signal
        # - Spoofed-like: sharper, more regular patterns
        if i < n_samples // 2:
            # Bonafide-like: natural noise + speech envelope
            duration = rng.uniform(1.5, 3.0)
            n = int(16000 * duration)
            t = np.linspace(0, duration, n)
            # Natural-sounding: multiple harmonics + noise
            freq = rng.uniform(80, 300)
            waveform = (
                0.3 * np.sin(2 * np.pi * freq * t)
                + 0.15 * np.sin(2 * np.pi * freq * 2 * t)
                + 0.1 * np.sin(2 * np.pi * freq * 3 * t)
                + 0.2 * rng.randn(n)
            ).astype(np.float32)
            # Apply random envelope
            envelope = np.clip(np.cumsum(rng.randn(n) * 0.001) + 0.5, 0.1, 1.0)
            waveform *= envelope.astype(np.float32)
            label = 0  # bonafide
        else:
            # Spoof-like: more synthetic/regular patterns
            duration = rng.uniform(1.5, 3.0)
            n = int(16000 * duration)
            t = np.linspace(0, duration, n)
            freq = rng.uniform(100, 400)
            # More regular, less natural variation
            waveform = (
                0.5 * np.sin(2 * np.pi * freq * t)
                + 0.25 * np.sin(2 * np.pi * freq * 2 * t + rng.uniform(0, 0.1))
                + 0.05 * rng.randn(n)
            ).astype(np.float32)
            label = 1  # spoof

        try:
            inputs = model.preprocess(waveform, 16000)
            input_values = inputs["input_values"].to(device)

            with torch.no_grad():
                emb = model.extract_embedding(input_values)
                embeddings.append(emb.cpu().squeeze(0).numpy())
                labels.append(label)
        except Exception:
            continue

        if i % 200 == 0:
            print(f"  Processed {i}/{n_samples}")

    embeddings = np.stack(embeddings)
    labels = np.array(labels)

    np.savez(output_path, embeddings=embeddings, labels=labels)
    print(f"  Saved {len(embeddings)} audio embeddings to {output_path}")
    print(f"  Shape: {embeddings.shape}, Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    return embeddings, labels


class RealFusionDataset(Dataset):
    """
    Creates cross-modal triplets from real embeddings with appropriate labels.

    Pairing strategy:
        safe (40%): bonafide audio + real video + safe text
        deepfake (20%): spoof audio + fake video + safe text
        coercion (20%): bonafide audio + real video + coercive text
        combined (20%): spoof audio + fake video + coercive text
    """

    def __init__(
        self,
        audio_embs: np.ndarray, audio_labels: np.ndarray,
        video_embs: np.ndarray, video_labels: np.ndarray,
        text_embs: np.ndarray, text_labels: np.ndarray,
        size: int = 5000, seed: int = 42,
    ):
        rng = np.random.RandomState(seed)

        # Split by class
        audio_real = audio_embs[audio_labels == 0]
        audio_fake = audio_embs[audio_labels == 1]
        video_real = video_embs[video_labels == 0]
        video_fake = video_embs[video_labels == 1]
        text_safe = text_embs[text_labels == 0]
        text_coerce = text_embs[text_labels > 0]  # any coercion type

        self.triplets = []
        self.labels = []

        for i in range(size):
            label = rng.choice([0, 1, 2, 3], p=[0.4, 0.2, 0.2, 0.2])

            if label == 0:  # Safe
                a = audio_real[rng.randint(len(audio_real))]
                v = video_real[rng.randint(len(video_real))]
                t = text_safe[rng.randint(len(text_safe))]
            elif label == 1:  # Deepfake only
                a = audio_fake[rng.randint(len(audio_fake))]
                v = video_fake[rng.randint(len(video_fake))]
                t = text_safe[rng.randint(len(text_safe))]
            elif label == 2:  # Coercion only
                a = audio_real[rng.randint(len(audio_real))]
                v = video_real[rng.randint(len(video_real))]
                t = text_coerce[rng.randint(len(text_coerce))]
            else:  # Combined
                a = audio_fake[rng.randint(len(audio_fake))]
                v = video_fake[rng.randint(len(video_fake))]
                t = text_coerce[rng.randint(len(text_coerce))]

            # Small noise augmentation to increase diversity
            a = a + rng.normal(0, 0.02, a.shape).astype(np.float32)
            v = v + rng.normal(0, 0.02, v.shape).astype(np.float32)
            t = t + rng.normal(0, 0.02, t.shape).astype(np.float32)

            self.triplets.append((
                torch.from_numpy(a.copy()),
                torch.from_numpy(v.copy()),
                torch.from_numpy(t.copy()),
            ))
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        a, v, t = self.triplets[idx]
        return a, v, t, self.labels[idx]


def train(args):
    from engine.fusion.cross_attention import MultimodalFusionNetwork

    print("=" * 60)
    print("Satya Drishti - Fusion Retraining on Real Embeddings")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Step 1: Extract or load embeddings ───
    text_path = EMBEDDINGS_DIR / "text_embeddings.npz"
    video_path = EMBEDDINGS_DIR / "video_embeddings.npz"
    audio_path = EMBEDDINGS_DIR / "audio_embeddings.npz"

    # Check if cached embeddings came from old models (different dimensions)
    # Video: old spatial_vit was 768d, new HuggingFace ViT is also 768d but different distribution
    # Audio: old AST was 768d, new Wav2Vec2 is also 768d but different distribution
    # Force re-extraction if a marker file indicates old model was used
    model_marker = EMBEDDINGS_DIR / "model_versions.json"
    current_versions = {
        "audio": "MelodyMachine/Deepfake-audio-detection-V2",
        "video": "prithivMLmods/Deep-Fake-Detector-v2-Model",
        "text": "DeBERTaV3+LoRA",
    }
    needs_reextract = False
    if model_marker.exists():
        with open(model_marker) as f:
            old_versions = json.load(f)
        if old_versions != current_versions:
            print("\n[!] Model versions changed since last extraction — forcing re-extraction")
            needs_reextract = True

    force = args.force_extract or needs_reextract

    if text_path.exists() and not force:
        print("\n[Text] Loading cached embeddings...")
        data = np.load(text_path)
        text_embs, text_labels = data["embeddings"], data["labels"]
        print(f"  Loaded {len(text_embs)} text embeddings")
    else:
        text_embs, text_labels = extract_text_embeddings(text_path, COERCION_TRAIN, args.max_samples)

    if video_path.exists() and not force:
        print("\n[Video] Loading cached embeddings...")
        data = np.load(video_path)
        video_embs, video_labels = data["embeddings"], data["labels"]
        print(f"  Loaded {len(video_embs)} video embeddings")
    else:
        video_embs, video_labels = extract_video_embeddings(video_path, args.max_samples)

    if audio_path.exists() and not force:
        print("\n[Audio] Loading cached embeddings...")
        data = np.load(audio_path)
        audio_embs, audio_labels = data["embeddings"], data["labels"]
        print(f"  Loaded {len(audio_embs)} audio embeddings")
    else:
        audio_embs, audio_labels = extract_audio_embeddings(audio_path, args.max_samples)

    # Save model version marker
    with open(model_marker, "w") as f:
        json.dump(current_versions, f, indent=2)

    if args.extract_only:
        print("\n[Done] Embeddings extracted. Use --extract_only=False to train.")
        return

    # Verify all embeddings are available
    if video_embs is None or audio_embs is None or text_embs is None:
        print("\n[Error] Could not extract all embeddings. Aborting.")
        return

    # ─── Step 2: Create datasets ───
    print(f"\n[Data] Creating {args.train_size} training and {args.train_size // 5} validation triplets...")

    train_dataset = RealFusionDataset(
        audio_embs, audio_labels,
        video_embs, video_labels,
        text_embs, text_labels,
        size=args.train_size, seed=42,
    )
    val_dataset = RealFusionDataset(
        audio_embs, audio_labels,
        video_embs, video_labels,
        text_embs, text_labels,
        size=args.train_size // 5, seed=123,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # ─── Step 3: Model ───
    # Check embedding dimensions
    audio_dim = audio_embs.shape[1]
    video_dim = video_embs.shape[1]
    text_dim = text_embs.shape[1]
    print(f"  Embedding dims: audio={audio_dim}, video={video_dim}, text={text_dim}")

    model = MultimodalFusionNetwork(
        audio_embed_dim=audio_dim,
        video_embed_dim=video_dim,
        text_embed_dim=text_dim,
        latent_dim=256,
        num_heads=8,
        num_layers=4,
        num_classes=4,
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # ─── Step 4: Training ───
    weights = torch.tensor([1.0, 2.5, 2.5, 4.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for a_emb, v_emb, t_emb, labels in train_loader:
            a_emb = a_emb.to(device)
            v_emb = v_emb.to(device)
            t_emb = t_emb.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(a_emb, v_emb, t_emb)
            loss = criterion(outputs["logits"], labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / n_batches
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_labels, all_preds = [], []

        with torch.no_grad():
            for a_emb, v_emb, t_emb, labels in val_loader:
                a_emb = a_emb.to(device)
                v_emb = v_emb.to(device)
                t_emb = t_emb.to(device)
                labels = labels.to(device)

                outputs = model(a_emb, v_emb, t_emb)
                val_loss += criterion(outputs["logits"], labels).item()
                val_batches += 1

                preds = torch.argmax(outputs["logits"], dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        avg_val_loss = val_loss / val_batches
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        acc = accuracy_score(all_labels, all_preds)

        print(f"Epoch {epoch}/{args.epochs}: train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | val_f1={f1:.4f} | val_acc={acc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            save_path = SAVE_DIR / "fusion_network_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_f1": best_f1,
                "trained_on": "real_embeddings",
            }, save_path)
            print(f"  -> New best F1={f1:.4f}! Saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # ─── Step 5: Final evaluation ───
    print("\n" + "=" * 60)
    print("Final Validation Report (Real Embeddings)")
    print("=" * 60)

    ckpt = torch.load(SAVE_DIR / "fusion_network_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_labels, all_preds = [], []
    with torch.no_grad():
        for a_emb, v_emb, t_emb, labels in val_loader:
            outputs = model(a_emb.to(device), v_emb.to(device), t_emb.to(device))
            preds = torch.argmax(outputs["logits"], dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())

    print(classification_report(
        all_labels, all_preds,
        target_names=["safe", "deepfake", "coercion", "combined"],
        digits=4,
    ))

    metrics = {
        "best_epoch": ckpt["epoch"],
        "best_f1": float(best_f1),
        "val_accuracy": float(accuracy_score(all_labels, all_preds)),
        "trained_on": "real_embeddings",
        "embedding_sources": {
            "audio": "Wav2Vec2 Deepfake-audio-detection-V2 (synthetic waveforms)",
            "video": "prithivMLmods/Deep-Fake-Detector-v2-Model ViT (deepfake frames)",
            "text": "DeBERTaV3+LoRA (coercion dataset)",
        },
    }
    with open(SAVE_DIR / "fusion_metrics_real.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[DONE] Best model at epoch {ckpt['epoch']}, F1={best_f1:.4f}")
    print("Trained on REAL embeddings from production models.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain fusion on real embeddings")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train_size", type=int, default=8000)
    parser.add_argument("--max_samples", type=int, default=2000, help="Max samples per modality for extraction")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--extract_only", action="store_true", help="Only extract embeddings, don't train")
    parser.add_argument("--force_extract", action="store_true", help="Force re-extraction even if cached")
    args = parser.parse_args()

    train(args)
