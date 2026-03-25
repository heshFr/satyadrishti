"""
Download diffusion-generated images from OpenFake dataset (HuggingFace)
for retraining the ViT deepfake detector to recognize modern AI images.

Covers: DALL-E 3, Midjourney 6/7, Flux, SD 1.5/2.1/XL/3.5, GPT Image, Imagen, etc.

Usage:
    python -m scripts.download_diffusion_data
    python -m scripts.download_diffusion_data --max_per_class 5000
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_DIR = PROJECT_ROOT / "datasets" / "diffusion_mixed"


def download(max_per_class: int = 10000, target_size: int = 224):
    from datasets import load_dataset
    from PIL import Image

    print("=" * 60)
    print("Downloading OpenFake diffusion dataset")
    print("=" * 60)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    real_dir = SAVE_DIR / "real"
    fake_dir = SAVE_DIR / "fake"
    real_dir.mkdir(exist_ok=True)
    fake_dir.mkdir(exist_ok=True)

    # Count existing files
    existing_real = len(list(real_dir.glob("*.jpg")))
    existing_fake = len(list(fake_dir.glob("*.jpg")))
    print(f"Existing: {existing_real} real, {existing_fake} fake")

    if existing_real >= max_per_class and existing_fake >= max_per_class:
        print("Already have enough images. Use --max_per_class to increase.")
        return

    ds = load_dataset("ComplexDataLab/OpenFake", split="train", streaming=True)

    real_count = existing_real
    fake_count = existing_fake
    skipped = 0

    # Track per-model counts for diversity
    model_counts = {}

    for sample in ds:
        if real_count >= max_per_class and fake_count >= max_per_class:
            break

        label = sample["label"]
        model_name = sample["model"]
        img = sample["image"]

        if label == "real" and real_count >= max_per_class:
            continue
        if label == "fake" and fake_count >= max_per_class:
            continue

        try:
            # Resize and save as JPEG
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = img.resize((target_size, target_size), Image.LANCZOS)

            if label == "real":
                out_path = real_dir / f"real_{real_count:06d}.jpg"
                real_count += 1
            else:
                out_path = fake_dir / f"fake_{fake_count:06d}_{model_name}.jpg"
                fake_count += 1
                model_counts[model_name] = model_counts.get(model_name, 0) + 1

            img.save(out_path, "JPEG", quality=95)

            total = real_count + fake_count
            if total % 500 == 0:
                print(f"  Downloaded {real_count} real + {fake_count} fake = {total} total")

        except Exception:
            skipped += 1
            continue

    print(f"\nDone: {real_count} real + {fake_count} fake")
    print(f"Skipped: {skipped}")
    print(f"\nFake images by model:")
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        print(f"  {model:30s}: {count}")
    print(f"\nSaved to: {SAVE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_per_class", type=int, default=10000)
    parser.add_argument("--target_size", type=int, default=224)
    args = parser.parse_args()
    download(args.max_per_class, args.target_size)
