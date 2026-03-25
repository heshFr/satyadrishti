"""
Generate manifest.json for the diffusion_mixed dataset.
Combines downloaded OpenFake images with existing GAN dataset for balanced training.

Usage:
    python -m scripts.prepare_diffusion_manifest
    python -m scripts.prepare_diffusion_manifest --include_gan_data
"""

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIFFUSION_DIR = PROJECT_ROOT / "datasets" / "diffusion_mixed"
GAN_MANIFEST = PROJECT_ROOT / "datasets" / "image_forensics" / "manifest.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include_gan_data", action="store_true",
                        help="Include original GAN dataset images for mixed training")
    parser.add_argument("--gan_sample", type=int, default=5000,
                        help="Max GAN images to include (per class)")
    args = parser.parse_args()

    manifest = []

    # Diffusion data
    real_dir = DIFFUSION_DIR / "real"
    fake_dir = DIFFUSION_DIR / "fake"

    real_count = 0
    fake_count = 0

    if real_dir.exists():
        for img_path in sorted(real_dir.glob("*.jpg")):
            manifest.append({"path": str(img_path), "label": 0})
            real_count += 1

    if fake_dir.exists():
        for img_path in sorted(fake_dir.glob("*.jpg")):
            manifest.append({"path": str(img_path), "label": 1})
            fake_count += 1

    print(f"Diffusion data: {real_count} real + {fake_count} fake")

    # Optionally include GAN data for mixed training
    if args.include_gan_data and GAN_MANIFEST.exists():
        with open(GAN_MANIFEST) as f:
            gan_data = json.load(f)

        import numpy as np
        rng = np.random.RandomState(42)
        rng.shuffle(gan_data)

        gan_real = [e for e in gan_data if e["label"] == 0]
        gan_fake = [e for e in gan_data if e["label"] == 1]

        # Sample equally
        gan_real = gan_real[:args.gan_sample]
        gan_fake = gan_fake[:args.gan_sample]

        # Verify paths exist
        gan_added = 0
        for entry in gan_real + gan_fake:
            if os.path.exists(entry["path"]):
                manifest.append(entry)
                gan_added += 1

        print(f"GAN data added: {len(gan_real)} real + {len(gan_fake)} fake = {gan_added} total")

    print(f"Total manifest: {len(manifest)} entries")

    out_path = DIFFUSION_DIR / "manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
