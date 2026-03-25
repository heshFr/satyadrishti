"""
Satya Drishti — Image Forensics Dataset Manifest Generator
==========================================================
Scans datasets/image_forensics/{authentic,ai_generated}/ and produces
a manifest.json compatible with the training pipeline.

Output format:
[
    {"path": "datasets/image_forensics/authentic/img001.jpg", "label": 0},
    {"path": "datasets/image_forensics/ai_generated/img002.png", "label": 1},
    ...
]

Usage:
    python scripts/prepare_image_forensics_manifest.py
    python scripts/prepare_image_forensics_manifest.py --dataset_dir datasets/image_forensics --output manifest.json
"""

import argparse
import json
import os
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def scan_directory(directory: str, label: int) -> list:
    """Recursively scan a directory for images, returning manifest entries."""
    entries = []
    dir_path = Path(directory)

    if not dir_path.exists():
        print(f"  Warning: Directory not found: {directory}")
        return entries

    for file_path in sorted(dir_path.rglob("*")):
        if file_path.suffix.lower() in IMAGE_EXTENSIONS and file_path.is_file():
            entries.append({
                "path": str(file_path).replace("\\", "/"),
                "label": label,
            })

    return entries


def main():
    parser = argparse.ArgumentParser(description="Generate image forensics dataset manifest")
    parser.add_argument("--dataset_dir", type=str, default="datasets/image_forensics",
                        help="Root directory containing authentic/ and ai_generated/ subdirs")
    parser.add_argument("--output", type=str, default=None,
                        help="Output manifest path (default: <dataset_dir>/manifest.json)")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    output_path = args.output or os.path.join(dataset_dir, "manifest.json")

    print(f"[Manifest] Scanning dataset directory: {dataset_dir}")

    # Label 0 = real/authentic, Label 1 = fake/ai-generated
    real_entries = scan_directory(os.path.join(dataset_dir, "authentic"), label=0)
    fake_entries = scan_directory(os.path.join(dataset_dir, "ai_generated"), label=1)

    manifest = real_entries + fake_entries

    print(f"[Manifest] Found {len(real_entries)} real images (label=0)")
    print(f"[Manifest] Found {len(fake_entries)} fake images (label=1)")
    print(f"[Manifest] Total: {len(manifest)} images")

    if not manifest:
        print("[Manifest] No images found! Check your dataset directory structure.")
        print(f"  Expected: {dataset_dir}/authentic/ and {dataset_dir}/ai_generated/")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Manifest] Saved to: {output_path}")


if __name__ == "__main__":
    main()
