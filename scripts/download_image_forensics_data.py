"""
Satya Drishti
=============
Downloads an AI-Generated Image forensics dataset.
Uses the HuggingFace datasets library to fetch a variety of
diffusion and GAN generated faces vs real faces.

This dataset is used to train the EfficientNet-B4 component of the
Media Scanner's `ImageForensicsDetector`.

Default dataset: OpenRL/DeepFakeFace (120K images, diffusion-generated celebrity faces)
Fallback: Any HuggingFace image classification dataset with 'image' and 'label' columns.
"""

import os
import argparse
from tqdm import tqdm

try:
    from datasets import load_dataset
    from PIL import Image
    HAS_HF = True
except ImportError:
    HAS_HF = False

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "image_forensics")

# Known dataset configs: maps repo ID to label parsing logic
DATASET_CONFIGS = {
    "Hemg/deepfake-and-real-images": {
        "split": "train",
        "image_col": "image",
        "label_col": "label",
        # 0 = fake, 1 = real (dataset-specific mapping)
        "fake_labels": [0],
    },
    "OpenRL/DeepFakeFace": {
        "split": "train",
        "image_col": "image",
        "label_col": "label",
        # 0 = real, 1 = fake (standard convention)
        "fake_labels": [1, "fake", "ai", "generated", True],
    },
    "danjacobellis/fake_faces": {
        "split": "train",
        "image_col": "image",
        "label_col": "label",
        "fake_labels": [1, "fake", "ai", True],
    },
}


def download_and_extract(dataset_name="OpenRL/DeepFakeFace", split=None, max_samples=10000):
    """
    Downloads images from a HuggingFace dataset and saves them locally.
    Default dataset: OpenRL/DeepFakeFace (diffusion-generated celebrity faces).
    """
    if not HAS_HF:
        print("Missing dependencies. Run: pip install datasets pillow")
        return

    config = DATASET_CONFIGS.get(dataset_name, {})
    if split is None:
        split = config.get("split", "train")
    image_col = config.get("image_col", "image")
    label_col = config.get("label_col", "label")
    fake_labels = config.get("fake_labels", [1, "fake", "ai", "generated", True])

    print(f"Downloading {dataset_name} ({split} split)...")
    try:
        ds = load_dataset(dataset_name, split=split, streaming=True)
    except Exception as e:
        print(f"Failed to load dataset '{dataset_name}': {e}")
        print("Trying fallback dataset 'danjacobellis/fake_faces'...")
        try:
            dataset_name = "danjacobellis/fake_faces"
            config = DATASET_CONFIGS[dataset_name]
            split = config["split"]
            image_col = config["image_col"]
            label_col = config["label_col"]
            fake_labels = config["fake_labels"]
            ds = load_dataset(dataset_name, split=split, streaming=True)
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return

    real_dir = os.path.join(DATA_DIR, "authentic")
    fake_dir = os.path.join(DATA_DIR, "ai_generated")

    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    # Two-pass approach: collect target count per class for balanced dataset
    per_class = max_samples // 2
    real_count = 0
    fake_count = 0

    print(f"Extracting up to {max_samples} images ({per_class} per class) to local disk...")
    print(f"  Dataset: {dataset_name}")
    print(f"  Image column: {image_col}, Label column: {label_col}")
    print(f"  Fake labels: {fake_labels}")
    print(f"  Output: {DATA_DIR}")

    with tqdm(total=max_samples, desc="Extracting") as pbar:
        for item in ds:
            if real_count >= per_class and fake_count >= per_class:
                break

            try:
                image = item.get(image_col)
                if image is None:
                    for col in ["image", "img", "photo"]:
                        if col in item:
                            image = item[col]
                            break
                if image is None:
                    continue

                label = item.get(label_col, 0)
                is_fake = label in fake_labels

                if is_fake and fake_count >= per_class:
                    continue
                if not is_fake and real_count >= per_class:
                    continue

                if is_fake:
                    out_path = os.path.join(fake_dir, f"ai_face_{fake_count:06d}.jpg")
                    fake_count += 1
                else:
                    out_path = os.path.join(real_dir, f"real_face_{real_count:06d}.jpg")
                    real_count += 1

                if image.mode != "RGB":
                    image = image.convert("RGB")

                image = image.resize((380, 380), Image.LANCZOS)
                image.save(out_path, quality=95)
                pbar.update(1)

            except Exception as e:
                print(f"  Warning: Skipping item due to Error: {e}")
                continue

    print("\nDataset preparation complete.")
    print(f"  Authentic faces extracted: {real_count}")
    print(f"  AI-Generated faces extracted: {fake_count}")
    print(f"  Total: {real_count + fake_count}")


def main():
    parser = argparse.ArgumentParser(description="Download image forensics dataset from HuggingFace")
    parser.add_argument(
        "--repo", type=str, default="Hemg/deepfake-and-real-images",
        help="HuggingFace dataset repo ID (default: Hemg/deepfake-and-real-images)"
    )
    parser.add_argument("--split", type=str, default=None, help="Dataset split (default: auto-detect)")
    parser.add_argument("--samples", type=int, default=10000, help="Max total samples to extract (default: 10000)")
    args = parser.parse_args()

    download_and_extract(dataset_name=args.repo, split=args.split, max_samples=args.samples)


if __name__ == "__main__":
    main()
