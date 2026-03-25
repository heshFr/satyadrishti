"""
Image Forensics — ViT-B/16 Deepfake Detector
=============================================
Binary classifier (real vs fake) built on google/vit-base-patch16-224-in21k
via HuggingFace Transformers. Loads pretrained backbone from local directory,
with optional fine-tuned checkpoint overlay.

Why ViT-B/16 over EfficientNet-B4:
- Pretrained on 14M images (ImageNet-21k) vs 1.2M (ImageNet-1k)
- Self-attention captures global manipulation artifacts across the image
- 86M params but only fine-tunes last 4 blocks + head (~30M trainable)
- Native 224x224 input with 16x16 patch embedding
"""

import os
import numpy as np
import cv2
from typing import Tuple, Dict, Any

try:
    import torch
    import torch.nn as nn
    from transformers import ViTForImageClassification, ViTImageProcessor
    HAS_VIT = True
except ImportError:
    HAS_VIT = False

# Default pretrained directory (local HuggingFace-format)
DEFAULT_PRETRAINED_DIR = os.path.join("models", "image_forensics", "pretrained_vit")


class ViTDetector:
    """
    ViT-B/16 based deepfake detector.

    Input: BGR image (numpy array) or file path
    Output: (fake_probability, details_dict)
    """

    INPUT_SIZE = 224

    def __init__(self, pretrained_dir: str = None, weights_path: str = None, device: str = None):
        """
        Args:
            pretrained_dir: Path to local HuggingFace model dir (config.json + model.safetensors).
            weights_path: Path to fine-tuned .pt checkpoint (overrides pretrained classifier head).
            device: Force device ("cuda" or "cpu"). Auto-detects if None.
        """
        if not HAS_VIT:
            raise RuntimeError("PyTorch and transformers are required for ViTDetector")

        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.processor = None

        effective_dir = pretrained_dir or DEFAULT_PRETRAINED_DIR
        self._build_model(effective_dir)

        if weights_path and os.path.exists(weights_path):
            self._load_finetuned(weights_path)
        else:
            print("[ViT] No fine-tuned weights found — using ImageNet-21k pretrained backbone.")

    def _build_model(self, pretrained_dir: str):
        """Load ViT-B/16 with a 2-class head from local pretrained directory."""
        try:
            # Try local dir first, fall back to HuggingFace hub
            model_source = pretrained_dir if os.path.isdir(pretrained_dir) else "google/vit-base-patch16-224-in21k"

            self.model = ViTForImageClassification.from_pretrained(
                model_source,
                num_labels=2,
                ignore_mismatched_sizes=True,
            )
            self.model.to(self.device)
            self.model.eval()

            # Processor handles resize + normalize (224x224, mean/std 0.5)
            self.processor = ViTImageProcessor(
                size={"height": self.INPUT_SIZE, "width": self.INPUT_SIZE},
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
                do_rescale=True,
                do_normalize=True,
                do_resize=True,
            )

            print(f"[ViT] ViT-B/16 loaded from {model_source} on {self.device}")
        except Exception as e:
            print(f"[ViT] Failed to build model: {e}")
            self.model = None

    def _load_finetuned(self, weights_path: str):
        """Load fine-tuned checkpoint (state_dict saved by training script)."""
        print(f"[ViT] Loading fine-tuned weights from {weights_path}...")
        try:
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("[ViT] Fine-tuned weights loaded successfully.")
        except Exception as e:
            print(f"[ViT] Error loading fine-tuned weights: {e}")
            print("[ViT] Falling back to pretrained backbone.")

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Run inference on a single BGR image (numpy array).

        Returns:
            (fake_probability, details) where fake_probability is 0.0-1.0
            and details contains raw logits and confidence info.
        """
        if self.model is None:
            return 0.0, {"status": "error", "reason": "model not initialized"}

        try:
            # Convert BGR -> RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # ViTImageProcessor expects PIL or numpy HWC RGB
            inputs = self.processor(images=rgb, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

            outputs = self.model(pixel_values=pixel_values)
            probs = torch.softmax(outputs.logits, dim=1)
            real_prob = probs[0][0].item()
            fake_prob = probs[0][1].item()

            return fake_prob, {
                "status": "success",
                "real_probability": round(real_prob, 4),
                "fake_probability": round(fake_prob, 4),
                "confidence": round(max(real_prob, fake_prob), 4),
            }
        except Exception as e:
            return 0.0, {"status": "error", "reason": str(e)}

    @torch.no_grad()
    def predict_tta(self, image: np.ndarray, num_augments: int = 5) -> Tuple[float, Dict[str, Any]]:
        """
        Test-Time Augmentation: run multiple augmented versions of the image
        and average the predictions for more robust results.

        Augmentations: original, horizontal flip, 4 corner crops + center crop.
        This significantly reduces false positives on compressed photos
        and improves detection of subtle AI artifacts.

        Returns:
            (avg_fake_probability, details) with per-augmentation breakdown.
        """
        if self.model is None:
            return 0.0, {"status": "error", "reason": "model not initialized"}

        try:
            from PIL import Image as PILImage

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)
            w, h = pil_img.size

            # Generate augmented views
            augmented_images = []
            aug_labels = []

            # 1. Original
            augmented_images.append(rgb)
            aug_labels.append("original")

            # 2. Horizontal flip
            flipped = cv2.flip(rgb, 1)
            augmented_images.append(flipped)
            aug_labels.append("h_flip")

            # 3-6. Four corner crops + center crop (90% of image)
            crop_ratio = 0.9
            crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
            if crop_h >= self.INPUT_SIZE and crop_w >= self.INPUT_SIZE:
                # Top-left
                tl = np.array(pil_img.crop((0, 0, crop_w, crop_h)))
                augmented_images.append(tl)
                aug_labels.append("crop_tl")

                # Top-right
                tr = np.array(pil_img.crop((w - crop_w, 0, w, crop_h)))
                augmented_images.append(tr)
                aug_labels.append("crop_tr")

                # Bottom-left
                bl = np.array(pil_img.crop((0, h - crop_h, crop_w, h)))
                augmented_images.append(bl)
                aug_labels.append("crop_bl")

                # Center crop
                cx, cy = w // 2, h // 2
                cc = np.array(pil_img.crop((
                    cx - crop_w // 2, cy - crop_h // 2,
                    cx + crop_w // 2, cy + crop_h // 2,
                )))
                augmented_images.append(cc)
                aug_labels.append("crop_center")

            # Limit to requested number
            augmented_images = augmented_images[:num_augments]
            aug_labels = aug_labels[:num_augments]

            # Batch inference
            all_probs = []
            per_aug = []
            for img_rgb, label in zip(augmented_images, aug_labels):
                inputs = self.processor(images=img_rgb, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)
                outputs = self.model(pixel_values=pixel_values)
                probs = torch.softmax(outputs.logits, dim=1)
                fake_p = probs[0][1].item()
                all_probs.append(fake_p)
                per_aug.append({"augmentation": label, "fake_prob": round(fake_p, 4)})

            # Robust aggregation: trimmed mean (remove single most extreme outlier)
            # This prevents a single bad crop from dominating the verdict
            if len(all_probs) >= 4:
                # Remove the value furthest from the median
                median_val = float(np.median(all_probs))
                distances = [abs(p - median_val) for p in all_probs]
                worst_idx = distances.index(max(distances))
                trimmed = [p for i, p in enumerate(all_probs) if i != worst_idx]
                avg_fake = float(np.mean(trimmed))
            elif len(all_probs) >= 2:
                avg_fake = float(np.mean(all_probs))
            else:
                avg_fake = all_probs[0] if all_probs else 0.5
            std_fake = float(np.std(all_probs))

            return avg_fake, {
                "status": "success",
                "method": "tta",
                "num_augmentations": len(all_probs),
                "avg_fake_probability": round(avg_fake, 4),
                "std_fake_probability": round(std_fake, 4),
                "real_probability": round(1 - avg_fake, 4),
                "fake_probability": round(avg_fake, 4),
                "confidence": round(max(avg_fake, 1 - avg_fake), 4),
                "per_augmentation": per_aug,
            }
        except Exception as e:
            # Fall back to single prediction
            return self.predict(image)

    @torch.no_grad()
    def predict_from_path(self, image_path: str) -> Tuple[float, Dict[str, Any]]:
        """Convenience wrapper that reads an image from disk."""
        image = cv2.imread(image_path)
        if image is None:
            return 0.0, {"status": "error", "reason": f"Could not read image: {image_path}"}
        return self.predict(image)
