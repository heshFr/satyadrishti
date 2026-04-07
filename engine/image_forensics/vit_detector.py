"""
Image Forensics — Pretrained Deepfake Detector (ViT-B/16)
=========================================================
Uses prithivMLmods/Deep-Fake-Detector-v2-Model from HuggingFace,
a ViT-B/16 fine-tuned on a large, diverse dataset of real and
AI-generated images (92%+ accuracy, F1: 0.925).

This replaces the previous custom ViT-B/16 that was trained on only
~9,000 images from a single parquet shard, which could not generalize
across modern AI generators (Midjourney, DALL-E 3, Flux, SDXL, etc.).

The pretrained model auto-downloads from HuggingFace on first use.
No local checkpoint files needed.
"""

import os
import numpy as np
import cv2
from typing import Tuple, Dict, Any

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModelForImageClassification, AutoImageProcessor
    HAS_VIT = True
except ImportError:
    HAS_VIT = False


class ViTDetector:
    """
    Pretrained ViT-B/16 deepfake detector from HuggingFace.

    Input: BGR image (numpy array) or file path
    Output: (fake_probability, details_dict)
    """

    INPUT_SIZE = 224
    MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-v2-Model"

    def __init__(self, pretrained_dir: str = None, weights_path: str = None, device: str = None):
        """
        Args:
            pretrained_dir: Ignored (kept for backward compatibility). Model loads from HuggingFace.
            weights_path: Ignored (kept for backward compatibility). Model loads from HuggingFace.
            device: Force device ("cuda" or "cpu"). Auto-detects if None.
        """
        if not HAS_VIT:
            raise RuntimeError("PyTorch and transformers are required for ViTDetector")

        if device and device != "cpu" and not torch.cuda.is_available():
            device = None  # fallback to auto-detect
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.processor = None
        self._fake_idx = None  # Index of the "fake/deepfake" class
        self._real_idx = None  # Index of the "real" class

        self._build_model()

    def _build_model(self):
        """Load pretrained deepfake detector from HuggingFace."""
        try:
            self.model = AutoModelForImageClassification.from_pretrained(
                self.MODEL_NAME,
            )
            self.model.to(self.device)
            self.model.eval()

            self.processor = AutoImageProcessor.from_pretrained(
                self.MODEL_NAME,
            )

            # Auto-detect label indices
            id2label = self.model.config.id2label
            self._detect_label_indices(id2label)

            print(f"[ViT] Loaded {self.MODEL_NAME} on {self.device}")
            print(f"[ViT] Labels: {id2label} (fake_idx={self._fake_idx}, real_idx={self._real_idx})")
        except Exception as e:
            print(f"[ViT] Failed to load model: {e}")
            self.model = None

    def _detect_label_indices(self, id2label: dict):
        """Auto-detect which label index is fake vs real."""
        fake_keywords = {"fake", "deepfake", "ai", "generated", "synthetic", "manipulated"}
        real_keywords = {"real", "realism", "authentic", "genuine", "original"}

        for idx, label in id2label.items():
            idx = int(idx)
            label_lower = str(label).lower().replace("-", "").replace("_", "")
            if any(kw in label_lower for kw in fake_keywords):
                self._fake_idx = idx
            elif any(kw in label_lower for kw in real_keywords):
                self._real_idx = idx

        # Fallback: if only one is found, infer the other
        num_labels = len(id2label)
        if self._fake_idx is not None and self._real_idx is None:
            self._real_idx = 1 - self._fake_idx if num_labels == 2 else 0
        elif self._real_idx is not None and self._fake_idx is None:
            self._fake_idx = 1 - self._real_idx if num_labels == 2 else 1
        elif self._fake_idx is None and self._real_idx is None:
            # Default: class 0 = real, class 1 = fake
            self._real_idx = 0
            self._fake_idx = 1

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Run inference on a single BGR image (numpy array).

        Returns:
            (fake_probability, details) where fake_probability is 0.0-1.0
        """
        if self.model is None:
            return 0.0, {"status": "error", "reason": "model not initialized"}

        try:
            # Convert BGR -> RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            inputs = self.processor(images=rgb, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

            outputs = self.model(pixel_values=pixel_values)
            probs = torch.softmax(outputs.logits, dim=1)

            fake_prob = probs[0][self._fake_idx].item()
            real_prob = probs[0][self._real_idx].item()

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

            # 3-6. Corner crops + center crop (90% of image)
            crop_ratio = 0.9
            crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
            if crop_h >= self.INPUT_SIZE and crop_w >= self.INPUT_SIZE:
                tl = np.array(pil_img.crop((0, 0, crop_w, crop_h)))
                augmented_images.append(tl)
                aug_labels.append("crop_tl")

                tr = np.array(pil_img.crop((w - crop_w, 0, w, crop_h)))
                augmented_images.append(tr)
                aug_labels.append("crop_tr")

                bl = np.array(pil_img.crop((0, h - crop_h, crop_w, h)))
                augmented_images.append(bl)
                aug_labels.append("crop_bl")

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
                fake_p = probs[0][self._fake_idx].item()
                all_probs.append(fake_p)
                per_aug.append({"augmentation": label, "fake_prob": round(fake_p, 4)})

            # Robust aggregation: trimmed mean
            if len(all_probs) >= 4:
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
    def extract_embedding(self, image: np.ndarray) -> "torch.Tensor":
        """
        Extract penultimate embedding for Cross-Modal Fusion.

        Uses the CLS token from the last hidden state of the ViT backbone.

        Args:
            image: BGR image (numpy array)

        Returns:
            embedding: (1, hidden_dim) tensor — typically (1, 768) for ViT-B/16
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
        # CLS token from the last hidden layer
        last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
        cls_embedding = last_hidden[:, 0, :]  # (batch, hidden_dim)
        return cls_embedding

    @torch.no_grad()
    def predict_from_path(self, image_path: str) -> Tuple[float, Dict[str, Any]]:
        """Convenience wrapper that reads an image from disk."""
        image = cv2.imread(image_path)
        if image is None:
            return 0.0, {"status": "error", "reason": f"Could not read image: {image_path}"}
        return self.predict(image)
