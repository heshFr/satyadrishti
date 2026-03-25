"""
Image Forensics — EfficientNet-B4 Deepfake Detector
====================================================
Binary classifier (real vs fake) built on EfficientNet-B4 pretrained
on ImageNet. The final classifier head is replaced with a 2-class
linear layer. Supports loading fine-tuned weights from disk.

Why EfficientNet-B4:
- Top performer on FaceForensics++ and DFDC benchmarks
- 19M params → fits easily in 4GB VRAM (RTX 3050)
- Fast inference (~20ms per image at 380x380)
- Excellent transfer learning from ImageNet
"""

import os
import numpy as np
import cv2
from typing import Tuple, Dict, Any

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class EfficientNetDetector:
    """
    EfficientNet-B4 based deepfake detector.

    Input: BGR image (numpy array) or file path
    Output: (fake_probability, details_dict)
    """

    # EfficientNet-B4 native input size
    INPUT_SIZE = 380

    def __init__(self, weights_path: str = None, device: str = None):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for EfficientNetDetector")

        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.INPUT_SIZE, self.INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self._build_model()

        if weights_path and os.path.exists(weights_path):
            self._load_weights(weights_path)
        else:
            print("[EfficientNet] No fine-tuned weights found — using ImageNet pretrained backbone.")

    def _build_model(self):
        """Build EfficientNet-B4 with a 2-class head."""
        self.model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        # Replace classifier: EfficientNet-B4 has 1792 features before the head
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 2),
        )
        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self, weights_path: str):
        """Load fine-tuned checkpoint."""
        print(f"[EfficientNet] Loading weights from {weights_path} on {self.device}...")
        try:
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
            # Support both raw state_dict and wrapped checkpoint formats
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("[EfficientNet] Weights loaded successfully.")
        except Exception as e:
            print(f"[EfficientNet] Error loading weights: {e}")
            print("[EfficientNet] Falling back to ImageNet pretrained backbone.")

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
            # Convert BGR → RGB
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)

            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
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
    def predict_from_path(self, image_path: str) -> Tuple[float, Dict[str, Any]]:
        """Convenience wrapper that reads an image from disk."""
        image = cv2.imread(image_path)
        if image is None:
            return 0.0, {"status": "error", "reason": f"Could not read image: {image_path}"}
        return self.predict(image)
