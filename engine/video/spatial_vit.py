"""
Video Spatial Stream — Vision Transformer for Deepfake Detection
=================================================================
Detects high-frequency blending artifacts at facial boundaries
using a ViT backbone fine-tuned for binary deepfake classification.
"""

import torch
import torch.nn as nn
from torchvision import models


class DeepfakeViT(nn.Module):
    """
    Vision Transformer (ViT-B/16) fine-tuned for deepfake detection.

    Focuses on spatial artifacts:
      - Blending boundaries where synthetic face meets background
      - High-frequency noise patterns from GAN/diffusion generation
      - Texture inconsistencies in skin regions

    Args:
        num_classes: Number of output classes (2 = real/fake)
        pretrained: Whether to load ImageNet pretrained weights
        embed_dim: ViT embedding dimension (768 for ViT-B/16)
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True, embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim

        # Load pretrained ViT-B/16 backbone
        self.backbone = models.vit_b_16(
            weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None
        )

        # Replace classification head
        self.backbone.heads = nn.Identity()

        # Custom classification head with dropout for forensic robustness
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (batch, 3, 224, 224) face-cropped RGB frames
        Returns:
            logits: (batch, num_classes)
        """
        features = self.backbone(frames)  # (batch, embed_dim)
        logits = self.classifier(features)
        return logits

    def extract_embedding(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract penultimate embedding for the Cross-Modal Fusion layer.

        Returns:
            embedding: (batch, embed_dim)
        """
        return self.backbone(frames)
