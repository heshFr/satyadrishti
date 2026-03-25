"""
Video Temporal Stream — Pretrained R3D-18 for Deepfake Detection
=================================================================
Uses ResNet3D-18 pretrained on Kinetics-400 (action recognition)
and fine-tunes for temporal deepfake detection.

Much more effective than training from scratch on small datasets.
"""

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class TemporalR3D(nn.Module):
    """
    ResNet3D-18 pretrained on Kinetics-400, fine-tuned for deepfake detection.

    Args:
        num_classes: Output classes (2 = real/fake)
        pretrained: Whether to load Kinetics-400 pretrained weights
        embed_dim: Backbone embedding dimension (512 for R3D-18)
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True, embed_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim

        self.backbone = r3d_18(
            weights=R3D_18_Weights.DEFAULT if pretrained else None
        )

        # Replace classification head
        self.backbone.fc = nn.Identity()

        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clips: (batch, 3, T, H, W) video clips
        Returns:
            logits: (batch, num_classes)
        """
        x = self.backbone(clips)
        return self.classifier(x)

    def extract_embedding(self, clips: torch.Tensor) -> torch.Tensor:
        """Extract penultimate embedding for fusion. Returns (batch, 512)."""
        return self.backbone(clips)
