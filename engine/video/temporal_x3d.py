"""
Video Temporal Stream — X3D for Physiological Inconsistency
=============================================================
Lightweight 3D CNN analyzing inter-frame temporal patterns
to detect physiological inconsistencies (fake heartbeats,
unnatural blinking, synthetic micro-expressions).
"""

import torch
import torch.nn as nn


class TemporalX3D(nn.Module):
    """
    X3D-S (Extremely Lightweight 3D CNN) adapted for deepfake
    temporal analysis.

    Uses depthwise-separable 3D convolutions for efficiency.
    Designed to run on edge devices while capturing inter-frame
    physiological signals that deepfakes cannot replicate.

    Args:
        clip_length: Number of frames per video clip
        num_classes: Output classes (2 = real/fake)
        embed_dim: Penultimate embedding dimension
    """

    def __init__(self, clip_length: int = 16, num_classes: int = 2, embed_dim: int = 432):
        super().__init__()
        self.embed_dim = embed_dim

        # Stem: initial 3D conv
        self.stem = nn.Sequential(
            nn.Conv3d(3, 24, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
        )

        # X3D stages with expanding channels and temporal convolutions
        self.stage1 = self._make_stage(24, 54, depth=3, temporal_kernel=3, spatial_stride=2)
        self.stage2 = self._make_stage(54, 108, depth=5, temporal_kernel=3, spatial_stride=2)
        self.stage3 = self._make_stage(108, 216, depth=11, temporal_kernel=3, spatial_stride=2)
        self.stage4 = self._make_stage(216, 432, depth=7, temporal_kernel=3, spatial_stride=2)

        # Global pooling → embedding
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(embed_dim, num_classes),
        )

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        temporal_kernel: int,
        spatial_stride: int,
    ) -> nn.Sequential:
        """Build a stage of depthwise-separable 3D blocks."""
        layers = []
        for i in range(depth):
            stride = (1, spatial_stride, spatial_stride) if i == 0 else (1, 1, 1)
            ch_in = in_channels if i == 0 else out_channels
            layers.append(X3DBlock(ch_in, out_channels, temporal_kernel, stride))
        return nn.Sequential(*layers)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clips: (batch, 3, T, H, W) video clips
        Returns:
            logits: (batch, num_classes)
        """
        x = self.stem(clips)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

    def extract_embedding(self, clips: torch.Tensor) -> torch.Tensor:
        """
        Extract penultimate embedding for Cross-Modal Fusion.

        Returns:
            embedding: (batch, embed_dim)
        """
        x = self.stem(clips)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.pool(x).flatten(1)


class X3DBlock(nn.Module):
    """
    X3D residual block with depthwise-separable 3D convolutions.
    Expands channels internally then projects back down.
    """

    def __init__(self, in_channels: int, out_channels: int, temporal_kernel: int, stride: tuple):
        super().__init__()
        expand = out_channels * 3  # channel expansion ratio

        self.conv1 = nn.Conv3d(in_channels, expand, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(expand)

        # Depthwise temporal conv
        self.conv2 = nn.Conv3d(
            expand, expand,
            kernel_size=(temporal_kernel, 3, 3),
            stride=stride,
            padding=(temporal_kernel // 2, 1, 1),
            groups=expand,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(expand)

        self.se = SqueezeExcitation3D(expand, expand // 4)

        self.conv3 = nn.Conv3d(expand, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Skip connection
        self.skip = nn.Identity()
        if in_channels != out_channels or stride != (1, 1, 1):
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class SqueezeExcitation3D(nn.Module):
    """3D Squeeze-and-Excitation block for channel recalibration."""

    def __init__(self, channels: int, reduction: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.size()
        scale = self.pool(x).view(b, c)
        scale = self.fc(scale).view(b, c, 1, 1, 1)
        return x * scale
