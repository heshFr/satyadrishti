"""
RawNet3 — Anti-Spoofing Network Architecture
===============================================
A deep residual network designed for raw waveform-based audio
anti-spoofing. Uses SincConv for learnable bandpass filters,
Res2Net-style multi-scale residual blocks, GRU aggregation,
and attentive statistical pooling.

Reference: Jung et al., "Improved RawNet with Feature Map Scaling
for Text-Independent Speaker Verification using Raw Waveforms"
(adapted for anti-spoofing).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv(nn.Module):
    """
    Sinc-based convolution layer with learnable bandpass filters.
    Learns the low and high cutoff frequencies directly from data,
    capturing frequency bands most discriminative for spoofing artifacts.
    """

    def __init__(self, out_channels: int = 128, kernel_size: int = 1024, sample_rate: int = 16000):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Initialize cutoff frequencies uniformly in mel scale
        low_hz = 0.0
        high_hz = sample_rate / 2.0
        mel_low = 2595.0 * math.log10(1.0 + low_hz / 700.0)
        mel_high = 2595.0 * math.log10(1.0 + high_hz / 700.0)
        mel_points = torch.linspace(mel_low, mel_high, out_channels + 1)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

        # Learnable parameters: low and band frequencies
        self.low_hz_ = nn.Parameter(hz_points[:-1].clone())
        self.band_hz_ = nn.Parameter((hz_points[1:] - hz_points[:-1]).clone())

        # Hamming window (fixed)
        n = torch.arange(0, kernel_size).float()
        window = 0.54 - 0.46 * torch.cos(2.0 * math.pi * n / kernel_size)
        self.register_buffer("window", window)

        # Time axis for sinc computation
        n_ = (2 * math.pi / sample_rate) * (torch.arange(-(kernel_size // 2), kernel_size // 2 + 1).float())
        self.register_buffer("n_", n_[: kernel_size])

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (batch, 1, time)
        Returns:
            (batch, out_channels, time')
        """
        low = torch.abs(self.low_hz_) + 1.0
        high = torch.clamp(low + torch.abs(self.band_hz_), min=2.0, max=self.sample_rate / 2.0)

        # Bandpass filter = lowpass(high) - lowpass(low)
        f_low = low / self.sample_rate
        f_high = high / self.sample_rate

        band_pass_left = (
            (torch.sin(f_high.unsqueeze(1) * self.n_) - torch.sin(f_low.unsqueeze(1) * self.n_))
            / (self.n_ / 2 + 1e-8)
        )
        band_pass_center = 2 * (f_high - f_low).unsqueeze(1)
        band_pass_right = band_pass_left

        # Construct symmetric filter
        band_pass = band_pass_left * self.window
        band_pass = band_pass / (2 * band_pass.sum(dim=1, keepdim=True) + 1e-8)

        filters = band_pass.unsqueeze(1)  # (out_channels, 1, kernel_size)
        return F.conv1d(waveform, filters, stride=1, padding=self.kernel_size // 2)


class Res2NetBlock(nn.Module):
    """
    Multi-scale residual block inspired by Res2Net.
    Splits channels into multiple groups and processes them
    hierarchically for multi-scale feature extraction.
    """

    def __init__(self, channels: int, scale: int = 4):
        super().__init__()
        self.scale = scale
        width = channels // scale
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm1d(width),
                nn.LeakyReLU(0.3),
                nn.Conv1d(width, width, kernel_size=3, padding=1),
            )
            for _ in range(scale - 1)
        ])
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        chunks = torch.chunk(x, self.scale, dim=1)
        outputs = [chunks[0]]
        for i, conv in enumerate(self.convs):
            if i == 0:
                h = conv(chunks[i + 1])
            else:
                h = conv(chunks[i + 1] + outputs[-1])
            outputs.append(h)
        out = torch.cat(outputs, dim=1)
        out = self.bn(out)
        return out + residual


class AttentiveStatisticsPooling(nn.Module):
    """
    Attentive statistics pooling: computes attention-weighted
    mean and standard deviation across the time dimension.
    """

    def __init__(self, channels: int, attention_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, attention_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attention_dim, channels, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time)
        Returns:
            (batch, channels * 2)  — concatenated weighted mean & std
        """
        attn = self.attention(x)
        mean = (attn * x).sum(dim=2)
        std = torch.sqrt(((attn * x**2).sum(dim=2) - mean**2).clamp(min=1e-8))
        return torch.cat([mean, std], dim=1)


class RawNet3(nn.Module):
    """
    RawNet3 Anti-Spoofing Model.

    Pipeline:
        Raw Waveform → SincConv → Res2Net Blocks → GRU → ASP → FC → Logits

    Args:
        sinc_filters: Number of SincConv output channels
        sinc_kernel_size: Kernel size for SincConv
        res_channels: Channel width for Res2Net blocks
        num_res_blocks: Number of Res2Net blocks per stage
        gru_hidden: GRU hidden dimension
        gru_layers: Number of GRU layers
        num_classes: Output classes (2 for bonafide/spoof)
    """

    def __init__(
        self,
        sinc_filters: int = 128,
        sinc_kernel_size: int = 1024,
        res_channels: int = 512,
        num_res_blocks: int = 6,
        gru_hidden: int = 1024,
        gru_layers: int = 3,
        num_classes: int = 2,
        sample_rate: int = 16000,
    ):
        super().__init__()

        # Stage 1: Learnable bandpass filters
        self.sinc_conv = SincConv(sinc_filters, sinc_kernel_size, sample_rate)
        self.bn_sinc = nn.BatchNorm1d(sinc_filters)

        # Stage 2: Channel expansion + Res2Net blocks
        self.conv_expand = nn.Conv1d(sinc_filters, res_channels, kernel_size=3, padding=1)
        self.bn_expand = nn.BatchNorm1d(res_channels)

        self.res_blocks = nn.Sequential(*[
            Res2NetBlock(res_channels, scale=4) for _ in range(num_res_blocks)
        ])

        # Stage 3: GRU temporal aggregation
        self.gru = nn.GRU(
            input_size=res_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False,
        )

        # Stage 4: Attentive statistics pooling
        self.asp = AttentiveStatisticsPooling(gru_hidden)

        # Stage 5: Classification head
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden * 2, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (batch, 1, time) raw audio signal
        Returns:
            logits: (batch, num_classes)
        """
        # SincConv → BN → LeakyReLU → MaxPool
        x = self.sinc_conv(waveform)
        x = F.leaky_relu(self.bn_sinc(x), 0.3)
        x = F.max_pool1d(x, kernel_size=3, stride=3)

        # Channel expansion → Res2Net blocks
        x = F.leaky_relu(self.bn_expand(self.conv_expand(x)), 0.3)
        x = self.res_blocks(x)

        # GRU: (batch, channels, time) → (batch, time, channels) → GRU
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        x = x.transpose(1, 2)  # back to (batch, channels, time)

        # Attentive statistics pooling → FC
        x = self.asp(x)
        logits = self.fc(x)

        return logits

    def extract_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract the penultimate embedding (before classification head).
        Used by the Cross-Modal Fusion layer.

        Returns:
            embedding: (batch, gru_hidden * 2)
        """
        x = self.sinc_conv(waveform)
        x = F.leaky_relu(self.bn_sinc(x), 0.3)
        x = F.max_pool1d(x, kernel_size=3, stride=3)
        x = F.leaky_relu(self.bn_expand(self.conv_expand(x)), 0.3)
        x = self.res_blocks(x)
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        x = x.transpose(1, 2)
        embedding = self.asp(x)
        return embedding
