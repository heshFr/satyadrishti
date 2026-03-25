"""
rPPG (Remote Photoplethysmography) Extraction
================================================
Extracts pulse/blood flow signals from facial video frames.
Deepfakes lack genuine physiological signals — their rPPG
waveforms are flat noise, which is a biometric barrier
generative models fundamentally cannot fake.

Implements the CHROM (Chrominance-based) method:
  De Haan & Jeanne, "Robust Pulse Rate from Chrominance-Based rPPG" (2013)
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import signal as scipy_signal


class RPPGExtractor:
    """
    Extract rPPG (remote photoplethysmography) pulse signals from
    a sequence of face-cropped video frames.

    The CHROM method exploits chrominance changes in skin pixels
    to extract a blood volume pulse (BVP) signal without contact.

    Args:
        fps: Frame rate of the input video
        window_size: Number of frames per analysis window
        bandpass_low: Low cutoff frequency in Hz (maps to min heartrate)
        bandpass_high: High cutoff frequency in Hz (maps to max heartrate)
    """

    def __init__(
        self,
        fps: float = 30.0,
        window_size: int = 300,
        bandpass_low: float = 0.7,
        bandpass_high: float = 4.0,
    ):
        self.fps = fps
        self.window_size = window_size
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high

    def extract_skin_signal(self, frames: np.ndarray) -> np.ndarray:
        """
        Extract spatially-averaged RGB skin signal from face frames.

        Args:
            frames: (num_frames, H, W, 3) face-cropped RGB frames [0-255]

        Returns:
            rgb_signal: (num_frames, 3) mean R, G, B values per frame
        """
        # Simple spatial averaging over the central face region
        # In production, use a skin segmentation mask
        h, w = frames.shape[1], frames.shape[2]
        # Focus on forehead + cheek region (central 60%)
        y1, y2 = int(h * 0.2), int(h * 0.8)
        x1, x2 = int(w * 0.2), int(w * 0.8)
        roi = frames[:, y1:y2, x1:x2, :]

        # Spatial average per channel
        rgb_signal = roi.mean(axis=(1, 2))  # (num_frames, 3)
        return rgb_signal

    def chrom_method(self, rgb_signal: np.ndarray) -> np.ndarray:
        """
        CHROM (Chrominance-based) rPPG extraction.

        Projects RGB signals onto chrominance axes to isolate
        the pulse component from motion and illumination noise.

        Args:
            rgb_signal: (num_frames, 3) mean R, G, B values

        Returns:
            bvp: (num_frames,) blood volume pulse signal
        """
        # Normalize RGB channels
        r = rgb_signal[:, 0]
        g = rgb_signal[:, 1]
        b = rgb_signal[:, 2]

        # Compute chrominance signals
        xs = 3.0 * r - 2.0 * g
        ys = 1.5 * r + g - 1.5 * b

        # Windowed processing for temporal stability
        bvp = np.zeros(len(r))
        step = self.window_size // 2  # 50% overlap

        for start in range(0, len(r) - self.window_size + 1, step):
            end = start + self.window_size
            x_win = xs[start:end]
            y_win = ys[start:end]

            # Remove mean
            x_win = x_win - x_win.mean()
            y_win = y_win - y_win.mean()

            # Standard deviation ratio for adaptive combination
            alpha = np.std(x_win) / (np.std(y_win) + 1e-8)

            # CHROM combination
            bvp_win = x_win - alpha * y_win

            # Hanning window to reduce spectral leakage
            bvp_win = bvp_win * np.hanning(len(bvp_win))

            bvp[start:end] += bvp_win

        return bvp

    def bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to isolate heartbeat frequencies."""
        nyquist = self.fps / 2.0
        low = self.bandpass_low / nyquist
        high = min(self.bandpass_high / nyquist, 0.99)

        b, a = scipy_signal.butter(4, [low, high], btype="band")
        filtered = scipy_signal.filtfilt(b, a, signal)
        return filtered

    def extract(self, frames: np.ndarray) -> dict:
        """
        Full rPPG extraction pipeline.

        Args:
            frames: (num_frames, H, W, 3) face-cropped RGB uint8

        Returns:
            dict with:
                - bvp: filtered blood volume pulse signal
                - heart_rate: estimated HR in bpm
                - signal_quality: SNR-based quality metric (0-1)
        """
        frames_float = frames.astype(np.float64)

        # Step 1: Extract spatially averaged RGB
        rgb_signal = self.extract_skin_signal(frames_float)

        # Step 2: CHROM method
        bvp_raw = self.chrom_method(rgb_signal)

        # Step 3: Bandpass filter
        bvp = self.bandpass_filter(bvp_raw)

        # Step 4: Estimate heart rate via peak frequency
        freqs, psd = scipy_signal.welch(bvp, fs=self.fps, nperseg=min(256, len(bvp)))
        mask = (freqs >= self.bandpass_low) & (freqs <= self.bandpass_high)
        if mask.any():
            peak_freq = freqs[mask][np.argmax(psd[mask])]
            heart_rate = peak_freq * 60.0
        else:
            heart_rate = 0.0

        # Step 5: Signal quality (SNR in pulse band vs. total)
        pulse_power = psd[mask].sum() if mask.any() else 0.0
        total_power = psd.sum() + 1e-8
        signal_quality = float(np.clip(pulse_power / total_power, 0.0, 1.0))

        return {
            "bvp": bvp,
            "heart_rate": heart_rate,
            "signal_quality": signal_quality,
        }


class RPPGFeatureEncoder(nn.Module):
    """
    Neural encoder that converts raw rPPG BVP signals into
    fixed-size feature vectors for the temporal stream.

    Takes the rPPG waveform and projects it into a dense embedding
    using 1D convolutions + temporal pooling.

    Args:
        signal_length: Expected BVP signal length (in samples)
        embed_dim: Output embedding dimension
    """

    def __init__(self, signal_length: int = 300, embed_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, bvp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bvp: (batch, signal_length) rPPG BVP signal
        Returns:
            embedding: (batch, embed_dim)
        """
        x = bvp.unsqueeze(1)  # (batch, 1, signal_length)
        x = self.encoder(x)  # (batch, 128, 1)
        x = x.squeeze(-1)  # (batch, 128)
        return self.fc(x)  # (batch, embed_dim)
