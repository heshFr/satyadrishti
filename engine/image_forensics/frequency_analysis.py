"""
Image Forensics — Frequency Analysis
====================================
Detects GAN fingerprints and AI generation artifacts by analyzing
the frequency domain (DCT and Power Spectrum) of the image.
AI models (especially GANs and early Diffusion models) often
leave high-frequency artifacts that are invisible to the eye
but show up as unnatural peaks in the power spectrum.
"""

import numpy as np
import cv2
from typing import Dict, Any, Tuple


class FrequencyAnalyzer:
    def __init__(self, target_size: int = 256):
        self.target_size = target_size

    def compute_power_spectrum(self, image: np.ndarray) -> np.ndarray:
        """
        Computes the 2D power spectrum of a grayscale image using FFT.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Resize for consistent FFT resolution
        gray = cv2.resize(gray, (self.target_size, self.target_size))

        # Perform 2D FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)

        # Calculate power spectrum (magnitude squared)
        magnitude_spectrum = np.abs(f_shift) ** 2

        # Log transform for better numerical stability and visualization
        # Adding 1e-8 to avoid log(0)
        power_spectrum = np.log10(magnitude_spectrum + 1e-8)

        return power_spectrum

    def get_azimuthal_average(self, power_spectrum: np.ndarray) -> np.ndarray:
        """
        Computes the 1D radially averaged power spectrum.
        Real images follow a smooth 1/f^2 decay pattern.
        AI images often deviate with anomalous peaks.
        """
        h, w = power_spectrum.shape
        center_y, center_x = h // 2, w // 2

        # Create coordinate grids
        y, x = np.indices((h, w))
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        r = r.astype(np.int32)

        # Average over concentric circles
        max_radius = min(center_x, center_y)
        radial_profile = np.zeros(max_radius)

        for radius in range(max_radius):
            mask = (r == radius)
            if np.any(mask):
                radial_profile[radius] = np.mean(power_spectrum[mask])

        return radial_profile

    def detect_artifacts(self, image: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Analyzes the image for frequency-domain anomalies typical of AI generation.
        Returns an anomaly score (0.0 to 1.0) and detailed metrics.
        """
        power_spectrum = self.compute_power_spectrum(image)
        radial_profile = self.get_azimuthal_average(power_spectrum)

        # 1. High-frequency Drop-off Ratio
        # Real images have energy mostly in low frequencies.
        # AI images sometimes have unnatural energy in the high-frequency band.
        total_energy = np.sum(radial_profile)
        hf_band = radial_profile[int(len(radial_profile) * 0.75):]
        hf_energy_ratio = np.sum(hf_band) / (total_energy + 1e-8)

        # 2. Spectral variance (roughness of the spectrum)
        # AI images may have grid-like artifacts causing periodic peaks in spectrum
        spectral_variance = np.var(power_spectrum)

        # Calculate a combined anomaly score (heuristic thresholds)
        score = 0.0

        if hf_energy_ratio > 0.15:
            score += 0.4
        elif hf_energy_ratio > 0.10:
            score += 0.2

        if spectral_variance > 1.5:
            score += 0.4
        elif spectral_variance > 1.2:
            score += 0.2

        # Cap score at 1.0
        score = min(1.0, score)

        details = {
            "hf_energy_ratio": float(hf_energy_ratio),
            "spectral_variance": float(spectral_variance),
            "anomaly_score": float(score)
        }

        return score, details
