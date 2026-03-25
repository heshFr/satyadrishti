"""
Image Forensics — Pixel Statistics Analysis
=============================================
Detects AI-generated images by analyzing pixel-level statistical properties
that differ between real camera captures and AI-generated images.

Modern diffusion models (Midjourney, DALL-E 3, SDXL, Flux) generate images
in floating-point space and quantize to uint8. This process leaves subtle
but detectable statistical fingerprints:

1. Color histogram smoothness — AI histograms are smoother (no gaps/spikes)
2. Saturation patterns — AI rarely produces true sensor saturation/clipping
3. Color channel correlation — AI has different cross-channel statistics
4. Bit-plane regularity — LSBs of AI images differ from sensor noise
5. Local variance distribution — AI has unnaturally consistent local contrast
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Any


class PixelStatisticsAnalyzer:
    """Analyzes pixel-level statistics to detect AI-generated images."""

    def __init__(self, target_size: int = 512):
        self.target_size = target_size

    def analyze(self, image: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Run pixel statistics analysis on a BGR image.
        Returns (anomaly_score, details).
        """
        image = cv2.resize(image, (self.target_size, self.target_size))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(image)

        score = 0.0
        details = {}

        # --- 1. Histogram smoothness ---
        # Real camera images have irregular histograms with gaps/spikes
        # AI images have smoother, more continuous histograms
        hist_smoothness = self._histogram_smoothness(gray)
        details["histogram_smoothness"] = round(hist_smoothness, 4)
        if hist_smoothness > 0.92:
            score += 0.25
        elif hist_smoothness > 0.85:
            score += 0.12

        # --- 2. Saturation/clipping analysis ---
        # Real photos have pixels at 0 and 255 (sensor clipping)
        # AI images rarely have true black/white pixels
        clip_ratio = self._clipping_analysis(gray)
        details["clipping_ratio"] = round(clip_ratio, 5)
        if clip_ratio < 0.001:
            score += 0.20
        elif clip_ratio < 0.005:
            score += 0.10

        # --- 3. Color channel correlation ---
        # Real sensor: R/G/B noise is partially correlated (Bayer demosaicing)
        # AI: channels are more independently generated
        rg_corr, rb_corr, gb_corr = self._channel_correlation(r, g, b)
        details["rg_correlation"] = round(rg_corr, 4)
        details["rb_correlation"] = round(rb_corr, 4)
        details["gb_correlation"] = round(gb_corr, 4)

        # AI images tend to have very high cross-channel correlation
        avg_corr = (rg_corr + rb_corr + gb_corr) / 3
        details["avg_channel_correlation"] = round(avg_corr, 4)
        if avg_corr > 0.97:
            score += 0.15
        elif avg_corr > 0.95:
            score += 0.07

        # --- 4. LSB (Least Significant Bit) analysis ---
        # Real sensors: LSBs are noisy (random)
        # AI quantization: LSBs can show patterns
        lsb_uniformity = self._lsb_analysis(gray)
        details["lsb_uniformity"] = round(lsb_uniformity, 4)
        if lsb_uniformity > 0.85:
            score += 0.15
        elif lsb_uniformity > 0.75:
            score += 0.07

        # --- 5. Local variance distribution ---
        # AI images have unnaturally consistent local contrast across the image
        # Real photos have high variance of local variance (some areas sharp, some smooth)
        variance_kurtosis = self._local_variance_analysis(gray)
        details["variance_kurtosis"] = round(variance_kurtosis, 3)
        # Low kurtosis = uniform local variance = suspicious
        if variance_kurtosis < 3.0:
            score += 0.15
        elif variance_kurtosis < 6.0:
            score += 0.07

        # --- 6. Color distribution unnaturalness ---
        # AI images often have subtly different color distributions
        # Check for unnaturally balanced saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        sat_cv = float(np.std(sat) / (np.mean(sat) + 1e-8))
        details["saturation_cv"] = round(sat_cv, 4)
        # Low saturation variance = unnaturally uniform colors
        if sat_cv < 0.4:
            score += 0.10
        elif sat_cv < 0.6:
            score += 0.05

        score = min(1.0, score)
        details["anomaly_score"] = round(score, 3)

        return score, details

    def _histogram_smoothness(self, gray: np.ndarray) -> float:
        """
        Compute histogram smoothness. Smooth histograms indicate AI generation.
        Real images have more irregular histograms with gaps and peaks.
        """
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-8)

        # Count zero bins (gaps in histogram) — real images have more gaps
        zero_bins = np.sum(hist == 0) / 256.0

        # Measure roughness: diff between adjacent bins
        diffs = np.abs(np.diff(hist))
        roughness = np.mean(diffs)

        # Smoothness = 1 - roughness_normalized
        max_roughness = 0.01  # typical max for normalized hist
        smoothness = 1.0 - min(1.0, roughness / max_roughness)

        # Penalize if very few zero bins (AI fills all bins)
        if zero_bins < 0.05:
            smoothness = max(smoothness, 0.9)

        return float(smoothness)

    def _clipping_analysis(self, gray: np.ndarray) -> float:
        """
        Check for sensor clipping (pixels at 0 or 255).
        Real photos commonly have clipped pixels, AI rarely does.
        """
        total = gray.size
        clipped = np.sum((gray == 0) | (gray == 255))
        return float(clipped / total)

    def _channel_correlation(self, r, g, b) -> Tuple[float, float, float]:
        """
        Compute Pearson correlation between color channels.
        """
        r_flat = r.flatten().astype(np.float32)
        g_flat = g.flatten().astype(np.float32)
        b_flat = b.flatten().astype(np.float32)

        rg = float(np.corrcoef(r_flat, g_flat)[0, 1])
        rb = float(np.corrcoef(r_flat, b_flat)[0, 1])
        gb = float(np.corrcoef(g_flat, b_flat)[0, 1])

        return rg, rb, gb

    def _lsb_analysis(self, gray: np.ndarray) -> float:
        """
        Analyze the Least Significant Bit plane.
        AI-generated images often have less random LSBs.
        """
        lsb = gray & 1  # Extract LSB plane

        # In a natural image, LSB should be close to 50/50 distribution
        ones_ratio = np.mean(lsb)

        # Check spatial autocorrelation of LSB
        # AI images have more structured LSB patterns
        lsb_float = lsb.astype(np.float32)
        h_shift = np.abs(lsb_float[:, 1:] - lsb_float[:, :-1])
        v_shift = np.abs(lsb_float[1:, :] - lsb_float[:-1, :])

        # Transition rate: in random noise, ~50% of adjacent pixels differ
        h_transition = float(np.mean(h_shift))
        v_transition = float(np.mean(v_shift))

        # If transition rate is far from 0.5, LSBs are structured (suspicious)
        avg_transition = (h_transition + v_transition) / 2
        uniformity = 1.0 - abs(avg_transition - 0.5) * 2

        return uniformity

    def _local_variance_analysis(self, gray: np.ndarray) -> float:
        """
        Analyze the distribution of local variance (contrast) across the image.
        Real images have high kurtosis (some areas very smooth, others very textured).
        AI images have more uniform local variance (lower kurtosis).
        """
        gray_f = gray.astype(np.float32)
        # Compute local variance using a sliding window
        kernel_size = 7
        mean_local = cv2.blur(gray_f, (kernel_size, kernel_size))
        mean_sq = cv2.blur(gray_f ** 2, (kernel_size, kernel_size))
        local_var = mean_sq - mean_local ** 2
        local_var = np.maximum(local_var, 0)  # numerical stability

        # Compute kurtosis of the local variance distribution
        var_mean = np.mean(local_var)
        var_std = np.std(local_var)
        if var_std < 1e-8:
            return 0.0

        centered = local_var - var_mean
        kurtosis = float(np.mean(centered ** 4) / (var_std ** 4))

        return kurtosis
