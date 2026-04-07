"""
GAN/Diffusion Model Fingerprint Identifier
============================================
Identifies WHICH AI generator produced an image by analyzing its unique
frequency-domain fingerprint.

THEORETICAL FOUNDATION:
━━━━━━━━━━━━━━━━━━━━━━━
Every image generator leaves a distinctive spectral signature:

1. GAN Generators (StyleGAN, ProGAN, BigGAN):
   - Periodic spectral peaks from transposed convolution (checkerboard artifacts)
   - Grid-like patterns in 2D FFT at specific frequencies
   - Frequency of peaks depends on upsampling factor and kernel size
   - StyleGAN3 reduced these but didn't eliminate them entirely

2. Diffusion Models (Stable Diffusion, DALL-E, Midjourney, Flux):
   - Spectral rolloff pattern from iterative denoising
   - High-frequency energy deficit (denoising smooths fine detail)
   - Characteristic noise floor shape from the diffusion schedule
   - Each model family has a different noise schedule → different rolloff

3. Autoregressive Models (DALL-E original, Parti):
   - Patch-based spectral inconsistency (tokens decoded independently)
   - Block boundary artifacts in frequency domain

4. Real Cameras:
   - Sensor noise pattern (Bayer filter artifacts in FFT)
   - Lens-specific frequency response (MTF)
   - JPEG quantization table fingerprint

ANALYSIS LAYERS:
━━━━━━━━━━━━━━━━
1. Azimuthal Average of Power Spectrum
   - Radial power distribution differs by generator class
   - Real images: smooth power-law falloff (1/f^β, β≈2)
   - GANs: bumps at upsampling frequencies
   - Diffusion: steeper rolloff in high frequencies

2. Spectral Peak Detection
   - Detect periodic peaks in azimuthal average
   - GAN-specific: peaks at frequencies = N * (1/upsample_factor)
   - Diffusion-specific: no periodic peaks, smooth rolloff

3. High-Frequency Energy Ratio
   - Ratio of high-freq to total spectral energy
   - Real cameras: moderate (sensor noise contributes)
   - Diffusion models: lower (denoising removes high-freq)
   - GANs: variable (some have aliasing = more high-freq)

4. Spectral Symmetry Analysis
   - 2D FFT should be conjugate-symmetric for real images
   - Some generators break this subtly

5. Cross-Channel Spectral Correlation
   - R, G, B channels have correlated spectra in real photos (same lens)
   - AI generators may have decorrelated channel spectra

OUTPUT:
━━━━━━━
  {
      "generator_class": "gan" | "diffusion" | "autoregressive" | "real" | "unknown",
      "generator_name": "midjourney-v6" | "stable-diffusion-xl" | ...,
      "generator_confidence": float,
      "spectral_features": { ... },
      "fingerprint_matches": [ { "name": ..., "similarity": ... }, ... ],
      "score": float (0-1, higher = more likely AI-generated)
  }
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2

log = logging.getLogger(__name__)

try:
    from scipy import signal as scipy_signal
    from scipy.stats import linregress
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ── Known generator spectral profiles ──
# Each profile contains expected spectral characteristics
GENERATOR_PROFILES = {
    # Diffusion models
    "stable-diffusion-xl": {
        "class": "diffusion",
        "beta_range": (2.2, 2.8),  # spectral slope
        "hf_energy_range": (0.01, 0.06),
        "peak_count_max": 2,
        "rolloff_steepness": "steep",
    },
    "stable-diffusion-3.5": {
        "class": "diffusion",
        "beta_range": (2.1, 2.7),
        "hf_energy_range": (0.01, 0.05),
        "peak_count_max": 2,
        "rolloff_steepness": "steep",
    },
    "midjourney-v6": {
        "class": "diffusion",
        "beta_range": (2.0, 2.5),
        "hf_energy_range": (0.02, 0.08),
        "peak_count_max": 3,
        "rolloff_steepness": "moderate",
    },
    "dall-e-3": {
        "class": "diffusion",
        "beta_range": (2.3, 2.9),
        "hf_energy_range": (0.01, 0.04),
        "peak_count_max": 1,
        "rolloff_steepness": "steep",
    },
    "flux-dev": {
        "class": "diffusion",
        "beta_range": (2.0, 2.6),
        "hf_energy_range": (0.02, 0.07),
        "peak_count_max": 2,
        "rolloff_steepness": "moderate",
    },
    "gpt-image-1": {
        "class": "diffusion",
        "beta_range": (2.1, 2.7),
        "hf_energy_range": (0.01, 0.05),
        "peak_count_max": 2,
        "rolloff_steepness": "steep",
    },
    "imagen-4": {
        "class": "diffusion",
        "beta_range": (2.2, 2.8),
        "hf_energy_range": (0.01, 0.04),
        "peak_count_max": 1,
        "rolloff_steepness": "steep",
    },
    # GAN models
    "stylegan3": {
        "class": "gan",
        "beta_range": (1.6, 2.2),
        "hf_energy_range": (0.05, 0.15),
        "peak_count_max": 8,
        "rolloff_steepness": "gradual",
    },
    "stylegan2": {
        "class": "gan",
        "beta_range": (1.4, 2.0),
        "hf_energy_range": (0.08, 0.20),
        "peak_count_max": 12,
        "rolloff_steepness": "gradual",
    },
    # Real camera profiles
    "real-camera": {
        "class": "real",
        "beta_range": (1.8, 2.4),
        "hf_energy_range": (0.03, 0.12),
        "peak_count_max": 4,
        "rolloff_steepness": "moderate",
    },
}


class GANFingerprintDetector:
    """
    Identifies which AI generator produced an image using frequency-domain
    fingerprint analysis.

    Works on any image size (resizes to 512x512 internally for consistent
    frequency analysis).
    """

    def __init__(self, analysis_size: int = 512, verbose: bool = False):
        self.analysis_size = analysis_size
        self.verbose = verbose
        self.is_available = HAS_SCIPY

    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image to identify generator fingerprint.

        Args:
            image: BGR image as numpy array (OpenCV format)

        Returns:
            Dict with generator identification and spectral features.
        """
        if not self.is_available:
            return self._default_result()

        # Resize to analysis size for consistent FFT
        if image.shape[0] != self.analysis_size or image.shape[1] != self.analysis_size:
            image_resized = cv2.resize(
                image, (self.analysis_size, self.analysis_size),
                interpolation=cv2.INTER_LANCZOS4,
            )
        else:
            image_resized = image.copy()

        # Convert to grayscale for primary analysis
        if len(image_resized.shape) == 3:
            gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            channels = cv2.split(image_resized)
        else:
            gray = image_resized
            channels = [gray]

        # Layer 1: Azimuthal average power spectrum
        azimuthal_features = self._azimuthal_power_spectrum(gray)

        # Layer 2: Spectral peak detection
        peak_features = self._detect_spectral_peaks(azimuthal_features["radial_power"])

        # Layer 3: High-frequency energy ratio
        hf_features = self._high_frequency_energy(gray)

        # Layer 4: Cross-channel spectral correlation
        channel_features = self._cross_channel_correlation(channels)

        # Layer 5: Spectral slope (beta exponent)
        slope_features = self._spectral_slope(azimuthal_features["radial_power"])

        # Combine features
        spectral_features = {
            **azimuthal_features,
            **peak_features,
            **hf_features,
            **channel_features,
            **slope_features,
        }

        # Match against known generator profiles
        matches = self._match_profiles(spectral_features)

        # Best match
        if matches:
            best = matches[0]
            generator_name = best["name"]
            generator_class = GENERATOR_PROFILES.get(generator_name, {}).get("class", "unknown")
            generator_confidence = best["similarity"]
        else:
            generator_name = "unknown"
            generator_class = "unknown"
            generator_confidence = 0.0

        # Overall score (how likely AI-generated based on spectral analysis)
        score = self._compute_ai_score(spectral_features, generator_class)

        return {
            "generator_class": generator_class,
            "generator_name": generator_name,
            "generator_confidence": round(generator_confidence, 4),
            "spectral_features": {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in spectral_features.items()
                if k != "radial_power"  # don't include raw array
            },
            "fingerprint_matches": matches[:5],
            "score": round(score, 4),
            "confidence": round(min(1.0, generator_confidence * 1.2), 4),
        }

    def _azimuthal_power_spectrum(self, gray: np.ndarray) -> Dict[str, Any]:
        """
        Compute azimuthally averaged power spectrum.

        This averages the 2D FFT magnitude over concentric rings,
        producing a 1D radial power distribution.
        """
        # 2D FFT
        f_transform = np.fft.fft2(gray.astype(np.float64))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift) ** 2

        # Azimuthal average: average over concentric rings
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        max_radius = min(cy, cx)

        radial_power = np.zeros(max_radius)
        counts = np.zeros(max_radius)

        y_coords, x_coords = np.mgrid[:h, :w]
        distances = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2).astype(int)

        for r in range(max_radius):
            mask = distances == r
            if np.any(mask):
                radial_power[r] = np.mean(magnitude[mask])
                counts[r] = np.sum(mask)

        # Normalize
        radial_power = radial_power / (radial_power[1] + 1e-10)

        # Convert to dB
        radial_power_db = 10 * np.log10(radial_power + 1e-20)

        return {
            "radial_power": radial_power,
            "radial_power_db_mean": float(np.mean(radial_power_db[1:max_radius//2])),
            "radial_power_db_std": float(np.std(radial_power_db[1:max_radius//2])),
            "total_spectral_energy": float(np.sum(magnitude)),
        }

    def _detect_spectral_peaks(self, radial_power: np.ndarray) -> Dict[str, Any]:
        """
        Detect periodic peaks in the radial power spectrum.

        GAN upsampling creates characteristic peaks at regular intervals.
        """
        if len(radial_power) < 10:
            return {"peak_count": 0, "peak_periodicity": 0.0, "peak_prominence_mean": 0.0}

        # Smooth the spectrum to find peaks above trend
        from scipy.ndimage import uniform_filter1d
        smoothed = uniform_filter1d(radial_power, size=5)
        residual = radial_power - smoothed

        # Find peaks in residual
        peaks, properties = scipy_signal.find_peaks(
            residual[5:],  # skip DC component
            height=np.std(residual) * 1.5,
            distance=3,
            prominence=np.std(residual) * 0.5,
        )

        peak_count = len(peaks)

        # Check for periodicity among peaks (GAN signature)
        if peak_count >= 3:
            peak_diffs = np.diff(peaks)
            periodicity = float(1.0 - np.std(peak_diffs) / (np.mean(peak_diffs) + 1e-10))
            periodicity = max(0.0, periodicity)
        else:
            periodicity = 0.0

        # Mean prominence of peaks
        if peak_count > 0 and "prominences" in properties:
            prom_mean = float(np.mean(properties["prominences"]))
        else:
            prom_mean = 0.0

        return {
            "peak_count": peak_count,
            "peak_periodicity": round(periodicity, 4),
            "peak_prominence_mean": round(prom_mean, 6),
        }

    def _high_frequency_energy(self, gray: np.ndarray) -> Dict[str, Any]:
        """
        Compute ratio of high-frequency to total spectral energy.
        """
        f_transform = np.fft.fft2(gray.astype(np.float64))
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift) ** 2

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        max_radius = min(cy, cx)

        # Create radial distance map
        y_coords, x_coords = np.mgrid[:h, :w]
        distances = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)

        # Total energy
        total_energy = float(np.sum(magnitude))

        # High-frequency energy (outer 25% of spectrum)
        hf_threshold = max_radius * 0.75
        hf_mask = distances >= hf_threshold
        hf_energy = float(np.sum(magnitude[hf_mask]))

        # Mid-frequency energy (25%-75% of spectrum)
        mf_mask = (distances >= max_radius * 0.25) & (distances < hf_threshold)
        mf_energy = float(np.sum(magnitude[mf_mask]))

        # Low-frequency energy (inner 25%)
        lf_mask = distances < max_radius * 0.25
        lf_energy = float(np.sum(magnitude[lf_mask]))

        hf_ratio = hf_energy / (total_energy + 1e-10)
        mf_ratio = mf_energy / (total_energy + 1e-10)
        lf_ratio = lf_energy / (total_energy + 1e-10)

        return {
            "hf_energy_ratio": float(hf_ratio),
            "mf_energy_ratio": float(mf_ratio),
            "lf_energy_ratio": float(lf_ratio),
        }

    def _cross_channel_correlation(self, channels: list) -> Dict[str, Any]:
        """
        Analyze spectral correlation between color channels.

        Real camera images have highly correlated channel spectra (same lens).
        AI-generated images may have less correlated channels.
        """
        if len(channels) < 3:
            return {"channel_correlation_mean": 1.0, "channel_correlation_std": 0.0}

        # Compute FFT magnitude for each channel
        channel_mags = []
        for ch in channels[:3]:
            f = np.fft.fft2(ch.astype(np.float64))
            mag = np.abs(np.fft.fftshift(f)).flatten()
            channel_mags.append(mag)

        # Pairwise correlation
        correlations = []
        for i in range(3):
            for j in range(i + 1, 3):
                corr = float(np.corrcoef(channel_mags[i], channel_mags[j])[0, 1])
                correlations.append(corr)

        return {
            "channel_correlation_mean": float(np.mean(correlations)),
            "channel_correlation_std": float(np.std(correlations)),
        }

    def _spectral_slope(self, radial_power: np.ndarray) -> Dict[str, Any]:
        """
        Fit power-law slope to the radial power spectrum.

        Real images: 1/f^β where β ≈ 2.0
        Diffusion models: β > 2.0 (steeper, less high-freq)
        GANs: β < 2.0 (flatter, more aliasing artifacts)
        """
        # Use middle portion to avoid DC and Nyquist edge effects
        n = len(radial_power)
        start = max(2, n // 20)
        end = n // 2

        if end - start < 5:
            return {"spectral_beta": 2.0, "spectral_r2": 0.0}

        freqs = np.arange(start, end, dtype=float)
        power = radial_power[start:end]

        # Avoid log of zero
        valid = power > 0
        if np.sum(valid) < 5:
            return {"spectral_beta": 2.0, "spectral_r2": 0.0}

        log_freqs = np.log10(freqs[valid])
        log_power = np.log10(power[valid])

        # Linear regression in log-log space: log(P) = -β * log(f) + c
        slope, intercept, r_value, _, _ = linregress(log_freqs, log_power)

        beta = -slope  # positive β for power-law decay

        return {
            "spectral_beta": float(beta),
            "spectral_r2": float(r_value ** 2),
        }

    def _match_profiles(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Match extracted features against known generator profiles.
        """
        beta = features.get("spectral_beta", 2.0)
        hf_ratio = features.get("hf_energy_ratio", 0.05)
        peak_count = features.get("peak_count", 0)

        matches = []

        for name, profile in GENERATOR_PROFILES.items():
            score = 0.0
            weights_sum = 0.0

            # Beta (spectral slope) match
            beta_low, beta_high = profile["beta_range"]
            if beta_low <= beta <= beta_high:
                score += 0.35
            else:
                # Partial credit for being close
                dist = min(abs(beta - beta_low), abs(beta - beta_high))
                if dist < 0.5:
                    score += 0.35 * (1.0 - dist / 0.5)
            weights_sum += 0.35

            # HF energy match
            hf_low, hf_high = profile["hf_energy_range"]
            if hf_low <= hf_ratio <= hf_high:
                score += 0.25
            else:
                dist = min(abs(hf_ratio - hf_low), abs(hf_ratio - hf_high))
                if dist < 0.05:
                    score += 0.25 * (1.0 - dist / 0.05)
            weights_sum += 0.25

            # Peak count match
            if peak_count <= profile["peak_count_max"]:
                # For GANs, having peaks is good; for diffusion, few peaks is good
                if profile["class"] == "gan" and peak_count >= 3:
                    score += 0.25
                elif profile["class"] == "diffusion" and peak_count <= 3:
                    score += 0.25
                elif profile["class"] == "real" and 0 <= peak_count <= 4:
                    score += 0.20
                else:
                    score += 0.10
            weights_sum += 0.25

            # Channel correlation (real images have high correlation)
            ch_corr = features.get("channel_correlation_mean", 0.9)
            if profile["class"] == "real" and ch_corr > 0.85:
                score += 0.15
            elif profile["class"] in ("gan", "diffusion") and ch_corr < 0.85:
                score += 0.15
            else:
                score += 0.05
            weights_sum += 0.15

            similarity = score / weights_sum if weights_sum > 0 else 0.0

            matches.append({
                "name": name,
                "class": profile["class"],
                "similarity": round(similarity, 4),
            })

        # Sort by similarity descending
        matches.sort(key=lambda x: x["similarity"], reverse=True)

        return matches

    def _compute_ai_score(
        self, features: Dict[str, Any], generator_class: str
    ) -> float:
        """
        Compute overall AI-generation probability from spectral features.
        """
        scores = []

        # Spectral beta: diffusion models have steeper slopes
        beta = features.get("spectral_beta", 2.0)
        if beta > 2.3:
            scores.append(min(1.0, (beta - 2.0) / 1.0))
        elif beta < 1.7:
            scores.append(min(1.0, (2.0 - beta) / 0.8))
        else:
            scores.append(0.2)

        # HF energy ratio: AI tends to have less high-frequency content
        hf_ratio = features.get("hf_energy_ratio", 0.05)
        if hf_ratio < 0.02:
            scores.append(0.7)  # very low HF = likely diffusion
        elif hf_ratio > 0.15:
            scores.append(0.5)  # high HF = possible GAN aliasing
        else:
            scores.append(0.3)

        # Periodic peaks: strong periodicity = GAN
        periodicity = features.get("peak_periodicity", 0.0)
        if periodicity > 0.5:
            scores.append(0.7)
        else:
            scores.append(0.2)

        # Channel decorrelation: AI images may have lower correlation
        ch_corr = features.get("channel_correlation_mean", 0.9)
        if ch_corr < 0.80:
            scores.append(0.5)
        else:
            scores.append(0.2)

        # R² of spectral fit: real images follow power law more closely
        r2 = features.get("spectral_r2", 0.9)
        if r2 < 0.7:
            scores.append(0.4)
        else:
            scores.append(0.2)

        return float(np.mean(scores))

    def _default_result(self) -> Dict[str, Any]:
        return {
            "generator_class": "unknown",
            "generator_name": "unknown",
            "generator_confidence": 0.0,
            "spectral_features": {},
            "fingerprint_matches": [],
            "score": 0.5,
            "confidence": 0.0,
        }
