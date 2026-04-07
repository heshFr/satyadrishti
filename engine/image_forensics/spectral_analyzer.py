"""
Spectral Decay & Upsampling Artifact Analyzer
==============================================
Two independent forensic signals in one module:

1. **Spectral Decay (1/f^beta)**: Real photographs follow a natural power-law
   decay in their frequency spectrum (1/f^beta where beta ~ 1.5-2.5).
   AI-generated images deviate from this law — diffusion models produce
   flatter high-frequency spectra, and GANs show periodic peaks.

2. **Upsampling Artifacts**: Most diffusion models operate in a latent space
   at lower resolution and upsample to output resolution. This leaves
   periodic artifacts detectable via autocorrelation in the frequency domain.

These are among the most reliable generator-agnostic forensic signals.
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class SpectralDecayAnalyzer:
    """Analyzes frequency domain power-law decay and upsampling artifacts."""

    # Real photograph beta exponent typically falls in this range
    NATURAL_BETA_RANGE = (1.4, 2.8)

    # AI-generated beta is often lower (flatter spectrum) or shows anomalous regions
    AI_BETA_RANGE = (0.3, 1.3)

    def analyze(self, image: np.ndarray) -> dict:
        """
        Run spectral decay + upsampling artifact analysis.

        Args:
            image: BGR image (numpy array).

        Returns:
            {
                "score": float (0=real, 1=AI),
                "confidence": float (0-1),
                "beta_exponent": float,
                "beta_deviation": float,
                "upsampling_score": float,
                "spectral_flatness": float,
                "anomalies": list[str],
            }
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        except Exception:
            gray = image[:, :, 0].astype(np.float64) if image.ndim == 3 else image.astype(np.float64)

        # Resize to standard size for consistent analysis
        target_size = 512
        gray = cv2.resize(gray, (target_size, target_size))

        anomalies = []

        # 1. Compute 2D FFT and radial power spectrum
        beta, beta_deviation, radial_power, freqs = self._compute_beta_exponent(gray)

        # 2. Spectral flatness (Wiener entropy)
        spectral_flatness = self._compute_spectral_flatness(radial_power)

        # 3. Upsampling artifact detection
        upsampling_score, upsampling_details = self._detect_upsampling_artifacts(gray)

        # 4. High-frequency energy ratio
        hf_ratio = self._high_frequency_ratio(radial_power, freqs)

        # 5. Spectral periodicity detection (GAN grid artifacts)
        periodicity_score = self._detect_spectral_periodicity(gray)

        # --- Score computation ---
        scores = []
        weights = []

        # Beta exponent analysis (highest weight — most reliable signal)
        beta_score = self._beta_to_score(beta, beta_deviation)
        scores.append(beta_score)
        weights.append(0.30)
        if beta_score > 0.6:
            anomalies.append(f"Spectral decay beta={beta:.2f} outside natural range")

        # Spectral flatness
        flatness_score = self._flatness_to_score(spectral_flatness)
        scores.append(flatness_score)
        weights.append(0.15)
        if flatness_score > 0.6:
            anomalies.append(f"High spectral flatness={spectral_flatness:.3f}")

        # Upsampling artifacts
        scores.append(upsampling_score)
        weights.append(0.25)
        if upsampling_score > 0.5:
            anomalies.append("Latent-space upsampling artifacts detected")

        # High-frequency energy ratio
        hf_score = self._hf_ratio_to_score(hf_ratio)
        scores.append(hf_score)
        weights.append(0.15)
        if hf_score > 0.6:
            anomalies.append(f"Anomalous high-frequency energy ratio={hf_ratio:.3f}")

        # Spectral periodicity (GAN grid)
        scores.append(periodicity_score)
        weights.append(0.15)
        if periodicity_score > 0.5:
            anomalies.append("Periodic spectral patterns (GAN grid artifacts)")

        # Weighted combination
        scores = np.array(scores)
        weights = np.array(weights)
        final_score = float(np.dot(scores, weights))

        # Confidence based on signal strength
        max_signal = float(np.max(scores))
        min_signal = float(np.min(scores))
        signal_agreement = 1.0 - float(np.std(scores))
        confidence = 0.5 * signal_agreement + 0.3 * max_signal + 0.2 * (1.0 - abs(final_score - 0.5) * 2)

        return {
            "score": float(np.clip(final_score, 0.0, 1.0)),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "beta_exponent": float(beta),
            "beta_deviation": float(beta_deviation),
            "upsampling_score": float(upsampling_score),
            "spectral_flatness": float(spectral_flatness),
            "hf_energy_ratio": float(hf_ratio),
            "periodicity_score": float(periodicity_score),
            "anomalies": anomalies,
        }

    def _compute_beta_exponent(self, gray: np.ndarray):
        """
        Compute the power-law exponent (beta) from the radial power spectrum.
        Real images: beta ~ 1.5-2.5
        AI images: beta often < 1.4 or shows curvature
        """
        h, w = gray.shape

        # Apply Hann window to reduce spectral leakage
        hann_y = np.hanning(h)
        hann_x = np.hanning(w)
        window = np.outer(hann_y, hann_x)
        windowed = gray * window

        # 2D FFT
        fft = np.fft.fft2(windowed)
        fft_shift = np.fft.fftshift(fft)
        power_spectrum = np.abs(fft_shift) ** 2

        # Compute radial average
        cy, cx = h // 2, w // 2
        max_radius = min(cy, cx)

        radial_power = np.zeros(max_radius)
        counts = np.zeros(max_radius)

        y_coords, x_coords = np.ogrid[:h, :w]
        radii = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2).astype(int)

        for r in range(1, max_radius):
            mask = radii == r
            if mask.any():
                radial_power[r] = np.mean(power_spectrum[mask])
                counts[r] = mask.sum()

        # Fit power law in log-log space (skip DC and very high frequencies)
        valid = (radial_power > 0) & (np.arange(max_radius) > 2) & (np.arange(max_radius) < max_radius * 0.8)
        freqs = np.arange(max_radius)[valid]
        power = radial_power[valid]

        if len(freqs) < 10:
            return 2.0, 0.0, radial_power, np.arange(max_radius)

        log_freqs = np.log10(freqs.astype(float))
        log_power = np.log10(power + 1e-10)

        # Linear regression in log-log space
        coeffs = np.polyfit(log_freqs, log_power, 1)
        beta = -coeffs[0]  # Negative slope = beta exponent

        # Measure deviation from linear fit (curvature = AI artifact)
        fitted = np.polyval(coeffs, log_freqs)
        residuals = log_power - fitted
        beta_deviation = float(np.std(residuals))

        return float(beta), beta_deviation, radial_power, np.arange(max_radius)

    def _compute_spectral_flatness(self, radial_power: np.ndarray) -> float:
        """
        Wiener entropy / spectral flatness.
        Real images: lower flatness (strong low-frequency dominance).
        AI images: higher flatness (more uniform power distribution).
        """
        valid = radial_power[radial_power > 0]
        if len(valid) < 2:
            return 0.5

        # Geometric mean / arithmetic mean
        log_mean = np.mean(np.log(valid + 1e-20))
        geometric_mean = np.exp(log_mean)
        arithmetic_mean = np.mean(valid)

        if arithmetic_mean < 1e-20:
            return 0.5

        flatness = geometric_mean / arithmetic_mean
        return float(np.clip(flatness, 0.0, 1.0))

    def _detect_upsampling_artifacts(self, gray: np.ndarray) -> tuple:
        """
        Detect periodic patterns from latent-space upsampling.

        Diffusion models typically generate at lower resolution in latent space
        then upsample (2x, 4x, 8x). This creates periodic peaks in the
        autocorrelation of the high-frequency residual.
        """
        h, w = gray.shape

        # Extract high-frequency residual
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        residual = gray - blurred

        # Compute autocorrelation of residual
        fft_res = np.fft.fft2(residual)
        power = np.abs(fft_res) ** 2
        autocorr = np.fft.ifft2(power).real
        autocorr = autocorr / (autocorr[0, 0] + 1e-10)  # Normalize

        # Look for peaks at regular intervals (upsampling factors: 2, 4, 8)
        peak_strengths = []
        for factor in [2, 4, 8]:
            step = h // factor
            if step < 2 or step >= h // 2:
                continue

            # Check for peaks at multiples of the factor
            peaks = []
            for mult in range(1, min(4, factor)):
                idx = step * mult
                if idx < h:
                    # Peak value relative to neighbors
                    neighborhood = autocorr[max(0, idx - 2):idx + 3, :3].mean()
                    local_mean = autocorr[max(0, idx - 10):idx + 10, :3].mean()
                    if local_mean > 1e-10:
                        peak_ratio = neighborhood / (local_mean + 1e-10)
                        peaks.append(peak_ratio)

            if peaks:
                peak_strengths.append(max(peaks))

        if not peak_strengths:
            return 0.0, {"upsampling_factor": None}

        max_peak = max(peak_strengths)
        best_factor = [2, 4, 8][peak_strengths.index(max_peak)] if len(peak_strengths) <= 3 else None

        # Score: strong periodic peaks indicate upsampling
        # Real images: max_peak typically < 1.5
        # AI upsampled: max_peak typically > 2.0
        if max_peak > 3.0:
            score = 0.9
        elif max_peak > 2.0:
            score = 0.7
        elif max_peak > 1.5:
            score = 0.4
        else:
            score = 0.1

        return float(score), {"upsampling_factor": best_factor, "peak_strength": float(max_peak)}

    def _high_frequency_ratio(self, radial_power: np.ndarray, freqs: np.ndarray) -> float:
        """
        Ratio of high-frequency to total power.
        AI images often have different high-frequency energy than real images.
        """
        total_power = np.sum(radial_power[1:])
        if total_power < 1e-10:
            return 0.5

        n = len(radial_power)
        mid = n // 2
        hf_power = np.sum(radial_power[mid:])

        return float(hf_power / total_power)

    def _detect_spectral_periodicity(self, gray: np.ndarray) -> float:
        """
        Detect periodic patterns in the power spectrum (GAN grid artifacts).
        GANs (especially StyleGAN, ProGAN) leave periodic peaks.
        """
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log(np.abs(fft_shift) + 1)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Exclude DC component region
        magnitude[cy - 3:cy + 4, cx - 3:cx + 4] = 0

        # Find peaks that are significantly above local mean
        # Use morphological operations to find local maxima
        kernel_size = 11
        local_mean = cv2.blur(magnitude, (kernel_size, kernel_size))
        local_std = np.sqrt(cv2.blur((magnitude - local_mean) ** 2, (kernel_size, kernel_size)))

        # Z-score threshold for peaks
        z_threshold = 4.0
        peaks = magnitude > (local_mean + z_threshold * (local_std + 1e-10))

        # Count significant peaks (excluding near-DC)
        mask = np.zeros_like(peaks, dtype=bool)
        r_min = max(cy, cx) // 8  # Minimum distance from center
        y_coords, x_coords = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)
        mask = dist_from_center > r_min

        num_peaks = np.sum(peaks & mask)
        total_pixels = np.sum(mask)

        if total_pixels == 0:
            return 0.0

        # Real images: very few spectral peaks (< 0.01% of pixels)
        # GANs: many periodic peaks (> 0.05% of pixels)
        peak_density = num_peaks / total_pixels

        if peak_density > 0.005:
            return 0.9
        elif peak_density > 0.002:
            return 0.6
        elif peak_density > 0.001:
            return 0.3
        else:
            return 0.1

    # --- Score mapping functions ---

    def _beta_to_score(self, beta: float, deviation: float) -> float:
        """Map beta exponent to AI probability score."""
        lo, hi = self.NATURAL_BETA_RANGE

        # Distance from natural range
        if lo <= beta <= hi:
            # Within natural range — lower score
            # But check deviation (curvature)
            base = 0.1 + 0.2 * min(1.0, deviation / 0.5)
        elif beta < lo:
            # Flatter than natural (common for diffusion models)
            distance = (lo - beta) / lo
            base = 0.5 + 0.4 * min(1.0, distance)
        else:
            # Steeper than natural (less common but possible for some generators)
            distance = (beta - hi) / hi
            base = 0.3 + 0.3 * min(1.0, distance)

        # High deviation from linear fit adds to score
        deviation_bonus = min(0.2, deviation * 0.4)
        return float(np.clip(base + deviation_bonus, 0.0, 1.0))

    def _flatness_to_score(self, flatness: float) -> float:
        """Map spectral flatness to AI probability score."""
        # Real images: flatness ~ 0.001-0.05
        # AI images: flatness ~ 0.05-0.3
        if flatness > 0.15:
            return 0.9
        elif flatness > 0.08:
            return 0.6
        elif flatness > 0.04:
            return 0.3
        else:
            return 0.1

    def _hf_ratio_to_score(self, hf_ratio: float) -> float:
        """Map high-frequency energy ratio to AI probability score."""
        # Real images: HF ratio ~ 0.01-0.08
        # AI images: variable, but often > 0.10 (diffusion) or < 0.005 (over-smooth GAN)
        if hf_ratio > 0.15:
            return 0.7
        elif hf_ratio < 0.003:
            return 0.6  # Suspiciously smooth
        elif 0.01 <= hf_ratio <= 0.08:
            return 0.15  # Natural range
        else:
            return 0.35
