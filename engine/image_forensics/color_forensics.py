"""
Color Space Anomaly Detector
=============================
AI-generated images have characteristic color space anomalies that differ
from natural photographs:

1. **LAB Color Distribution**: AI images often have non-natural distributions
   in the LAB color space — oversaturated a/b channels, unusual lightness
   histograms, and color gamut that doesn't match camera sensor output.

2. **HSV Hue Distribution**: Natural images have multi-modal hue distributions
   driven by scene content. AI images often have smoother, more uniform
   hue distributions or specific modes from training data bias.

3. **Color Coherence**: AI images may have color bleeding between regions,
   inconsistent white balance across the image, or color banding in gradients.

4. **Channel Correlation**: In real photographs, R/G/B channels are highly
   correlated (driven by physics). AI images may have lower inter-channel
   correlation patterns.

These signals are generator-agnostic and work across Midjourney, DALL-E,
Flux, Stable Diffusion, and other generators.
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class ColorForensicsAnalyzer:
    """Detects AI generation artifacts in color space distributions."""

    def analyze(self, image: np.ndarray) -> dict:
        """
        Run color space forensic analysis.

        Args:
            image: BGR image (numpy array).

        Returns:
            {
                "score": float (0=real, 1=AI),
                "confidence": float,
                "lab_anomaly": float,
                "hsv_anomaly": float,
                "channel_correlation": float,
                "color_coherence": float,
                "anomalies": list[str],
            }
        """
        anomalies = []

        # Resize for consistent analysis
        small = cv2.resize(image, (384, 384))

        # 1. LAB color distribution analysis
        lab_score, lab_details = self._analyze_lab_distribution(small)

        # 2. HSV hue distribution analysis
        hsv_score, hsv_details = self._analyze_hsv_distribution(small)

        # 3. Inter-channel correlation
        corr_score, corr_details = self._analyze_channel_correlation(small)

        # 4. Color coherence (gradient banding + white balance)
        coherence_score, coherence_details = self._analyze_color_coherence(small)

        # 5. Saturation distribution analysis
        sat_score, sat_details = self._analyze_saturation_profile(small)

        # Build anomaly list
        if lab_score > 0.55:
            anomalies.append(f"LAB color distribution anomaly (score={lab_score:.2f})")
        if hsv_score > 0.55:
            anomalies.append(f"HSV hue distribution anomaly (score={hsv_score:.2f})")
        if corr_score > 0.55:
            anomalies.append(f"Channel correlation anomaly (score={corr_score:.2f})")
        if coherence_score > 0.55:
            anomalies.append(f"Color coherence anomaly (score={coherence_score:.2f})")
        if sat_score > 0.55:
            anomalies.append(f"Saturation profile anomaly (score={sat_score:.2f})")

        # Weighted combination
        weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
        scores = np.array([lab_score, hsv_score, corr_score, coherence_score, sat_score])
        final_score = float(np.dot(scores, weights))

        # Confidence
        agreement = 1.0 - float(np.std(scores))
        confidence = 0.5 * agreement + 0.3 * float(np.max(scores)) + 0.2 * min(1.0, len(anomalies) / 3)

        return {
            "score": float(np.clip(final_score, 0.0, 1.0)),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "lab_anomaly": float(lab_score),
            "hsv_anomaly": float(hsv_score),
            "channel_correlation": float(corr_score),
            "color_coherence": float(coherence_score),
            "saturation_anomaly": float(sat_score),
            "anomalies": anomalies,
        }

    def _analyze_lab_distribution(self, image: np.ndarray) -> tuple:
        """
        Analyze LAB color space distribution.

        Real photos: L channel follows scene-dependent distribution;
        a/b channels are tightly clustered around neutral with scene-specific modes.
        AI images: Often have wider a/b distributions, unusual L histograms.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

        scores = []

        # L channel entropy (AI images often have more uniform lightness)
        l_hist = np.histogram(L, bins=64, range=(0, 255))[0].astype(float)
        l_hist = l_hist / (l_hist.sum() + 1e-10)
        l_entropy = -np.sum(l_hist[l_hist > 0] * np.log2(l_hist[l_hist > 0] + 1e-10))
        # Natural images: entropy ~ 4.0-5.5; AI: often > 5.5 or < 3.5
        if l_entropy > 5.8 or l_entropy < 3.2:
            scores.append(0.7)
        elif l_entropy > 5.5 or l_entropy < 3.5:
            scores.append(0.4)
        else:
            scores.append(0.15)

        # a/b channel spread (kurtosis)
        # Real photos: a/b channels tend to be leptokurtic (peaked)
        # AI images: more platykurtic (flatter) or irregular
        a_flat = a.flatten()
        b_flat = b.flatten()

        a_kurtosis = self._kurtosis(a_flat)
        b_kurtosis = self._kurtosis(b_flat)

        # Natural images: kurtosis > 3 (leptokurtic)
        # AI images: kurtosis often < 2.5 or > 15
        avg_kurtosis = (a_kurtosis + b_kurtosis) / 2
        if avg_kurtosis < 2.0 or avg_kurtosis > 18:
            scores.append(0.7)
        elif avg_kurtosis < 2.8 or avg_kurtosis > 12:
            scores.append(0.4)
        else:
            scores.append(0.15)

        # a*b joint distribution: check for unnaturally uniform spread
        ab_std = np.sqrt(np.std(a_flat) ** 2 + np.std(b_flat) ** 2)
        # Real photos: ab_std varies widely but usually < 25
        # AI tends to have more consistent color spread
        ab_cov = np.corrcoef(a_flat[:10000], b_flat[:10000])[0, 1] if len(a_flat) > 100 else 0
        if abs(ab_cov) < 0.1:
            # Very low a-b correlation: unusual for natural scenes
            scores.append(0.5)
        else:
            scores.append(0.15)

        return float(np.mean(scores)), {"l_entropy": float(l_entropy), "ab_kurtosis": float(avg_kurtosis)}

    def _analyze_hsv_distribution(self, image: np.ndarray) -> tuple:
        """
        Analyze HSV hue distribution.

        Real photos: Multi-modal hue histogram driven by distinct objects.
        AI images: Smoother hue distribution or specific biases.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Only consider sufficiently saturated pixels
        sat_mask = s > 30
        if sat_mask.sum() < 100:
            return 0.3, {"reason": "too few saturated pixels"}

        hue_values = h[sat_mask].flatten()

        scores = []

        # Hue histogram smoothness
        hue_hist = np.histogram(hue_values, bins=36, range=(0, 180))[0].astype(float)
        hue_hist = hue_hist / (hue_hist.sum() + 1e-10)

        # Entropy of hue distribution
        hue_entropy = -np.sum(hue_hist[hue_hist > 0] * np.log2(hue_hist[hue_hist > 0] + 1e-10))
        # Real: entropy varies 2.0-4.5; AI: often higher (more uniform) or shows specific peaks
        max_entropy = np.log2(36)  # ~5.17

        uniformity_ratio = hue_entropy / max_entropy
        if uniformity_ratio > 0.92:
            scores.append(0.7)  # Suspiciously uniform
        elif uniformity_ratio > 0.85:
            scores.append(0.4)
        else:
            scores.append(0.15)

        # Number of dominant hue modes
        # Smooth histogram to find modes
        from scipy.ndimage import gaussian_filter1d
        smooth_hist = gaussian_filter1d(hue_hist, sigma=2)
        # Find peaks (local maxima)
        peaks = []
        for i in range(1, len(smooth_hist) - 1):
            if smooth_hist[i] > smooth_hist[i - 1] and smooth_hist[i] > smooth_hist[i + 1]:
                if smooth_hist[i] > 0.02:  # Minimum height
                    peaks.append(i)

        # Real images: 1-4 dominant modes; AI: often 0 or >5
        n_peaks = len(peaks)
        if n_peaks == 0 or n_peaks > 8:
            scores.append(0.6)
        elif n_peaks > 5:
            scores.append(0.4)
        else:
            scores.append(0.15)

        return float(np.mean(scores)), {"hue_entropy": float(hue_entropy), "n_hue_modes": n_peaks}

    def _analyze_channel_correlation(self, image: np.ndarray) -> tuple:
        """
        Analyze inter-channel correlation patterns.

        In real photographs, RGB channels are driven by the same physical
        illumination and reflectance, creating high inter-channel correlation.
        AI images may have different correlation structure.
        """
        b, g, r = image[:, :, 0].flatten().astype(float), image[:, :, 1].flatten().astype(float), image[:, :, 2].flatten().astype(float)

        # Subsample for efficiency
        n = min(50000, len(r))
        idx = np.random.RandomState(42).choice(len(r), n, replace=False)
        r_s, g_s, b_s = r[idx], g[idx], b[idx]

        # Pairwise correlations
        rg_corr = np.corrcoef(r_s, g_s)[0, 1]
        rb_corr = np.corrcoef(r_s, b_s)[0, 1]
        gb_corr = np.corrcoef(g_s, b_s)[0, 1]

        avg_corr = (abs(rg_corr) + abs(rb_corr) + abs(gb_corr)) / 3

        # Correlation variance (real images have consistent cross-channel correlation)
        corr_values = [abs(rg_corr), abs(rb_corr), abs(gb_corr)]
        corr_std = np.std(corr_values)

        scores = []

        # Real images: avg_corr typically > 0.75
        # AI images: can have lower or more variable correlation
        if avg_corr < 0.50:
            scores.append(0.8)
        elif avg_corr < 0.65:
            scores.append(0.5)
        elif avg_corr < 0.75:
            scores.append(0.3)
        else:
            scores.append(0.1)

        # High correlation variance is suspicious
        if corr_std > 0.20:
            scores.append(0.7)
        elif corr_std > 0.12:
            scores.append(0.4)
        else:
            scores.append(0.15)

        # Check for regional correlation inconsistency
        regional_score = self._regional_correlation_check(image)
        scores.append(regional_score)

        return float(np.mean(scores)), {
            "avg_correlation": float(avg_corr),
            "correlation_std": float(corr_std),
        }

    def _regional_correlation_check(self, image: np.ndarray) -> float:
        """
        Check if channel correlations are consistent across image regions.
        Real images: consistent due to same camera sensor.
        AI images: may vary across generated regions.
        """
        h, w = image.shape[:2]
        grid_size = 4
        cell_h, cell_w = h // grid_size, w // grid_size

        regional_corrs = []
        for gy in range(grid_size):
            for gx in range(grid_size):
                region = image[gy * cell_h:(gy + 1) * cell_h, gx * cell_w:(gx + 1) * cell_w]
                r = region[:, :, 2].flatten().astype(float)
                g = region[:, :, 1].flatten().astype(float)
                if len(r) < 50 or np.std(r) < 1 or np.std(g) < 1:
                    continue
                corr = abs(np.corrcoef(r, g)[0, 1])
                if not np.isnan(corr):
                    regional_corrs.append(corr)

        if len(regional_corrs) < 4:
            return 0.3

        # High variance in regional correlations = suspicious
        std_regional = np.std(regional_corrs)
        if std_regional > 0.25:
            return 0.7
        elif std_regional > 0.15:
            return 0.4
        else:
            return 0.15

    def _analyze_color_coherence(self, image: np.ndarray) -> tuple:
        """
        Detect color banding in gradients and white balance inconsistency.
        """
        scores = []

        # 1. Color banding detection
        banding_score = self._detect_color_banding(image)
        scores.append(banding_score)

        # 2. White balance consistency
        wb_score = self._check_white_balance_consistency(image)
        scores.append(wb_score)

        return float(np.mean(scores)), {"banding": float(banding_score), "white_balance": float(wb_score)}

    def _detect_color_banding(self, image: np.ndarray) -> float:
        """
        Detect color banding in smooth gradients.
        AI images (especially from 8-bit latent spaces) show quantization bands.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)

        # Find smooth gradient regions (low local variance)
        kernel = np.ones((5, 5)) / 25
        local_mean = cv2.filter2D(gray, -1, kernel)
        local_var = cv2.filter2D((gray - local_mean) ** 2, -1, kernel)

        # Smooth regions: local variance < threshold
        smooth_mask = local_var < 20.0

        if smooth_mask.sum() < 1000:
            return 0.2  # Not enough smooth regions to analyze

        # In smooth regions, check for quantization steps
        # Compute gradient magnitude in smooth areas
        dx = np.abs(np.diff(gray, axis=1))
        dy = np.abs(np.diff(gray, axis=0))

        # Trim to match smooth_mask dimensions
        dx_mask = smooth_mask[:, :-1]
        dy_mask = smooth_mask[:-1, :]

        if dx_mask.sum() < 100:
            return 0.2

        # In smooth gradients, real photos have continuous gradients
        # AI images have quantized steps (integer values)
        dx_smooth = dx[dx_mask]
        dy_smooth = dy[dy_mask]

        # Fraction of integer-valued gradients (exactly 0 or 1)
        integer_frac_x = np.mean(np.isin(dx_smooth, [0, 1, 2]))
        integer_frac_y = np.mean(np.isin(dy_smooth, [0, 1, 2]))

        avg_integer_frac = (integer_frac_x + integer_frac_y) / 2

        # Real images (from camera with dithering): < 60% integer gradients
        # AI images (quantized latent): > 75% integer gradients
        if avg_integer_frac > 0.85:
            return 0.8
        elif avg_integer_frac > 0.75:
            return 0.5
        elif avg_integer_frac > 0.65:
            return 0.3
        else:
            return 0.1

    def _check_white_balance_consistency(self, image: np.ndarray) -> float:
        """
        Check if white balance is consistent across the image.
        Real photos: single illuminant → consistent WB.
        AI: may have inconsistent color temperature across regions.
        """
        h, w = image.shape[:2]
        grid = 3
        cell_h, cell_w = h // grid, w // grid

        wb_ratios = []
        for gy in range(grid):
            for gx in range(grid):
                cell = image[gy * cell_h:(gy + 1) * cell_h, gx * cell_w:(gx + 1) * cell_w]
                mean_bgr = cell.mean(axis=(0, 1))
                if mean_bgr[1] > 10:  # Avoid dark regions
                    wb_ratio = mean_bgr[2] / (mean_bgr[0] + 1e-5)  # R/B ratio
                    wb_ratios.append(wb_ratio)

        if len(wb_ratios) < 4:
            return 0.3

        wb_std = np.std(wb_ratios)
        wb_range = np.ptp(wb_ratios)

        # Real images: WB std typically < 0.3
        # AI with inconsistent lighting: > 0.5
        if wb_std > 0.6:
            return 0.7
        elif wb_std > 0.4:
            return 0.5
        elif wb_std > 0.25:
            return 0.3
        else:
            return 0.1

    def _analyze_saturation_profile(self, image: np.ndarray) -> tuple:
        """
        Analyze saturation distribution.
        AI images often have characteristic saturation profiles:
        - Midjourney: tends to oversaturate
        - DALL-E: moderate saturation with specific patterns
        - Flux: can have unusual saturation in highlights
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1].flatten().astype(float)
        v = hsv[:, :, 2].flatten().astype(float)

        scores = []

        # Saturation statistics
        s_mean = np.mean(s)
        s_std = np.std(s)
        s_skew = self._skewness(s)

        # Real photos: mean saturation varies widely (30-120)
        # AI often clusters around specific ranges
        # Check saturation-value relationship
        # In real photos: dark areas have lower saturation (physical constraint)
        # AI may violate this
        dark_mask = v < 80
        bright_mask = v > 180

        if dark_mask.sum() > 100 and bright_mask.sum() > 100:
            dark_sat = np.mean(s[dark_mask])
            bright_sat = np.mean(s[bright_mask])

            # Real: dark areas have lower saturation
            # If dark areas have unusually high saturation: suspicious
            if dark_sat > bright_sat * 0.9 and dark_sat > 60:
                scores.append(0.65)
            else:
                scores.append(0.15)
        else:
            scores.append(0.3)

        # Saturation histogram modality
        s_hist = np.histogram(s, bins=32, range=(0, 255))[0].astype(float)
        s_hist = s_hist / (s_hist.sum() + 1e-10)
        s_entropy = -np.sum(s_hist[s_hist > 0] * np.log2(s_hist[s_hist > 0] + 1e-10))

        # Very high entropy (uniform saturation) is suspicious
        max_s_entropy = np.log2(32)
        if s_entropy / max_s_entropy > 0.90:
            scores.append(0.6)
        elif s_entropy / max_s_entropy > 0.80:
            scores.append(0.35)
        else:
            scores.append(0.15)

        return float(np.mean(scores)), {"s_mean": float(s_mean), "s_entropy": float(s_entropy)}

    @staticmethod
    def _kurtosis(arr: np.ndarray) -> float:
        """Compute excess kurtosis."""
        n = len(arr)
        if n < 4:
            return 0.0
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-10:
            return 0.0
        m4 = np.mean((arr - mean) ** 4)
        return float(m4 / (std ** 4) - 3.0)

    @staticmethod
    def _skewness(arr: np.ndarray) -> float:
        """Compute skewness."""
        n = len(arr)
        if n < 3:
            return 0.0
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-10:
            return 0.0
        m3 = np.mean((arr - mean) ** 3)
        return float(m3 / (std ** 3))
