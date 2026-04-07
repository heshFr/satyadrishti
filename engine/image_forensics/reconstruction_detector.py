"""
Reconstruction Consistency Detector
=====================================
Detects AI-generated images by measuring how they respond to
JPEG recompression. This exploits a fundamental difference:

- **Real camera images**: Already contain sensor noise and natural
  high-frequency detail. JPEG recompression at similar quality
  produces minimal change (the image is already "JPEG-native").

- **AI-generated images**: Contain synthetic high-frequency patterns
  that are not JPEG-optimized. Recompression changes these patterns
  significantly, producing measurable reconstruction error.

Additionally:
- **Blur-Reblur Consistency**: Applying Gaussian blur to a real photo
  and measuring the difference reveals natural detail. Applying it to
  AI images reveals different residual patterns.

- **Noise Injection Response**: Adding Gaussian noise and denoising
  produces different residual patterns for real vs AI images.

This is one of the most adversarially robust detection methods because
it doesn't rely on specific generator artifacts — it exploits the
fundamental difference in how real vs synthetic images encode information.
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class ReconstructionDetector:
    """Detects AI generation through reconstruction consistency analysis."""

    def analyze(self, image: np.ndarray) -> dict:
        """
        Run reconstruction consistency analysis.

        Args:
            image: BGR image (numpy array).

        Returns:
            {
                "score": float (0=real, 1=AI),
                "confidence": float,
                "jpeg_consistency": float,
                "blur_consistency": float,
                "denoise_consistency": float,
                "anomalies": list[str],
            }
        """
        anomalies = []

        # Resize for consistent analysis
        target_h, target_w = 384, 384
        resized = cv2.resize(image, (target_w, target_h))

        # 1. JPEG recompression consistency
        jpeg_score = self._jpeg_reconstruction_test(resized)

        # 2. Blur-reblur consistency
        blur_score = self._blur_reconstruction_test(resized)

        # 3. Denoise reconstruction
        denoise_score = self._denoise_reconstruction_test(resized)

        # 4. Resize reconstruction (downsample + upsample)
        resize_score = self._resize_reconstruction_test(resized)

        # 5. Multi-quality JPEG ladder
        ladder_score = self._jpeg_quality_ladder(resized)

        if jpeg_score > 0.5:
            anomalies.append(f"JPEG reconstruction inconsistency ({jpeg_score:.2f})")
        if blur_score > 0.5:
            anomalies.append(f"Blur reconstruction anomaly ({blur_score:.2f})")
        if denoise_score > 0.5:
            anomalies.append(f"Denoise reconstruction anomaly ({denoise_score:.2f})")
        if resize_score > 0.5:
            anomalies.append(f"Resize reconstruction anomaly ({resize_score:.2f})")
        if ladder_score > 0.5:
            anomalies.append(f"JPEG quality ladder anomaly ({ladder_score:.2f})")

        # Weighted combination
        weights = np.array([0.30, 0.20, 0.20, 0.15, 0.15])
        scores = np.array([jpeg_score, blur_score, denoise_score, resize_score, ladder_score])
        final_score = float(np.dot(scores, weights))

        confidence = 0.5 * (1.0 - float(np.std(scores))) + 0.3 * float(np.max(scores)) + 0.2 * min(1.0, len(anomalies) / 3)

        return {
            "score": float(np.clip(final_score, 0.0, 1.0)),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "jpeg_consistency": float(jpeg_score),
            "blur_consistency": float(blur_score),
            "denoise_consistency": float(denoise_score),
            "resize_consistency": float(resize_score),
            "quality_ladder": float(ladder_score),
            "anomalies": anomalies,
        }

    def _jpeg_reconstruction_test(self, image: np.ndarray) -> float:
        """
        Compress image to JPEG at multiple qualities and measure reconstruction error.

        Real images: Low error at similar quality (already JPEG-native).
        AI images: Higher error because synthetic patterns are disrupted.
        """
        qualities = [85, 75, 60]
        errors = []

        for q in qualities:
            # Encode to JPEG in memory
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, q]
            _, encoded = cv2.imencode('.jpg', image, encode_params)
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

            if decoded is None:
                continue

            # Compute reconstruction error
            diff = np.abs(image.astype(float) - decoded.astype(float))
            mse = np.mean(diff ** 2)
            mae = np.mean(diff)

            # Also compute structural difference
            # Focus on high-frequency components (most affected by recompression)
            orig_hf = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
            recon_hf = cv2.Laplacian(cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
            hf_diff = np.mean(np.abs(orig_hf - recon_hf))

            errors.append({
                "quality": q,
                "mse": mse,
                "mae": mae,
                "hf_diff": hf_diff,
            })

        if not errors:
            return 0.3

        # Key metric: high-frequency reconstruction error at Q=85
        # Real images: hf_diff typically < 5.0 at Q=85
        # AI images: hf_diff typically > 7.0 at Q=85
        hf_85 = errors[0]["hf_diff"]

        # Also check the ratio of errors across qualities
        # Real images: error increases smoothly as quality drops
        # AI images: error may spike at certain qualities
        if len(errors) >= 3:
            error_ratios = []
            for i in range(len(errors) - 1):
                if errors[i]["hf_diff"] > 1e-5:
                    error_ratios.append(errors[i + 1]["hf_diff"] / errors[i]["hf_diff"])
            ratio_std = np.std(error_ratios) if error_ratios else 0

        scores = []

        # HF error at Q=85
        if hf_85 > 12.0:
            scores.append(0.8)
        elif hf_85 > 8.0:
            scores.append(0.6)
        elif hf_85 > 5.0:
            scores.append(0.35)
        else:
            scores.append(0.1)

        # MAE pattern
        mae_85 = errors[0]["mae"]
        if mae_85 > 6.0:
            scores.append(0.7)
        elif mae_85 > 3.5:
            scores.append(0.4)
        else:
            scores.append(0.15)

        return float(np.mean(scores))

    def _blur_reconstruction_test(self, image: np.ndarray) -> float:
        """
        Apply Gaussian blur and measure the residual pattern.

        Real images: Residual (detail) has natural, varied structure.
        AI images: Residual has more uniform, synthetic patterns.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)

        # Apply multiple blur levels
        blurred_3 = cv2.GaussianBlur(gray, (3, 3), 1.0)
        blurred_7 = cv2.GaussianBlur(gray, (7, 7), 2.0)

        residual_3 = gray - blurred_3
        residual_7 = gray - blurred_7

        scores = []

        # Feature 1: Kurtosis of fine residual
        # Real images: leptokurtic (heavy tails — natural texture)
        # AI images: more Gaussian (lighter tails)
        r3_flat = residual_3.flatten()
        kurtosis_3 = self._kurtosis(r3_flat)

        if kurtosis_3 < 2.0:
            scores.append(0.7)  # Too Gaussian for a real photo
        elif kurtosis_3 < 4.0:
            scores.append(0.4)
        else:
            scores.append(0.15)  # Natural heavy-tailed residual

        # Feature 2: Spatial uniformity of residual energy
        h, w = residual_7.shape
        block_size = 48
        block_energies = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = residual_7[y:y + block_size, x:x + block_size]
                block_energies.append(np.mean(block ** 2))

        if len(block_energies) > 4:
            energy_cv = np.std(block_energies) / (np.mean(block_energies) + 1e-10)
            # Real images: high CV (varied detail across regions)
            # AI: lower CV (more uniform)
            if energy_cv < 0.3:
                scores.append(0.65)
            elif energy_cv < 0.5:
                scores.append(0.35)
            else:
                scores.append(0.15)

        # Feature 3: Cross-scale residual correlation
        # Real: moderate correlation between fine and coarse residuals
        # AI: often higher correlation (similar patterns at all scales)
        r3_sub = residual_3[:h, :w].flatten()[:50000]
        r7_sub = residual_7.flatten()[:50000]
        if len(r3_sub) > 100 and np.std(r3_sub) > 0.1 and np.std(r7_sub) > 0.1:
            cross_corr = abs(np.corrcoef(r3_sub, r7_sub)[0, 1])
            if cross_corr > 0.90:
                scores.append(0.6)  # Too correlated
            elif cross_corr > 0.80:
                scores.append(0.35)
            else:
                scores.append(0.15)

        return float(np.mean(scores)) if scores else 0.3

    def _denoise_reconstruction_test(self, image: np.ndarray) -> float:
        """
        Add noise, denoise, and compare with original.

        The denoised version reveals the "structure" of the image.
        Real vs AI images have different structure-noise separation.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)

        # Add Gaussian noise
        rng = np.random.RandomState(42)
        noise = rng.randn(*gray.shape) * 15.0
        noisy = np.clip(gray + noise, 0, 255).astype(np.uint8)

        # Denoise with bilateral filter (edge-preserving)
        denoised = cv2.bilateralFilter(noisy, 9, 75, 75).astype(float)

        # Compute difference between original and denoised
        diff = gray - denoised
        diff_std = np.std(diff)
        diff_mean = np.mean(np.abs(diff))

        scores = []

        # Real images: denoised version is close to original (natural structure preserved)
        # AI images: denoised version may differ more (synthetic patterns disrupted)
        if diff_mean > 12.0:
            scores.append(0.7)
        elif diff_mean > 7.0:
            scores.append(0.4)
        else:
            scores.append(0.15)

        # Check spatial structure of the difference
        # Real: difference concentrated around edges/texture
        # AI: more uniform difference
        h, w = diff.shape
        block_size = 48
        diff_blocks = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block_diff = np.mean(np.abs(diff[y:y + block_size, x:x + block_size]))
                diff_blocks.append(block_diff)

        if len(diff_blocks) > 4:
            cv_diff = np.std(diff_blocks) / (np.mean(diff_blocks) + 1e-10)
            if cv_diff < 0.25:
                scores.append(0.6)  # Too uniform
            elif cv_diff < 0.4:
                scores.append(0.35)
            else:
                scores.append(0.15)

        return float(np.mean(scores)) if scores else 0.3

    def _resize_reconstruction_test(self, image: np.ndarray) -> float:
        """
        Downsample → upsample and measure reconstruction error.

        Tests how much information is lost/changed during resolution change.
        AI images respond differently to resolution changes.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
        h, w = gray.shape

        # Downsample by 2x then upsample back
        small = cv2.resize(gray.astype(np.uint8), (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        restored = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC).astype(float)

        diff = np.abs(gray - restored)
        mean_diff = np.mean(diff)

        # Measure high-frequency loss
        orig_lap = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
        rest_lap = cv2.Laplacian(restored.astype(np.uint8), cv2.CV_64F)
        hf_loss = np.mean(np.abs(orig_lap)) - np.mean(np.abs(rest_lap))

        scores = []

        # Real images: more high-frequency detail loss (natural detail)
        # AI images: less loss (synthetic detail is more "redundant")
        if hf_loss < 1.0:
            scores.append(0.6)  # Suspiciously little HF loss
        elif hf_loss < 3.0:
            scores.append(0.35)
        else:
            scores.append(0.15)

        # Mean reconstruction error
        if mean_diff < 5.0:
            scores.append(0.55)  # Too easy to reconstruct
        elif mean_diff < 10.0:
            scores.append(0.3)
        else:
            scores.append(0.15)

        return float(np.mean(scores))

    def _jpeg_quality_ladder(self, image: np.ndarray) -> float:
        """
        Compress at increasing JPEG qualities and check error curve shape.

        Real images: smooth error curve.
        AI images: may have non-monotonic error (inflection points).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        qualities = [95, 85, 75, 65, 55, 45, 35]
        errors = []

        for q in qualities:
            _, encoded = cv2.imencode('.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, q])
            decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
            if decoded is not None:
                mse = np.mean((gray.astype(float) - decoded.astype(float)) ** 2)
                errors.append(mse)

        if len(errors) < 5:
            return 0.3

        # Check monotonicity (error should increase as quality decreases)
        # Count violations
        violations = 0
        for i in range(len(errors) - 1):
            if errors[i + 1] < errors[i] * 0.95:  # Allow 5% tolerance
                violations += 1

        # Check smoothness of error curve
        if len(errors) >= 3:
            second_diff = np.diff(np.diff(errors))
            curve_smoothness = np.std(second_diff) / (np.mean(np.abs(second_diff)) + 1e-10)
        else:
            curve_smoothness = 0

        scores = []

        if violations > 2:
            scores.append(0.7)  # Non-monotonic = suspicious
        elif violations > 0:
            scores.append(0.4)
        else:
            scores.append(0.15)

        if curve_smoothness > 3.0:
            scores.append(0.6)  # Irregular error curve
        elif curve_smoothness > 1.5:
            scores.append(0.35)
        else:
            scores.append(0.15)

        return float(np.mean(scores))

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
