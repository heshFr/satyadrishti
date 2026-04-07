"""
Upsampling & Latent Space Artifact Detector
=============================================
Detects artifacts from the latent-space decoding process used by
diffusion models (Stable Diffusion, Flux, DALL-E, Midjourney).

Most modern AI image generators operate in a compressed latent space
and decode to full resolution using a VAE decoder. This process creates:

1. **Grid-aligned artifacts**: VAE decoders process patches (typically 8x8
   or 16x16), creating subtle grid-aligned patterns at patch boundaries.

2. **Checkerboard artifacts**: Transposed convolution (deconvolution) layers
   in the decoder create characteristic checkerboard patterns.

3. **Interpolation artifacts**: Latent-space interpolation produces
   smoother gradients than natural images at specific spatial frequencies.

4. **Patch boundary discontinuities**: Where decoder patches meet,
   there are subtle discontinuities in gradient and noise statistics.

These artifacts are fundamental to how diffusion models work and are
very difficult to remove without significantly degrading image quality.
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class UpsamplingDetector:
    """Detects latent-space decoding artifacts from AI image generators."""

    # Common VAE patch sizes used by different generators
    VAE_PATCH_SIZES = [8, 16, 32]

    def analyze(self, image: np.ndarray) -> dict:
        """
        Run latent-space artifact detection.

        Args:
            image: BGR image (numpy array).

        Returns:
            {
                "score": float (0=real, 1=AI),
                "confidence": float,
                "grid_artifact_score": float,
                "checkerboard_score": float,
                "patch_boundary_score": float,
                "interpolation_score": float,
                "detected_patch_size": int or None,
                "anomalies": list[str],
            }
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        anomalies = []

        # 1. Grid-aligned artifact detection
        grid_score, detected_patch = self._detect_grid_artifacts(gray)

        # 2. Checkerboard pattern detection
        checker_score = self._detect_checkerboard(gray)

        # 3. Patch boundary discontinuity
        boundary_score = self._detect_patch_boundaries(gray, detected_patch)

        # 4. Interpolation smoothness analysis
        interp_score = self._detect_interpolation_artifacts(gray)

        # 5. Deconvolution spectral signature
        deconv_score = self._detect_deconv_signature(gray)

        if grid_score > 0.5:
            anomalies.append(f"Grid artifacts at {detected_patch}px patch boundaries")
        if checker_score > 0.5:
            anomalies.append("Checkerboard deconvolution pattern detected")
        if boundary_score > 0.5:
            anomalies.append("Patch boundary discontinuities detected")
        if interp_score > 0.5:
            anomalies.append("Latent interpolation smoothness anomaly")
        if deconv_score > 0.5:
            anomalies.append("Deconvolution spectral signature detected")

        # Weighted combination
        weights = np.array([0.25, 0.20, 0.20, 0.15, 0.20])
        scores = np.array([grid_score, checker_score, boundary_score, interp_score, deconv_score])
        final_score = float(np.dot(scores, weights))

        confidence = 0.5 * (1.0 - float(np.std(scores))) + 0.3 * float(np.max(scores)) + 0.2 * min(1.0, len(anomalies) / 3)

        return {
            "score": float(np.clip(final_score, 0.0, 1.0)),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "grid_artifact_score": float(grid_score),
            "checkerboard_score": float(checker_score),
            "patch_boundary_score": float(boundary_score),
            "interpolation_score": float(interp_score),
            "deconv_signature_score": float(deconv_score),
            "detected_patch_size": detected_patch,
            "anomalies": anomalies,
        }

    def _detect_grid_artifacts(self, gray: np.ndarray) -> tuple:
        """
        Detect grid-aligned artifacts at VAE patch boundaries.

        The VAE decoder processes the latent in patches. At patch boundaries,
        there are subtle differences in gradient statistics.
        """
        h, w = gray.shape
        best_score = 0.0
        best_patch = None

        for patch_size in self.VAE_PATCH_SIZES:
            if h < patch_size * 4 or w < patch_size * 4:
                continue

            # Extract gradients at patch boundaries vs centers
            boundary_grads = []
            center_grads = []

            # Horizontal boundaries
            for y in range(patch_size, h - 1, patch_size):
                grad_at_boundary = np.abs(gray[y] - gray[y - 1]).mean()
                # Center gradient (halfway between boundaries)
                center_y = y - patch_size // 2
                if 0 < center_y < h - 1:
                    grad_at_center = np.abs(gray[center_y] - gray[center_y - 1]).mean()
                    boundary_grads.append(grad_at_boundary)
                    center_grads.append(grad_at_center)

            # Vertical boundaries
            for x in range(patch_size, w - 1, patch_size):
                grad_at_boundary = np.abs(gray[:, x] - gray[:, x - 1]).mean()
                center_x = x - patch_size // 2
                if 0 < center_x < w - 1:
                    grad_at_center = np.abs(gray[:, center_x] - gray[:, center_x - 1]).mean()
                    boundary_grads.append(grad_at_boundary)
                    center_grads.append(grad_at_center)

            if len(boundary_grads) < 5:
                continue

            boundary_mean = np.mean(boundary_grads)
            center_mean = np.mean(center_grads)

            # If boundary gradients are systematically different from center gradients,
            # this indicates grid-aligned processing
            if center_mean > 1e-5:
                ratio = boundary_mean / center_mean
                # Real images: ratio ≈ 1.0 (no grid alignment)
                # AI images: ratio > 1.1 or < 0.9 (grid artifacts)
                deviation = abs(ratio - 1.0)
                score = min(1.0, deviation * 5.0)  # Scale: 0.2 deviation → score 1.0
            else:
                score = 0.0

            if score > best_score:
                best_score = score
                best_patch = patch_size

        return float(best_score), best_patch

    def _detect_checkerboard(self, gray: np.ndarray) -> float:
        """
        Detect checkerboard artifacts from transposed convolution.

        Transposed conv layers create 2x2 or 4x4 checkerboard patterns
        visible in the frequency domain as peaks at Nyquist frequencies.
        """
        h, w = gray.shape

        # Use a smaller region for efficiency
        size = min(256, h, w)
        crop = gray[:size, :size]

        # High-pass filter to isolate artifacts
        blurred = cv2.GaussianBlur(crop, (5, 5), 1.5)
        residual = crop - blurred

        # FFT of residual
        fft = np.fft.fft2(residual)
        magnitude = np.abs(fft)

        # Look for peaks at Nyquist-related frequencies
        # Checkerboard: strong power at (N/2, N/2), (N/2, 0), (0, N/2)
        n = size
        nyquist_power = 0
        total_power = magnitude.sum()

        if total_power < 1e-10:
            return 0.1

        # Check power at checkerboard-related frequencies
        checkerboard_indices = [
            (n // 2, n // 2),
            (n // 2, 0),
            (0, n // 2),
            (n // 4, n // 4),
            (n // 4, 0),
            (0, n // 4),
        ]

        for dy, dx in checkerboard_indices:
            # Sum power in a small neighborhood around the frequency
            for oy in range(-1, 2):
                for ox in range(-1, 2):
                    ny_idx = (dy + oy) % n
                    nx_idx = (dx + ox) % n
                    nyquist_power += magnitude[ny_idx, nx_idx]

        # Ratio of checkerboard-frequency power to total power
        cb_ratio = nyquist_power / total_power

        # Real images: cb_ratio typically < 0.005
        # Checkerboard artifacts: > 0.01
        if cb_ratio > 0.03:
            return 0.85
        elif cb_ratio > 0.015:
            return 0.6
        elif cb_ratio > 0.008:
            return 0.35
        else:
            return 0.1

    def _detect_patch_boundaries(self, gray: np.ndarray, patch_size: int = None) -> float:
        """
        Detect statistical discontinuities at patch boundaries.

        At VAE decode patch borders, noise statistics change subtly.
        """
        if patch_size is None:
            patch_size = 8  # Default: most common VAE patch size

        h, w = gray.shape
        if h < patch_size * 6 or w < patch_size * 6:
            return 0.2

        # Compute local noise (high-frequency residual)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0.8)
        noise = gray - blurred

        # Compare noise statistics in patches vs across boundaries
        within_patch_vars = []
        across_boundary_vars = []

        for y in range(0, h - patch_size, patch_size):
            for x in range(0, w - patch_size, patch_size):
                # Within-patch noise variance
                patch_noise = noise[y:y + patch_size, x:x + patch_size]
                within_patch_vars.append(np.var(patch_noise))

                # Across-boundary noise variance (straddling the boundary)
                if y + patch_size + patch_size // 2 < h:
                    boundary_patch = noise[y + patch_size - patch_size // 4:
                                           y + patch_size + patch_size // 4,
                                           x:x + patch_size]
                    across_boundary_vars.append(np.var(boundary_patch))

        if len(within_patch_vars) < 10 or len(across_boundary_vars) < 5:
            return 0.2

        # In real images: within-patch and across-boundary variances should be similar
        # In AI images: across-boundary variance may be systematically different
        within_mean = np.mean(within_patch_vars)
        across_mean = np.mean(across_boundary_vars)

        if within_mean < 1e-5:
            return 0.2

        ratio = across_mean / within_mean
        # Deviation from 1.0 indicates boundary effects
        deviation = abs(ratio - 1.0)

        if deviation > 0.30:
            return 0.75
        elif deviation > 0.15:
            return 0.5
        elif deviation > 0.08:
            return 0.3
        else:
            return 0.1

    def _detect_interpolation_artifacts(self, gray: np.ndarray) -> float:
        """
        Detect overly smooth interpolation from latent space decoding.

        Latent-space interpolation produces smoother gradients than
        natural camera images at specific spatial frequencies.
        """
        h, w = gray.shape

        # Compute gradient at multiple scales
        dx1 = np.abs(np.diff(gray, axis=1))
        dy1 = np.abs(np.diff(gray, axis=0))

        # Second-order gradients (curvature)
        dx2 = np.abs(np.diff(gray, n=2, axis=1))
        dy2 = np.abs(np.diff(gray, n=2, axis=0))

        # Ratio of second-order to first-order gradient energy
        # Real images: higher ratio (natural texture has varying curvature)
        # AI images: lower ratio (smoother interpolation)
        first_order_energy = np.mean(dx1) + np.mean(dy1)
        second_order_energy = np.mean(dx2) + np.mean(dy2)

        if first_order_energy < 1e-5:
            return 0.3

        curvature_ratio = second_order_energy / first_order_energy

        # Real images: ratio typically > 0.6
        # AI with smooth interpolation: ratio < 0.5
        if curvature_ratio < 0.35:
            return 0.75
        elif curvature_ratio < 0.50:
            return 0.5
        elif curvature_ratio < 0.60:
            return 0.3
        else:
            return 0.1

    def _detect_deconv_signature(self, gray: np.ndarray) -> float:
        """
        Detect spectral signature of transposed convolution / upsampling layers.

        The frequency spectrum of images processed by deconvolution layers
        shows characteristic patterns at harmonics of the upsampling factor.
        """
        h, w = gray.shape
        size = min(512, h, w)
        crop = gray[:size, :size]

        # Compute power spectrum
        fft = np.fft.fft2(crop)
        fft_shift = np.fft.fftshift(fft)
        power = np.abs(fft_shift) ** 2

        cy, cx = size // 2, size // 2

        # Compute radial power profile
        max_r = min(cy, cx)
        radial = np.zeros(max_r)

        y_coords, x_coords = np.ogrid[:size, :size]
        radii = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2).astype(int)

        for r in range(1, max_r):
            mask = radii == r
            if mask.any():
                radial[r] = np.mean(power[mask])

        if radial[1:].max() < 1e-10:
            return 0.1

        # Normalize
        radial = radial / (radial[1:].max() + 1e-10)

        # Look for dips/peaks at harmonics of common upsampling factors (2x, 4x, 8x)
        harmonic_scores = []
        for factor in [2, 4, 8]:
            harmonic_freq = max_r // factor
            if harmonic_freq < 5 or harmonic_freq >= max_r - 5:
                continue

            # Compare power at harmonic vs neighbors
            harmonic_power = np.mean(radial[harmonic_freq - 1:harmonic_freq + 2])
            neighbor_power = np.mean(radial[harmonic_freq - 5:harmonic_freq - 2])
            neighbor_power2 = np.mean(radial[harmonic_freq + 2:harmonic_freq + 5])
            avg_neighbor = (neighbor_power + neighbor_power2) / 2

            if avg_neighbor > 1e-10:
                ratio = harmonic_power / avg_neighbor
                # Peaks or dips at harmonics indicate deconvolution processing
                deviation = abs(ratio - 1.0)
                harmonic_scores.append(deviation)

        if not harmonic_scores:
            return 0.2

        max_deviation = max(harmonic_scores)

        if max_deviation > 0.5:
            return 0.75
        elif max_deviation > 0.25:
            return 0.5
        elif max_deviation > 0.12:
            return 0.3
        else:
            return 0.1
