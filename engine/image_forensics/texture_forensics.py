"""
Texture Synthesis Detector
==========================
Detects AI-generated textures using Local Binary Patterns (LBP) and
Gray-Level Co-occurrence Matrix (GLCM) analysis.

Key insight: AI-generated textures have statistical regularities that differ
from natural textures captured by camera sensors:

1. **LBP Distribution**: Real textures have characteristic LBP histograms
   driven by sensor noise and natural surface properties. AI textures have
   different LBP distributions, especially in uniform regions.

2. **GLCM Features**: Co-occurrence statistics (contrast, correlation,
   energy, homogeneity) differ between real and AI textures. AI images
   tend to have higher homogeneity and lower contrast in micro-texture.

3. **Texture Repetition**: AI generators sometimes produce subtle repetitive
   patterns (especially in backgrounds) that don't occur in natural scenes.

4. **Multi-Scale Consistency**: Natural textures are self-similar across
   scales (fractal property). AI textures may break this at certain scales.
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class TextureForensicsAnalyzer:
    """Detects AI-generated images through texture analysis."""

    def analyze(self, image: np.ndarray) -> dict:
        """
        Run texture forensic analysis.

        Args:
            image: BGR image (numpy array).

        Returns:
            {
                "score": float (0=real, 1=AI),
                "confidence": float,
                "lbp_anomaly": float,
                "glcm_anomaly": float,
                "repetition_score": float,
                "multiscale_score": float,
                "anomalies": list[str],
            }
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        small_gray = cv2.resize(gray, (384, 384))
        anomalies = []

        # 1. LBP analysis
        lbp_score = self._analyze_lbp(small_gray)

        # 2. GLCM analysis
        glcm_score = self._analyze_glcm(small_gray)

        # 3. Texture repetition detection
        repetition_score = self._detect_texture_repetition(small_gray)

        # 4. Multi-scale consistency
        multiscale_score = self._multiscale_consistency(gray)

        # 5. Micro-texture uniformity
        uniformity_score = self._micro_texture_uniformity(small_gray)

        # Build anomalies
        if lbp_score > 0.55:
            anomalies.append(f"LBP distribution anomaly ({lbp_score:.2f})")
        if glcm_score > 0.55:
            anomalies.append(f"GLCM texture anomaly ({glcm_score:.2f})")
        if repetition_score > 0.5:
            anomalies.append(f"Texture repetition detected ({repetition_score:.2f})")
        if multiscale_score > 0.55:
            anomalies.append(f"Multi-scale texture inconsistency ({multiscale_score:.2f})")
        if uniformity_score > 0.55:
            anomalies.append(f"Micro-texture uniformity anomaly ({uniformity_score:.2f})")

        # Weighted combination
        weights = np.array([0.25, 0.25, 0.15, 0.20, 0.15])
        scores = np.array([lbp_score, glcm_score, repetition_score, multiscale_score, uniformity_score])
        final_score = float(np.dot(scores, weights))

        confidence = 0.5 * (1.0 - float(np.std(scores))) + 0.3 * float(np.max(scores)) + 0.2 * min(1.0, len(anomalies) / 3)

        return {
            "score": float(np.clip(final_score, 0.0, 1.0)),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "lbp_anomaly": float(lbp_score),
            "glcm_anomaly": float(glcm_score),
            "repetition_score": float(repetition_score),
            "multiscale_score": float(multiscale_score),
            "uniformity_score": float(uniformity_score),
            "anomalies": anomalies,
        }

    def _compute_lbp(self, gray: np.ndarray, radius: int = 1) -> np.ndarray:
        """Compute Local Binary Pattern image."""
        h, w = gray.shape
        n_points = 8 * radius
        lbp = np.zeros((h - 2 * radius, w - 2 * radius), dtype=np.uint8)

        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            dx = int(round(radius * np.cos(angle)))
            dy = int(round(-radius * np.sin(angle)))

            # Get neighbor values
            ny = radius + dy
            nx = radius + dx
            neighbor = gray[ny:ny + h - 2 * radius, nx:nx + w - 2 * radius]
            center = gray[radius:h - radius, radius:w - radius]

            lbp |= ((neighbor >= center).astype(np.uint8) << i)

        return lbp

    def _analyze_lbp(self, gray: np.ndarray) -> float:
        """
        Analyze LBP histogram for AI generation artifacts.

        Real photos: LBP histogram has characteristic shape driven by sensor noise.
        AI images: Different LBP distribution, especially in uniform patterns.
        """
        lbp = self._compute_lbp(gray, radius=1)

        # Compute normalized histogram
        hist, _ = np.histogram(lbp, bins=256, range=(0, 255))
        hist = hist.astype(float) / (hist.sum() + 1e-10)

        # Feature 1: Uniform patterns ratio
        # LBP patterns with at most 2 transitions (0↔1) are called "uniform"
        # Real images: ~58-70% uniform patterns; AI: often different
        uniform_count = 0
        for val in range(256):
            binary = format(val, '08b')
            transitions = sum(1 for i in range(7) if binary[i] != binary[i + 1])
            transitions += (1 if binary[7] != binary[0] else 0)
            if transitions <= 2:
                uniform_count += hist[val]

        # Feature 2: Entropy of LBP histogram
        lbp_entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-10))
        max_entropy = np.log2(256)
        normalized_entropy = lbp_entropy / max_entropy

        # Feature 3: Concentration (how much mass is in top-10 bins)
        sorted_hist = np.sort(hist)[::-1]
        top10_mass = sorted_hist[:10].sum()

        scores = []

        # Uniform pattern ratio
        if uniform_count < 0.50 or uniform_count > 0.78:
            scores.append(0.6)
        elif uniform_count < 0.55 or uniform_count > 0.73:
            scores.append(0.35)
        else:
            scores.append(0.15)

        # Entropy
        if normalized_entropy > 0.90 or normalized_entropy < 0.55:
            scores.append(0.6)
        elif normalized_entropy > 0.85 or normalized_entropy < 0.60:
            scores.append(0.35)
        else:
            scores.append(0.15)

        # Concentration
        if top10_mass > 0.65:
            scores.append(0.6)  # Too concentrated
        elif top10_mass < 0.20:
            scores.append(0.5)  # Too spread out
        else:
            scores.append(0.15)

        return float(np.mean(scores))

    def _analyze_glcm(self, gray: np.ndarray) -> float:
        """
        Analyze GLCM features for AI generation artifacts.

        Computes contrast, correlation, energy, homogeneity at multiple
        distances and angles.
        """
        # Quantize to 32 levels for tractable GLCM
        quantized = (gray // 8).astype(np.uint8)
        n_levels = 32

        scores = []

        for distance in [1, 3, 5]:
            # Compute GLCM for horizontal direction
            glcm = self._compute_glcm(quantized, distance, 0, n_levels)

            # Normalize
            glcm_sum = glcm.sum()
            if glcm_sum < 1:
                continue
            glcm = glcm / glcm_sum

            # Compute features
            contrast = self._glcm_contrast(glcm, n_levels)
            correlation = self._glcm_correlation(glcm, n_levels)
            energy = self._glcm_energy(glcm)
            homogeneity = self._glcm_homogeneity(glcm, n_levels)

            # AI detection: AI images tend to have higher homogeneity
            # and energy in micro-texture (distance=1)
            if distance == 1:
                if homogeneity > 0.85:
                    scores.append(0.65)  # Suspiciously homogeneous
                elif homogeneity > 0.75:
                    scores.append(0.4)
                else:
                    scores.append(0.15)

                if energy > 0.15:
                    scores.append(0.6)  # High energy = repetitive texture
                elif energy > 0.08:
                    scores.append(0.35)
                else:
                    scores.append(0.15)

            # At larger distances, AI images often show less contrast than real
            if distance == 5:
                if contrast < 5.0:
                    scores.append(0.55)  # Suspiciously smooth at distance
                elif contrast < 15.0:
                    scores.append(0.3)
                else:
                    scores.append(0.15)

        return float(np.mean(scores)) if scores else 0.3

    def _compute_glcm(self, quantized: np.ndarray, dx: int, dy: int, n_levels: int) -> np.ndarray:
        """Compute Gray-Level Co-occurrence Matrix."""
        h, w = quantized.shape
        glcm = np.zeros((n_levels, n_levels), dtype=np.float64)

        y_start = max(0, -dy)
        y_end = min(h, h - dy)
        x_start = max(0, -dx)
        x_end = min(w, w - dx)

        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                i = quantized[y, x]
                j = quantized[y + dy, x + dx]
                if i < n_levels and j < n_levels:
                    glcm[i, j] += 1

        # Make symmetric
        glcm = glcm + glcm.T
        return glcm

    @staticmethod
    def _glcm_contrast(glcm: np.ndarray, n_levels: int) -> float:
        i, j = np.meshgrid(range(n_levels), range(n_levels), indexing='ij')
        return float(np.sum(glcm * (i - j) ** 2))

    @staticmethod
    def _glcm_correlation(glcm: np.ndarray, n_levels: int) -> float:
        i, j = np.meshgrid(range(n_levels), range(n_levels), indexing='ij')
        mu_i = np.sum(i * glcm)
        mu_j = np.sum(j * glcm)
        sigma_i = np.sqrt(np.sum((i - mu_i) ** 2 * glcm))
        sigma_j = np.sqrt(np.sum((j - mu_j) ** 2 * glcm))
        if sigma_i < 1e-10 or sigma_j < 1e-10:
            return 0.0
        return float(np.sum((i - mu_i) * (j - mu_j) * glcm) / (sigma_i * sigma_j))

    @staticmethod
    def _glcm_energy(glcm: np.ndarray) -> float:
        return float(np.sum(glcm ** 2))

    @staticmethod
    def _glcm_homogeneity(glcm: np.ndarray, n_levels: int) -> float:
        i, j = np.meshgrid(range(n_levels), range(n_levels), indexing='ij')
        return float(np.sum(glcm / (1 + np.abs(i - j))))

    def _detect_texture_repetition(self, gray: np.ndarray) -> float:
        """
        Detect repeating texture patterns (AI artifacts).
        Uses autocorrelation of texture features.
        """
        h, w = gray.shape

        # Divide into overlapping patches
        patch_size = 64
        stride = 32
        patches = []

        for y in range(0, h - patch_size, stride):
            for x in range(0, w - patch_size, stride):
                patch = gray[y:y + patch_size, x:x + patch_size]
                patches.append(patch.flatten().astype(float))

        if len(patches) < 10:
            return 0.2

        patches = np.array(patches)

        # Normalize patches
        norms = np.linalg.norm(patches, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        patches_norm = patches / norms

        # Compute pairwise cosine similarities
        n = min(100, len(patches_norm))
        selected = patches_norm[:n]
        similarities = selected @ selected.T

        # Remove diagonal
        np.fill_diagonal(similarities, 0)

        # Count highly similar non-adjacent patches
        high_sim_threshold = 0.95
        high_sim_count = np.sum(similarities > high_sim_threshold) / 2  # Symmetric

        # Fraction of highly similar pairs
        total_pairs = n * (n - 1) / 2
        repetition_ratio = high_sim_count / (total_pairs + 1e-10)

        # Real images: very few identical patches (< 1%)
        # AI with repetitive texture: > 5%
        if repetition_ratio > 0.10:
            return 0.8
        elif repetition_ratio > 0.05:
            return 0.6
        elif repetition_ratio > 0.02:
            return 0.35
        else:
            return 0.1

    def _multiscale_consistency(self, gray: np.ndarray) -> float:
        """
        Check texture consistency across scales.
        Natural images are approximately self-similar (fractal).
        AI images may break this at certain scales.
        """
        # Compute texture energy at multiple scales
        scales = [1.0, 0.5, 0.25]
        energies = []

        for scale in scales:
            h, w = gray.shape
            sh, sw = int(h * scale), int(w * scale)
            if sh < 64 or sw < 64:
                continue

            scaled = cv2.resize(gray, (sw, sh))

            # Compute Laplacian energy (texture measure)
            laplacian = cv2.Laplacian(scaled, cv2.CV_64F)
            energy = np.mean(laplacian ** 2)
            energies.append(energy)

        if len(energies) < 3:
            return 0.3

        # In natural images, texture energy follows a power law across scales
        # Compute ratios between consecutive scales
        ratios = []
        for i in range(len(energies) - 1):
            if energies[i + 1] > 1e-10:
                ratios.append(energies[i] / energies[i + 1])

        if not ratios:
            return 0.3

        # Natural images: consistent ratio (2-6x between 2x scale changes)
        # AI images: inconsistent ratios
        ratio_std = np.std(ratios)
        mean_ratio = np.mean(ratios)

        if ratio_std > 2.0:
            return 0.7  # Very inconsistent
        elif mean_ratio < 1.5 or mean_ratio > 10.0:
            return 0.55  # Unusual energy scaling
        elif ratio_std > 1.0:
            return 0.4
        else:
            return 0.15

    def _micro_texture_uniformity(self, gray: np.ndarray) -> float:
        """
        Check if micro-texture (high-frequency detail) is too uniform.
        AI images often have spatially uniform noise/texture patterns.
        """
        # High-pass filter to extract micro-texture
        blurred = cv2.GaussianBlur(gray.astype(float), (7, 7), 2.0)
        residual = gray.astype(float) - blurred

        h, w = residual.shape
        block_size = 32
        block_stds = []

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = residual[y:y + block_size, x:x + block_size]
                block_stds.append(np.std(block))

        if len(block_stds) < 10:
            return 0.3

        block_stds = np.array(block_stds)
        mean_std = np.mean(block_stds)
        std_of_stds = np.std(block_stds)

        # Coefficient of variation of micro-texture energy
        if mean_std < 1e-5:
            return 0.6  # Perfectly smooth = suspicious

        cv = std_of_stds / mean_std

        # Real images: CV typically 0.4-1.5 (variable micro-texture)
        # AI images: CV often < 0.3 (uniform micro-texture)
        if cv < 0.20:
            return 0.75  # Very uniform micro-texture
        elif cv < 0.30:
            return 0.55
        elif cv < 0.40:
            return 0.35
        else:
            return 0.1
