"""
Adversarial Preprocessing Pipeline
====================================
Applies preprocessing transformations to images/audio before detection
to improve robustness against adversarial attacks.

Key insight: Adversarial attacks on deepfake detectors work by adding
tiny perturbations that fool neural networks. These perturbations are
fragile and can be disrupted by simple preprocessing:

1. **JPEG Recompression**: Destroys adversarial perturbations that rely
   on specific pixel values. Also standardizes compression artifacts.

2. **Gaussian Blur**: Removes high-frequency adversarial noise while
   preserving structural AI artifacts (which are low-frequency).

3. **Resize + Restore**: Downsampling and upsampling disrupts pixel-level
   perturbations while preserving semantic-level features.

4. **Bit Depth Reduction**: Quantizing to fewer bits removes subtle
   perturbations while preserving large-scale AI artifacts.

5. **Ensemble Preprocessing**: Running detection on multiple preprocessed
   versions and aggregating results improves robustness.

This module provides preprocessing for BOTH images and audio.
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class AdversarialPreprocessor:
    """
    Applies adversarial robustness preprocessing to media.

    Use detect_with_preprocessing() for ensemble preprocessing —
    runs detection on multiple preprocessed versions and aggregates.
    """

    # Image preprocessing pipeline steps
    IMAGE_PREPROCESS_CONFIGS = [
        {"name": "original", "jpeg_quality": None, "blur_sigma": 0, "resize_factor": 1.0, "bit_depth": 8},
        {"name": "jpeg_90", "jpeg_quality": 90, "blur_sigma": 0, "resize_factor": 1.0, "bit_depth": 8},
        {"name": "jpeg_75", "jpeg_quality": 75, "blur_sigma": 0, "resize_factor": 1.0, "bit_depth": 8},
        {"name": "blur_light", "jpeg_quality": None, "blur_sigma": 0.5, "resize_factor": 1.0, "bit_depth": 8},
        {"name": "resize_75", "jpeg_quality": None, "blur_sigma": 0, "resize_factor": 0.75, "bit_depth": 8},
        {"name": "bit_6", "jpeg_quality": None, "blur_sigma": 0, "resize_factor": 1.0, "bit_depth": 6},
    ]

    def preprocess_image(self, image: np.ndarray, config: dict) -> np.ndarray:
        """
        Apply a single preprocessing configuration to an image.

        Args:
            image: BGR image.
            config: Preprocessing configuration dict.

        Returns:
            Preprocessed BGR image.
        """
        result = image.copy()

        # JPEG recompression
        if config.get("jpeg_quality") is not None:
            q = config["jpeg_quality"]
            _, encoded = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, q])
            result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        # Gaussian blur
        sigma = config.get("blur_sigma", 0)
        if sigma > 0:
            ksize = int(sigma * 6) | 1  # Ensure odd
            ksize = max(3, ksize)
            result = cv2.GaussianBlur(result, (ksize, ksize), sigma)

        # Resize + restore
        factor = config.get("resize_factor", 1.0)
        if factor != 1.0:
            h, w = result.shape[:2]
            small = cv2.resize(result, (int(w * factor), int(h * factor)))
            result = cv2.resize(small, (w, h))

        # Bit depth reduction
        bit_depth = config.get("bit_depth", 8)
        if bit_depth < 8:
            shift = 8 - bit_depth
            result = ((result >> shift) << shift).astype(np.uint8)

        return result

    def preprocess_image_ensemble(self, image: np.ndarray) -> list:
        """
        Generate multiple preprocessed versions of an image.

        Returns:
            List of (config_name, preprocessed_image) tuples.
        """
        results = []
        for config in self.IMAGE_PREPROCESS_CONFIGS:
            try:
                preprocessed = self.preprocess_image(image, config)
                results.append((config["name"], preprocessed))
            except Exception as e:
                logger.warning("Preprocessing '%s' failed: %s", config["name"], e)
        return results

    def aggregate_scores(self, scores: list, method: str = "trimmed_mean") -> dict:
        """
        Aggregate detection scores from multiple preprocessed versions.

        Args:
            scores: List of score dicts from different preprocessed versions.
            method: Aggregation method ("trimmed_mean", "median", "max").

        Returns:
            Aggregated score dict.
        """
        if not scores:
            return {"score": 0.5, "confidence": 0.0, "method": method}

        score_values = [s.get("score", 0.5) for s in scores]
        confidence_values = [s.get("confidence", 0.5) for s in scores]

        if method == "trimmed_mean":
            # Remove highest and lowest, average the rest
            if len(score_values) > 2:
                sorted_scores = sorted(score_values)
                trimmed = sorted_scores[1:-1]
                agg_score = float(np.mean(trimmed))
            else:
                agg_score = float(np.mean(score_values))
        elif method == "median":
            agg_score = float(np.median(score_values))
        elif method == "max":
            agg_score = float(np.max(score_values))
        else:
            agg_score = float(np.mean(score_values))

        # Confidence: higher when preprocessed versions agree
        score_std = float(np.std(score_values))
        agreement_bonus = max(0, 1.0 - score_std * 3)
        agg_confidence = float(np.mean(confidence_values)) * 0.5 + agreement_bonus * 0.5

        # Score consistency analysis
        consistent = all(
            (s > 0.5) == (agg_score > 0.5) for s in score_values
        )

        return {
            "score": float(np.clip(agg_score, 0.0, 1.0)),
            "confidence": float(np.clip(agg_confidence, 0.0, 1.0)),
            "method": method,
            "n_versions": len(scores),
            "score_std": score_std,
            "consistent_across_preprocessing": consistent,
            "per_version_scores": {
                s.get("version", f"v{i}"): s.get("score", 0.5)
                for i, s in enumerate(scores)
            },
        }

    # Audio preprocessing
    @staticmethod
    def preprocess_audio(waveform: np.ndarray, sr: int, method: str = "all") -> list:
        """
        Generate preprocessed versions of audio for robust detection.

        Args:
            waveform: Audio waveform (1D float array).
            sr: Sample rate.
            method: "all" for full ensemble, "basic" for minimal.

        Returns:
            List of (name, preprocessed_waveform) tuples.
        """
        results = [("original", waveform)]

        if method == "basic":
            return results

        # Low-pass filter (remove adversarial HF noise)
        try:
            from scipy.signal import butter, filtfilt
            nyq = sr / 2
            cutoff = min(7000, nyq * 0.9)
            b, a = butter(4, cutoff / nyq, btype='low')
            lowpass = filtfilt(b, a, waveform)
            results.append(("lowpass_7k", lowpass))
        except ImportError:
            pass

        # Add tiny noise (disrupts adversarial perturbations)
        noise = np.random.RandomState(42).randn(*waveform.shape) * 0.001
        noisy = waveform + noise
        results.append(("noise_added", noisy))

        # Resample (downsample + upsample)
        try:
            import librosa
            down = librosa.resample(waveform, orig_sr=sr, target_sr=sr // 2)
            up = librosa.resample(down, orig_sr=sr // 2, target_sr=sr)
            # Trim to match original length
            if len(up) > len(waveform):
                up = up[:len(waveform)]
            elif len(up) < len(waveform):
                up = np.pad(up, (0, len(waveform) - len(up)))
            results.append(("resample_half", up))
        except ImportError:
            pass

        # Quantize (reduce bit depth)
        quantized = np.round(waveform * 32768) / 32768  # 16-bit quantization
        quantized = np.round(quantized * 256) / 256  # 8-bit
        results.append(("quantize_8bit", quantized))

        return results
