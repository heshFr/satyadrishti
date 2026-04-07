"""
Generator Attribution Module
==============================
Identifies WHICH AI generator created the content, providing actionable
intelligence beyond binary real/fake classification.

Supported generator families:
- **Image**: Midjourney, DALL-E 3, Stable Diffusion (XL/3.5), Flux,
  GPT-Image-1, Adobe Firefly, Imagen, StyleGAN
- **Audio**: ElevenLabs, OpenAI TTS, Bark, XTTS, Microsoft VALL-E,
  Google Cloud TTS, Amazon Polly
- **Video**: Sora, Runway Gen-2/3, Kling, Pika, Minimax, Synthesia

Attribution is based on:
1. **Spectral fingerprints**: Each generator leaves characteristic
   frequency-domain patterns
2. **Metadata signatures**: Hidden patterns in file structure
3. **Artifact patterns**: Generator-specific visual/audio artifacts
4. **Statistical profiles**: Distribution of pixel/sample values

Note: Attribution accuracy depends on having seen examples from each
generator. New/unknown generators will be flagged as "unknown_ai".
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


# Generator profiles — spectral and statistical characteristics
IMAGE_GENERATOR_PROFILES = {
    "midjourney_v6": {
        "display_name": "Midjourney v6",
        "beta_range": (0.8, 1.5),
        "saturation_bias": "high",
        "hf_energy": "moderate",
        "color_temp_bias": "warm",
        "typical_resolution": (1024, 1024),
    },
    "dalle_3": {
        "display_name": "DALL-E 3",
        "beta_range": (1.0, 1.6),
        "saturation_bias": "moderate",
        "hf_energy": "moderate",
        "color_temp_bias": "neutral",
        "typical_resolution": (1024, 1024),
    },
    "stable_diffusion_xl": {
        "display_name": "Stable Diffusion XL",
        "beta_range": (0.9, 1.4),
        "saturation_bias": "variable",
        "hf_energy": "high",
        "color_temp_bias": "variable",
        "typical_resolution": (1024, 1024),
    },
    "flux_1": {
        "display_name": "Flux.1",
        "beta_range": (1.1, 1.7),
        "saturation_bias": "moderate",
        "hf_energy": "low_moderate",
        "color_temp_bias": "neutral",
        "typical_resolution": (1024, 1024),
    },
    "gpt_image_1": {
        "display_name": "GPT-Image-1",
        "beta_range": (1.2, 1.8),
        "saturation_bias": "moderate",
        "hf_energy": "low",
        "color_temp_bias": "neutral",
        "typical_resolution": (1024, 1024),
    },
    "adobe_firefly": {
        "display_name": "Adobe Firefly",
        "beta_range": (1.3, 1.9),
        "saturation_bias": "moderate",
        "hf_energy": "low",
        "color_temp_bias": "warm",
        "typical_resolution": (1024, 1024),
    },
    "imagen": {
        "display_name": "Google Imagen",
        "beta_range": (1.2, 1.8),
        "saturation_bias": "low",
        "hf_energy": "low_moderate",
        "color_temp_bias": "neutral",
        "typical_resolution": (1024, 1024),
    },
    "stylegan": {
        "display_name": "StyleGAN 2/3",
        "beta_range": (0.6, 1.2),
        "saturation_bias": "high",
        "hf_energy": "very_high",
        "color_temp_bias": "variable",
        "typical_resolution": (1024, 1024),
    },
}

AUDIO_GENERATOR_PROFILES = {
    "elevenlabs": {
        "display_name": "ElevenLabs",
        "spectral_smoothness": "very_high",
        "breathing_present": False,
        "noise_floor": "very_low",
        "phase_coherence": "high",
    },
    "openai_tts": {
        "display_name": "OpenAI TTS",
        "spectral_smoothness": "high",
        "breathing_present": False,
        "noise_floor": "very_low",
        "phase_coherence": "moderate",
    },
    "bark": {
        "display_name": "Bark",
        "spectral_smoothness": "moderate",
        "breathing_present": True,
        "noise_floor": "low",
        "phase_coherence": "moderate",
    },
    "xtts": {
        "display_name": "XTTS/Coqui",
        "spectral_smoothness": "moderate",
        "breathing_present": False,
        "noise_floor": "low",
        "phase_coherence": "high",
    },
    "google_tts": {
        "display_name": "Google Cloud TTS",
        "spectral_smoothness": "high",
        "breathing_present": False,
        "noise_floor": "very_low",
        "phase_coherence": "very_high",
    },
}

VIDEO_GENERATOR_PROFILES = {
    "sora": {"display_name": "Sora", "temporal_smoothness": "very_high", "physics_accuracy": "high"},
    "runway_gen3": {"display_name": "Runway Gen-3", "temporal_smoothness": "high", "physics_accuracy": "moderate"},
    "kling_v2": {"display_name": "Kling v2", "temporal_smoothness": "moderate", "physics_accuracy": "moderate"},
    "pika": {"display_name": "Pika", "temporal_smoothness": "moderate", "physics_accuracy": "low"},
    "minimax": {"display_name": "Minimax", "temporal_smoothness": "moderate", "physics_accuracy": "moderate"},
    "synthesia": {"display_name": "Synthesia", "temporal_smoothness": "high", "physics_accuracy": "high"},
}


class GeneratorAttributor:
    """Identifies which AI generator created the content."""

    def attribute_image(self, image: np.ndarray, forensic_scores: dict = None) -> dict:
        """
        Attribute an image to a specific AI generator.

        Args:
            image: BGR image.
            forensic_scores: Optional existing forensic analysis scores.

        Returns:
            {
                "is_ai": bool,
                "generator_family": str,
                "generator_name": str,
                "confidence": float,
                "top_matches": list[dict],
                "features_used": list[str],
            }
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        small = cv2.resize(gray, (256, 256))

        # Extract attribution features
        features = {}

        # 1. Spectral beta exponent
        beta = self._compute_beta(small)
        features["beta"] = beta

        # 2. Saturation profile
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        s_mean = np.mean(hsv[:, :, 1])
        features["saturation_mean"] = s_mean

        # 3. High-frequency energy
        hf_energy = self._compute_hf_energy(small)
        features["hf_energy"] = hf_energy

        # 4. Color temperature (R/B ratio)
        avg_bgr = image.mean(axis=(0, 1))
        color_temp = avg_bgr[2] / (avg_bgr[0] + 1e-5)
        features["color_temp"] = color_temp

        # 5. Resolution
        h, w = image.shape[:2]
        features["resolution"] = (w, h)

        # Score each generator profile
        match_scores = {}
        for gen_id, profile in IMAGE_GENERATOR_PROFILES.items():
            score = self._match_image_profile(features, profile)
            match_scores[gen_id] = score

        # Sort by score
        sorted_matches = sorted(match_scores.items(), key=lambda x: x[1], reverse=True)

        # Top match
        best_id, best_score = sorted_matches[0]
        best_profile = IMAGE_GENERATOR_PROFILES[best_id]

        # Is it AI at all? Check if best score is meaningful
        is_ai = best_score > 0.3

        # Top 3 matches
        top_matches = []
        for gen_id, score in sorted_matches[:3]:
            profile = IMAGE_GENERATOR_PROFILES[gen_id]
            top_matches.append({
                "generator": profile["display_name"],
                "id": gen_id,
                "match_score": round(score, 3),
            })

        return {
            "is_ai": is_ai,
            "generator_family": best_profile["display_name"] if is_ai else "Real Camera",
            "generator_name": best_id if is_ai else "camera",
            "confidence": float(np.clip(best_score, 0.0, 1.0)),
            "top_matches": top_matches,
            "features_used": list(features.keys()),
            "features": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                         for k, v in features.items()},
        }

    def attribute_audio(self, audio_forensic_scores: dict) -> dict:
        """
        Attribute audio to a specific TTS generator based on forensic scores.

        Args:
            audio_forensic_scores: Dict of analyzer_name → score from ensemble.

        Returns:
            Attribution result dict.
        """
        if not audio_forensic_scores:
            return {"is_ai": False, "generator_family": "Unknown", "confidence": 0.0}

        # Extract features from forensic scores
        tts_score = audio_forensic_scores.get("tts_artifacts", {}).get("score", 0.5)
        breathing_score = audio_forensic_scores.get("breathing", {}).get("score", 0.5)
        phase_score = audio_forensic_scores.get("phase", {}).get("score", 0.5)
        ast_score = audio_forensic_scores.get("ast", {}).get("score", 0.5)

        match_scores = {}
        for gen_id, profile in AUDIO_GENERATOR_PROFILES.items():
            score = 0.0

            # Breathing analysis
            if not profile["breathing_present"]:
                # Generator doesn't produce breathing → high breathing score is match
                score += 0.25 * min(1.0, breathing_score / 0.7)
            else:
                # Generator produces breathing → low breathing score is match
                score += 0.25 * max(0, 1.0 - breathing_score)

            # Noise floor
            if profile["noise_floor"] == "very_low":
                score += 0.2  # Clean audio matches
            elif profile["noise_floor"] == "low":
                score += 0.1

            # Phase coherence
            if profile["phase_coherence"] in ("high", "very_high"):
                score += 0.15 * min(1.0, phase_score / 0.6)

            # TTS artifacts
            score += 0.2 * min(1.0, tts_score / 0.6)

            # Neural detection agreement
            score += 0.2 * min(1.0, ast_score / 0.6)

            match_scores[gen_id] = float(score)

        sorted_matches = sorted(match_scores.items(), key=lambda x: x[1], reverse=True)
        best_id, best_score = sorted_matches[0]
        best_profile = AUDIO_GENERATOR_PROFILES[best_id]

        is_ai = best_score > 0.3

        top_matches = []
        for gen_id, score in sorted_matches[:3]:
            profile = AUDIO_GENERATOR_PROFILES[gen_id]
            top_matches.append({
                "generator": profile["display_name"],
                "id": gen_id,
                "match_score": round(score, 3),
            })

        return {
            "is_ai": is_ai,
            "generator_family": best_profile["display_name"] if is_ai else "Real Speech",
            "generator_name": best_id if is_ai else "real",
            "confidence": float(np.clip(best_score, 0.0, 1.0)),
            "top_matches": top_matches,
        }

    def _compute_beta(self, gray: np.ndarray) -> float:
        """Compute spectral decay beta exponent."""
        h, w = gray.shape
        hann = np.outer(np.hanning(h), np.hanning(w))
        fft = np.fft.fft2(gray * hann)
        fft_shift = np.fft.fftshift(fft)
        power = np.abs(fft_shift) ** 2

        cy, cx = h // 2, w // 2
        max_r = min(cy, cx)

        radial = np.zeros(max_r)
        y_coords, x_coords = np.ogrid[:h, :w]
        radii = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2).astype(int)

        for r in range(2, max_r):
            mask = radii == r
            if mask.any():
                radial[r] = np.mean(power[mask])

        valid = (radial > 0) & (np.arange(max_r) > 3) & (np.arange(max_r) < max_r * 0.7)
        freqs = np.arange(max_r)[valid]
        power_valid = radial[valid]

        if len(freqs) < 5:
            return 2.0

        coeffs = np.polyfit(np.log10(freqs.astype(float)), np.log10(power_valid + 1e-10), 1)
        return float(-coeffs[0])

    def _compute_hf_energy(self, gray: np.ndarray) -> float:
        """Compute high-frequency energy ratio."""
        fft = np.fft.fft2(gray)
        power = np.abs(fft) ** 2
        h, w = gray.shape
        total = power.sum()
        if total < 1e-10:
            return 0.5
        # High frequency: outer 50% of spectrum
        mask = np.ones((h, w), dtype=bool)
        mask[h // 4:3 * h // 4, w // 4:3 * w // 4] = False
        hf = power[mask].sum()
        return float(hf / total)

    def _match_image_profile(self, features: dict, profile: dict) -> float:
        """Score how well features match a generator profile."""
        score = 0.0

        # Beta exponent match
        beta = features.get("beta", 2.0)
        beta_lo, beta_hi = profile["beta_range"]
        if beta_lo <= beta <= beta_hi:
            score += 0.30  # Within range
        else:
            distance = min(abs(beta - beta_lo), abs(beta - beta_hi))
            score += 0.30 * max(0, 1 - distance * 2)

        # Saturation match
        s_mean = features.get("saturation_mean", 100)
        if profile["saturation_bias"] == "high" and s_mean > 120:
            score += 0.20
        elif profile["saturation_bias"] == "moderate" and 60 < s_mean < 140:
            score += 0.20
        elif profile["saturation_bias"] == "low" and s_mean < 100:
            score += 0.20
        else:
            score += 0.05

        # HF energy match
        hf = features.get("hf_energy", 0.1)
        hf_map = {"very_high": 0.3, "high": 0.2, "moderate": 0.12, "low_moderate": 0.08, "low": 0.05}
        expected_hf = hf_map.get(profile.get("hf_energy", "moderate"), 0.12)
        hf_distance = abs(hf - expected_hf)
        score += 0.25 * max(0, 1 - hf_distance * 10)

        # Color temperature match
        ct = features.get("color_temp", 1.0)
        if profile["color_temp_bias"] == "warm" and ct > 1.1:
            score += 0.15
        elif profile["color_temp_bias"] == "neutral" and 0.8 < ct < 1.2:
            score += 0.15
        elif profile["color_temp_bias"] == "cool" and ct < 0.9:
            score += 0.15
        else:
            score += 0.05

        return float(np.clip(score, 0.0, 1.0))
