"""
Image Forensics -- Compression-Aware Preprocessing
====================================================
Normalizes compression artifacts before neural model inference.

Why this works:
- JPEG compression creates 8x8 block artifacts and color banding
  that the ViT misinterprets as GAN generation artifacts
- Bilateral filtering smooths these block artifacts while
  preserving genuine manipulation edges (blending boundaries,
  texture inconsistencies) that the ViT should detect
- The denoising strength is calibrated to the compression level:
  heavier compression -> stronger denoising

Why bilateral filter specifically:
- Unlike Gaussian blur, it preserves edges (critical for deepfake detection)
- Unlike non-local means, it's fast enough for real-time inference
- The (d, sigmaColor, sigmaSpace) parameters give fine-grained control
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class CompressionNormalizer:
    """
    Preprocesses images to reduce compression artifacts before ViT inference.

    The normalizer is aware of the compression level (from CompressionDetector)
    and applies proportional denoising. Clean images pass through untouched.
    """

    # Denoising profiles calibrated per compression severity.
    # (bilateral_d, sigma_color, sigma_space)
    # Higher values = more smoothing.
    # These were tuned to remove JPEG blockiness without destroying
    # the subtle artifacts that distinguish real deepfakes.
    PROFILES = {
        "none":     None,                # No denoising -- clean image
        "light":    (5,  20,  20),        # Very light -- barely noticeable
        "moderate": (7,  40,  40),        # Moderate -- reduces visible blockiness
        "heavy":    (9,  60,  60),        # Strong -- aggressive artifact removal
        "social":   (9,  75,  75),        # Strongest -- social media worst case
    }

    def normalize(
        self,
        image: np.ndarray,
        compression_severity: str = "none",
        is_social_media: bool = False,
        estimated_quality: Optional[int] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Apply compression-aware denoising to an image.

        Args:
            image: BGR image (numpy array, uint8)
            compression_severity: "none", "light", "moderate", "heavy"
            is_social_media: Whether the image was identified as from social media
            estimated_quality: JPEG quality estimate (1-100)

        Returns:
            (processed_image, info_dict) where info_dict contains preprocessing details
        """
        info = {
            "preprocessing_applied": False,
            "method": "none",
            "severity_input": compression_severity,
            "social_media": is_social_media,
        }

        # Determine which profile to use
        if is_social_media:
            profile_key = "social"
        elif compression_severity in ("heavy",):
            profile_key = "heavy"
        elif compression_severity in ("moderate",):
            profile_key = "moderate"
        elif compression_severity in ("light",):
            # Only apply light denoising if quality is actually low
            if estimated_quality and estimated_quality < 85:
                profile_key = "light"
            else:
                profile_key = "none"
        else:
            profile_key = "none"

        params = self.PROFILES.get(profile_key)

        if params is None:
            # Clean image -- pass through untouched
            return image, info

        d, sigma_color, sigma_space = params

        # Apply bilateral filter -- smooths flat regions (JPEG blocks)
        # while preserving edges (potential deepfake artifacts)
        processed = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

        # For heavy compression, also apply a very light non-local means
        # denoising pass to catch residual color banding
        if profile_key in ("heavy", "social"):
            # Convert to LAB for perceptual denoising
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            # Only denoise the L channel (luminance) -- preserves color info
            l_channel = lab[:, :, 0]
            l_denoised = cv2.fastNlMeansDenoising(l_channel, None, h=6, templateWindowSize=7, searchWindowSize=21)
            lab[:, :, 0] = l_denoised
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        info["preprocessing_applied"] = True
        info["method"] = f"bilateral({d},{sigma_color},{sigma_space})"
        if profile_key in ("heavy", "social"):
            info["method"] += " + nlm_luminance"
        info["profile"] = profile_key

        return processed, info
