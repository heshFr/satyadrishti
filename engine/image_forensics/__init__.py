"""
Satya Drishti — Image Forensics Module
======================================
This module provides a robust, multi-layered approach to detecting
AI-generated images and deepfakes. It powers the Media Scanner
feature by combining:
1. Frequency Analysis (GAN Fingerprints)
2. Metadata Verification (EXIF & C2PA Provenance)
3. Neural Deepfake Detection (ViT-B/16)
"""

from .detector import ImageForensicsDetector
from .frequency_analysis import FrequencyAnalyzer
from .metadata_checker import MetadataChecker
from .vit_detector import ViTDetector

__all__ = [
    "ImageForensicsDetector",
    "FrequencyAnalyzer",
    "MetadataChecker",
    "ViTDetector",
]
