"""
Image Forensics -- Error Level Analysis (ELA)
=============================================
Detects AI-generated and manipulated images by analyzing JPEG compression
error patterns. When a real photo is re-saved at a known quality, different
regions show varying error levels (textured areas differ from smooth areas).
AI-generated images tend to show UNIFORM error levels across the entire image
because they were never captured by a real camera sensor.

V2 improvements:
- Quality-adaptive ELA: uses estimated JPEG quality instead of fixed quality=90
- Handles double-compressed images properly (social media recompression)
- Returns richer diagnostic details
"""

import io
import numpy as np
import cv2
from typing import Tuple, Dict, Any
from PIL import Image


class ELAAnalyzer:
    """Error Level Analysis for detecting AI-generated images."""

    def __init__(self, quality: int = 90, target_size: int = 512):
        self.default_quality = quality
        self.target_size = target_size

    def compute_ela(self, image: np.ndarray, quality: int = None) -> np.ndarray:
        """
        Compute ELA by re-compressing the image and measuring the difference.
        Returns the ELA difference map (grayscale, 0-255).

        If quality is provided, uses quality+5 for resave (quality-adaptive).
        This gives more meaningful ELA for heavily compressed images.
        """
        resave_quality = quality if quality else self.default_quality

        # Convert BGR to RGB PIL
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Resize for consistency
        pil_img = pil_img.resize((self.target_size, self.target_size), Image.LANCZOS)

        # Re-compress at known quality
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=resave_quality)
        buffer.seek(0)
        recompressed = Image.open(buffer)

        # Compute absolute difference
        original_arr = np.array(pil_img, dtype=np.float32)
        recompressed_arr = np.array(recompressed, dtype=np.float32)
        diff = np.abs(original_arr - recompressed_arr)

        # Convert to grayscale difference
        ela_map = np.mean(diff, axis=2)
        return ela_map

    def analyze(
        self, image: np.ndarray, estimated_quality: int = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze image using ELA to detect AI generation.

        Key insight: AI-generated images show unusually UNIFORM error levels,
        while real photos show HIGH VARIANCE in error levels (different textures,
        edges, smooth areas all compress differently).

        Quality-adaptive: if estimated_quality is provided (from compression detector),
        we use it to select the optimal resave quality. For heavily compressed images
        (quality < 80), the ELA signal is inherently weak and we reduce our confidence.

        Returns (anomaly_score, details).
        """
        # Quality-adaptive resave quality
        if estimated_quality is not None and estimated_quality > 0:
            # Use quality slightly above the estimated original
            # This maximizes the ELA signal for that compression level
            resave_quality = min(98, estimated_quality + 5)
        else:
            resave_quality = self.default_quality

        ela_map = self.compute_ela(image, quality=resave_quality)

        # --- Metric 1: Error uniformity ---
        # AI images have very uniform ELA (low std relative to mean)
        ela_mean = float(np.mean(ela_map))
        ela_std = float(np.std(ela_map))
        # Coefficient of variation -- low = uniform = suspicious
        cv = ela_std / (ela_mean + 1e-8)

        # --- Metric 2: Grid pattern detection ---
        # Divide image into blocks and check variance between blocks
        block_size = 32
        h, w = ela_map.shape
        block_means = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = ela_map[y:y + block_size, x:x + block_size]
                block_means.append(np.mean(block))
        block_means = np.array(block_means)
        block_variance = float(np.std(block_means)) if len(block_means) > 1 else 0.0

        # --- Metric 3: Edge vs smooth region contrast ---
        # Real photos: edges have high ELA, smooth areas have low ELA
        # AI images: more uniform across edges and smooth areas
        # Use Canny to find edges
        ela_uint8 = np.clip(ela_map * 10, 0, 255).astype(np.uint8)
        edges = cv2.Canny(ela_uint8, 30, 100)
        edge_mask = edges > 0
        smooth_mask = ~edge_mask

        if np.any(edge_mask) and np.any(smooth_mask):
            edge_ela = float(np.mean(ela_map[edge_mask]))
            smooth_ela = float(np.mean(ela_map[smooth_mask]))
            edge_contrast = abs(edge_ela - smooth_ela) / (max(edge_ela, smooth_ela) + 1e-8)
        else:
            edge_contrast = 0.5  # neutral

        # --- Metric 4: Regional consistency (new in v2) ---
        # Divide image into quadrants and check if ELA is suspiciously uniform
        quad_h, quad_w = h // 2, w // 2
        quadrants = [
            ela_map[:quad_h, :quad_w],
            ela_map[:quad_h, quad_w:],
            ela_map[quad_h:, :quad_w],
            ela_map[quad_h:, quad_w:],
        ]
        quad_means = [float(np.mean(q)) for q in quadrants]
        quad_range = max(quad_means) - min(quad_means)

        # --- Scoring ---
        score = 0.0

        # Low coefficient of variation = uniform = suspicious
        if cv < 0.3:
            score += 0.35
        elif cv < 0.5:
            score += 0.20
        elif cv < 0.7:
            score += 0.10

        # Low block variance = uniform = suspicious
        if block_variance < 1.0:
            score += 0.35
        elif block_variance < 2.0:
            score += 0.20
        elif block_variance < 3.0:
            score += 0.10

        # Low edge contrast = uniform = suspicious
        if edge_contrast < 0.15:
            score += 0.30
        elif edge_contrast < 0.25:
            score += 0.15
        elif edge_contrast < 0.35:
            score += 0.05

        score = min(1.0, score)

        # --- Compression attenuation ---
        # For heavily compressed images, the ELA signal is inherently unreliable
        # because recompression noise dominates the real signal.
        # Scale down the score based on estimated quality.
        compression_note = ""
        if estimated_quality is not None:
            if estimated_quality <= 60:
                # Very heavy compression -- ELA is basically noise
                score *= 0.3
                compression_note = f"Heavy compression (Q={estimated_quality}) -- ELA unreliable"
            elif estimated_quality <= 75:
                # Moderate compression -- reduce confidence
                score *= 0.55
                compression_note = f"Moderate compression (Q={estimated_quality}) -- ELA dampened"
            elif estimated_quality <= 85:
                score *= 0.8
                compression_note = f"Light compression (Q={estimated_quality}) -- minor dampening"

        details = {
            "ela_mean": round(ela_mean, 3),
            "ela_std": round(ela_std, 3),
            "coefficient_of_variation": round(cv, 3),
            "block_variance": round(block_variance, 3),
            "edge_contrast": round(edge_contrast, 3),
            "quadrant_range": round(quad_range, 3),
            "resave_quality": resave_quality,
            "anomaly_score": round(score, 3),
        }
        if compression_note:
            details["compression_note"] = compression_note

        return score, details
