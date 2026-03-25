"""
Advanced Image Forensics — Spatial Rich Model (SRM) Noise Analysis
==================================================================
State-of-the-art AI generators (Midjourney v6, Flux) mimic semantic structures perfectly
but fail to synthesize physically accurate, continuous camera sensor noise at the pixel level.

This module applies SRM high-pass kernels to strip out the image content, leaving only
invisible noise residuals. We dissect these residuals in a granular grid (patch-by-patch)
to detect regions of suspicious smoothness, extreme discontinuities, and unnatural uniformity.
"""

import numpy as np
import cv2
import logging
from scipy.signal import convolve2d
from typing import Tuple, Dict, Any, List

logger = logging.getLogger(__name__)

class NoiseAnalyzer:
    """Analyzes noise patterns to distinguish real photos from SOTA AI-generated images using SRM."""

    def __init__(self, target_size: int = 1024):
        self.target_size = target_size
        
        # Spatial Rich Model High-Pass Kernels
        # These kernels suppress image contours/color and isolate micro-noise.
        # SRM KB Kernel (5x5)
        self.srm_kb = np.array([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8, -12, 8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=np.float32) / 12.0
        
        # SRM KV Kernel (3x3)
        self.srm_kv = np.array([
            [-1,  2, -1],
            [ 2, -4,  2],
            [-1,  2, -1]
        ], dtype=np.float32) / 4.0

    def extract_srm_residual(self, image_gray: np.ndarray) -> np.ndarray:
        """
        Applies SRM high-pass kernels to extract noise residuals.
        """
        # Convolve with KB kernel to get strong noise artifact map
        residual_kb = convolve2d(image_gray, self.srm_kb, mode='same', boundary='symm')
        
        # Combine with a simpler KV kernel to catch edge-adjacent noise
        residual_kv = convolve2d(image_gray, self.srm_kv, mode='same', boundary='symm')
        
        # Average the magnitude of the residuals
        srm_noise = (np.abs(residual_kb) + np.abs(residual_kv)) / 2.0
        return srm_noise

    def analyze(self, image: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Performs in-depth patch-based SRM variance mapping.
        """
        try:
            # Upscale if too small, or cap at target size
            h, w = image.shape[:2]
            if w > self.target_size or h > self.target_size:
                ratio = self.target_size / max(h, w)
                image = cv2.resize(image, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LANCZOS4)
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            
            # Step 1: Extract pure noise residuals (destroy the image semantics)
            noise_map = self.extract_srm_residual(gray)
            
            # Step 2: Divide into robust 32x32 patches for analysis
            patch_size = 32
            h, w = noise_map.shape
            
            patch_variances = []
            patch_means = []
            
            # Extract statistics for every single local patch
            for y in range(0, h - patch_size + 1, patch_size):
                for x in range(0, w - patch_size + 1, patch_size):
                    patch = noise_map[y:y+patch_size, x:x+patch_size]
                    var = np.var(patch)
                    mean = np.mean(patch)
                    
                    # Ignore pure black/featureless corner patches
                    if mean > 0.1: 
                        patch_variances.append(var)
                        patch_means.append(mean)

            if not patch_variances:
                return 0.5, {"status": "error", "reason": "No valid noise patches"}

            variances = np.array(patch_variances)
            means = np.array(patch_means)
            
            # Sub-Metric A: Global Uniformity
            # Deepfakes often have unnaturally low median noise globally because of generation-denoising.
            median_var = float(np.median(variances))
            
            # Sub-Metric B: Variance Volatility (Inconsistency)
            # AI struggles to balance foreground vs background noise. Real photos have consistent PRNU sensor noise.
            # We measure how wildly the noise variance swings from block to block.
            var_std = float(np.std(variances))
            volatility_index = var_std / (median_var + 1e-6)
            
            # Sub-Metric C: Dead Zones
            # Extreme smoothness (e.g., highly compressed or AI "perfect skin" / flat backgrounds)
            # Count the percentage of blocks with practically zero high-frequency variance.
            dead_zones_pct = float(np.sum(variances < 0.5) / len(variances))

            # --- SOTA Anomaly Scoring Logic ---
            score = 0.0
            
            # 1. Heavily penalize absolute dead zones (plastic/CGI smoothness)
            if dead_zones_pct > 0.40:
                score += 0.45
            elif dead_zones_pct > 0.20:
                score += 0.25
                
            # 2. Penalize completely uniform noise (lack of natural variance)
            if median_var < 1.0:
                score += 0.35
            elif median_var < 3.0:
                score += 0.15
                
            # 3. Penalize extreme volatility (sharp disconnect between objects in image)
            # If volatility is massive, parts of the image were generated with different localized diffusion steps.
            if volatility_index > 5.0:
                score += 0.20
            elif volatility_index > 3.0:
                score += 0.10
                
            # Final anomaly clipping
            final_score = min(1.0, score)
            
            details = {
                "srm_median_variance": round(median_var, 4),
                "srm_variance_std": round(var_std, 4),
                "volatility_index": round(volatility_index, 4),
                "dead_zones_pct": round(dead_zones_pct, 4),
                "patches_analyzed": len(variances),
                "anomaly_score": round(final_score, 3)
            }
            
            return final_score, details
            
        except Exception as e:
            logger.error(f"SRM Noise Analysis Error: {e}")
            return 0.0, {"status": "error", "reason": str(e)}
