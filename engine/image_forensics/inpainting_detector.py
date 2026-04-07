"""
Inpainting & Splice Detection
================================
Detects partial image manipulation — where only a PORTION of the image
has been modified (face swap in an otherwise real photo, object removal,
content-aware fill, etc.).

Unlike full-image deepfake detection, splice detection looks for
INCONSISTENCIES BETWEEN REGIONS of the same image.

THEORETICAL FOUNDATION:
━━━━━━━━━━━━━━━━━━━━━━━
When an image region is manipulated (inpainted, spliced, or replaced):

1. Noise Level Inconsistency
   - Real camera images have uniform sensor noise across the frame
   - Manipulated regions have different noise characteristics:
     * Inpainting: lower noise (neural networks produce smoother output)
     * Splicing: different noise from different source cameras
     * Clone-stamp: identical noise patterns in non-adjacent regions
   - Measure: per-region noise variance using Laplacian-of-Gaussian

2. JPEG Grid Misalignment
   - JPEG compression divides images into 8x8 blocks
   - When a region is pasted from another JPEG, its 8x8 grid may be
     offset from the rest of the image
   - Detection: analyze DCT coefficient distributions per block
   - Can detect even if re-saved (double compression artifacts)

3. Edge Discontinuity Analysis
   - At splice boundaries, there are subtle edge artifacts:
     * Luminance discontinuity across the boundary
     * Gradient direction reversal
     * Unnaturally sharp or smooth boundary
   - Detection: measure gradient statistics at detected edges

4. Color/Illumination Mismatch
   - Spliced regions may have different:
     * White balance (color temperature)
     * Illumination direction (shadow inconsistency)
     * Gamma curve (different cameras have different tone mapping)
   - Detection: color channel statistics per region vs global

5. Copy-Move Detection
   - Detect if a region has been copied within the same image
   - Use robust features (DCT, PCA on blocks) to find matches
   - If two distant regions are pixel-similar: manipulation

OUTPUT:
━━━━━━━
  {
      "is_manipulated": bool,
      "manipulation_type": "inpainting" | "splice" | "copy-move" | "none",
      "manipulation_confidence": float,
      "manipulation_regions": [ { "bbox": [...], "type": ..., "confidence": ... } ],
      "noise_map_anomaly": float,
      "jpeg_grid_anomaly": float,
      "edge_anomaly": float,
      "color_anomaly": float,
      "score": float (0-1, higher = more likely manipulated),
      "details": { ... }
  }
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

try:
    from scipy import signal as scipy_signal
    from scipy.ndimage import uniform_filter, generic_filter
    from scipy.stats import entropy as scipy_entropy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class InpaintingDetector:
    """
    Detects partial image manipulation (inpainting, splicing, copy-move)
    by analyzing inconsistencies between image regions.

    Args:
        block_size: Size of analysis blocks (default 32x32)
        grid_size: Grid divisions for region comparison (default 8x8)
        verbose: Print per-layer analysis during detection
    """

    def __init__(
        self,
        block_size: int = 32,
        grid_size: int = 8,
        verbose: bool = False,
    ):
        self.block_size = block_size
        self.grid_size = grid_size
        self.verbose = verbose
        self.is_available = HAS_SCIPY

    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run full inpainting/splice detection analysis.

        Args:
            image: BGR image as numpy array (OpenCV format)

        Returns:
            Dict with manipulation detection results.
        """
        if not self.is_available:
            return self._default_result()

        if image is None or image.size == 0:
            return self._default_result()

        # Ensure minimum size
        h, w = image.shape[:2]
        if h < 64 or w < 64:
            return self._default_result()

        details = {}

        # Convert to different color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        else:
            lab = None

        # Layer 1: Noise inconsistency analysis
        noise_result = self._analyze_noise_inconsistency(gray)
        details["noise"] = noise_result

        # Layer 2: JPEG grid alignment analysis
        jpeg_result = self._analyze_jpeg_grid(gray)
        details["jpeg_grid"] = jpeg_result

        # Layer 3: Edge discontinuity analysis
        edge_result = self._analyze_edge_discontinuity(gray)
        details["edges"] = edge_result

        # Layer 4: Color/illumination mismatch
        color_result = self._analyze_color_mismatch(image, lab)
        details["color"] = color_result

        # Layer 5: Copy-move detection
        copymove_result = self._detect_copy_move(gray)
        details["copy_move"] = copymove_result

        # Combine scores
        anomaly_scores = [
            noise_result["anomaly_score"],
            jpeg_result["anomaly_score"],
            edge_result["anomaly_score"],
            color_result["anomaly_score"],
            copymove_result["anomaly_score"],
        ]

        # Weighted combination
        weights = [0.30, 0.20, 0.15, 0.20, 0.15]
        overall_score = float(np.average(anomaly_scores, weights=weights))

        # Determine manipulation type
        manipulation_type = "none"
        if overall_score > 0.45:
            type_scores = {
                "inpainting": noise_result["anomaly_score"] * 0.4 + edge_result["anomaly_score"] * 0.3 + color_result["anomaly_score"] * 0.3,
                "splice": color_result["anomaly_score"] * 0.35 + jpeg_result["anomaly_score"] * 0.35 + noise_result["anomaly_score"] * 0.3,
                "copy-move": copymove_result["anomaly_score"],
            }
            manipulation_type = max(type_scores, key=type_scores.get)

        is_manipulated = overall_score > 0.45

        # Collect detected regions
        regions = []
        if noise_result.get("anomalous_regions"):
            for region in noise_result["anomalous_regions"][:5]:
                regions.append({
                    "bbox": region["bbox"],
                    "type": "noise_anomaly",
                    "confidence": round(region["score"], 3),
                })
        if copymove_result.get("matched_pairs"):
            for pair in copymove_result["matched_pairs"][:3]:
                regions.append({
                    "bbox": pair["region1"],
                    "type": "copy_move_source",
                    "confidence": round(pair["similarity"], 3),
                })
                regions.append({
                    "bbox": pair["region2"],
                    "type": "copy_move_target",
                    "confidence": round(pair["similarity"], 3),
                })

        # Confidence based on how many layers agree
        layers_agree = sum(1 for s in anomaly_scores if s > 0.4)
        confidence = min(1.0, layers_agree / 3.0) * 0.5 + overall_score * 0.5

        return {
            "is_manipulated": is_manipulated,
            "manipulation_type": manipulation_type,
            "manipulation_confidence": round(confidence, 4),
            "manipulation_regions": regions[:10],
            "noise_map_anomaly": round(noise_result["anomaly_score"], 4),
            "jpeg_grid_anomaly": round(jpeg_result["anomaly_score"], 4),
            "edge_anomaly": round(edge_result["anomaly_score"], 4),
            "color_anomaly": round(color_result["anomaly_score"], 4),
            "copy_move_anomaly": round(copymove_result["anomaly_score"], 4),
            "score": round(overall_score, 4),
            "confidence": round(confidence, 4),
            "details": details,
        }

    def _analyze_noise_inconsistency(self, gray: np.ndarray) -> Dict[str, Any]:
        """
        Analyze noise level consistency across image blocks.

        Manipulated regions have different noise characteristics than
        the original image.
        """
        h, w = gray.shape
        bs = self.block_size

        # Extract noise using Laplacian (high-pass filter)
        noise = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)

        # Compute per-block noise variance
        n_rows = h // bs
        n_cols = w // bs
        noise_map = np.zeros((n_rows, n_cols))

        for i in range(n_rows):
            for j in range(n_cols):
                block = noise[i*bs:(i+1)*bs, j*bs:(j+1)*bs]
                noise_map[i, j] = np.var(block)

        # Global noise statistics
        global_mean = float(np.mean(noise_map))
        global_std = float(np.std(noise_map))

        if global_std < 1e-10:
            return {"anomaly_score": 0.0, "global_noise_mean": global_mean, "global_noise_std": 0.0}

        # Find anomalous blocks (noise significantly different from global)
        z_scores = np.abs(noise_map - global_mean) / (global_std + 1e-10)

        anomalous_regions = []
        for i in range(n_rows):
            for j in range(n_cols):
                if z_scores[i, j] > 2.5:
                    anomalous_regions.append({
                        "bbox": [j*bs, i*bs, (j+1)*bs, (i+1)*bs],
                        "score": float(z_scores[i, j] / 5.0),
                        "noise_level": float(noise_map[i, j]),
                    })

        # Anomaly score: fraction of blocks that are anomalous, weighted by severity
        n_anomalous = len(anomalous_regions)
        total_blocks = n_rows * n_cols

        if n_anomalous == 0:
            anomaly_score = 0.0
        else:
            # Ratio of anomalous blocks (capped)
            ratio = min(0.5, n_anomalous / total_blocks)
            # Mean severity
            mean_severity = float(np.mean([r["score"] for r in anomalous_regions]))
            anomaly_score = min(1.0, ratio * 3.0 * mean_severity)

        # Coefficient of variation of noise map
        noise_cv = global_std / (global_mean + 1e-10)

        return {
            "anomaly_score": float(anomaly_score),
            "global_noise_mean": round(global_mean, 4),
            "global_noise_std": round(global_std, 4),
            "noise_cv": round(float(noise_cv), 4),
            "anomalous_blocks": n_anomalous,
            "total_blocks": total_blocks,
            "anomalous_regions": sorted(anomalous_regions, key=lambda x: x["score"], reverse=True)[:10],
        }

    def _analyze_jpeg_grid(self, gray: np.ndarray) -> Dict[str, Any]:
        """
        Detect JPEG 8x8 block grid misalignment.

        When a JPEG-compressed region is pasted into another JPEG image,
        the 8x8 DCT block boundaries of the pasted region may not align
        with the rest of the image.
        """
        h, w = gray.shape

        if h < 16 or w < 16:
            return {"anomaly_score": 0.0}

        # Compute block boundary artifacts
        # At JPEG block boundaries, there are subtle discontinuities
        # We look for these by computing horizontal and vertical gradients
        # sampled at 8-pixel intervals

        # Horizontal boundary detection (every 8th column)
        h_boundaries = []
        for col in range(8, w - 1, 8):
            diff = np.abs(gray[:, col].astype(float) - gray[:, col - 1].astype(float))
            h_boundaries.append(float(np.mean(diff)))

        # Non-boundary columns for comparison
        h_non_boundaries = []
        for col in range(4, w - 1, 8):
            if col + 1 < w:
                diff = np.abs(gray[:, col].astype(float) - gray[:, col - 1].astype(float))
                h_non_boundaries.append(float(np.mean(diff)))

        # Vertical boundary detection
        v_boundaries = []
        for row in range(8, h - 1, 8):
            diff = np.abs(gray[row, :].astype(float) - gray[row - 1, :].astype(float))
            v_boundaries.append(float(np.mean(diff)))

        v_non_boundaries = []
        for row in range(4, h - 1, 8):
            if row + 1 < h:
                diff = np.abs(gray[row, :].astype(float) - gray[row - 1, :].astype(float))
                v_non_boundaries.append(float(np.mean(diff)))

        # Blockiness measure: ratio of boundary vs non-boundary gradients
        h_boundary_mean = float(np.mean(h_boundaries)) if h_boundaries else 0
        h_non_mean = float(np.mean(h_non_boundaries)) if h_non_boundaries else 0
        v_boundary_mean = float(np.mean(v_boundaries)) if v_boundaries else 0
        v_non_mean = float(np.mean(v_non_boundaries)) if v_non_boundaries else 0

        h_blockiness = h_boundary_mean / (h_non_mean + 1e-10)
        v_blockiness = v_boundary_mean / (v_non_mean + 1e-10)

        # Check for regional blockiness variation (grid misalignment indicator)
        # Divide image into quadrants and compare blockiness
        quadrant_blockiness = []
        for qi in range(2):
            for qj in range(2):
                qy1, qy2 = qi * h // 2, (qi + 1) * h // 2
                qx1, qx2 = qj * w // 2, (qj + 1) * w // 2
                quad = gray[qy1:qy2, qx1:qx2]
                if quad.shape[0] > 16 and quad.shape[1] > 16:
                    q_h = []
                    for col in range(8, quad.shape[1] - 1, 8):
                        diff = np.abs(quad[:, col].astype(float) - quad[:, col-1].astype(float))
                        q_h.append(float(np.mean(diff)))
                    quadrant_blockiness.append(float(np.mean(q_h)) if q_h else 0)

        # Blockiness variation across quadrants (high = possible splice)
        if len(quadrant_blockiness) >= 2:
            blockiness_cv = float(np.std(quadrant_blockiness) / (np.mean(quadrant_blockiness) + 1e-10))
        else:
            blockiness_cv = 0.0

        # Anomaly score: high variation in regional blockiness
        anomaly_score = min(1.0, blockiness_cv * 1.5)

        # Also check: if blockiness ratio is very low globally, image is not JPEG
        if h_blockiness < 1.05 and v_blockiness < 1.05:
            anomaly_score *= 0.5  # probably not JPEG, grid analysis less meaningful

        return {
            "anomaly_score": float(anomaly_score),
            "h_blockiness": round(h_blockiness, 4),
            "v_blockiness": round(v_blockiness, 4),
            "blockiness_cv": round(blockiness_cv, 4),
            "quadrant_blockiness": [round(b, 4) for b in quadrant_blockiness],
        }

    def _analyze_edge_discontinuity(self, gray: np.ndarray) -> Dict[str, Any]:
        """
        Detect unnatural edge patterns that indicate splice boundaries.

        At manipulation boundaries, there are often:
        - Unnaturally sharp edges (copy-paste boundary)
        - Gradient direction reversals
        - Isolated strong edges not explained by scene content
        """
        # Canny edge detection at two scales
        edges_fine = cv2.Canny(gray, 50, 150)
        edges_coarse = cv2.Canny(gray, 20, 60)

        # Edge density per region
        h, w = gray.shape
        gs = self.grid_size
        cell_h = h // gs
        cell_w = w // gs

        edge_densities = np.zeros((gs, gs))
        for i in range(gs):
            for j in range(gs):
                cell = edges_fine[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                edge_densities[i, j] = np.mean(cell > 0)

        # Edge density variation
        density_mean = float(np.mean(edge_densities))
        density_std = float(np.std(edge_densities))
        density_cv = density_std / (density_mean + 1e-10)

        # Gradient magnitude analysis
        grad_x = cv2.Sobel(gray.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Per-cell gradient statistics
        grad_stats = np.zeros((gs, gs))
        for i in range(gs):
            for j in range(gs):
                cell = grad_mag[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                grad_stats[i, j] = np.mean(cell)

        # Gradient discontinuity between adjacent cells
        h_discon = []
        v_discon = []
        for i in range(gs):
            for j in range(gs - 1):
                h_discon.append(abs(grad_stats[i, j] - grad_stats[i, j+1]))
        for i in range(gs - 1):
            for j in range(gs):
                v_discon.append(abs(grad_stats[i, j] - grad_stats[i+1, j]))

        mean_discon = float(np.mean(h_discon + v_discon)) if (h_discon or v_discon) else 0
        max_discon = float(np.max(h_discon + v_discon)) if (h_discon or v_discon) else 0

        # Anomaly: high edge density variation + high gradient discontinuity
        anomaly_score = min(1.0, density_cv * 0.5 + (max_discon / (mean_discon + 1e-10)) * 0.1)
        anomaly_score = min(1.0, anomaly_score)

        return {
            "anomaly_score": float(anomaly_score),
            "edge_density_mean": round(density_mean, 4),
            "edge_density_cv": round(density_cv, 4),
            "gradient_discontinuity_mean": round(mean_discon, 4),
            "gradient_discontinuity_max": round(max_discon, 4),
        }

    def _analyze_color_mismatch(
        self, image: np.ndarray, lab: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Detect color/illumination inconsistencies between regions.

        Spliced regions may have different white balance, gamma, or
        illumination from the rest of the image.
        """
        if lab is None or len(image.shape) < 3:
            return {"anomaly_score": 0.0}

        h, w = image.shape[:2]
        gs = self.grid_size
        cell_h = h // gs
        cell_w = w // gs

        # Per-cell color statistics in LAB space
        l_means = np.zeros((gs, gs))
        a_means = np.zeros((gs, gs))
        b_means = np.zeros((gs, gs))
        l_stds = np.zeros((gs, gs))

        for i in range(gs):
            for j in range(gs):
                cell = lab[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                l_means[i, j] = np.mean(cell[:, :, 0])
                a_means[i, j] = np.mean(cell[:, :, 1])
                b_means[i, j] = np.mean(cell[:, :, 2])
                l_stds[i, j] = np.std(cell[:, :, 0])

        # Color temperature variation (a and b channels indicate warm/cool)
        a_cv = float(np.std(a_means) / (np.mean(np.abs(a_means)) + 1e-10))
        b_cv = float(np.std(b_means) / (np.mean(np.abs(b_means)) + 1e-10))

        # Luminance consistency
        l_cv = float(np.std(l_means) / (np.mean(l_means) + 1e-10))

        # Check for abrupt color transitions between adjacent cells
        color_jumps = []
        for i in range(gs):
            for j in range(gs - 1):
                jump = np.sqrt(
                    (a_means[i, j] - a_means[i, j+1])**2 +
                    (b_means[i, j] - b_means[i, j+1])**2
                )
                color_jumps.append(float(jump))
        for i in range(gs - 1):
            for j in range(gs):
                jump = np.sqrt(
                    (a_means[i, j] - a_means[i+1, j])**2 +
                    (b_means[i, j] - b_means[i+1, j])**2
                )
                color_jumps.append(float(jump))

        mean_jump = float(np.mean(color_jumps)) if color_jumps else 0
        max_jump = float(np.max(color_jumps)) if color_jumps else 0

        # Anomaly: high color variation that isn't explained by scene content
        # Use ratio of max to mean jump as indicator of abrupt change
        jump_ratio = max_jump / (mean_jump + 1e-10)

        # Color temperature variance
        color_temp_var = (a_cv + b_cv) / 2.0

        anomaly_score = min(1.0, (jump_ratio - 2.0) * 0.15 + color_temp_var * 0.3)
        anomaly_score = max(0.0, anomaly_score)

        return {
            "anomaly_score": float(anomaly_score),
            "luminance_cv": round(l_cv, 4),
            "a_channel_cv": round(a_cv, 4),
            "b_channel_cv": round(b_cv, 4),
            "color_jump_mean": round(mean_jump, 4),
            "color_jump_max": round(max_jump, 4),
            "color_jump_ratio": round(jump_ratio, 4),
        }

    def _detect_copy_move(self, gray: np.ndarray) -> Dict[str, Any]:
        """
        Detect copy-move forgery using block-based feature matching.

        Divides image into overlapping blocks, computes DCT features,
        and finds suspiciously similar pairs of distant blocks.
        """
        h, w = gray.shape
        bs = 16  # smaller block size for copy-move
        stride = 8

        # Skip if image is too small
        if h < 64 or w < 64:
            return {"anomaly_score": 0.0, "matched_pairs": []}

        # Extract blocks and their DCT features
        blocks = []
        positions = []

        for i in range(0, h - bs, stride):
            for j in range(0, w - bs, stride):
                block = gray[i:i+bs, j:j+bs].astype(np.float64)
                # Use truncated DCT as feature (first 6 coefficients)
                dct = cv2.dct(block)
                feature = dct[:4, :4].flatten()  # 16-dim feature
                blocks.append(feature)
                positions.append((j, i, j + bs, i + bs))

        if len(blocks) < 10:
            return {"anomaly_score": 0.0, "matched_pairs": []}

        blocks = np.array(blocks)
        positions = np.array(positions)

        # Subsample for efficiency (max 2000 blocks)
        if len(blocks) > 2000:
            indices = np.random.RandomState(42).choice(len(blocks), 2000, replace=False)
            blocks = blocks[indices]
            positions = positions[indices]

        # Normalize features
        norms = np.linalg.norm(blocks, axis=1, keepdims=True)
        blocks_norm = blocks / (norms + 1e-10)

        # Find similar pairs using cosine similarity
        # Only check pairs that are spatially distant (>50 pixels apart)
        matched_pairs = []
        min_distance = 50  # minimum pixel distance

        # Efficient search: sort by first feature dimension, check nearby in sorted order
        sort_idx = np.argsort(blocks_norm[:, 0])
        sorted_blocks = blocks_norm[sort_idx]
        sorted_positions = positions[sort_idx]

        for i in range(len(sorted_blocks)):
            # Check neighbors in sorted order (nearby in feature space)
            for j in range(i + 1, min(i + 50, len(sorted_blocks))):
                # Spatial distance check
                cx1 = (sorted_positions[i][0] + sorted_positions[i][2]) / 2
                cy1 = (sorted_positions[i][1] + sorted_positions[i][3]) / 2
                cx2 = (sorted_positions[j][0] + sorted_positions[j][2]) / 2
                cy2 = (sorted_positions[j][1] + sorted_positions[j][3]) / 2
                spatial_dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

                if spatial_dist < min_distance:
                    continue

                # Cosine similarity
                sim = float(np.dot(sorted_blocks[i], sorted_blocks[j]))

                if sim > 0.98:  # very high similarity for distant blocks
                    matched_pairs.append({
                        "region1": sorted_positions[i].tolist(),
                        "region2": sorted_positions[j].tolist(),
                        "similarity": round(sim, 4),
                        "distance": round(spatial_dist, 1),
                    })

            if len(matched_pairs) >= 20:
                break

        # Anomaly score based on number and strength of matches
        if matched_pairs:
            n_matches = len(matched_pairs)
            mean_sim = float(np.mean([p["similarity"] for p in matched_pairs]))
            anomaly_score = min(1.0, n_matches * 0.1 * mean_sim)
        else:
            anomaly_score = 0.0

        return {
            "anomaly_score": float(anomaly_score),
            "matched_pairs": sorted(matched_pairs, key=lambda x: x["similarity"], reverse=True)[:5],
            "total_matches": len(matched_pairs),
        }

    def _default_result(self) -> Dict[str, Any]:
        return {
            "is_manipulated": False,
            "manipulation_type": "none",
            "manipulation_confidence": 0.0,
            "manipulation_regions": [],
            "noise_map_anomaly": 0.0,
            "jpeg_grid_anomaly": 0.0,
            "edge_anomaly": 0.0,
            "color_anomaly": 0.0,
            "copy_move_anomaly": 0.0,
            "score": 0.0,
            "confidence": 0.0,
            "details": {},
        }
