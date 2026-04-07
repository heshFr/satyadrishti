"""
Shadow & Lighting Physics Verifier
====================================
Estimates light source direction per frame and checks temporal consistency
to detect AI-generated video.

THEORETICAL FOUNDATION:
━━━━━━━━━━━━━━━━━━━━━━━
In real-world video:
  - Light sources are physically fixed (sun, lamps, etc.)
  - The apparent lighting direction changes only due to camera motion
    (which is smooth and predictable)
  - Shadow directions are consistent with the dominant light source
  - Color temperature is consistent across the scene
  - Specular highlights move consistently with surface orientation

In AI-generated video:
  - Lighting direction may shift between frames (no physical light model)
  - Shadow directions can be inconsistent
  - Color temperature may vary across regions or frames
  - Specular highlights may appear/disappear or move unnaturally
  - "Ambient" lighting without clear directional source

ANALYSIS LAYERS:
━━━━━━━━━━━━━━━━
1. Dominant Light Direction Estimation
   - Estimate per-frame using image gradients (simplified photometric stereo)
   - Track direction over time → should be smooth

2. Shadow Direction Consistency
   - Detect shadow edges (strong luminance gradients near dark regions)
   - Check if shadow directions agree across the frame

3. Color Temperature Stability
   - Estimate white balance / color temperature per frame
   - Real: consistent (same light source), AI: may vary

4. Luminance Distribution Consistency
   - Histogram of luminance values per frame
   - Real: smooth evolution, AI: jumpy distribution changes

5. Specular Highlight Tracking
   - Detect bright specular regions
   - Track their position and intensity → should be physically consistent
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, List


class LightingConsistencyAnalyzer:
    """
    Analyzes lighting and shadow consistency across video frames
    to detect AI-generated content.
    """

    def __init__(self, sample_frames: int = 32, verbose: bool = False):
        self.sample_frames = sample_frames
        self.verbose = verbose

    def analyze(self, video_path: str) -> Dict[str, Any]:
        """
        Run full lighting consistency analysis.

        Returns dict with ai_probability and detailed metrics.
        """
        frames = self._extract_frames(video_path)
        if len(frames) < 4:
            return {
                "ai_probability": 0.5,
                "metrics": {"error": "Too few frames"},
            }

        # Convert to different color spaces
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        lab_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2LAB) for f in frames]

        # Layer 1: Light Direction Consistency
        light_score, light_metrics = self._analyze_light_direction(gray_frames)

        # Layer 2: Shadow Direction Consistency
        shadow_score, shadow_metrics = self._analyze_shadow_consistency(gray_frames)

        # Layer 3: Color Temperature Stability
        color_score, color_metrics = self._analyze_color_temperature(frames, lab_frames)

        # Layer 4: Luminance Distribution Consistency
        lum_score, lum_metrics = self._analyze_luminance_distribution(gray_frames)

        # Layer 5: Specular Highlight Tracking
        spec_score, spec_metrics = self._analyze_specular_highlights(gray_frames)

        # Weighted ensemble
        weights = {
            "light_direction": (light_score, 1.5),
            "shadow_consistency": (shadow_score, 1.2),
            "color_temperature": (color_score, 1.3),
            "luminance_distribution": (lum_score, 1.0),
            "specular_highlights": (spec_score, 0.8),
        }

        total_w = sum(w for _, w in weights.values())
        ai_probability = sum(s * w for s, w in weights.values()) / total_w
        ai_probability = float(np.clip(ai_probability, 0, 1))

        if self.verbose:
            for name, (score, _) in weights.items():
                print(f"  [Lighting] {name}: {score:.4f}")

        return {
            "ai_probability": round(ai_probability, 4),
            "layer_scores": {
                name: round(score, 4) for name, (score, _) in weights.items()
            },
            "metrics": {
                "frames_analyzed": len(frames),
                "light": light_metrics,
                "shadow": shadow_metrics,
                "color": color_metrics,
                "luminance": lum_metrics,
                "specular": spec_metrics,
            },
        }

    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n = min(self.sample_frames, total)
        indices = np.linspace(0, total - 1, n, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                if max(h, w) > 640:
                    scale = 640.0 / max(h, w)
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)
                frames.append(frame)
        cap.release()
        return frames

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 1: Light Direction Estimation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_light_direction(
        self, gray_frames: List[np.ndarray]
    ) -> Tuple[float, Dict]:
        """
        Estimate dominant light direction per frame using gradient analysis.

        Method:
        1. Compute Sobel gradients (dx, dy) per frame
        2. Weight gradients by magnitude (strong gradients = illumination boundaries)
        3. Compute weighted average gradient direction → light direction estimate
        4. Track direction across frames → should be smooth for real video

        The key insight: in a real scene, the gradient field is dominated by
        the primary light source direction. AI generators don't maintain this
        consistency because they don't model physical lighting.
        """
        light_angles = []

        for frame in gray_frames:
            # Compute gradients
            gx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
            gy = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)

            # Gradient magnitude as weight
            mag = np.sqrt(gx ** 2 + gy ** 2)

            # Only use strong gradients (top 20% by magnitude)
            threshold = np.percentile(mag, 80)
            strong_mask = mag > threshold

            if strong_mask.sum() < 100:
                continue

            # Weighted average direction
            weighted_gx = float(np.sum(gx[strong_mask] * mag[strong_mask]))
            weighted_gy = float(np.sum(gy[strong_mask] * mag[strong_mask]))

            angle = np.arctan2(weighted_gy, weighted_gx)
            light_angles.append(angle)

        if len(light_angles) < 3:
            return 0.5, {"error": "insufficient gradient data"}

        light_angles = np.array(light_angles)

        # Angular differences between consecutive frames
        angle_diffs = []
        for i in range(len(light_angles) - 1):
            diff = light_angles[i + 1] - light_angles[i]
            # Normalize to [-π, π]
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            angle_diffs.append(abs(diff))

        angle_diffs = np.array(angle_diffs)

        mean_diff = float(np.mean(angle_diffs))
        max_diff = float(np.max(angle_diffs))
        std_diff = float(np.std(angle_diffs))

        # Circular variance of all light directions
        sin_sum = float(np.sum(np.sin(light_angles)))
        cos_sum = float(np.sum(np.cos(light_angles)))
        r = np.sqrt(sin_sum ** 2 + cos_sum ** 2) / len(light_angles)
        circular_variance = 1.0 - r  # 0 = all same direction, 1 = uniform

        # Score
        # mean_diff: real ~0.01-0.08, AI ~0.08-0.30+
        diff_score = np.clip((mean_diff - 0.03) / 0.20, 0, 1)

        # circular_variance: real ~0.01-0.10, AI ~0.10-0.40+
        var_score = np.clip((circular_variance - 0.03) / 0.25, 0, 1)

        # max_diff: real ~0.05-0.15, AI ~0.15-0.50+
        max_score = np.clip((max_diff - 0.08) / 0.30, 0, 1)

        combined = diff_score * 0.35 + var_score * 0.40 + max_score * 0.25

        metrics = {
            "mean_angle_diff": round(mean_diff, 4),
            "max_angle_diff": round(max_diff, 4),
            "std_angle_diff": round(std_diff, 4),
            "circular_variance": round(circular_variance, 4),
            "n_frames": len(light_angles),
        }

        return float(np.clip(combined, 0, 1)), metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 2: Shadow Direction Consistency
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_shadow_consistency(
        self, gray_frames: List[np.ndarray]
    ) -> Tuple[float, Dict]:
        """
        Analyze shadow edge directions for consistency.

        Shadows are produced by occluders blocking light. In a real scene with
        a dominant light source, all shadow edges should have a consistent
        orientation (perpendicular to light direction). AI generators often
        produce shadows that point in different directions within the same frame.

        Method:
        1. Detect dark regions (potential shadows)
        2. Find edges of these dark regions
        3. Compute orientation of shadow edges
        4. Measure within-frame consistency of shadow edge directions
        5. Track across frames
        """
        per_frame_consistency = []

        for frame in gray_frames:
            # Find dark regions (shadows tend to be in lower 25% of luminance)
            dark_threshold = np.percentile(frame, 25)
            dark_mask = (frame < dark_threshold).astype(np.uint8)

            # Find edges of dark regions
            dark_edges = cv2.Canny(dark_mask * 255, 50, 150)

            # Compute edge orientations using Sobel
            gx = cv2.Sobel(dark_edges, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(dark_edges, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(gx ** 2 + gy ** 2)

            strong = mag > np.percentile(mag, 90) if mag.max() > 0 else mag > 0
            if strong.sum() < 50:
                continue

            # Edge orientations
            angles = np.arctan2(gy[strong], gx[strong])

            # Circular variance of shadow edge directions
            sin_mean = np.mean(np.sin(2 * angles))  # ×2 because edges are undirected
            cos_mean = np.mean(np.cos(2 * angles))
            r = np.sqrt(sin_mean ** 2 + cos_mean ** 2)
            consistency = r  # 0 = random directions, 1 = all aligned

            per_frame_consistency.append(consistency)

        if len(per_frame_consistency) < 3:
            return 0.5, {"error": "insufficient shadow data"}

        avg_consistency = float(np.mean(per_frame_consistency))
        std_consistency = float(np.std(per_frame_consistency))

        # Also check temporal variance of shadow consistency
        # Real: consistent consistency (shadow directions stable)
        # AI: varying consistency (shadows randomly organized)
        temporal_var = std_consistency / (avg_consistency + 1e-6)

        # Score
        # avg_consistency: real ~0.3-0.6, AI ~0.1-0.3 (more random)
        consist_score = np.clip((0.4 - avg_consistency) / 0.25, 0, 1)

        # temporal_var: real ~0.1-0.3, AI ~0.3-0.8+
        var_score = np.clip((temporal_var - 0.15) / 0.45, 0, 1)

        combined = consist_score * 0.5 + var_score * 0.5

        metrics = {
            "avg_shadow_consistency": round(avg_consistency, 4),
            "std_shadow_consistency": round(std_consistency, 4),
            "temporal_variance": round(temporal_var, 4),
            "frames_with_shadows": len(per_frame_consistency),
        }

        return float(np.clip(combined, 0, 1)), metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 3: Color Temperature Stability
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_color_temperature(
        self, color_frames: List[np.ndarray],
        lab_frames: List[np.ndarray],
    ) -> Tuple[float, Dict]:
        """
        Analyze color temperature stability across frames.

        Real video: consistent white balance (camera's AWB algorithm
        produces smooth adjustments). AI: color temperature may shift
        randomly between frames.

        Uses LAB color space: L = lightness, A = green-red, B = blue-yellow.
        The A and B channels encode color temperature information.
        """
        a_means = []
        b_means = []
        ab_ratios = []  # Color temperature proxy

        for lab in lab_frames:
            # Global mean of A and B channels (128 = neutral in OpenCV LAB)
            a_mean = float(lab[:, :, 1].mean()) - 128
            b_mean = float(lab[:, :, 2].mean()) - 128
            a_means.append(a_mean)
            b_means.append(b_mean)
            ab_ratios.append(b_mean / (abs(a_mean) + 1e-6))

        a_means = np.array(a_means)
        b_means = np.array(b_means)
        ab_ratios = np.array(ab_ratios)

        # Temporal variance of color channels
        a_var = float(np.var(a_means))
        b_var = float(np.var(b_means))
        ab_ratio_var = float(np.var(ab_ratios))

        # Frame-to-frame differences
        a_diffs = np.abs(np.diff(a_means))
        b_diffs = np.abs(np.diff(b_means))

        mean_a_diff = float(np.mean(a_diffs))
        mean_b_diff = float(np.mean(b_diffs))
        max_a_diff = float(np.max(a_diffs))
        max_b_diff = float(np.max(b_diffs))

        # Also check spatial color temperature consistency per frame
        spatial_vars = []
        for lab in lab_frames:
            # Divide into 4 quadrants
            h, w = lab.shape[:2]
            quads = [
                lab[:h // 2, :w // 2],
                lab[:h // 2, w // 2:],
                lab[h // 2:, :w // 2],
                lab[h // 2:, w // 2:],
            ]
            quad_b_means = [float(q[:, :, 2].mean()) for q in quads]
            spatial_vars.append(float(np.var(quad_b_means)))

        avg_spatial_var = float(np.mean(spatial_vars))
        spatial_var_var = float(np.var(spatial_vars))

        # Score
        # ab_ratio_var: real ~0.01-0.10, AI ~0.10-0.50+
        ratio_score = np.clip((ab_ratio_var - 0.03) / 0.30, 0, 1)

        # mean color diffs: real ~0.1-0.5, AI ~0.5-2.0+
        diff_score = np.clip(((mean_a_diff + mean_b_diff) / 2 - 0.2) / 1.0, 0, 1)

        # spatial_var_var: real ~0.1-1.0, AI ~1.0-5.0+
        spatial_score = np.clip((spatial_var_var - 0.5) / 3.0, 0, 1)

        combined = ratio_score * 0.40 + diff_score * 0.35 + spatial_score * 0.25

        metrics = {
            "a_channel_variance": round(a_var, 4),
            "b_channel_variance": round(b_var, 4),
            "ab_ratio_variance": round(ab_ratio_var, 4),
            "mean_a_diff": round(mean_a_diff, 4),
            "mean_b_diff": round(mean_b_diff, 4),
            "avg_spatial_variance": round(avg_spatial_var, 4),
            "spatial_var_variance": round(spatial_var_var, 4),
        }

        return float(np.clip(combined, 0, 1)), metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 4: Luminance Distribution Consistency
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_luminance_distribution(
        self, gray_frames: List[np.ndarray]
    ) -> Tuple[float, Dict]:
        """
        Analyze temporal consistency of luminance histograms.

        Real video: luminance distribution evolves smoothly (gradual exposure
        changes, smooth lighting transitions).
        AI video: histogram may jump between frames as the generator produces
        frames with different brightness distributions.
        """
        histograms = []
        for frame in gray_frames:
            hist = cv2.calcHist([frame], [0], None, [64], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-6)
            histograms.append(hist)

        # Histogram distances between consecutive frames
        distances = []
        for i in range(len(histograms) - 1):
            # Bhattacharyya distance
            dist = cv2.compareHist(
                histograms[i].astype(np.float32),
                histograms[i + 1].astype(np.float32),
                cv2.HISTCMP_BHATTACHARYYA,
            )
            distances.append(dist)

        distances = np.array(distances)

        mean_dist = float(np.mean(distances))
        std_dist = float(np.std(distances))
        max_dist = float(np.max(distances))

        # Second-order: variance of distances (smoothness of distribution changes)
        dist_var = float(np.var(distances)) if len(distances) >= 2 else 0

        # Score
        # mean_dist: real ~0.01-0.04, AI ~0.04-0.12+
        dist_score = np.clip((mean_dist - 0.015) / 0.08, 0, 1)

        # dist_var: real ~0.0001-0.001, AI ~0.001-0.005+
        var_score = np.clip((dist_var - 0.0003) / 0.003, 0, 1)

        # max_dist: real ~0.03-0.08, AI ~0.08-0.20+
        max_score = np.clip((max_dist - 0.04) / 0.12, 0, 1)

        combined = dist_score * 0.40 + var_score * 0.30 + max_score * 0.30

        metrics = {
            "mean_hist_distance": round(mean_dist, 6),
            "std_hist_distance": round(std_dist, 6),
            "max_hist_distance": round(max_dist, 6),
            "hist_distance_variance": round(dist_var, 8),
        }

        return float(np.clip(combined, 0, 1)), metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 5: Specular Highlight Tracking
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_specular_highlights(
        self, gray_frames: List[np.ndarray]
    ) -> Tuple[float, Dict]:
        """
        Track specular highlights (bright reflections) across frames.

        Real specular highlights:
        - Move consistently with surface and camera motion
        - Maintain consistent brightness relative to surroundings
        - Follow physics of reflection (angle of incidence = angle of reflection)

        AI specular highlights:
        - May appear/disappear between frames
        - May jump positions unnaturally
        - Brightness may be inconsistent
        """
        highlight_counts = []
        highlight_intensities = []
        highlight_positions = []  # centroid of highlight region

        for frame in gray_frames:
            # Detect specular highlights: top 2% brightest pixels
            threshold = np.percentile(frame, 98)
            highlight_mask = (frame > threshold).astype(np.uint8)

            # Count highlight pixels
            n_highlight = int(highlight_mask.sum())
            total_pixels = frame.shape[0] * frame.shape[1]
            highlight_counts.append(n_highlight / total_pixels)

            # Mean intensity of highlights
            if n_highlight > 0:
                highlight_intensities.append(float(frame[highlight_mask > 0].mean()))
            else:
                highlight_intensities.append(0.0)

            # Centroid of highlight region
            if n_highlight > 10:
                coords = np.where(highlight_mask > 0)
                cy = float(np.mean(coords[0])) / frame.shape[0]
                cx = float(np.mean(coords[1])) / frame.shape[1]
                highlight_positions.append((cx, cy))
            else:
                highlight_positions.append(None)

        # Temporal stability of highlight count
        count_var = float(np.var(highlight_counts))
        count_mean = float(np.mean(highlight_counts)) + 1e-8
        count_cv = count_var / (count_mean ** 2)

        # Temporal stability of highlight intensity
        intensity_var = float(np.var(highlight_intensities))
        intensity_mean = float(np.mean(highlight_intensities)) + 1e-8
        intensity_cv = intensity_var / (intensity_mean ** 2)

        # Highlight position stability (how much do highlights jump?)
        position_jumps = []
        for i in range(len(highlight_positions) - 1):
            p1 = highlight_positions[i]
            p2 = highlight_positions[i + 1]
            if p1 is not None and p2 is not None:
                jump = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                position_jumps.append(jump)

        avg_jump = float(np.mean(position_jumps)) if position_jumps else 0
        jump_var = float(np.var(position_jumps)) if len(position_jumps) >= 2 else 0

        # Highlight appearance/disappearance rate
        n_appear_disappear = 0
        for i in range(len(highlight_positions) - 1):
            p1_exists = highlight_positions[i] is not None
            p2_exists = highlight_positions[i + 1] is not None
            if p1_exists != p2_exists:
                n_appear_disappear += 1
        flicker_rate = n_appear_disappear / max(1, len(highlight_positions) - 1)

        # Score
        # count_cv: real ~0.01-0.10, AI ~0.10-0.50+
        count_score = np.clip((count_cv - 0.03) / 0.30, 0, 1)

        # avg_jump: real ~0.01-0.05, AI ~0.05-0.15+
        jump_score = np.clip((avg_jump - 0.02) / 0.10, 0, 1)

        # flicker_rate: real ~0.0-0.05, AI ~0.05-0.20+
        flicker_score = np.clip((flicker_rate - 0.02) / 0.15, 0, 1)

        combined = count_score * 0.30 + jump_score * 0.35 + flicker_score * 0.35

        metrics = {
            "highlight_count_cv": round(count_cv, 4),
            "highlight_intensity_cv": round(intensity_cv, 4),
            "avg_position_jump": round(avg_jump, 4),
            "position_jump_variance": round(jump_var, 6),
            "highlight_flicker_rate": round(flicker_rate, 4),
            "n_appear_disappear": n_appear_disappear,
        }

        return float(np.clip(combined, 0, 1)), metrics
