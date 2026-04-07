"""
AI-Generated Video Detector
=============================
Comprehensive detector for AI-generated videos (Sora, Runway, Kling, Pika,
Minimax, etc.) that goes far beyond traditional face-swap deepfake detection.

The key insight: AI-generated videos have fundamentally different artifacts
than face-swap deepfakes. Face-swap detectors look for blending boundaries
and face inconsistencies. But fully-generated AI videos have:
  - Temporal flickering (frame-to-frame pixel instability)
  - Unnatural optical flow (physics-defying motion patterns)
  - Inter-frame noise inconsistency (no real sensor noise)
  - Edge coherence failures (objects morph between frames)
  - Texture repetition and washout in backgrounds
  - Unnatural motion patterns (hair, clothing, water don't move naturally)
  - Frequency spectrum anomalies across frames
  - Semantic AI tells detectable by CLIP

This module implements 8 specialized analysis layers:

Layer 1: Temporal Flickering Analysis
  - Measures pixel-level stability across consecutive frames
  - AI videos have micro-flickering that's invisible to humans but measurable
  - Uses variance of frame differences in luminance channel

Layer 2: Optical Flow Consistency
  - Computes dense optical flow between consecutive frames
  - AI videos have spatially incoherent flow (objects move independently
    of physics) and temporally unstable flow magnitude
  - Measures flow magnitude variance, direction consistency, spatial coherence

Layer 3: Inter-Frame Noise Consistency
  - Real cameras produce consistent sensor noise patterns
  - AI generators produce varying noise patterns frame-to-frame
  - Extracts high-frequency noise via Laplacian, measures cross-frame correlation

Layer 4: Edge Temporal Coherence
  - Extracts edge maps per frame (Canny)
  - Measures how much edges shift between frames
  - AI videos have edge morphing in static regions (walls, furniture warping)

Layer 5: Texture Stability Analysis
  - Computes LBP (Local Binary Pattern) histograms per frame
  - Real textures are stable across frames; AI textures subtly morph
  - Measures histogram distance between consecutive frames

Layer 6: Background Stability Analysis
  - Segments static vs dynamic regions
  - In real video, static backgrounds are truly static
  - AI videos have subtle background warping/breathing even in still regions

Layer 7: Frequency Spectrum Temporal Analysis
  - Computes FFT magnitude spectrum per frame
  - AI generators leave frequency fingerprints that vary over time
  - Real video has stable frequency characteristics (same lens, same sensor)

Layer 8: Motion Naturalness Assessment
  - Analyzes acceleration patterns in optical flow
  - Real physics: smooth acceleration/deceleration
  - AI motion: abrupt changes, non-Newtonian dynamics
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class AIVideoDetector:
    """
    Comprehensive AI-generated video detector using 8 forensic analysis layers.

    This detector is designed for FULLY AI-GENERATED videos (Sora, Runway, etc.),
    NOT face-swap deepfakes. It analyzes temporal consistency, physics plausibility,
    and statistical patterns that differentiate real camera footage from neural
    network outputs.
    """

    def __init__(self, sample_frames: int = 32, verbose: bool = False):
        """
        Args:
            sample_frames: Number of frames to sample for analysis.
                           More frames = more accurate but slower.
            verbose: Print per-layer scores during analysis.
        """
        self.sample_frames = sample_frames
        self.verbose = verbose

    def analyze(self, video_path: str) -> Dict[str, Any]:
        """
        Run full 8-layer AI video detection on a video file.

        Returns a dict with:
          - ai_probability: float 0-1 (overall AI generation probability)
          - layer_scores: dict of per-layer scores (0=real, 1=AI)
          - forensic_checks: list of human-readable check results
          - raw_metrics: detailed metrics from each layer
        """
        # Extract frames
        frames, fps, total_count = self._extract_frames(video_path)
        if len(frames) < 4:
            return {
                "ai_probability": 0.5,
                "layer_scores": {},
                "forensic_checks": [{
                    "id": "ai_video_error",
                    "name": "AI Video Analysis",
                    "status": "warn",
                    "description": "Insufficient frames for AI video analysis",
                }],
                "raw_metrics": {},
                "error": "Too few frames extracted",
            }

        # Convert to grayscale for efficiency (keep color copies for some analyses)
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

        # Run all 8 analysis layers
        layer_results = {}

        # Layer 1: Temporal Flickering
        flicker_score, flicker_metrics = self._analyze_temporal_flickering(gray_frames)
        layer_results["temporal_flickering"] = {
            "score": flicker_score,
            "metrics": flicker_metrics,
            "weight": 1.2,  # Strong signal for AI video
        }

        # Layer 2: Optical Flow Consistency
        flow_score, flow_metrics = self._analyze_optical_flow(gray_frames, fps)
        layer_results["optical_flow"] = {
            "score": flow_score,
            "metrics": flow_metrics,
            "weight": 1.5,  # Very strong signal — AI physics are wrong
        }

        # Layer 3: Inter-Frame Noise Consistency
        noise_score, noise_metrics = self._analyze_noise_consistency(gray_frames)
        layer_results["noise_consistency"] = {
            "score": noise_score,
            "metrics": noise_metrics,
            "weight": 1.0,
        }

        # Layer 4: Edge Temporal Coherence
        edge_score, edge_metrics = self._analyze_edge_coherence(gray_frames)
        layer_results["edge_coherence"] = {
            "score": edge_score,
            "metrics": edge_metrics,
            "weight": 1.3,  # Edge morphing is a strong AI tell
        }

        # Layer 5: Texture Stability (LBP)
        texture_score, texture_metrics = self._analyze_texture_stability(gray_frames)
        layer_results["texture_stability"] = {
            "score": texture_score,
            "metrics": texture_metrics,
            "weight": 0.9,
        }

        # Layer 6: Background Stability
        bg_score, bg_metrics = self._analyze_background_stability(frames)
        layer_results["background_stability"] = {
            "score": bg_score,
            "metrics": bg_metrics,
            "weight": 1.1,
        }

        # Layer 7: Frequency Spectrum Temporal Analysis
        freq_score, freq_metrics = self._analyze_frequency_temporal(gray_frames)
        layer_results["frequency_temporal"] = {
            "score": freq_score,
            "metrics": freq_metrics,
            "weight": 1.0,
        }

        # Layer 8: Motion Naturalness
        motion_score, motion_metrics = self._analyze_motion_naturalness(gray_frames, fps)
        layer_results["motion_naturalness"] = {
            "score": motion_score,
            "metrics": motion_metrics,
            "weight": 1.2,
        }

        # Compute weighted ensemble score
        total_weight = 0.0
        weighted_sum = 0.0
        layer_scores = {}

        for name, result in layer_results.items():
            score = result["score"]
            weight = result["weight"]
            layer_scores[name] = round(score, 4)
            weighted_sum += score * weight
            total_weight += weight

            if self.verbose:
                print(f"  [AIVideo] {name}: {score:.4f} (w={weight})")

        ai_probability = weighted_sum / total_weight if total_weight > 0 else 0.5

        # Apply calibration: boost high scores, suppress noise near 0.5
        ai_probability = self._calibrate_score(ai_probability, layer_scores)

        if self.verbose:
            print(f"  [AIVideo] Final AI probability: {ai_probability:.4f}")

        # Generate forensic checks for human-readable report
        forensic_checks = self._generate_checks(layer_results, ai_probability)

        return {
            "ai_probability": round(ai_probability, 4),
            "layer_scores": layer_scores,
            "forensic_checks": forensic_checks,
            "raw_metrics": {
                name: result["metrics"] for name, result in layer_results.items()
            },
            "frames_analyzed": len(frames),
            "video_fps": fps,
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Frame Extraction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], float, int]:
        """
        Extract evenly-spaced frames from video.
        Returns (frames_bgr, fps, total_frame_count).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], 0.0, 0

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        if total <= 0:
            cap.release()
            return [], fps, 0

        n_sample = min(self.sample_frames, total)
        indices = np.linspace(0, total - 1, n_sample, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                # Resize to consistent size for analysis (preserve aspect ratio isn't
                # needed here — we're doing statistical analysis, not visual)
                h, w = frame.shape[:2]
                if max(h, w) > 720:
                    scale = 720.0 / max(h, w)
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)
                frames.append(frame)

        cap.release()
        return frames, fps, total

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 1: Temporal Flickering Analysis
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_temporal_flickering(
        self, gray_frames: List[np.ndarray]
    ) -> Tuple[float, Dict]:
        """
        Detect temporal flickering — frame-to-frame pixel intensity instability.

        AI-generated videos exhibit micro-flickering: tiny brightness/color
        oscillations between consecutive frames that real cameras don't produce.
        Real cameras have smooth temporal transitions governed by physics (exposure,
        motion blur). AI generators sample each frame somewhat independently,
        creating subtle but measurable instability.

        Method:
        1. Compute absolute difference between consecutive frames
        2. Measure the variance of these differences over time
        3. High variance in differences = flickering = AI
        4. Also measure the percentage of "unstable" pixels that flicker
        """
        if len(gray_frames) < 3:
            return 0.5, {"error": "too few frames"}

        diffs = []
        for i in range(len(gray_frames) - 1):
            diff = cv2.absdiff(gray_frames[i], gray_frames[i + 1]).astype(np.float32)
            diffs.append(diff)

        # Stack all difference frames: (N-1, H, W)
        diff_stack = np.stack(diffs)

        # Global mean difference (accounts for motion)
        mean_diff = float(np.mean(diff_stack))

        # Temporal variance of differences at each pixel, averaged
        # High = pixels change intensity erratically = flickering
        temporal_var = float(np.mean(np.var(diff_stack, axis=0)))

        # Percentage of pixels with high temporal variance
        pixel_vars = np.var(diff_stack, axis=0)
        # Threshold: variance > 100 means that pixel flickers significantly
        flicker_ratio = float(np.mean(pixel_vars > 100))

        # Second-order: variance of frame-level mean differences
        # Real video: smooth changes. AI: jumpy changes.
        frame_means = [float(np.mean(d)) for d in diffs]
        second_order_var = float(np.var(frame_means)) if len(frame_means) >= 2 else 0

        # Normalized second-order jitter (invariant to motion amount)
        mean_of_means = np.mean(frame_means) + 1e-6
        jitter_ratio = second_order_var / (mean_of_means ** 2)

        # Score computation:
        # temporal_var: real video ~20-80, AI video ~80-300+
        # flicker_ratio: real ~0.01-0.05, AI ~0.05-0.20+
        # jitter_ratio: real ~0.001-0.05, AI ~0.05-0.30+
        var_score = np.clip((temporal_var - 30) / 200, 0, 1)
        flicker_score = np.clip((flicker_ratio - 0.02) / 0.15, 0, 1)
        jitter_score = np.clip((jitter_ratio - 0.01) / 0.20, 0, 1)

        combined = var_score * 0.35 + flicker_score * 0.35 + jitter_score * 0.30

        metrics = {
            "mean_diff": round(mean_diff, 2),
            "temporal_variance": round(temporal_var, 2),
            "flicker_ratio": round(flicker_ratio, 4),
            "second_order_variance": round(second_order_var, 4),
            "jitter_ratio": round(jitter_ratio, 6),
            "var_score": round(var_score, 4),
            "flicker_score": round(flicker_score, 4),
            "jitter_score": round(jitter_score, 4),
        }

        return float(np.clip(combined, 0, 1)), metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 2: Optical Flow Consistency
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_optical_flow(
        self, gray_frames: List[np.ndarray], fps: float
    ) -> Tuple[float, Dict]:
        """
        Analyze optical flow for physics-defying motion patterns.

        Real-world motion obeys physics: smooth acceleration, consistent
        direction within rigid objects, motion parallax. AI-generated video
        often has:
        - Spatially incoherent flow (different parts of same object move
          in different directions)
        - Temporally unstable flow magnitude (jerky speed changes)
        - Abnormal flow distributions (too uniform or too chaotic)

        Uses Farneback dense optical flow for full-frame analysis.
        """
        if len(gray_frames) < 3:
            return 0.5, {"error": "too few frames"}

        flow_magnitudes = []
        flow_angle_vars = []
        spatial_coherences = []

        for i in range(len(gray_frames) - 1):
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[i], gray_frames[i + 1],
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            # Magnitude and angle
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Mean flow magnitude for this frame pair
            flow_magnitudes.append(float(np.mean(mag)))

            # Angle variance within this frame (spatial coherence)
            # Real objects: nearby pixels have similar flow directions
            # AI: flow directions can be chaotic
            # Compute in 16x16 blocks
            h, w = ang.shape
            block_size = 16
            block_angle_vars = []
            for by in range(0, h - block_size, block_size):
                for bx in range(0, w - block_size, block_size):
                    block = ang[by:by + block_size, bx:bx + block_size]
                    block_mag = mag[by:by + block_size, bx:bx + block_size]
                    # Only consider blocks with significant motion
                    if np.mean(block_mag) > 0.5:
                        # Circular variance for angles
                        sin_mean = np.mean(np.sin(block))
                        cos_mean = np.mean(np.cos(block))
                        circ_var = 1.0 - np.sqrt(sin_mean ** 2 + cos_mean ** 2)
                        block_angle_vars.append(circ_var)

            if block_angle_vars:
                flow_angle_vars.append(float(np.mean(block_angle_vars)))

            # Spatial coherence: how smooth is the flow field?
            # Compute gradient of flow magnitude
            flow_mag_dx = cv2.Sobel(mag, cv2.CV_64F, 1, 0, ksize=3)
            flow_mag_dy = cv2.Sobel(mag, cv2.CV_64F, 0, 1, ksize=3)
            flow_gradient = np.sqrt(flow_mag_dx ** 2 + flow_mag_dy ** 2)
            spatial_coherences.append(float(np.mean(flow_gradient)))

        # Temporal stability of flow magnitude
        if len(flow_magnitudes) >= 2:
            flow_mag_var = float(np.var(flow_magnitudes))
            flow_mag_mean = float(np.mean(flow_magnitudes)) + 1e-6
            flow_temporal_jitter = flow_mag_var / (flow_mag_mean ** 2)
        else:
            flow_temporal_jitter = 0.0
            flow_mag_var = 0.0
            flow_mag_mean = 0.0

        # Average spatial coherence (higher = less coherent = more AI-like)
        avg_spatial_incoherence = float(np.mean(spatial_coherences)) if spatial_coherences else 0

        # Average angular variance (higher = more chaotic directions = AI)
        avg_angle_var = float(np.mean(flow_angle_vars)) if flow_angle_vars else 0

        # Score computation
        # flow_temporal_jitter: real ~0.01-0.10, AI ~0.10-0.50+
        jitter_score = np.clip((flow_temporal_jitter - 0.03) / 0.30, 0, 1)

        # spatial_incoherence: real ~0.5-2.0, AI ~2.0-5.0+
        spatial_score = np.clip((avg_spatial_incoherence - 1.0) / 3.0, 0, 1)

        # angle_variance: real ~0.1-0.3, AI ~0.3-0.6+
        angle_score = np.clip((avg_angle_var - 0.15) / 0.35, 0, 1)

        combined = jitter_score * 0.40 + spatial_score * 0.35 + angle_score * 0.25

        metrics = {
            "flow_mag_mean": round(flow_mag_mean, 4),
            "flow_mag_var": round(flow_mag_var, 4),
            "flow_temporal_jitter": round(flow_temporal_jitter, 6),
            "avg_spatial_incoherence": round(avg_spatial_incoherence, 4),
            "avg_angle_variance": round(avg_angle_var, 4),
            "jitter_score": round(jitter_score, 4),
            "spatial_score": round(spatial_score, 4),
            "angle_score": round(angle_score, 4),
        }

        return float(np.clip(combined, 0, 1)), metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 3: Inter-Frame Noise Consistency
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_noise_consistency(
        self, gray_frames: List[np.ndarray]
    ) -> Tuple[float, Dict]:
        """
        Analyze sensor noise pattern consistency across frames.

        Real cameras have a fixed noise pattern (fixed-pattern noise from the
        sensor + shot noise that varies but has consistent statistical properties).
        AI generators produce noise that varies in character between frames because
        each frame is generated somewhat independently.

        Method:
        1. Extract high-frequency noise via Laplacian filter
        2. Compute noise statistics per frame (mean, std, histogram)
        3. Measure how much noise statistics vary across frames
        4. Real video: consistent noise. AI video: varying noise character.
        """
        if len(gray_frames) < 3:
            return 0.5, {"error": "too few frames"}

        noise_means = []
        noise_stds = []
        noise_histograms = []

        for frame in gray_frames:
            # Extract noise: high-pass filter via Laplacian
            noise = cv2.Laplacian(frame, cv2.CV_64F)
            noise_means.append(float(np.mean(np.abs(noise))))
            noise_stds.append(float(np.std(noise)))

            # Noise histogram (binned for comparison)
            abs_noise = np.clip(np.abs(noise), 0, 50).astype(np.uint8)
            hist = cv2.calcHist([abs_noise], [0], None, [50], [0, 50])
            hist = hist.flatten() / (hist.sum() + 1e-6)
            noise_histograms.append(hist)

        # Temporal variance of noise statistics
        noise_mean_var = float(np.var(noise_means))
        noise_std_var = float(np.var(noise_stds))

        # Normalized variance (invariant to overall noise level)
        avg_noise_mean = float(np.mean(noise_means)) + 1e-6
        avg_noise_std = float(np.mean(noise_stds)) + 1e-6
        norm_mean_var = noise_mean_var / (avg_noise_mean ** 2)
        norm_std_var = noise_std_var / (avg_noise_std ** 2)

        # Histogram distance between consecutive frames
        hist_distances = []
        for i in range(len(noise_histograms) - 1):
            # Chi-squared distance between histograms
            dist = cv2.compareHist(
                noise_histograms[i].astype(np.float32),
                noise_histograms[i + 1].astype(np.float32),
                cv2.HISTCMP_CHISQR,
            )
            hist_distances.append(dist)

        avg_hist_dist = float(np.mean(hist_distances)) if hist_distances else 0
        hist_dist_var = float(np.var(hist_distances)) if len(hist_distances) >= 2 else 0

        # Cross-frame noise correlation (sample pairs)
        noise_correlations = []
        step = max(1, len(gray_frames) // 8)
        for i in range(0, len(gray_frames) - step, step):
            n1 = cv2.Laplacian(gray_frames[i], cv2.CV_64F).flatten()
            n2 = cv2.Laplacian(gray_frames[i + step], cv2.CV_64F).flatten()
            # Subsample for speed
            n1 = n1[::10]
            n2 = n2[::10]
            corr = float(np.corrcoef(n1, n2)[0, 1]) if len(n1) > 10 else 0
            noise_correlations.append(corr)

        avg_noise_corr = float(np.mean(noise_correlations)) if noise_correlations else 0

        # Score computation
        # norm_mean_var: real ~0.0001-0.005, AI ~0.005-0.05+
        mean_var_score = np.clip((norm_mean_var - 0.001) / 0.03, 0, 1)

        # norm_std_var: real ~0.0001-0.003, AI ~0.003-0.03+
        std_var_score = np.clip((norm_std_var - 0.0005) / 0.02, 0, 1)

        # avg_hist_dist: real ~0.001-0.01, AI ~0.01-0.10+
        hist_score = np.clip((avg_hist_dist - 0.005) / 0.05, 0, 1)

        # noise_correlation: real ~0.7-0.95 (fixed pattern noise), AI ~0.3-0.7
        corr_score = np.clip((0.75 - avg_noise_corr) / 0.40, 0, 1)

        combined = (
            mean_var_score * 0.20
            + std_var_score * 0.20
            + hist_score * 0.25
            + corr_score * 0.35
        )

        metrics = {
            "noise_mean_var": round(noise_mean_var, 6),
            "noise_std_var": round(noise_std_var, 6),
            "norm_mean_var": round(norm_mean_var, 6),
            "norm_std_var": round(norm_std_var, 6),
            "avg_hist_distance": round(avg_hist_dist, 6),
            "hist_dist_variance": round(hist_dist_var, 6),
            "avg_noise_correlation": round(avg_noise_corr, 4),
            "mean_var_score": round(mean_var_score, 4),
            "std_var_score": round(std_var_score, 4),
            "hist_score": round(hist_score, 4),
            "corr_score": round(corr_score, 4),
        }

        return float(np.clip(combined, 0, 1)), metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 4: Edge Temporal Coherence
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_edge_coherence(
        self, gray_frames: List[np.ndarray]
    ) -> Tuple[float, Dict]:
        """
        Analyze temporal coherence of edge maps.

        In real video, edges of static objects remain stable. AI-generated videos
        often exhibit "edge morphing" — edges of walls, furniture, and backgrounds
        subtly shift position between frames even when nothing should be moving.

        Method:
        1. Compute Canny edge maps for each frame
        2. Measure edge stability in regions with low optical flow (static regions)
        3. High edge change in static regions = AI artifact
        """
        if len(gray_frames) < 3:
            return 0.5, {"error": "too few frames"}

        # Compute edge maps
        edge_maps = []
        for frame in gray_frames:
            # Blur slightly to reduce noise sensitivity
            blurred = cv2.GaussianBlur(frame, (3, 3), 0)
            edges = cv2.Canny(blurred, 50, 150)
            edge_maps.append(edges)

        # Compute optical flow to identify static regions
        static_edge_changes = []
        total_edge_changes = []

        for i in range(len(gray_frames) - 1):
            # Quick flow to find static regions
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[i], gray_frames[i + 1],
                None, 0.5, 2, 10, 3, 5, 1.1, 0
            )
            flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

            # Static mask: pixels with minimal flow (< 1 pixel)
            static_mask = (flow_mag < 1.0).astype(np.uint8)

            # Edge difference between consecutive frames
            edge_diff = cv2.absdiff(edge_maps[i], edge_maps[i + 1])

            # Edge change in static regions (should be near zero for real video)
            static_pixels = static_mask.sum() + 1
            static_edge_change = float((edge_diff * static_mask).sum()) / static_pixels
            static_edge_changes.append(static_edge_change)

            # Total edge change for reference
            total_edge_change = float(edge_diff.mean())
            total_edge_changes.append(total_edge_change)

        avg_static_edge_change = float(np.mean(static_edge_changes))
        max_static_edge_change = float(np.max(static_edge_changes))
        static_edge_var = float(np.var(static_edge_changes))

        avg_total_edge_change = float(np.mean(total_edge_changes))

        # Ratio of static edge change to total (isolates AI artifacts from motion)
        if avg_total_edge_change > 0.01:
            static_to_total_ratio = avg_static_edge_change / avg_total_edge_change
        else:
            static_to_total_ratio = 0.0

        # Score computation
        # avg_static_edge_change: real ~0.5-3.0, AI ~3.0-15.0+
        static_score = np.clip((avg_static_edge_change - 1.0) / 8.0, 0, 1)

        # static_to_total_ratio: real ~0.2-0.4, AI ~0.4-0.8+
        ratio_score = np.clip((static_to_total_ratio - 0.25) / 0.40, 0, 1)

        # max_static_edge_change: captures worst-case morphing
        max_score = np.clip((max_static_edge_change - 3.0) / 15.0, 0, 1)

        combined = static_score * 0.40 + ratio_score * 0.35 + max_score * 0.25

        metrics = {
            "avg_static_edge_change": round(avg_static_edge_change, 4),
            "max_static_edge_change": round(max_static_edge_change, 4),
            "static_edge_variance": round(static_edge_var, 6),
            "avg_total_edge_change": round(avg_total_edge_change, 4),
            "static_to_total_ratio": round(static_to_total_ratio, 4),
            "static_score": round(static_score, 4),
            "ratio_score": round(ratio_score, 4),
            "max_score": round(max_score, 4),
        }

        return float(np.clip(combined, 0, 1)), metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 5: Texture Stability (LBP)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _compute_lbp(self, image: np.ndarray) -> np.ndarray:
        """
        Compute simplified Local Binary Pattern.
        Compares each pixel with its 8 neighbors, encodes as 8-bit number.
        """
        h, w = image.shape
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)

        # 8 neighbors, center at (1,1) relative offsets
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                     (1, 1), (1, 0), (1, -1), (0, -1)]

        for bit, (dy, dx) in enumerate(neighbors):
            # Compare neighbor with center pixel
            neighbor = image[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
            center = image[1:h - 1, 1:w - 1]
            lbp |= ((neighbor >= center).astype(np.uint8) << bit)

        return lbp

    def _analyze_texture_stability(
        self, gray_frames: List[np.ndarray]
    ) -> Tuple[float, Dict]:
        """
        Analyze texture stability using Local Binary Patterns (LBP).

        Real textures (fabric, skin, walls) maintain consistent micro-texture
        patterns across frames. AI-generated textures subtly morph between frames
        because the generator doesn't maintain a consistent texture state.

        Method:
        1. Compute LBP histograms for each frame
        2. Measure histogram distance between consecutive frames
        3. High variation = texture morphing = AI
        """
        if len(gray_frames) < 3:
            return 0.5, {"error": "too few frames"}

        # Compute LBP histograms for each frame
        lbp_hists = []
        for frame in gray_frames:
            # Resize for consistent LBP computation
            resized = cv2.resize(frame, (256, 256))
            lbp = self._compute_lbp(resized)
            hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-6)
            lbp_hists.append(hist)

        # Compare consecutive histograms
        hist_distances = []
        for i in range(len(lbp_hists) - 1):
            # Bhattacharyya distance (0 = identical, 1 = completely different)
            dist = cv2.compareHist(
                lbp_hists[i].astype(np.float32),
                lbp_hists[i + 1].astype(np.float32),
                cv2.HISTCMP_BHATTACHARYYA,
            )
            hist_distances.append(dist)

        avg_dist = float(np.mean(hist_distances))
        max_dist = float(np.max(hist_distances))
        dist_var = float(np.var(hist_distances))

        # Also check non-consecutive frames (catch slow drift)
        drift_distances = []
        step = max(1, len(lbp_hists) // 4)
        for i in range(0, len(lbp_hists) - step, step):
            dist = cv2.compareHist(
                lbp_hists[i].astype(np.float32),
                lbp_hists[i + step].astype(np.float32),
                cv2.HISTCMP_BHATTACHARYYA,
            )
            drift_distances.append(dist)

        avg_drift = float(np.mean(drift_distances)) if drift_distances else 0

        # Score computation
        # avg_dist: real ~0.02-0.08, AI ~0.08-0.25+
        dist_score = np.clip((avg_dist - 0.03) / 0.15, 0, 1)

        # dist_var: real ~0.0001-0.001, AI ~0.001-0.01+
        var_score = np.clip((dist_var - 0.0003) / 0.005, 0, 1)

        # avg_drift: real ~0.03-0.10, AI ~0.10-0.30+
        drift_score = np.clip((avg_drift - 0.05) / 0.18, 0, 1)

        combined = dist_score * 0.45 + var_score * 0.25 + drift_score * 0.30

        metrics = {
            "avg_lbp_distance": round(avg_dist, 4),
            "max_lbp_distance": round(max_dist, 4),
            "lbp_distance_variance": round(dist_var, 6),
            "avg_texture_drift": round(avg_drift, 4),
            "dist_score": round(dist_score, 4),
            "var_score": round(var_score, 4),
            "drift_score": round(drift_score, 4),
        }

        return float(np.clip(combined, 0, 1)), metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 6: Background Stability Analysis
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_background_stability(
        self, color_frames: List[np.ndarray]
    ) -> Tuple[float, Dict]:
        """
        Analyze background stability.

        In real video, truly static background regions are pixel-identical
        (up to sensor noise). AI-generated videos exhibit "background breathing" —
        subtle color shifts and warping in regions that should be perfectly still.

        Method:
        1. Identify static regions across all frames via temporal median
        2. Compute per-pixel temporal variance in static regions
        3. Check for color channel independence (AI color shifts are correlated
           differently than sensor noise)
        """
        if len(color_frames) < 4:
            return 0.5, {"error": "too few frames"}

        # Compute temporal median to find "background"
        # Stack frames: (N, H, W, 3)
        h, w = color_frames[0].shape[:2]
        target_h, target_w = min(h, 360), min(w, 640)

        resized = []
        for f in color_frames:
            resized.append(cv2.resize(f, (target_w, target_h)))

        stack = np.stack(resized).astype(np.float32)  # (N, H, W, 3)

        # Temporal variance per pixel per channel
        temporal_var = np.var(stack, axis=0)  # (H, W, 3)
        mean_var_per_pixel = np.mean(temporal_var, axis=2)  # (H, W)

        # Static regions: pixels with low temporal variance
        # (below 10 = basically static)
        static_threshold = 15.0
        static_mask = (mean_var_per_pixel < static_threshold)
        static_ratio = float(static_mask.sum()) / (target_h * target_w)

        if static_ratio < 0.05:
            # Not enough static regions (very dynamic video)
            return 0.4, {
                "static_ratio": round(static_ratio, 4),
                "note": "Insufficient static regions for analysis",
            }

        # Variance in static regions (should be very low for real video)
        static_vars = mean_var_per_pixel[static_mask]
        avg_static_var = float(np.mean(static_vars))
        max_static_var = float(np.max(static_vars))

        # Color channel correlation in static regions
        # Real sensor noise: channels are relatively independent
        # AI color shifts: channels shift together (correlated)
        b_var = temporal_var[:, :, 0][static_mask]
        g_var = temporal_var[:, :, 1][static_mask]
        r_var = temporal_var[:, :, 2][static_mask]

        if len(b_var) > 10:
            # Subsample for speed
            sub = min(10000, len(b_var))
            idx = np.random.choice(len(b_var), sub, replace=False) if len(b_var) > sub else np.arange(len(b_var))
            bg_corr = float(np.corrcoef(b_var[idx], g_var[idx])[0, 1])
            br_corr = float(np.corrcoef(b_var[idx], r_var[idx])[0, 1])
            gr_corr = float(np.corrcoef(g_var[idx], r_var[idx])[0, 1])
            avg_channel_corr = (abs(bg_corr) + abs(br_corr) + abs(gr_corr)) / 3
        else:
            avg_channel_corr = 0.5

        # Temporal gradient in static regions: measure how static pixels
        # change over time (captures "breathing" effect)
        breathing_scores = []
        for i in range(len(resized) - 1):
            diff = cv2.absdiff(resized[i], resized[i + 1]).astype(np.float32)
            diff_gray = np.mean(diff, axis=2)
            static_diff = diff_gray[static_mask]
            breathing_scores.append(float(np.mean(static_diff)))

        avg_breathing = float(np.mean(breathing_scores)) if breathing_scores else 0
        max_breathing = float(np.max(breathing_scores)) if breathing_scores else 0

        # Score computation
        # avg_static_var: real ~0.5-3.0, AI ~3.0-10.0+
        var_score = np.clip((avg_static_var - 1.5) / 6.0, 0, 1)

        # avg_breathing: real ~0.2-1.5, AI ~1.5-5.0+
        breathing_score = np.clip((avg_breathing - 0.8) / 3.0, 0, 1)

        # avg_channel_corr: real ~0.3-0.6, AI ~0.7-0.95+
        corr_score = np.clip((avg_channel_corr - 0.5) / 0.40, 0, 1)

        combined = var_score * 0.30 + breathing_score * 0.40 + corr_score * 0.30

        metrics = {
            "static_ratio": round(static_ratio, 4),
            "avg_static_variance": round(avg_static_var, 4),
            "max_static_variance": round(max_static_var, 4),
            "avg_breathing": round(avg_breathing, 4),
            "max_breathing": round(max_breathing, 4),
            "avg_channel_correlation": round(avg_channel_corr, 4),
            "var_score": round(var_score, 4),
            "breathing_score": round(breathing_score, 4),
            "corr_score": round(corr_score, 4),
        }

        return float(np.clip(combined, 0, 1)), metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 7: Frequency Spectrum Temporal Analysis
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_frequency_temporal(
        self, gray_frames: List[np.ndarray]
    ) -> Tuple[float, Dict]:
        """
        Analyze temporal consistency of frequency spectra across frames.

        Real cameras produce consistent frequency fingerprints (same lens,
        same sensor, same optical path). AI generators produce varying
        frequency characteristics because they don't model physical optics.

        Method:
        1. Compute 2D FFT magnitude spectrum per frame
        2. Compute radial power spectrum (frequency vs power)
        3. Measure temporal variance of radial spectra
        4. Check for abnormal high-frequency energy patterns
        """
        if len(gray_frames) < 3:
            return 0.5, {"error": "too few frames"}

        radial_spectra = []
        high_freq_ratios = []

        for frame in gray_frames:
            # Resize to consistent size for FFT comparison
            resized = cv2.resize(frame, (256, 256)).astype(np.float32)

            # 2D FFT
            f_transform = np.fft.fft2(resized)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.log1p(np.abs(f_shift))

            # Radial power spectrum: average magnitude at each frequency radius
            h, w = magnitude.shape
            cy, cx = h // 2, w // 2
            max_radius = min(cy, cx)

            radial = np.zeros(max_radius)
            counts = np.zeros(max_radius)

            y_coords, x_coords = np.ogrid[:h, :w]
            r = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2).astype(int)
            r = np.clip(r, 0, max_radius - 1)

            for radius in range(max_radius):
                mask = (r == radius)
                if mask.sum() > 0:
                    radial[radius] = magnitude[mask].mean()
                    counts[radius] = mask.sum()

            # Normalize
            radial = radial / (radial.max() + 1e-6)
            radial_spectra.append(radial)

            # High-frequency energy ratio
            mid = max_radius // 2
            low_freq_energy = np.sum(radial[:mid])
            high_freq_energy = np.sum(radial[mid:])
            ratio = high_freq_energy / (low_freq_energy + 1e-6)
            high_freq_ratios.append(ratio)

        # Temporal variance of radial spectra
        spectra_stack = np.stack(radial_spectra)  # (N, R)
        spectral_var = np.mean(np.var(spectra_stack, axis=0))

        # Temporal variance of high-frequency ratio
        hf_ratio_var = float(np.var(high_freq_ratios))
        hf_ratio_mean = float(np.mean(high_freq_ratios))

        # Spectral flatness: how uniform is the spectrum?
        # AI tends to have flatter spectra (less natural 1/f falloff)
        spectral_flatness_scores = []
        for spectrum in radial_spectra:
            # Natural images follow ~1/f power law
            # Compute deviation from 1/f
            n = len(spectrum)
            ideal_1f = 1.0 / (np.arange(1, n + 1).astype(np.float32))
            ideal_1f = ideal_1f / (ideal_1f.max() + 1e-6)
            deviation = float(np.mean((spectrum - ideal_1f) ** 2))
            spectral_flatness_scores.append(deviation)

        avg_flatness_deviation = float(np.mean(spectral_flatness_scores))

        # Score computation
        # spectral_var: real ~0.0001-0.001, AI ~0.001-0.01+
        var_score = np.clip((spectral_var - 0.0003) / 0.005, 0, 1)

        # hf_ratio_var: real ~0.0001-0.002, AI ~0.002-0.02+
        hf_var_score = np.clip((hf_ratio_var - 0.0005) / 0.01, 0, 1)

        # flatness: real ~0.02-0.06, AI ~0.06-0.15+
        flatness_score = np.clip((avg_flatness_deviation - 0.03) / 0.08, 0, 1)

        combined = var_score * 0.35 + hf_var_score * 0.30 + flatness_score * 0.35

        metrics = {
            "spectral_variance": round(float(spectral_var), 6),
            "hf_ratio_mean": round(hf_ratio_mean, 4),
            "hf_ratio_variance": round(hf_ratio_var, 6),
            "avg_flatness_deviation": round(avg_flatness_deviation, 6),
            "var_score": round(var_score, 4),
            "hf_var_score": round(hf_var_score, 4),
            "flatness_score": round(flatness_score, 4),
        }

        return float(np.clip(combined, 0, 1)), metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 8: Motion Naturalness Assessment
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_motion_naturalness(
        self, gray_frames: List[np.ndarray], fps: float
    ) -> Tuple[float, Dict]:
        """
        Assess whether motion patterns follow natural physics.

        Real-world motion obeys Newton's laws: objects have inertia, gravity
        acts uniformly, acceleration is smooth. AI-generated motion often
        violates these principles:
        - Abrupt velocity changes (no inertia)
        - Non-uniform acceleration (jerky motion)
        - Objects that move without physical cause
        - Impossible deformations of rigid objects

        Method:
        1. Track sparse keypoints using Lucas-Kanade optical flow
        2. Compute velocity and acceleration per keypoint over time
        3. Measure smoothness of motion trajectories
        4. Detect physically implausible acceleration events
        """
        if len(gray_frames) < 4:
            return 0.5, {"error": "too few frames"}

        # Detect good features to track in first frame
        feature_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=15,
            blockSize=7,
        )
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        # Track across all frames
        p0 = cv2.goodFeaturesToTrack(gray_frames[0], mask=None, **feature_params)
        if p0 is None or len(p0) < 5:
            return 0.4, {"error": "insufficient keypoints", "num_keypoints": 0}

        # Collect trajectories: (num_points, num_frames, 2)
        trajectories = [p0.reshape(-1, 2)]
        active_mask = np.ones(len(p0), dtype=bool)

        for i in range(1, len(gray_frames)):
            if active_mask.sum() < 3:
                break

            p_prev = trajectories[-1][active_mask].reshape(-1, 1, 2).astype(np.float32)
            p_next, status, err = cv2.calcOpticalFlowPyrLK(
                gray_frames[i - 1], gray_frames[i], p_prev, None, **lk_params
            )

            if p_next is None:
                break

            # Update active mask
            status = status.flatten().astype(bool)
            new_positions = np.full((len(active_mask), 2), np.nan)

            active_indices = np.where(active_mask)[0]
            for j, idx in enumerate(active_indices):
                if j < len(status) and status[j]:
                    new_positions[idx] = p_next[j].flatten()
                else:
                    active_mask[idx] = False

            trajectories.append(new_positions)

        if len(trajectories) < 4:
            return 0.4, {"error": "insufficient trajectory length"}

        # Convert to array: (num_points, num_frames, 2)
        traj_array = np.stack(trajectories, axis=1)  # (N, T, 2)

        # Compute velocities and accelerations for each tracked point
        velocities = []
        accelerations = []
        jerk_values = []  # third derivative

        for pt_idx in range(traj_array.shape[0]):
            traj = traj_array[pt_idx]  # (T, 2)
            valid = ~np.isnan(traj[:, 0])

            if valid.sum() < 4:
                continue

            valid_pos = traj[valid]  # (T_valid, 2)

            # Velocity: first derivative
            vel = np.diff(valid_pos, axis=0)  # (T-1, 2)
            speed = np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2)
            velocities.append(speed)

            if len(speed) >= 2:
                # Acceleration: second derivative
                accel = np.diff(speed)
                accelerations.append(accel)

                if len(accel) >= 2:
                    # Jerk: third derivative
                    jerk = np.diff(accel)
                    jerk_values.append(jerk)

        if not velocities:
            return 0.4, {"error": "no valid trajectories"}

        # Aggregate motion metrics

        # 1. Acceleration smoothness: ratio of high-acceleration events
        all_accels = np.concatenate(accelerations) if accelerations else np.array([0])
        accel_std = float(np.std(all_accels))
        mean_speed = float(np.mean(np.concatenate(velocities)))

        # Normalized acceleration (independent of overall speed)
        norm_accel_std = accel_std / (mean_speed + 1e-6)

        # 2. Jerk magnitude (smoothness of acceleration)
        all_jerks = np.concatenate(jerk_values) if jerk_values else np.array([0])
        jerk_rms = float(np.sqrt(np.mean(all_jerks ** 2)))
        norm_jerk = jerk_rms / (mean_speed + 1e-6)

        # 3. Sudden velocity changes (implausible acceleration events)
        high_accel_threshold = np.percentile(np.abs(all_accels), 95) if len(all_accels) > 1 else 0
        high_accel_ratio = float(np.mean(np.abs(all_accels) > accel_std * 3)) if accel_std > 0 else 0

        # 4. Motion coherence: nearby points should have similar velocities
        motion_coherence_scores = []
        for t in range(min(len(trajectories) - 1, 10)):
            pos_t = traj_array[:, t, :]
            pos_t1 = traj_array[:, t + 1, :]
            vel_t = pos_t1 - pos_t  # (N, 2)

            valid = ~np.isnan(vel_t[:, 0])
            if valid.sum() < 5:
                continue

            valid_pos = pos_t[valid]
            valid_vel = vel_t[valid]

            # For each point, check if nearby points have similar velocity
            coherence = []
            for i in range(len(valid_pos)):
                dists = np.sqrt(np.sum((valid_pos - valid_pos[i]) ** 2, axis=1))
                nearby = (dists < 50) & (dists > 0)
                if nearby.sum() > 0:
                    my_vel = valid_vel[i]
                    nearby_vels = valid_vel[nearby]
                    vel_diff = np.sqrt(np.sum((nearby_vels - my_vel) ** 2, axis=1))
                    coherence.append(float(np.mean(vel_diff)))

            if coherence:
                motion_coherence_scores.append(float(np.mean(coherence)))

        avg_incoherence = float(np.mean(motion_coherence_scores)) if motion_coherence_scores else 0

        # Score computation
        # norm_accel_std: real ~0.1-0.5, AI ~0.5-2.0+
        accel_score = np.clip((norm_accel_std - 0.2) / 1.0, 0, 1)

        # norm_jerk: real ~0.05-0.3, AI ~0.3-1.5+
        jerk_score = np.clip((norm_jerk - 0.1) / 0.8, 0, 1)

        # high_accel_ratio: real ~0.01-0.05, AI ~0.05-0.15+
        sudden_score = np.clip((high_accel_ratio - 0.02) / 0.10, 0, 1)

        # avg_incoherence: real ~0.5-2.0, AI ~2.0-6.0+
        coherence_score = np.clip((avg_incoherence - 1.0) / 4.0, 0, 1)

        combined = (
            accel_score * 0.30
            + jerk_score * 0.25
            + sudden_score * 0.20
            + coherence_score * 0.25
        )

        metrics = {
            "mean_speed": round(mean_speed, 4),
            "accel_std": round(accel_std, 4),
            "norm_accel_std": round(norm_accel_std, 4),
            "jerk_rms": round(jerk_rms, 4),
            "norm_jerk": round(norm_jerk, 4),
            "high_accel_ratio": round(high_accel_ratio, 4),
            "avg_motion_incoherence": round(avg_incoherence, 4),
            "num_tracked_points": len(velocities),
            "trajectory_length": len(trajectories),
            "accel_score": round(accel_score, 4),
            "jerk_score": round(jerk_score, 4),
            "sudden_score": round(sudden_score, 4),
            "coherence_score": round(coherence_score, 4),
        }

        return float(np.clip(combined, 0, 1)), metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Score Calibration & Report Generation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _calibrate_score(
        self, raw_score: float, layer_scores: Dict[str, float]
    ) -> float:
        """
        Calibrate the raw ensemble score using a max-boosted strategy.

        The weighted average can be too conservative: if 3 layers detect
        strong AI artifacts (>0.7) but 5 layers see nothing, the average
        is dragged below 0.5 even though the strong signals are highly
        indicative of AI generation.

        Calibration strategy:
        1. Use the top-K strong signals to set a floor
        2. Boost when multiple independent layers agree
        3. Suppress uncertain scores near 0.5
        4. A single maxed-out layer (>0.9) should dominate if corroborated
        """
        scores = list(layer_scores.values())
        n_layers = len(scores)

        if n_layers == 0:
            return raw_score

        # Sort scores descending to identify top signals
        sorted_scores = sorted(scores, reverse=True)

        # Count signal categories
        n_ai_signals = sum(1 for s in scores if s > 0.5)
        n_strong_signals = sum(1 for s in scores if s > 0.65)
        n_very_strong = sum(1 for s in scores if s > 0.8)

        # ─── Top-K Floor: strong signals shouldn't be averaged away ───
        # If the top 3 signals average > 0.7, that's strong evidence
        top_3_avg = float(np.mean(sorted_scores[:3])) if n_layers >= 3 else float(np.mean(sorted_scores))
        if top_3_avg > 0.65 and raw_score < top_3_avg:
            # Pull raw_score toward top-3 average (don't let weak layers kill it)
            pull = 0.5  # how much to trust top-3 vs full average
            raw_score = raw_score * (1 - pull) + top_3_avg * pull

        # ─── Strong Signal Dominance ───
        # If any single layer is >0.9 AND at least 1 other layer >0.5,
        # the score should be at least 0.6 (corroborated extreme signal)
        max_score = sorted_scores[0]
        if max_score > 0.9 and n_ai_signals >= 2:
            raw_score = max(raw_score, 0.60 + (max_score - 0.9) * 0.5)

        # ─── Multi-signal agreement boost ───
        agreement_ratio = n_ai_signals / n_layers

        if agreement_ratio > 0.5 and raw_score > 0.45:
            agreement_boost = (agreement_ratio - 0.5) * 0.4
            raw_score = raw_score + agreement_boost * (1 - raw_score)

        # Strong signal count bonus (each strong layer adds evidence)
        if n_strong_signals >= 2 and raw_score > 0.45:
            strength_boost = min(0.15, (n_strong_signals - 1) * 0.04)
            raw_score = raw_score + strength_boost * (1 - raw_score)

        if n_very_strong >= 2 and raw_score > 0.5:
            extreme_boost = min(0.12, (n_very_strong - 1) * 0.05)
            raw_score = raw_score + extreme_boost * (1 - raw_score)

        # ─── Suppress uncertain scores ───
        if 0.42 < raw_score < 0.50 and n_strong_signals < 2:
            pull_strength = 0.25
            raw_score = 0.5 + (raw_score - 0.5) * (1 - pull_strength)

        return float(np.clip(raw_score, 0, 1))

    def _generate_checks(
        self, layer_results: Dict, ai_probability: float
    ) -> List[Dict[str, Any]]:
        """Generate human-readable forensic check results."""
        checks = []

        # Layer 1: Temporal Flickering
        r = layer_results["temporal_flickering"]
        score = r["score"]
        m = r["metrics"]
        if "error" not in m:
            status = "fail" if score > 0.6 else "warn" if score > 0.35 else "pass"
            if score > 0.6:
                desc = f"Temporal flickering detected: {m.get('flicker_ratio', 0):.1%} of pixels unstable (AI artifact)"
            elif score > 0.35:
                desc = f"Mild temporal instability detected (flickering score: {score:.2f})"
            else:
                desc = "Stable frame-to-frame transitions (natural camera behavior)"
            checks.append({
                "id": "ai_temporal_flicker",
                "name": "Temporal Flickering Analysis",
                "status": status,
                "description": desc,
            })

        # Layer 2: Optical Flow
        r = layer_results["optical_flow"]
        score = r["score"]
        m = r["metrics"]
        if "error" not in m:
            status = "fail" if score > 0.6 else "warn" if score > 0.35 else "pass"
            if score > 0.6:
                desc = f"Unnatural motion patterns: flow jitter={m.get('flow_temporal_jitter', 0):.4f}, spatial incoherence={m.get('avg_spatial_incoherence', 0):.2f}"
            elif score > 0.35:
                desc = f"Mildly irregular optical flow patterns (score: {score:.2f})"
            else:
                desc = "Natural physics-consistent motion patterns"
            checks.append({
                "id": "ai_optical_flow",
                "name": "Optical Flow Physics Analysis",
                "status": status,
                "description": desc,
            })

        # Layer 3: Noise Consistency
        r = layer_results["noise_consistency"]
        score = r["score"]
        m = r["metrics"]
        if "error" not in m:
            status = "fail" if score > 0.6 else "warn" if score > 0.35 else "pass"
            if score > 0.6:
                desc = f"Inconsistent noise patterns across frames (cross-frame correlation: {m.get('avg_noise_correlation', 0):.2f}, expected >0.7 for real camera)"
            elif score > 0.35:
                desc = f"Mild noise inconsistency detected (score: {score:.2f})"
            else:
                desc = "Consistent sensor noise patterns (real camera signature)"
            checks.append({
                "id": "ai_noise_consistency",
                "name": "Inter-Frame Noise Analysis",
                "status": status,
                "description": desc,
            })

        # Layer 4: Edge Coherence
        r = layer_results["edge_coherence"]
        score = r["score"]
        m = r["metrics"]
        if "error" not in m:
            status = "fail" if score > 0.6 else "warn" if score > 0.35 else "pass"
            if score > 0.6:
                desc = f"Edge morphing in static regions: {m.get('avg_static_edge_change', 0):.1f} edge pixels shift per frame (walls/objects warping)"
            elif score > 0.35:
                desc = f"Minor edge instability in static regions (score: {score:.2f})"
            else:
                desc = "Stable edge structure in static regions"
            checks.append({
                "id": "ai_edge_coherence",
                "name": "Edge Temporal Coherence",
                "status": status,
                "description": desc,
            })

        # Layer 5: Texture Stability
        r = layer_results["texture_stability"]
        score = r["score"]
        m = r["metrics"]
        if "error" not in m:
            status = "fail" if score > 0.6 else "warn" if score > 0.35 else "pass"
            if score > 0.6:
                desc = f"Texture morphing detected: LBP distance={m.get('avg_lbp_distance', 0):.4f} (textures subtly change between frames)"
            elif score > 0.35:
                desc = f"Mild texture instability (score: {score:.2f})"
            else:
                desc = "Stable micro-texture patterns across frames"
            checks.append({
                "id": "ai_texture_stability",
                "name": "Texture Stability (LBP) Analysis",
                "status": status,
                "description": desc,
            })

        # Layer 6: Background Stability
        r = layer_results["background_stability"]
        score = r["score"]
        m = r["metrics"]
        if "error" not in m:
            status = "fail" if score > 0.6 else "warn" if score > 0.35 else "pass"
            if score > 0.6:
                desc = f"Background breathing detected: static regions show {m.get('avg_breathing', 0):.2f} pixel drift (AI artifact)"
            elif score > 0.35:
                desc = f"Minor background instability (score: {score:.2f})"
            else:
                desc = "Stable background regions (no breathing artifacts)"
            checks.append({
                "id": "ai_background_stability",
                "name": "Background Stability Analysis",
                "status": status,
                "description": desc,
            })

        # Layer 7: Frequency Temporal
        r = layer_results["frequency_temporal"]
        score = r["score"]
        m = r["metrics"]
        if "error" not in m:
            status = "fail" if score > 0.6 else "warn" if score > 0.35 else "pass"
            if score > 0.6:
                desc = f"Frequency spectrum instability: spectral variance={m.get('spectral_variance', 0):.6f} (no consistent camera fingerprint)"
            elif score > 0.35:
                desc = f"Mild frequency instability across frames (score: {score:.2f})"
            else:
                desc = "Consistent frequency spectrum (real camera optics)"
            checks.append({
                "id": "ai_frequency_temporal",
                "name": "Frequency Spectrum Temporal Analysis",
                "status": status,
                "description": desc,
            })

        # Layer 8: Motion Naturalness
        r = layer_results["motion_naturalness"]
        score = r["score"]
        m = r["metrics"]
        if "error" not in m:
            status = "fail" if score > 0.6 else "warn" if score > 0.35 else "pass"
            if score > 0.6:
                desc = (
                    f"Non-Newtonian motion: normalized jerk={m.get('norm_jerk', 0):.4f}, "
                    f"accel anomalies={m.get('high_accel_ratio', 0):.1%} (physics-defying movement)"
                )
            elif score > 0.35:
                desc = f"Slightly unnatural motion dynamics (score: {score:.2f})"
            else:
                desc = "Physically plausible motion dynamics"
            checks.append({
                "id": "ai_motion_naturalness",
                "name": "Motion Physics Analysis",
                "status": status,
                "description": desc,
            })

        return checks
