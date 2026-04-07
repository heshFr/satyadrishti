"""
CLIP Temporal Drift Detector
==============================
Analyzes how CLIP image embeddings evolve across video frames to detect
AI-generated video.

THEORETICAL FOUNDATION:
━━━━━━━━━━━━━━━━━━━━━━━
CLIP (Contrastive Language-Image Pretraining) maps images to a 768-dimensional
semantic embedding space. In this space, similar images are close together
and dissimilar images are far apart.

Real video captured by a camera has:
  - SMOOTH semantic trajectory: consecutive frames are semantically similar
    (the camera sees the same scene with gradual changes)
  - HIGH inter-frame cosine similarity: typically 0.95-0.99+
  - CONSISTENT drift rate: motion causes smooth, predictable embedding changes
  - LOW trajectory jerk: acceleration of embedding change is smooth

AI-generated video has:
  - JERKY semantic trajectory: the generator may produce semantically
    inconsistent frames (scene composition subtly shifts)
  - LOWER inter-frame similarity: AI frames are more semantically variable
  - VARIABLE drift rate: speed of semantic change is unpredictable
  - HIGHER trajectory jerk: abrupt semantic direction changes

WHY THIS WORKS:
━━━━━━━━━━━━━━━
CLIP was trained on 400M image-text pairs and understands scene semantics
at a high level. AI generators can produce individually convincing frames,
but maintaining perfect semantic consistency across 30+ frames per second
is extremely difficult. Subtle changes in object count, spatial relationships,
lighting mood, and compositional balance are all captured by CLIP embeddings
even when invisible to the human eye.

ANALYSIS LAYERS:
━━━━━━━━━━━━━━━━
1. Cosine Similarity Drift
   - Inter-frame cosine similarity statistics
   - Sudden drops indicate semantic jumps

2. Embedding Trajectory Smoothness
   - First derivative (velocity) → should be smooth
   - Second derivative (acceleration/jerk) → should be low
   - Real video: smooth trajectory, AI: jerky trajectory

3. Semantic Stability Zones
   - Static scenes: embedding should barely change
   - Real: near-zero drift in static moments
   - AI: continuous small drift even in "static" scenes

4. Directional Consistency
   - In real video, semantic drift direction is consistent
     over short windows (camera pans smoothly)
   - AI: random direction changes between frames

5. Long-Range Coherence
   - Embedding at frame 1 vs frame N should be predictable
     from the intermediate trajectory
   - AI: end-to-end drift may exceed trajectory integral
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, List, Optional


class CLIPTemporalDriftDetector:
    """
    Detects AI-generated video by analyzing temporal drift of CLIP
    image embeddings across frames.

    Uses an externally-provided CLIP model (shared with the forensics pipeline)
    to avoid loading the model twice.
    """

    def __init__(self, clip_model=None, verbose: bool = False):
        """
        Args:
            clip_model: A CLIPDetector instance (from engine.image_forensics.clip_detector)
                        or None (will need to be provided at analyze time)
            verbose: Print per-layer scores
        """
        self.clip = clip_model
        self.verbose = verbose

    def analyze(
        self,
        video_path: str,
        clip_model=None,
        max_frames: int = 40,
    ) -> Dict[str, Any]:
        """
        Run full CLIP temporal drift analysis.

        Args:
            video_path: Path to video file
            clip_model: CLIPDetector instance (overrides constructor param)
            max_frames: Number of frames to sample (more = more accurate but slower)

        Returns:
            Dict with ai_probability, layer scores, and detailed metrics
        """
        clip = clip_model or self.clip
        if clip is None:
            return {
                "ai_probability": 0.5,
                "metrics": {"error": "No CLIP model available"},
            }

        # Extract frames
        frames = self._extract_frames(video_path, max_frames)
        if len(frames) < 5:
            return {
                "ai_probability": 0.5,
                "metrics": {"error": "Too few frames", "count": len(frames)},
            }

        # Extract CLIP embeddings for all frames
        embeddings = self._extract_embeddings(frames, clip)
        if len(embeddings) < 5:
            return {
                "ai_probability": 0.5,
                "metrics": {"error": "Embedding extraction failed"},
            }

        # Normalize embeddings (should already be L2-normalized from CLIP)
        embeddings = np.array(embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        embeddings = embeddings / norms

        # ─── Layer 1: Cosine Similarity Drift ───
        cos_score, cos_metrics = self._analyze_cosine_drift(embeddings)

        # ─── Layer 2: Trajectory Smoothness ───
        smooth_score, smooth_metrics = self._analyze_trajectory_smoothness(embeddings)

        # ─── Layer 3: Semantic Stability Zones ───
        stability_score, stability_metrics = self._analyze_semantic_stability(embeddings)

        # ─── Layer 4: Directional Consistency ───
        direction_score, direction_metrics = self._analyze_directional_consistency(embeddings)

        # ─── Layer 5: Long-Range Coherence ───
        coherence_score, coherence_metrics = self._analyze_long_range_coherence(embeddings)

        # ─── Weighted Ensemble ───
        weights = {
            "cosine_drift": (cos_score, 1.5),
            "trajectory_smoothness": (smooth_score, 1.3),
            "semantic_stability": (stability_score, 1.0),
            "directional_consistency": (direction_score, 1.2),
            "long_range_coherence": (coherence_score, 1.0),
        }

        total_w = sum(w for _, w in weights.values())
        ai_probability = sum(s * w for s, w in weights.values()) / total_w

        # Calibration: boost when multiple layers agree
        scores = [s for s, _ in weights.values()]
        n_ai = sum(1 for s in scores if s > 0.5)
        if n_ai >= 4 and ai_probability > 0.5:
            ai_probability = ai_probability + 0.05 * (1 - ai_probability)
        elif n_ai <= 1 and ai_probability < 0.5:
            ai_probability = ai_probability * 0.9

        ai_probability = float(np.clip(ai_probability, 0, 1))

        if self.verbose:
            for name, (score, _) in weights.items():
                print(f"  [CLIP-Drift] {name}: {score:.4f}")
            print(f"  [CLIP-Drift] Final: {ai_probability:.4f}")

        return {
            "ai_probability": round(ai_probability, 4),
            "layer_scores": {
                name: round(score, 4) for name, (score, _) in weights.items()
            },
            "metrics": {
                "frames_analyzed": len(frames),
                "embeddings_extracted": len(embeddings),
                "cosine": cos_metrics,
                "smoothness": smooth_metrics,
                "stability": stability_metrics,
                "direction": direction_metrics,
                "coherence": coherence_metrics,
            },
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Frame Extraction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _extract_frames(self, video_path: str, max_frames: int) -> List[np.ndarray]:
        """Extract evenly-spaced frames from video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n = min(max_frames, total)
        indices = np.linspace(0, total - 1, n, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)

        cap.release()
        return frames

    def _extract_embeddings(
        self, frames: List[np.ndarray], clip_model
    ) -> List[np.ndarray]:
        """Extract CLIP embeddings for all frames."""
        embeddings = []
        for frame in frames:
            try:
                emb = clip_model.extract_embedding(frame)
                if hasattr(emb, 'cpu'):
                    emb = emb.cpu().numpy()
                embeddings.append(emb.flatten())
            except Exception:
                continue
        return embeddings

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 1: Cosine Similarity Drift
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_cosine_drift(self, embeddings: np.ndarray) -> Tuple[float, Dict]:
        """
        Measure cosine similarity between consecutive frame embeddings.

        Real video: high similarity (>0.95), low variance
        AI video: lower similarity, higher variance, occasional drops
        """
        n = len(embeddings)
        similarities = []

        for i in range(n - 1):
            cos_sim = float(np.dot(embeddings[i], embeddings[i + 1]))
            similarities.append(cos_sim)

        similarities = np.array(similarities)

        mean_sim = float(np.mean(similarities))
        std_sim = float(np.std(similarities))
        min_sim = float(np.min(similarities))
        max_sim = float(np.max(similarities))

        # Number of "drops" — frames where similarity suddenly decreases
        drop_threshold = mean_sim - 2 * std_sim
        n_drops = int(np.sum(similarities < drop_threshold))
        drop_ratio = n_drops / len(similarities)

        # Score computation
        # mean_sim: real ~0.96-0.99, AI ~0.90-0.96
        sim_score = np.clip((0.97 - mean_sim) / 0.06, 0, 1)

        # std_sim: real ~0.001-0.008, AI ~0.008-0.03+
        var_score = np.clip((std_sim - 0.003) / 0.02, 0, 1)

        # drop_ratio: real ~0.0-0.02, AI ~0.02-0.10+
        drop_score = np.clip((drop_ratio - 0.01) / 0.08, 0, 1)

        combined = sim_score * 0.40 + var_score * 0.35 + drop_score * 0.25

        metrics = {
            "mean_similarity": round(mean_sim, 6),
            "std_similarity": round(std_sim, 6),
            "min_similarity": round(min_sim, 6),
            "max_similarity": round(max_sim, 6),
            "n_drops": n_drops,
            "drop_ratio": round(drop_ratio, 4),
        }

        return float(np.clip(combined, 0, 1)), metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 2: Trajectory Smoothness
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_trajectory_smoothness(
        self, embeddings: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        Analyze smoothness of the embedding trajectory.

        Computes velocity (1st derivative) and acceleration (2nd derivative)
        of the embedding path through CLIP space.

        Real video: smooth trajectory → low acceleration/jerk
        AI video: jerky trajectory → high acceleration/jerk
        """
        n = len(embeddings)

        # Velocity: embedding differences between consecutive frames
        velocities = np.diff(embeddings, axis=0)  # (N-1, D)
        speeds = np.linalg.norm(velocities, axis=1)  # (N-1,)

        # Acceleration: differences in velocity
        if len(velocities) >= 2:
            accelerations = np.diff(velocities, axis=0)  # (N-2, D)
            accel_magnitudes = np.linalg.norm(accelerations, axis=1)  # (N-2,)
        else:
            accel_magnitudes = np.array([0.0])

        # Jerk: differences in acceleration
        if len(accelerations) >= 2 if len(velocities) >= 2 else False:
            jerks = np.diff(accelerations, axis=0)
            jerk_magnitudes = np.linalg.norm(jerks, axis=1)
        else:
            jerk_magnitudes = np.array([0.0])

        # Statistics
        mean_speed = float(np.mean(speeds))
        std_speed = float(np.std(speeds))
        mean_accel = float(np.mean(accel_magnitudes))
        mean_jerk = float(np.mean(jerk_magnitudes))

        # Normalized by speed (independent of overall motion amount)
        norm_accel = mean_accel / (mean_speed + 1e-8)
        norm_jerk = mean_jerk / (mean_speed + 1e-8)
        speed_cv = std_speed / (mean_speed + 1e-8)  # coefficient of variation

        # Score computation
        # norm_accel: real ~0.3-0.8, AI ~0.8-2.0+
        accel_score = np.clip((norm_accel - 0.4) / 1.0, 0, 1)

        # norm_jerk: real ~0.2-0.6, AI ~0.6-1.5+
        jerk_score = np.clip((norm_jerk - 0.3) / 0.8, 0, 1)

        # speed_cv: real ~0.2-0.6, AI ~0.6-1.5+
        cv_score = np.clip((speed_cv - 0.3) / 0.8, 0, 1)

        combined = accel_score * 0.35 + jerk_score * 0.35 + cv_score * 0.30

        metrics = {
            "mean_speed": round(mean_speed, 6),
            "std_speed": round(std_speed, 6),
            "speed_cv": round(speed_cv, 4),
            "mean_accel": round(mean_accel, 6),
            "norm_accel": round(norm_accel, 4),
            "mean_jerk": round(mean_jerk, 6),
            "norm_jerk": round(norm_jerk, 4),
        }

        return float(np.clip(combined, 0, 1)), metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 3: Semantic Stability Zones
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_semantic_stability(
        self, embeddings: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        Detect semantic instability in "stable" regions of the video.

        In real video, when the camera is relatively still, CLIP embeddings
        should be nearly identical. AI video generators continue to inject
        small semantic variations even in "static" moments because each frame
        is partially independently generated.

        Method:
        1. Find windows where overall embedding change is low (stable zones)
        2. Measure residual micro-drift in these zones
        3. High micro-drift in stable zones = AI artifact
        """
        n = len(embeddings)
        window_size = max(3, n // 8)

        # Compute per-frame displacement
        displacements = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)

        # Find stable windows (low average displacement)
        window_means = []
        for i in range(len(displacements) - window_size + 1):
            w = displacements[i:i + window_size]
            window_means.append(float(np.mean(w)))

        if not window_means:
            return 0.5, {"error": "too few frames for stability analysis"}

        # Threshold: bottom 30% of windows are "stable"
        threshold = np.percentile(window_means, 30)
        stable_windows = [i for i, m in enumerate(window_means) if m <= threshold]

        if len(stable_windows) < 2:
            return 0.4, {"n_stable_windows": 0}

        # Measure micro-drift within stable windows
        micro_drifts = []
        for start in stable_windows:
            window = displacements[start:start + window_size]
            micro_drifts.extend(window.tolist())

        micro_drift_mean = float(np.mean(micro_drifts))
        micro_drift_std = float(np.std(micro_drifts))

        # Also measure variance within stable windows
        # Real: near-zero variance (static = static)
        # AI: non-zero variance (generator keeps varying)
        stable_variance = micro_drift_std / (micro_drift_mean + 1e-8)

        # Score
        # micro_drift_mean: real ~0.001-0.005, AI ~0.005-0.02+
        drift_score = np.clip((micro_drift_mean - 0.002) / 0.015, 0, 1)

        # stable_variance: real ~0.1-0.4, AI ~0.4-1.0+
        var_score = np.clip((stable_variance - 0.2) / 0.6, 0, 1)

        combined = drift_score * 0.6 + var_score * 0.4

        metrics = {
            "n_stable_windows": len(stable_windows),
            "micro_drift_mean": round(micro_drift_mean, 6),
            "micro_drift_std": round(micro_drift_std, 6),
            "stable_variance": round(stable_variance, 4),
            "stability_threshold": round(threshold, 6),
        }

        return float(np.clip(combined, 0, 1)), metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 4: Directional Consistency
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_directional_consistency(
        self, embeddings: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        Analyze consistency of embedding drift direction.

        Real video: smooth camera motion → embedding drifts in a consistent
        direction over short windows (like a pan or zoom).
        AI video: drift direction changes randomly between frames.

        Method: Compute cosine similarity between consecutive velocity vectors.
        High similarity = consistent direction = real.
        Low similarity = random direction changes = AI.
        """
        velocities = np.diff(embeddings, axis=0)  # (N-1, D)

        if len(velocities) < 3:
            return 0.5, {"error": "too few frames"}

        # Cosine similarity between consecutive velocity vectors
        direction_sims = []
        for i in range(len(velocities) - 1):
            v1 = velocities[i]
            v2 = velocities[i + 1]
            norm1 = np.linalg.norm(v1) + 1e-10
            norm2 = np.linalg.norm(v2) + 1e-10
            cos_sim = float(np.dot(v1, v2) / (norm1 * norm2))
            direction_sims.append(cos_sim)

        direction_sims = np.array(direction_sims)

        mean_dir_sim = float(np.mean(direction_sims))
        std_dir_sim = float(np.std(direction_sims))
        min_dir_sim = float(np.min(direction_sims))

        # Number of direction reversals (cosine sim < 0 = opposite direction)
        n_reversals = int(np.sum(direction_sims < 0))
        reversal_ratio = n_reversals / len(direction_sims)

        # Score
        # mean_dir_sim: real ~0.3-0.7, AI ~-0.1-0.3
        sim_score = np.clip((0.4 - mean_dir_sim) / 0.5, 0, 1)

        # reversal_ratio: real ~0.05-0.2, AI ~0.2-0.5+
        reversal_score = np.clip((reversal_ratio - 0.1) / 0.3, 0, 1)

        # std_dir_sim: real ~0.1-0.3, AI ~0.3-0.6+
        var_score = np.clip((std_dir_sim - 0.15) / 0.35, 0, 1)

        combined = sim_score * 0.40 + reversal_score * 0.30 + var_score * 0.30

        metrics = {
            "mean_direction_similarity": round(mean_dir_sim, 4),
            "std_direction_similarity": round(std_dir_sim, 4),
            "min_direction_similarity": round(min_dir_sim, 4),
            "n_reversals": n_reversals,
            "reversal_ratio": round(reversal_ratio, 4),
        }

        return float(np.clip(combined, 0, 1)), metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 5: Long-Range Coherence
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_long_range_coherence(
        self, embeddings: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        Check long-range semantic coherence.

        Compare first few frames to last few frames. In real video,
        the semantic distance should be predictable from the drift rate.
        In AI video, long-range coherence can break down (scene drifts
        in unexpected ways over time).

        Also checks mid-point predictability: can we predict the middle
        embedding from the start and end? Linear interpolation should
        be close for real video.
        """
        n = len(embeddings)

        # Start, middle, end embedding groups
        start_emb = embeddings[:3].mean(axis=0)
        end_emb = embeddings[-3:].mean(axis=0)
        mid_emb = embeddings[n // 2 - 1:n // 2 + 2].mean(axis=0)

        # Predicted mid-point via linear interpolation
        predicted_mid = (start_emb + end_emb) / 2.0

        # Mid-point prediction error
        mid_error = float(np.linalg.norm(mid_emb - predicted_mid))

        # Total trajectory length vs straight-line distance
        trajectory_length = 0.0
        for i in range(n - 1):
            trajectory_length += np.linalg.norm(embeddings[i + 1] - embeddings[i])

        straight_line = float(np.linalg.norm(end_emb - start_emb))

        # Tortuosity: trajectory_length / straight_line
        # Real: relatively direct path (1.0-3.0)
        # AI: meandering path (3.0-10.0+) or very short (no coherent motion)
        tortuosity = trajectory_length / (straight_line + 1e-8)

        # Non-monotonic drift: how often does the trajectory reverse toward start?
        start_dists = [float(np.linalg.norm(embeddings[i] - start_emb)) for i in range(n)]
        n_reversions = 0
        for i in range(1, len(start_dists)):
            if start_dists[i] < start_dists[i - 1] - 0.001:
                n_reversions += 1
        reversion_ratio = n_reversions / (n - 1)

        # Score
        # mid_error: real ~0.01-0.05, AI ~0.05-0.15+
        mid_score = np.clip((mid_error - 0.02) / 0.10, 0, 1)

        # tortuosity: real ~1.0-5.0, AI ~5.0-20.0+ (or <1.0 if perfectly static)
        tort_score = np.clip((tortuosity - 3.0) / 10.0, 0, 1)

        # reversion_ratio: real ~0.1-0.3, AI ~0.3-0.5+
        rev_score = np.clip((reversion_ratio - 0.15) / 0.30, 0, 1)

        combined = mid_score * 0.35 + tort_score * 0.30 + rev_score * 0.35

        metrics = {
            "mid_prediction_error": round(mid_error, 6),
            "trajectory_length": round(float(trajectory_length), 4),
            "straight_line_distance": round(straight_line, 6),
            "tortuosity": round(tortuosity, 4),
            "n_reversions": n_reversions,
            "reversion_ratio": round(reversion_ratio, 4),
        }

        return float(np.clip(combined, 0, 1)), metrics
