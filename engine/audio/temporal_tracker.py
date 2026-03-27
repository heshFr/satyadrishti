"""
Temporal Speaker Consistency Tracker
=====================================
Tracks speaker embedding consistency and prosodic stability over the
duration of a live call. AI-generated voices exhibit either unnaturally
stable speaker embeddings (cosine similarity >0.99, near-zero variance)
or sudden instabilities (embedding shifts mid-call), both of which
diverge from natural speech patterns.

This is a stateful class that accumulates evidence over a call and
produces an anomaly score with increasing confidence as more audio
chunks arrive.
"""

import logging
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class TemporalTracker:
    """
    Tracks speaker consistency over time during a live call.

    Called every ~3 seconds with a new speaker embedding and optional
    prosodic/score data. Maintains a sliding window of recent chunks
    and accumulates long-range statistics.

    Natural speakers:
        - Embedding cosine similarity 0.85–0.98, variance 0.001–0.01
        - Stable mean F0 with natural micro-variation
    AI voices:
        - Either too stable (similarity >0.99, variance <0.001)
          or unstable (sudden embedding drops)
        - F0 drift or unnatural consistency

    Args:
        window_size: Number of chunks to keep in the sliding window.
    """

    # --- Thresholds ---
    NATURAL_SIM_LOW = 0.85
    NATURAL_SIM_HIGH = 0.97
    NATURAL_VAR_LOW = 0.001
    NATURAL_VAR_HIGH = 0.01
    SUDDEN_CHANGE_THRESHOLD = 0.2  # cosine sim drop triggering a "sudden change"
    TOO_STABLE_SIM = 0.985
    TOO_STABLE_VAR = 0.001
    MIN_CONFIDENCE_CHUNKS = 5  # minimum chunks before confidence is reasonable

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        embedding: np.ndarray,
        f0_stats: Optional[dict] = None,
        chunk_score: Optional[float] = None,
    ) -> dict:
        """
        Ingest a new chunk of audio analysis data.

        Args:
            embedding: 192d or 768d speaker embedding from ECAPA-TDNN or AST.
            f0_stats: Optional dict with ``f0_mean`` and ``f0_std`` from
                      the prosodic analyzer.
            chunk_score: Optional per-chunk spoof score from the ensemble
                         (0 = bonafide, 1 = spoof).

        Returns:
            Dictionary with temporal anomaly assessment::

                {
                    "score": float,         # 0 = consistent/bonafide, 1 = anomalous/spoof
                    "confidence": float,     # increases with more data
                    "features": {
                        "embedding_stability": float,
                        "embedding_variance": float,
                        "f0_drift": float,
                        "score_consistency": float,
                        "sudden_changes": int,
                        "chunks_analyzed": int,
                    },
                    "anomalies": list[str],
                }
        """
        embedding = np.asarray(embedding, dtype=np.float64).ravel()

        # Store data
        self._embeddings.append(embedding)
        if f0_stats is not None:
            self._f0_means.append(f0_stats.get("f0_mean", 0.0))
            self._f0_stds.append(f0_stats.get("f0_std", 0.0))
        if chunk_score is not None:
            self._chunk_scores.append(float(chunk_score))

        self._chunks_analyzed += 1

        # Compute features
        features = self._compute_features()
        anomalies = self._detect_anomalies(features)
        score = self._compute_anomaly_score(features, anomalies)
        confidence = self._compute_confidence()

        result = {
            "score": float(np.clip(score, 0.0, 1.0)),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "features": features,
            "anomalies": anomalies,
        }

        logger.debug(
            "TemporalTracker chunk %d: score=%.3f confidence=%.3f anomalies=%s",
            self._chunks_analyzed,
            result["score"],
            result["confidence"],
            anomalies,
        )

        return result

    def reset(self):
        """Reset all state for a new call."""
        self._embeddings: deque[np.ndarray] = deque(maxlen=self.window_size)
        self._f0_means: deque[float] = deque(maxlen=self.window_size)
        self._f0_stds: deque[float] = deque(maxlen=self.window_size)
        self._chunk_scores: deque[float] = deque(maxlen=self.window_size)
        self._cosine_sims: deque[float] = deque(maxlen=self.window_size)
        self._sudden_changes: int = 0
        self._chunks_analyzed: int = 0
        logger.debug("TemporalTracker reset")

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors, safe against zero norms."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _compute_features(self) -> dict:
        """Compute all temporal features from the current window."""
        n = len(self._embeddings)

        # --- Embedding stability & variance ---
        if n >= 2:
            latest_sim = self._cosine_similarity(
                self._embeddings[-1], self._embeddings[-2]
            )
            self._cosine_sims.append(latest_sim)

            # Check for sudden change
            if latest_sim < (1.0 - self.SUDDEN_CHANGE_THRESHOLD):
                self._sudden_changes += 1
                logger.info(
                    "Sudden embedding shift detected: cosine_sim=%.4f (chunk %d)",
                    latest_sim,
                    self._chunks_analyzed,
                )

        if len(self._cosine_sims) > 0:
            sims = np.array(self._cosine_sims)
            embedding_stability = float(np.mean(sims))
            embedding_variance = float(np.var(sims))
        else:
            embedding_stability = 1.0
            embedding_variance = 0.0

        # --- F0 drift ---
        f0_drift = 0.0
        if len(self._f0_means) >= 3:
            f0_arr = np.array(self._f0_means)
            # Linear regression slope as drift indicator (Hz per chunk)
            x = np.arange(len(f0_arr), dtype=np.float64)
            if np.std(x) > 0 and np.std(f0_arr) > 0:
                slope = np.polyfit(x, f0_arr, 1)[0]
                # Normalize by mean F0 to get relative drift
                mean_f0 = np.mean(f0_arr)
                if mean_f0 > 1.0:
                    f0_drift = float(abs(slope) / mean_f0)
                else:
                    f0_drift = float(abs(slope))

        # --- Score consistency ---
        score_consistency = 1.0
        if len(self._chunk_scores) >= 2:
            scores_arr = np.array(self._chunk_scores)
            score_consistency = float(1.0 - np.std(scores_arr))
            score_consistency = max(0.0, score_consistency)

        return {
            "embedding_stability": embedding_stability,
            "embedding_variance": embedding_variance,
            "f0_drift": f0_drift,
            "score_consistency": score_consistency,
            "sudden_changes": self._sudden_changes,
            "chunks_analyzed": self._chunks_analyzed,
        }

    def _detect_anomalies(self, features: dict) -> list:
        """Identify specific anomaly patterns from computed features."""
        anomalies = []

        if self._chunks_analyzed < 2:
            return anomalies

        stability = features["embedding_stability"]
        variance = features["embedding_variance"]

        # Unnaturally stable (robotic / cloned voice)
        if stability > self.TOO_STABLE_SIM and variance < self.TOO_STABLE_VAR:
            anomalies.append(
                f"unnaturally_stable_embeddings (sim={stability:.4f}, var={variance:.6f})"
            )

        # Unnaturally unstable
        if stability < self.NATURAL_SIM_LOW:
            anomalies.append(
                f"unstable_embeddings (sim={stability:.4f})"
            )

        # Variance outside natural range (but not the stable case above)
        if variance > self.NATURAL_VAR_HIGH:
            anomalies.append(
                f"high_embedding_variance (var={variance:.6f})"
            )

        # Sudden changes
        if features["sudden_changes"] > 0:
            anomalies.append(
                f"sudden_embedding_shifts (count={features['sudden_changes']})"
            )

        # F0 drift
        if features["f0_drift"] > 0.05:
            anomalies.append(
                f"f0_drift (relative_slope={features['f0_drift']:.4f})"
            )

        # Inconsistent chunk scores (analyzer disagreeing with itself over time)
        if features["score_consistency"] < 0.6 and len(self._chunk_scores) >= 3:
            anomalies.append(
                f"inconsistent_chunk_scores (consistency={features['score_consistency']:.3f})"
            )

        return anomalies

    def _compute_anomaly_score(self, features: dict, anomalies: list) -> float:
        """
        Combine feature signals into a single anomaly score.

        0 = consistent / bonafide, 1 = anomalous / likely spoof.
        """
        if self._chunks_analyzed < 2:
            # Not enough data — return neutral
            return 0.5

        sub_scores = []

        # --- Embedding stability sub-score ---
        stability = features["embedding_stability"]
        variance = features["embedding_variance"]

        if stability > self.TOO_STABLE_SIM and variance < self.TOO_STABLE_VAR:
            # Unnaturally stable → high anomaly
            stability_score = 0.7 + 0.3 * min(1.0, (stability - self.TOO_STABLE_SIM) / 0.009)
        elif stability < self.NATURAL_SIM_LOW:
            # Unstable → high anomaly
            stability_score = 0.6 + 0.4 * min(1.0, (self.NATURAL_SIM_LOW - stability) / 0.3)
        elif self.NATURAL_SIM_LOW <= stability <= self.NATURAL_SIM_HIGH:
            # Natural range
            stability_score = 0.1
        else:
            # Between NATURAL_SIM_HIGH and TOO_STABLE_SIM — slightly suspicious
            stability_score = 0.1 + 0.5 * (stability - self.NATURAL_SIM_HIGH) / (
                self.TOO_STABLE_SIM - self.NATURAL_SIM_HIGH
            )
        sub_scores.append(("stability", stability_score, 0.35))

        # --- Variance sub-score ---
        if variance < self.NATURAL_VAR_LOW:
            var_score = 0.5 + 0.5 * min(1.0, (self.NATURAL_VAR_LOW - variance) / self.NATURAL_VAR_LOW)
        elif variance > self.NATURAL_VAR_HIGH:
            var_score = 0.4 + 0.6 * min(1.0, (variance - self.NATURAL_VAR_HIGH) / 0.05)
        else:
            var_score = 0.1
        sub_scores.append(("variance", var_score, 0.15))

        # --- Sudden changes sub-score ---
        sudden_score = min(1.0, features["sudden_changes"] * 0.3)
        sub_scores.append(("sudden_changes", sudden_score, 0.20))

        # --- F0 drift sub-score ---
        drift = features["f0_drift"]
        drift_score = min(1.0, drift / 0.15)
        sub_scores.append(("f0_drift", drift_score, 0.15))

        # --- Score consistency sub-score ---
        if len(self._chunk_scores) >= 2:
            consistency_score = 1.0 - features["score_consistency"]
        else:
            consistency_score = 0.0
        sub_scores.append(("score_consistency", consistency_score, 0.15))

        # Weighted average
        total_weight = sum(w for _, _, w in sub_scores)
        if total_weight < 1e-12:
            return 0.5
        score = sum(s * w for _, s, w in sub_scores) / total_weight

        return score

    def _compute_confidence(self) -> float:
        """
        Confidence ramps up as more chunks are analyzed.

        Low confidence for the first MIN_CONFIDENCE_CHUNKS chunks,
        reaching ~0.9 at 2x MIN_CONFIDENCE_CHUNKS, and saturating
        near 1.0 for long calls.
        """
        n = self._chunks_analyzed
        if n <= 1:
            return 0.1
        # Logarithmic ramp: reaches ~0.5 at MIN_CONFIDENCE_CHUNKS, ~0.9 at 2x
        confidence = 1.0 - 1.0 / (1.0 + 0.3 * (n - 1))
        return min(confidence, 0.98)
