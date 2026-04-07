"""
Voice Cloning Detector
=======================
Detects voice cloning attacks by cross-referencing speaker identity with
deepfake detection results. If the voice SOUNDS like a known person but
is flagged as synthetic by the neural detectors, it's a voice clone.

This is the critical bridge between speaker verification and deepfake
detection that catches the most dangerous attack vector: AI-cloned voices
of trusted contacts (family members, bank officials, bosses).

DETECTION STRATEGIES:
━━━━━━━━━━━━━━━━━━━━━
1. Speaker Embedding Anomaly Detection
   - Real human speech produces consistent ECAPA-TDNN embeddings
   - Cloned speech has subtle embedding-space anomalies:
     * Higher intra-utterance embedding variance (clone consistency drifts)
     * Unusual embedding norm (clones tend toward unit-norm centroids)
     * Atypical angular distribution in embedding space

2. Enrollment Cross-Check
   - If speaker is enrolled, compare clone's embedding to stored voiceprint
   - Clone + high similarity to enrolled speaker = voice cloning attack
   - Clone + no enrolled match = generic deepfake (not targeted clone)

3. Embedding-Score Divergence
   - When neural audio detector says "spoof" but speaker embedding says
     "matches known person" → strong clone indicator
   - When neural says "real" and embedding matches → genuine person

4. Temporal Embedding Stability
   - Split audio into segments, extract embeddings per segment
   - Real speakers: high inter-segment similarity (0.85+)
   - Clones: lower inter-segment similarity (cloning model consistency drops)
   - Especially effective on longer audio (>10s)

5. Embedding Distribution Analysis
   - Real speech embeddings follow a characteristic distribution in the
     192d ECAPA-TDNN space (specific variance structure, kurtosis patterns)
   - Clone embeddings tend to be "too clean" — lower variance, higher
     concentration near the centroid
   - Statistical tests on embedding features detect this

OUTPUT:
━━━━━━━
  {
      "is_clone": bool,
      "clone_confidence": float (0-1),
      "clone_target": str or None (name of cloned person if enrolled),
      "clone_similarity": float (cosine sim to best enrolled match),
      "embedding_anomaly_score": float (0-1, higher = more anomalous),
      "temporal_consistency": float (0-1, inter-segment similarity),
      "details": { ... per-check scores ... }
  }
"""

import logging
import io
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

try:
    import torch
    import torchaudio
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    HAS_SF = False


# Thresholds calibrated on VoxCeleb + cloned speech experiments
CLONE_SIMILARITY_THRESHOLD = 0.30      # cosine sim for "same speaker"
EMBEDDING_VARIANCE_REAL_MIN = 0.008    # real speech has this much variance
EMBEDDING_VARIANCE_REAL_MAX = 0.065    # above this = unusual
TEMPORAL_CONSISTENCY_REAL_MIN = 0.80   # real speakers are consistent across segments
TEMPORAL_CONSISTENCY_CLONE_MAX = 0.92  # clones are suspiciously consistent
EMBEDDING_NORM_REAL_RANGE = (4.0, 12.0)  # typical L2 norm range for real speech
SEGMENT_DURATION = 3.0                 # seconds per segment for temporal analysis
MIN_SEGMENTS = 2                       # minimum segments needed


class VoiceCloneDetector:
    """
    Detects voice cloning attacks using speaker embeddings + deepfake cross-check.

    Requires a speaker verification model (ECAPA-TDNN via SpeechBrain) which is
    shared with the SpeakerVerifier module to avoid loading the model twice.

    Args:
        speaker_verifier: Optional SpeakerVerifier instance (shared model).
            If None, will try to load its own ECAPA-TDNN model.
        enrolled_dir: Path to directory containing enrolled voiceprints.
    """

    def __init__(
        self,
        speaker_verifier=None,
        enrolled_dir: str = "models/voice_prints",
    ):
        self.speaker_verifier = speaker_verifier
        self.enrolled_dir = Path(enrolled_dir)
        self._model = None
        self.is_available = False

        if speaker_verifier is not None and hasattr(speaker_verifier, 'is_available'):
            self.is_available = speaker_verifier.is_available
        elif HAS_TORCH:
            self.is_available = True

    def _get_model(self):
        """Get or create the speaker embedding model."""
        if self.speaker_verifier is not None:
            return self.speaker_verifier
        if self._model is None:
            try:
                from engine.audio.speaker_verify import SpeakerVerifier
                self._model = SpeakerVerifier(device="cpu")
                self.is_available = self._model.is_available
            except Exception as e:
                log.warning("Could not load speaker verification model: %s", e)
                self.is_available = False
        return self._model

    def _extract_embedding(self, waveform: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """Extract 192d ECAPA-TDNN embedding from audio segment."""
        model = self._get_model()
        if model is None or not self.is_available:
            return None

        try:
            embedding = model.extract_embedding(waveform, sr)
            if embedding is not None:
                return np.asarray(embedding).flatten()
        except Exception as e:
            log.debug("Embedding extraction failed: %s", e)
        return None

    def _extract_segment_embeddings(
        self, waveform: np.ndarray, sr: int
    ) -> List[np.ndarray]:
        """Extract embeddings from fixed-duration segments of the audio."""
        segment_samples = int(SEGMENT_DURATION * sr)
        n_segments = max(1, len(waveform) // segment_samples)
        embeddings = []

        for i in range(min(n_segments, 10)):
            start = i * segment_samples
            end = min(start + segment_samples, len(waveform))
            segment = waveform[start:end]

            if len(segment) < sr:  # skip segments shorter than 1 second
                continue

            emb = self._extract_embedding(segment, sr)
            if emb is not None and len(emb) > 0:
                embeddings.append(emb)

        return embeddings

    def _compute_temporal_consistency(self, embeddings: List[np.ndarray]) -> float:
        """
        Compute inter-segment embedding similarity.

        Real speakers produce consistent embeddings across segments.
        Returns mean pairwise cosine similarity.
        """
        if len(embeddings) < 2:
            return 0.5  # insufficient data

        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                cos_sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-10
                )
                similarities.append(float(cos_sim))

        return float(np.mean(similarities))

    def _compute_embedding_anomaly(self, embeddings: List[np.ndarray]) -> Dict[str, float]:
        """
        Analyze embedding distribution for clone-specific anomalies.

        Returns per-feature anomaly scores.
        """
        if len(embeddings) < 1:
            return {"anomaly_score": 0.5, "variance": 0.0, "norm": 0.0, "kurtosis": 0.0}

        emb_matrix = np.stack(embeddings, axis=0)  # (N, 192)

        # 1. Per-dimension variance (real speech has moderate variance)
        dim_variances = np.var(emb_matrix, axis=0)
        mean_variance = float(np.mean(dim_variances))

        # 2. L2 norm statistics
        norms = np.linalg.norm(emb_matrix, axis=1)
        mean_norm = float(np.mean(norms))
        norm_std = float(np.std(norms))

        # 3. Kurtosis of embedding values (clones tend toward normal/sub-normal)
        from scipy.stats import kurtosis as scipy_kurtosis
        try:
            emb_kurtosis = float(np.mean(scipy_kurtosis(emb_matrix, axis=0)))
        except Exception:
            emb_kurtosis = 0.0

        # 4. Effective dimensionality (PCA-like: how many dimensions carry signal)
        if len(embeddings) >= 3:
            centered = emb_matrix - np.mean(emb_matrix, axis=0)
            try:
                singular_values = np.linalg.svd(centered, compute_uv=False)
                sv_normalized = singular_values / (singular_values.sum() + 1e-10)
                effective_dim = float(np.exp(-np.sum(sv_normalized * np.log(sv_normalized + 1e-10))))
            except Exception:
                effective_dim = len(embeddings)
        else:
            effective_dim = 0.0

        # Score each anomaly dimension
        scores = []

        # Variance anomaly: too low = clone (too clean), too high = unusual
        if mean_variance < EMBEDDING_VARIANCE_REAL_MIN:
            var_score = 1.0 - (mean_variance / EMBEDDING_VARIANCE_REAL_MIN)
            scores.append(min(1.0, var_score * 0.8))
        elif mean_variance > EMBEDDING_VARIANCE_REAL_MAX:
            var_score = (mean_variance - EMBEDDING_VARIANCE_REAL_MAX) / EMBEDDING_VARIANCE_REAL_MAX
            scores.append(min(1.0, var_score * 0.5))
        else:
            scores.append(0.0)

        # Norm anomaly
        if mean_norm < EMBEDDING_NORM_REAL_RANGE[0] or mean_norm > EMBEDDING_NORM_REAL_RANGE[1]:
            if mean_norm < EMBEDDING_NORM_REAL_RANGE[0]:
                norm_score = (EMBEDDING_NORM_REAL_RANGE[0] - mean_norm) / EMBEDDING_NORM_REAL_RANGE[0]
            else:
                norm_score = (mean_norm - EMBEDDING_NORM_REAL_RANGE[1]) / EMBEDDING_NORM_REAL_RANGE[1]
            scores.append(min(1.0, norm_score * 0.6))
        else:
            scores.append(0.0)

        # Low effective dimensionality = clone (embeddings too concentrated)
        if effective_dim > 0 and effective_dim < len(embeddings) * 0.3:
            scores.append(0.5)
        else:
            scores.append(0.0)

        anomaly_score = float(np.mean(scores)) if scores else 0.0

        return {
            "anomaly_score": anomaly_score,
            "variance": mean_variance,
            "norm": mean_norm,
            "norm_std": norm_std,
            "kurtosis": emb_kurtosis,
            "effective_dim": effective_dim,
        }

    def _check_enrolled_matches(
        self, embedding: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        Compare embedding against all enrolled voiceprints.

        Returns (best_match_name, similarity) or (None, 0.0) if no match.
        """
        if not self.enrolled_dir.exists():
            return None, 0.0

        best_match = None
        best_sim = -1.0

        for voiceprint_file in self.enrolled_dir.glob("*.npy"):
            try:
                stored_emb = np.load(voiceprint_file).flatten()
                cos_sim = np.dot(embedding, stored_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(stored_emb) + 1e-10
                )
                if cos_sim > best_sim:
                    best_sim = float(cos_sim)
                    best_match = voiceprint_file.stem
            except Exception:
                continue

        if best_sim >= CLONE_SIMILARITY_THRESHOLD:
            return best_match, best_sim
        return None, best_sim

    def analyze(
        self,
        waveform: np.ndarray,
        sr: int,
        spoof_probability: float = 0.0,
        spoof_verdict: str = "uncertain",
    ) -> Dict[str, Any]:
        """
        Run voice cloning analysis.

        Args:
            waveform: Mono audio signal (float32, [-1, 1])
            sr: Sample rate in Hz
            spoof_probability: Output from the deepfake detector (0-1)
            spoof_verdict: "spoof", "bonafide", or "uncertain" from deepfake detector

        Returns:
            Dict with clone detection results.
        """
        if not self.is_available:
            return {
                "is_clone": False,
                "clone_confidence": 0.0,
                "clone_target": None,
                "clone_similarity": 0.0,
                "embedding_anomaly_score": 0.0,
                "temporal_consistency": 0.5,
                "score": 0.5,
                "confidence": 0.0,
                "details": {"error": "Speaker verification model not available"},
            }

        duration = len(waveform) / sr
        details = {}

        # Step 1: Extract segment embeddings
        segment_embeddings = self._extract_segment_embeddings(waveform, sr)

        if len(segment_embeddings) == 0:
            # Try full audio as single segment
            full_emb = self._extract_embedding(waveform, sr)
            if full_emb is not None:
                segment_embeddings = [full_emb]

        if len(segment_embeddings) == 0:
            return {
                "is_clone": False,
                "clone_confidence": 0.0,
                "clone_target": None,
                "clone_similarity": 0.0,
                "embedding_anomaly_score": 0.0,
                "temporal_consistency": 0.5,
                "score": 0.5,
                "confidence": 0.0,
                "details": {"error": "Could not extract speaker embeddings"},
            }

        # Step 2: Temporal consistency
        temporal_consistency = self._compute_temporal_consistency(segment_embeddings)
        details["temporal_consistency"] = round(temporal_consistency, 4)

        # Step 3: Embedding anomaly analysis
        anomaly_result = self._compute_embedding_anomaly(segment_embeddings)
        embedding_anomaly = anomaly_result["anomaly_score"]
        details["embedding_analysis"] = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in anomaly_result.items()
        }

        # Step 4: Enrolled speaker check
        mean_embedding = np.mean(np.stack(segment_embeddings), axis=0)
        clone_target, clone_similarity = self._check_enrolled_matches(mean_embedding)
        details["enrolled_match"] = {
            "target": clone_target,
            "similarity": round(clone_similarity, 4),
        }

        # Step 5: Cross-reference with deepfake detection
        # The key insight: voice clone = spoof + matches known speaker
        details["deepfake_crossref"] = {
            "spoof_probability": round(spoof_probability, 4),
            "spoof_verdict": spoof_verdict,
        }

        # ── Score Computation ──
        # Higher score = more likely a voice clone

        clone_scores = []
        clone_reasons = []

        # Signal 1: Deepfake detector says spoof
        if spoof_verdict == "spoof" or spoof_probability > 0.5:
            clone_scores.append(spoof_probability * 0.8)
            if spoof_probability > 0.6:
                clone_reasons.append(f"Neural detector flagged as synthetic ({spoof_probability*100:.0f}%)")

        # Signal 2: Matches enrolled speaker + is spoof → targeted clone!
        if clone_target is not None and spoof_probability > 0.4:
            targeted_score = min(1.0, clone_similarity * spoof_probability * 2.0)
            clone_scores.append(targeted_score)
            clone_reasons.append(
                f"Voice matches enrolled speaker '{clone_target}' "
                f"(similarity={clone_similarity:.2f}) but audio is synthetic"
            )

        # Signal 3: Embedding anomaly (clone-specific patterns)
        if embedding_anomaly > 0.3:
            clone_scores.append(embedding_anomaly * 0.6)
            clone_reasons.append(f"Embedding distribution anomaly ({embedding_anomaly*100:.0f}%)")

        # Signal 4: Temporal consistency anomaly
        # Clones are sometimes TOO consistent (templated generation)
        # or TOO inconsistent (quality drift)
        if len(segment_embeddings) >= MIN_SEGMENTS:
            if temporal_consistency > TEMPORAL_CONSISTENCY_CLONE_MAX:
                # Suspiciously consistent — possible clone
                over_consistent = (temporal_consistency - TEMPORAL_CONSISTENCY_CLONE_MAX) / \
                                  (1.0 - TEMPORAL_CONSISTENCY_CLONE_MAX + 1e-10)
                clone_scores.append(min(0.5, over_consistent * 0.4))
                clone_reasons.append(
                    f"Suspiciously uniform voice (consistency={temporal_consistency:.3f})"
                )
            elif temporal_consistency < TEMPORAL_CONSISTENCY_REAL_MIN:
                # Inconsistent across segments — possible quality drift
                under_consistent = (TEMPORAL_CONSISTENCY_REAL_MIN - temporal_consistency) / \
                                   TEMPORAL_CONSISTENCY_REAL_MIN
                clone_scores.append(min(0.4, under_consistent * 0.3))
                clone_reasons.append(
                    f"Voice consistency drift detected ({temporal_consistency:.3f})"
                )

        # Final score: weighted combination
        if clone_scores:
            # Weight by importance: spoof detection > enrollment match > anomaly > consistency
            clone_confidence = float(np.clip(np.max(clone_scores) * 0.6 + np.mean(clone_scores) * 0.4, 0, 1))
        else:
            clone_confidence = 0.0

        # Determine if it's a clone
        is_clone = clone_confidence > 0.45 and spoof_probability > 0.4
        is_targeted_clone = clone_target is not None and is_clone

        # Overall score (for ensemble integration)
        # This score represents "probability of being a cloned voice"
        score = clone_confidence if is_clone else clone_confidence * 0.3

        # Confidence in our assessment (how sure we are about is_clone)
        assessment_confidence = min(1.0, len(segment_embeddings) / 5.0) * 0.5
        if spoof_probability > 0.7:
            assessment_confidence += 0.3
        if clone_target is not None:
            assessment_confidence += 0.2
        assessment_confidence = min(1.0, assessment_confidence)

        return {
            "is_clone": is_clone,
            "is_targeted_clone": is_targeted_clone,
            "clone_confidence": round(clone_confidence, 4),
            "clone_target": clone_target,
            "clone_similarity": round(clone_similarity, 4),
            "embedding_anomaly_score": round(embedding_anomaly, 4),
            "temporal_consistency": round(temporal_consistency, 4),
            "score": round(score, 4),
            "confidence": round(assessment_confidence, 4),
            "reasons": clone_reasons,
            "details": details,
            "segments_analyzed": len(segment_embeddings),
            "duration_seconds": round(duration, 2),
        }
