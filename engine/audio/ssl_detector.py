"""
XLS-R 300M Self-Supervised Learning Detector (Layer 2)
=======================================================
Uses Facebook's XLS-R 300M (wav2vec2-based) model to extract deep
speech representations for deepfake detection.

Self-supervised speech models learn rich acoustic representations from
680K+ hours of unlabeled speech across 128 languages. Synthetic speech
from vocoders/TTS produces subtly different representation patterns:

  - **Temporal smoothness**: TTS outputs are smoother frame-to-frame
    in SSL space; real speech has higher micro-variation.
  - **Layer dynamics**: Real speech representation evolves differently
    across transformer layers than vocoder output.
  - **Distributional shape**: Real speech hidden states tend toward
    leptokurtic distributions; synthesis is often more platykurtic.
  - **Inter-frame consistency**: Consecutive frames in real speech
    have moderate cosine similarity (~0.85-0.95); synthetic can be
    unnaturally high (>0.98) or structurally repetitive.

Model: facebook/wav2vec2-xls-r-300m
  - 24 transformer layers, hidden_size=1024
  - 7-layer CNN feature extractor (raw waveform → 512d)
  - 300M parameters, ~1.2GB
  - Sampling rate: 16kHz
  - Runs on CPU (adequate for single-file analysis)
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


# ── Known norms for real speech in XLS-R feature space ──
# Derived from literature on SSL representations of natural speech.
# These serve as reference anchors; scores measure deviation from them.
REAL_SPEECH_NORMS = {
    # Temporal: frame-to-frame cosine similarity between consecutive hidden states
    "temporal_similarity_mean": (0.88, 0.96),   # (low, high) for real speech
    "temporal_similarity_std": (0.015, 0.06),    # real speech has moderate variation

    # Kurtosis of hidden state activations (across embedding dims)
    "activation_kurtosis": (2.0, 6.0),           # real speech: meso- to leptokurtic

    # Variance of hidden states across time (per-dim, then averaged)
    "temporal_variance": (0.02, 0.15),            # real speech: moderate variance

    # Cross-layer cosine similarity (between early and late layers)
    "cross_layer_similarity": (0.3, 0.75),        # real speech: moderate divergence

    # Entropy of attention-like features (cosine similarity matrix)
    "frame_diversity": (0.6, 0.95),               # real speech: diverse frame representations
}


def _score_from_range(value: float, low: float, high: float) -> float:
    """
    Score how far a value is from the expected real-speech range.
    Returns 0.0 (perfectly within range) to 1.0 (far outside).
    """
    if low <= value <= high:
        return 0.0
    if value < low:
        dist = low - value
        range_size = high - low
    else:
        dist = value - high
        range_size = high - low
    # Normalize: distance as fraction of range, capped at 1.0
    return min(1.0, dist / max(range_size, 1e-8))


class SSLDetector:
    """
    XLS-R 300M-based deepfake detector.

    Extracts multi-layer hidden states from a wav2vec2 model and computes
    statistical features that discriminate real from synthetic speech.

    Usage:
        detector = SSLDetector(model_path="models/audio/xls_r_300m")
        result = detector.analyze(waveform, sr=16000)
        print(result["score"], result["anomalies"])
    """

    def __init__(
        self,
        model_path: str = "models/audio/xls_r_300m",
        device: str = "cpu",
        extract_layers: Optional[List[int]] = None,
    ):
        """
        Args:
            model_path: Path to local XLS-R 300M directory (config.json + pytorch_model.bin).
            device: "cpu" or "cuda". CPU recommended (1.2GB model, analysis not latency-critical).
            extract_layers: Which transformer layers to analyze (0-indexed).
                            Default: [0, 5, 11, 17, 23] (early/mid/late spread across 24 layers).
        """
        self.model: Optional[Any] = None
        self.feature_extractor: Optional[Any] = None
        self.device = device
        self.extract_layers = extract_layers or [0, 5, 11, 17, 23]
        self._available = False

        if not HAS_TORCH or not HAS_TRANSFORMERS:
            logger.warning("SSLDetector requires torch and transformers. Unavailable.")
            return

        if not os.path.isdir(model_path):
            logger.warning("XLS-R model not found at %s", model_path)
            return

        try:
            logger.info("Loading XLS-R 300M from %s on %s...", model_path, device)
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
            self.model = Wav2Vec2Model.from_pretrained(
                model_path,
                output_hidden_states=True,
            )
            self.model.to(device)
            self.model.eval()
            self._available = True
            logger.info("XLS-R 300M ready (%d transformer layers).", self.model.config.num_hidden_layers)
        except Exception as e:
            logger.error("Failed to load XLS-R 300M: %s", e)

    @property
    def is_available(self) -> bool:
        return self._available

    @torch.no_grad()
    def _extract_features(
        self, waveform: np.ndarray, sr: int
    ) -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray]]:
        """
        Run XLS-R forward pass and extract hidden states from selected layers.

        Returns:
            (layer_hidden_states, final_hidden_state) or (None, None) on failure.
            Each hidden state shape: (time_frames, hidden_size=1024)
        """
        if not self._available:
            return None, None

        # Resample to 16kHz if needed
        if sr != 16000:
            try:
                import librosa
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
                sr = 16000
            except ImportError:
                # Manual resample via scipy
                from scipy.signal import resample
                target_len = int(len(waveform) * 16000 / sr)
                waveform = resample(waveform, target_len)
                sr = 16000

        # Cap at 30 seconds to limit memory/compute
        max_samples = 30 * sr
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]

        # Preprocess
        inputs = self.feature_extractor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(self.device)

        # Forward pass
        outputs = self.model(input_values, output_hidden_states=True)

        # outputs.hidden_states: tuple of (1, T, 1024) for each layer (0=CNN out, 1-24=transformer layers)
        all_hidden = outputs.hidden_states  # 25 tensors (CNN output + 24 transformer layers)

        # Extract selected layers
        layer_states = []
        for layer_idx in self.extract_layers:
            # +1 because index 0 is CNN feature extractor output
            actual_idx = layer_idx + 1
            if actual_idx < len(all_hidden):
                hs = all_hidden[actual_idx][0].cpu().numpy()  # (T, 1024)
                layer_states.append(hs)

        final_hidden = outputs.last_hidden_state[0].cpu().numpy()  # (T, 1024)

        return layer_states, final_hidden

    def _compute_temporal_features(self, hidden_states: np.ndarray) -> Dict[str, float]:
        """
        Compute frame-to-frame temporal statistics from hidden states.

        Args:
            hidden_states: (T, D) array of hidden state vectors.
        """
        T, D = hidden_states.shape
        if T < 3:
            return {}

        # Frame-to-frame cosine similarity
        norms = np.linalg.norm(hidden_states, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = hidden_states / norms

        # Consecutive frame similarities
        sims = np.sum(normalized[:-1] * normalized[1:], axis=1)  # (T-1,)

        # Temporal variance: per-dimension variance across time, then averaged
        temporal_var = np.mean(np.var(hidden_states, axis=0))

        return {
            "temporal_similarity_mean": float(np.mean(sims)),
            "temporal_similarity_std": float(np.std(sims)),
            "temporal_similarity_min": float(np.min(sims)),
            "temporal_variance": float(temporal_var),
        }

    def _compute_distributional_features(self, hidden_states: np.ndarray) -> Dict[str, float]:
        """
        Compute distributional statistics of hidden state activations.
        """
        T, D = hidden_states.shape

        # Flatten all activations for kurtosis/skewness
        flat = hidden_states.flatten()

        # Kurtosis (excess kurtosis: normal=0, leptokurtic>0)
        mean = np.mean(flat)
        std = np.std(flat)
        if std > 1e-8:
            kurtosis = float(np.mean(((flat - mean) / std) ** 4) - 3.0)
            skewness = float(np.mean(((flat - mean) / std) ** 3))
        else:
            kurtosis = 0.0
            skewness = 0.0

        # Per-dimension statistics
        dim_means = np.mean(hidden_states, axis=0)  # (D,)
        dim_stds = np.std(hidden_states, axis=0)     # (D,)

        # Activation sparsity: fraction of near-zero activations
        sparsity = float(np.mean(np.abs(hidden_states) < 0.01))

        # Energy concentration: how much energy is in top-k dimensions
        dim_energy = np.mean(hidden_states ** 2, axis=0)  # (D,)
        sorted_energy = np.sort(dim_energy)[::-1]
        total_energy = np.sum(sorted_energy) + 1e-8
        top_10pct = int(D * 0.1)
        energy_concentration = float(np.sum(sorted_energy[:top_10pct]) / total_energy)

        return {
            "activation_kurtosis": kurtosis,
            "activation_skewness": skewness,
            "activation_sparsity": sparsity,
            "dim_std_mean": float(np.mean(dim_stds)),
            "dim_std_std": float(np.std(dim_stds)),
            "energy_concentration_top10pct": energy_concentration,
        }

    def _compute_cross_layer_features(self, layer_states: List[np.ndarray]) -> Dict[str, float]:
        """
        Measure how representations evolve across transformer layers.
        Real speech shows specific layer-wise dynamics; synthesis may differ.
        """
        if len(layer_states) < 2:
            return {}

        # Mean-pool each layer to get a single vector per layer
        layer_means = []
        for hs in layer_states:
            layer_means.append(np.mean(hs, axis=0))  # (D,)

        layer_means = np.array(layer_means)  # (n_layers, D)

        # Cross-layer cosine similarities (early vs late)
        norms = np.linalg.norm(layer_means, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = layer_means / norms

        # Early vs late layer similarity
        early_late_sim = float(np.dot(normalized[0], normalized[-1]))

        # Consecutive layer similarities
        layer_sims = []
        for i in range(len(normalized) - 1):
            sim = float(np.dot(normalized[i], normalized[i + 1]))
            layer_sims.append(sim)

        # Layer-wise variance: how much representation changes across layers
        layer_variance = float(np.mean(np.var(layer_means, axis=0)))

        return {
            "cross_layer_similarity": early_late_sim,
            "layer_transition_mean": float(np.mean(layer_sims)),
            "layer_transition_std": float(np.std(layer_sims)),
            "layer_variance": layer_variance,
        }

    def _compute_frame_diversity(self, hidden_states: np.ndarray) -> Dict[str, float]:
        """
        Measure diversity of frame representations.
        TTS/VC may produce less diverse frames than real speech.
        """
        T, D = hidden_states.shape
        if T < 5:
            return {}

        # Subsample for efficiency (max 100 frames for pairwise computation)
        if T > 100:
            indices = np.linspace(0, T - 1, 100, dtype=int)
            sub = hidden_states[indices]
        else:
            sub = hidden_states

        # Pairwise cosine similarity matrix
        norms = np.linalg.norm(sub, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = sub / norms
        sim_matrix = normalized @ normalized.T  # (n, n)

        # Frame diversity: 1 - mean off-diagonal similarity
        n = sim_matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)
        off_diag = sim_matrix[mask]
        frame_diversity = float(1.0 - np.mean(off_diag))

        # Effective rank via singular values (higher = more diverse)
        try:
            sv = np.linalg.svd(normalized, compute_uv=False)
            sv = sv / (np.sum(sv) + 1e-8)
            sv = sv[sv > 1e-10]
            effective_rank = float(np.exp(-np.sum(sv * np.log(sv))))
        except Exception:
            effective_rank = 0.0

        return {
            "frame_diversity": frame_diversity,
            "frame_similarity_mean": float(np.mean(off_diag)),
            "frame_similarity_std": float(np.std(off_diag)),
            "effective_rank": effective_rank,
        }

    def _score_features(self, features: Dict[str, float]) -> Tuple[float, float, List[str]]:
        """
        Convert raw features into a spoof score (0=real, 1=spoof) with anomaly list.

        Uses deviation from known real-speech norms in SSL feature space.
        """
        sub_scores = []
        weights = []
        anomalies = []

        # Temporal similarity
        if "temporal_similarity_mean" in features:
            val = features["temporal_similarity_mean"]
            lo, hi = REAL_SPEECH_NORMS["temporal_similarity_mean"]
            s = _score_from_range(val, lo, hi)
            sub_scores.append(s)
            weights.append(2.0)  # high weight — strong discriminator
            if val > hi:
                anomalies.append(f"unnaturally_high_temporal_consistency ({val:.3f} > {hi})")
            elif val < lo:
                anomalies.append(f"low_temporal_coherence ({val:.3f} < {lo})")

        if "temporal_similarity_std" in features:
            val = features["temporal_similarity_std"]
            lo, hi = REAL_SPEECH_NORMS["temporal_similarity_std"]
            s = _score_from_range(val, lo, hi)
            sub_scores.append(s)
            weights.append(1.5)
            if val < lo:
                anomalies.append(f"too_uniform_frame_transitions ({val:.4f} < {lo})")

        # Temporal variance
        if "temporal_variance" in features:
            val = features["temporal_variance"]
            lo, hi = REAL_SPEECH_NORMS["temporal_variance"]
            s = _score_from_range(val, lo, hi)
            sub_scores.append(s)
            weights.append(1.5)
            if val < lo:
                anomalies.append(f"low_temporal_variance ({val:.4f})")
            elif val > hi:
                anomalies.append(f"high_temporal_variance ({val:.4f})")

        # Kurtosis
        if "activation_kurtosis" in features:
            val = features["activation_kurtosis"]
            lo, hi = REAL_SPEECH_NORMS["activation_kurtosis"]
            s = _score_from_range(val, lo, hi)
            sub_scores.append(s)
            weights.append(1.0)
            if val < lo:
                anomalies.append(f"platykurtic_activations ({val:.2f})")
            elif val > hi:
                anomalies.append(f"hyperleptokurtic_activations ({val:.2f})")

        # Cross-layer similarity
        if "cross_layer_similarity" in features:
            val = features["cross_layer_similarity"]
            lo, hi = REAL_SPEECH_NORMS["cross_layer_similarity"]
            s = _score_from_range(val, lo, hi)
            sub_scores.append(s)
            weights.append(1.2)
            if val > hi:
                anomalies.append(f"layers_too_similar ({val:.3f} > {hi})")
            elif val < lo:
                anomalies.append(f"layers_too_divergent ({val:.3f} < {lo})")

        # Frame diversity
        if "frame_diversity" in features:
            val = features["frame_diversity"]
            lo, hi = REAL_SPEECH_NORMS["frame_diversity"]
            s = _score_from_range(val, lo, hi)
            sub_scores.append(s)
            weights.append(1.3)
            if val < lo:
                anomalies.append(f"low_frame_diversity ({val:.3f})")

        if not sub_scores:
            return 0.5, 0.3, ["insufficient_features"]

        weights = np.array(weights)
        sub_scores = np.array(sub_scores)
        score = float(np.average(sub_scores, weights=weights))

        # Confidence: higher when we have more features and they agree
        n_features = len(sub_scores)
        agreement = 1.0 - float(np.std(sub_scores))  # higher when scores are consistent
        confidence = min(1.0, (n_features / 6.0) * 0.6 + agreement * 0.4)

        return score, confidence, anomalies

    def analyze(self, waveform: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
        """
        Full SSL-based analysis of an audio waveform.

        Args:
            waveform: 1D float32 array of audio samples.
            sr: Sample rate (will be resampled to 16kHz if different).

        Returns:
            dict with keys: score, confidence, features, anomalies, embedding
        """
        if not self._available:
            return {
                "score": 0.5,
                "confidence": 0.0,
                "features": {},
                "anomalies": ["ssl_model_unavailable"],
                "embedding": None,
            }

        try:
            layer_states, final_hidden = self._extract_features(waveform, sr)
            if layer_states is None or final_hidden is None:
                return {
                    "score": 0.5,
                    "confidence": 0.0,
                    "features": {},
                    "anomalies": ["feature_extraction_failed"],
                    "embedding": None,
                }

            # Compute all feature groups
            features: Dict[str, float] = {}
            features.update(self._compute_temporal_features(final_hidden))
            features.update(self._compute_distributional_features(final_hidden))
            features.update(self._compute_cross_layer_features(layer_states))
            features.update(self._compute_frame_diversity(final_hidden))

            # Score
            score, confidence, anomalies = self._score_features(features)

            # Mean-pooled embedding from final layer (1024d) for downstream use
            embedding = np.mean(final_hidden, axis=0)  # (1024,)

            return {
                "score": round(score, 4),
                "confidence": round(confidence, 4),
                "features": {k: round(v, 6) for k, v in features.items()},
                "anomalies": anomalies,
                "embedding": embedding,
            }

        except Exception as e:
            logger.error("SSL analysis failed: %s", e, exc_info=True)
            return {
                "score": 0.5,
                "confidence": 0.0,
                "features": {},
                "anomalies": [f"analysis_error: {e}"],
                "embedding": None,
            }

    def extract_embedding(self, waveform: np.ndarray, sr: int = 16000) -> Optional[np.ndarray]:
        """
        Extract only the mean-pooled 1024d embedding (for fusion/similarity tasks).
        """
        if not self._available:
            return None
        try:
            _, final_hidden = self._extract_features(waveform, sr)
            if final_hidden is None:
                return None
            return np.mean(final_hidden, axis=0)  # (1024,)
        except Exception as e:
            logger.error("SSL embedding extraction failed: %s", e)
            return None
