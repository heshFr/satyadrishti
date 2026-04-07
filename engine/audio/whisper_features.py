"""
Whisper Encoder Feature Extractor (Layer 3)
=============================================
Extracts acoustic features from Whisper's encoder for deepfake detection.

Whisper was trained on 680,000 hours of multilingual speech, making its
encoder one of the most robust acoustic feature extractors available.
Its internal representations capture subtle acoustic patterns that differ
between real and synthetic speech:

  - **Log-mel spectrogram statistics**: Whisper's 80-band mel computation
    reveals synthesis artifacts in spectral energy distribution.
  - **Encoder hidden states**: The transformer encoder output captures
    high-level acoustic patterns; TTS/VC produces representations with
    different distributional properties than real speech.
  - **Temporal coherence**: Frame-to-frame consistency in encoder space
    differs between real and synthetic speech.
  - **Spectral flatness & tilt**: Real speech has characteristic spectral
    shapes from the vocal tract; synthesizers produce flatter/different
    spectral profiles.
  - **Modulation spectrum**: Natural speech has ~4Hz syllabic modulation
    and ~30Hz phonetic modulation; synthesis may lack these.

Uses faster-whisper (CTranslate2) for efficient inference. Shares the
model instance with the transcriber when possible.

Model: faster-whisper "small" (encoder: 12 layers, 768d hidden states)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel
    from faster_whisper.feature_extractor import FeatureExtractor
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

try:
    import ctranslate2
    HAS_CT2 = True
except ImportError:
    HAS_CT2 = False


# ── Known norms for real speech in Whisper feature space ──
REAL_SPEECH_NORMS = {
    # Widened ranges to reduce false positives on real-world audio
    # (phone recordings, noisy environments, varied microphones)

    # Log-mel spectrogram statistics
    "mel_mean": (-4.5, -0.5),                # wider: varied gain/recording levels
    "mel_std": (0.5, 3.0),                   # wider: quiet rooms vs noisy streets
    "mel_spectral_flatness": (0.03, 0.45),   # wider: breathy/noisy speech is OK
    "mel_spectral_tilt": (-0.12, 0.005),     # wider: recording equipment varies

    # Encoder hidden state statistics
    "encoder_temporal_sim_mean": (0.75, 0.98),  # wider: noisy vs clean recordings
    "encoder_temporal_sim_std": (0.01, 0.10),   # wider: monotone vs expressive
    "encoder_activation_std": (0.2, 2.0),       # wider: varied speaker styles

    # Modulation spectrum
    "modulation_4hz_power": (0.01, 0.35),       # wider: slow vs fast speakers
    "modulation_30hz_power": (0.003, 0.12),     # wider: recording quality varies

    # Delta features (rate of change in mel spectrogram)
    "delta_energy_std": (0.05, 1.0),            # wider: more tolerance
}


def _score_from_range(value: float, low: float, high: float) -> float:
    """Score how far a value is from the expected real-speech range."""
    if low <= value <= high:
        return 0.0
    if value < low:
        dist = low - value
    else:
        dist = value - high
    range_size = high - low
    return min(1.0, dist / max(range_size, 1e-8))


class WhisperFeatureExtractor:
    """
    Whisper-based acoustic feature extractor for deepfake detection.

    Extracts log-mel spectrogram features and encoder hidden states
    from Whisper's encoder, computing statistical measures that
    distinguish real from synthetic speech.

    Can share a WhisperModel instance with RealTimeTranscriber to
    avoid loading the model twice.

    Usage:
        extractor = WhisperFeatureExtractor()
        result = extractor.analyze(waveform, sr=16000)
        print(result["score"], result["anomalies"])
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "float16",
        shared_model: Optional[Any] = None,
    ):
        """
        Args:
            model_size: Whisper model size ("tiny", "base", "small").
            device: "cpu", "cuda", or "auto".
            compute_type: "float16" (GPU), "int8" (CPU), or "float32".
            shared_model: Optional WhisperModel instance to reuse (from transcriber).
        """
        self._model: Optional[Any] = None
        self._feature_extractor: Optional[Any] = None
        self._available = False
        self._model_size = model_size

        if not HAS_WHISPER:
            logger.warning("WhisperFeatureExtractor requires faster-whisper. Unavailable.")
            return

        try:
            if shared_model is not None and hasattr(shared_model, 'model'):
                # Reuse existing WhisperModel instance
                self._model = shared_model
                self._feature_extractor = shared_model.feature_extractor
                self._available = True
                logger.info("WhisperFeatureExtractor sharing model with transcriber.")
            else:
                if device == "auto":
                    try:
                        import torch
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                    except ImportError:
                        device = "cpu"

                if device == "cpu":
                    compute_type = "int8"

                logger.info("Loading Whisper '%s' for feature extraction on %s...", model_size, device)
                self._model = WhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type,
                )
                self._feature_extractor = self._model.feature_extractor
                self._available = True
                logger.info("WhisperFeatureExtractor ready.")
        except Exception as e:
            logger.error("Failed to load Whisper for feature extraction: %s", e)

    @property
    def is_available(self) -> bool:
        return self._available

    def _compute_log_mel(self, waveform: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """
        Compute Whisper-style 80-band log-mel spectrogram.

        Returns:
            (n_mels=80, T) log-mel spectrogram, or None on failure.
        """
        if self._feature_extractor is None:
            return None

        # Resample to 16kHz if needed
        if sr != 16000:
            try:
                import librosa
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
            except ImportError:
                from scipy.signal import resample
                target_len = int(len(waveform) * 16000 / sr)
                waveform = resample(waveform, target_len).astype(np.float32)

        # Use Whisper's own feature extractor (80-band mel, 25ms window, 10ms hop)
        log_mel = self._feature_extractor(waveform)  # (80, T)
        return log_mel

    def _compute_encoder_output(self, log_mel: np.ndarray) -> Optional[np.ndarray]:
        """
        Run Whisper encoder and return hidden states.

        Args:
            log_mel: (80, T) log-mel spectrogram.

        Returns:
            (T', D) encoder output array, or None on failure.
        """
        if self._model is None:
            return None

        try:
            # CTranslate2 on CUDA: np.array() on a GPU StorageView returns scalar.
            # Use model.model.encode() with to_cpu=True to get CPU-side numpy.
            if log_mel.ndim == 2:
                log_mel_batched = log_mel[np.newaxis, :, :]  # (1, 80, T)
            else:
                log_mel_batched = log_mel

            try:
                import ctranslate2
                features_sv = ctranslate2.StorageView.from_array(log_mel_batched)
                encoder_output = self._model.model.encode(features_sv, to_cpu=True)
            except (ImportError, AttributeError):
                # Fallback: use the high-level encode (works on CPU)
                encoder_output = self._model.encode(log_mel)

            output_np = np.array(encoder_output)
            # Remove batch dimension(s) until we get (T', D)
            while output_np.ndim > 2:
                output_np = output_np[0]
            if output_np.ndim < 2 or output_np.shape[0] < 3:
                logger.warning("Encoder output too short: shape=%s", output_np.shape)
                return None
            return output_np
        except Exception as e:
            logger.warning("Whisper encoder forward failed: %s", e)
            return None

    def _analyze_mel_spectrogram(self, log_mel: np.ndarray) -> Dict[str, float]:
        """
        Compute statistical features from the log-mel spectrogram.
        """
        n_mels, T = log_mel.shape
        if T < 5:
            return {}

        features: Dict[str, float] = {}

        # ── Global statistics ──
        features["mel_mean"] = float(np.mean(log_mel))
        features["mel_std"] = float(np.std(log_mel))
        features["mel_max"] = float(np.max(log_mel))
        features["mel_min"] = float(np.min(log_mel))

        # ── Per-band energy ──
        band_energy = np.mean(log_mel, axis=1)  # (80,)

        # Spectral tilt: linear regression slope across mel bands
        x = np.arange(n_mels, dtype=np.float64)
        y = band_energy.astype(np.float64)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        slope = float(np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean) ** 2) + 1e-8))
        features["mel_spectral_tilt"] = slope

        # Spectral flatness (geometric mean / arithmetic mean)
        # Work in linear domain — clamp band_energy to avoid overflow in 10^x
        clamped_energy = np.clip(band_energy, -20.0, 20.0)
        linear_energy = np.power(10.0, clamped_energy / 10.0).astype(np.float64)
        linear_energy = np.maximum(linear_energy, 1e-20)
        geo_mean = np.exp(np.mean(np.log(linear_energy)))
        arith_mean = np.mean(linear_energy)
        spectral_flatness = float(geo_mean / (arith_mean + 1e-8))
        features["mel_spectral_flatness"] = spectral_flatness

        # Spectral centroid
        freqs = np.arange(n_mels, dtype=np.float64)
        centroid = float(np.sum(freqs * linear_energy) / (np.sum(linear_energy) + 1e-8))
        features["mel_spectral_centroid"] = centroid

        # Spectral bandwidth
        bandwidth = float(np.sqrt(
            np.sum(linear_energy * (freqs - centroid) ** 2) / (np.sum(linear_energy) + 1e-8)
        ))
        features["mel_spectral_bandwidth"] = bandwidth

        # ── Temporal dynamics (per-frame energy) ──
        frame_energy = np.mean(log_mel, axis=0)  # (T,)

        # Delta energy (first derivative)
        delta_energy = np.diff(frame_energy)
        features["delta_energy_mean"] = float(np.mean(np.abs(delta_energy)))
        features["delta_energy_std"] = float(np.std(delta_energy))

        # Delta-delta (second derivative / acceleration)
        if len(delta_energy) > 1:
            delta_delta = np.diff(delta_energy)
            features["delta_delta_energy_std"] = float(np.std(delta_delta))

        # ── Modulation spectrum ──
        # Compute modulation spectrum from frame energy envelope
        if T >= 32:
            # Frame rate depends on hop_length (default 160 samples at 16kHz = 100 fps)
            fps = 100.0
            # FFT of energy envelope
            fft_energy = np.fft.rfft(frame_energy - np.mean(frame_energy))
            power_spectrum = np.abs(fft_energy) ** 2
            freqs_mod = np.fft.rfftfreq(len(frame_energy), d=1.0 / fps)

            total_power = np.sum(power_spectrum) + 1e-8

            # Syllabic modulation band (3-5 Hz)
            mask_4hz = (freqs_mod >= 3.0) & (freqs_mod <= 5.0)
            features["modulation_4hz_power"] = float(np.sum(power_spectrum[mask_4hz]) / total_power)

            # Phonetic modulation band (25-35 Hz)
            mask_30hz = (freqs_mod >= 25.0) & (freqs_mod <= 35.0)
            features["modulation_30hz_power"] = float(np.sum(power_spectrum[mask_30hz]) / total_power)

            # Low modulation (<2 Hz, typical of static noise/tone)
            mask_low = freqs_mod < 2.0
            features["modulation_low_power"] = float(np.sum(power_spectrum[mask_low]) / total_power)

        # ── Mel band correlation (cross-band coherence) ──
        # Real speech has correlated harmonics; synthesis may have different patterns
        if T >= 10:
            # Subsample bands for efficiency
            band_indices = [5, 15, 25, 35, 45, 55, 65, 75]
            band_indices = [b for b in band_indices if b < n_mels]
            if len(band_indices) >= 4:
                sub_mel = log_mel[band_indices, :]  # (n_bands, T)
                # Filter out constant bands (zero std) to avoid NaN in corrcoef
                band_stds = np.std(sub_mel, axis=1)
                valid = band_stds > 1e-8
                if np.sum(valid) >= 3:
                    sub_mel = sub_mel[valid]
                    corr_matrix = np.corrcoef(sub_mel)  # (n_bands, n_bands)
                    mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                    off_diag_corr = corr_matrix[mask]
                    off_diag_corr = off_diag_corr[~np.isnan(off_diag_corr)]
                    if len(off_diag_corr) > 0:
                        features["mel_band_correlation_mean"] = float(np.mean(off_diag_corr))
                        features["mel_band_correlation_std"] = float(np.std(off_diag_corr))

        return features

    def _analyze_encoder_output(self, encoder_output: np.ndarray) -> Dict[str, float]:
        """
        Compute statistical features from Whisper encoder hidden states.
        """
        T, D = encoder_output.shape
        if T < 3:
            return {}

        features: Dict[str, float] = {}

        # ── Global activation statistics ──
        features["encoder_activation_mean"] = float(np.mean(encoder_output))
        features["encoder_activation_std"] = float(np.std(encoder_output))

        # Kurtosis
        flat = encoder_output.flatten()
        mean = np.mean(flat)
        std = np.std(flat)
        if std > 1e-8:
            z = np.clip((flat - mean) / std, -10.0, 10.0)  # clamp to avoid overflow in **4
            features["encoder_kurtosis"] = float(np.mean(z ** 4) - 3.0)
        else:
            features["encoder_kurtosis"] = 0.0

        # ── Temporal consistency ──
        norms = np.linalg.norm(encoder_output, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = encoder_output / norms

        # Frame-to-frame cosine similarity
        sims = np.sum(normalized[:-1] * normalized[1:], axis=1)
        features["encoder_temporal_sim_mean"] = float(np.mean(sims))
        features["encoder_temporal_sim_std"] = float(np.std(sims))
        features["encoder_temporal_sim_min"] = float(np.min(sims))

        # ── Temporal variance ──
        temporal_var = np.mean(np.var(encoder_output, axis=0))
        features["encoder_temporal_variance"] = float(temporal_var)

        # ── Frame diversity (subsampled pairwise similarity) ──
        if T > 50:
            indices = np.linspace(0, T - 1, 50, dtype=int)
            sub = normalized[indices]
        else:
            sub = normalized

        sim_matrix = sub @ sub.T
        n = sim_matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)
        off_diag = sim_matrix[mask]
        features["encoder_frame_diversity"] = float(1.0 - np.mean(off_diag))
        features["encoder_frame_sim_std"] = float(np.std(off_diag))

        # ── Dimension-wise statistics ──
        dim_vars = np.var(encoder_output, axis=0)  # (D,)
        features["encoder_dim_var_mean"] = float(np.mean(dim_vars))
        features["encoder_dim_var_std"] = float(np.std(dim_vars))

        # Energy concentration in top dimensions
        sorted_vars = np.sort(dim_vars)[::-1]
        total_var = np.sum(sorted_vars) + 1e-8
        top_10pct = int(D * 0.1)
        features["encoder_energy_concentration"] = float(np.sum(sorted_vars[:top_10pct]) / total_var)

        # ── Attention-like self-similarity structure ──
        # Compute "burstiness" — how much the encoder output clusters in time
        if T >= 10:
            # Split into segments and compare their mean representations
            n_segments = min(10, T // 3)
            seg_len = T // n_segments
            seg_means = []
            for i in range(n_segments):
                start = i * seg_len
                end = min(start + seg_len, T)
                seg_mean = np.mean(encoder_output[start:end], axis=0)
                seg_means.append(seg_mean)
            seg_means = np.array(seg_means)

            # Segment-level variance (how much the representation drifts over time)
            seg_var = float(np.mean(np.var(seg_means, axis=0)))
            features["encoder_segment_drift"] = seg_var

        return features

    def _score_features(self, features: Dict[str, float]) -> Tuple[float, float, List[str]]:
        """
        Convert raw features into a spoof score (0=real, 1=spoof).
        """
        sub_scores = []
        weights = []
        anomalies = []

        # Mel spectrogram features
        if "mel_spectral_flatness" in features:
            val = features["mel_spectral_flatness"]
            lo, hi = REAL_SPEECH_NORMS["mel_spectral_flatness"]
            s = _score_from_range(val, lo, hi)
            sub_scores.append(s)
            weights.append(1.2)
            if val > hi:
                anomalies.append(f"flat_spectrum_noise_like ({val:.3f})")
            elif val < lo:
                anomalies.append(f"overly_tonal_spectrum ({val:.3f})")

        if "mel_spectral_tilt" in features:
            val = features["mel_spectral_tilt"]
            lo, hi = REAL_SPEECH_NORMS["mel_spectral_tilt"]
            s = _score_from_range(val, lo, hi)
            sub_scores.append(s)
            weights.append(1.0)
            if val > hi:
                anomalies.append(f"abnormal_spectral_tilt ({val:.4f})")

        if "mel_std" in features:
            val = features["mel_std"]
            lo, hi = REAL_SPEECH_NORMS["mel_std"]
            s = _score_from_range(val, lo, hi)
            sub_scores.append(s)
            weights.append(0.8)
            if val < lo:
                anomalies.append(f"low_spectral_variance ({val:.3f})")

        # Encoder features
        if "encoder_temporal_sim_mean" in features:
            val = features["encoder_temporal_sim_mean"]
            lo, hi = REAL_SPEECH_NORMS["encoder_temporal_sim_mean"]
            s = _score_from_range(val, lo, hi)
            sub_scores.append(s)
            weights.append(2.0)  # strong discriminator
            if val > hi:
                anomalies.append(f"encoder_too_smooth ({val:.3f})")
            elif val < lo:
                anomalies.append(f"encoder_incoherent ({val:.3f})")

        if "encoder_temporal_sim_std" in features:
            val = features["encoder_temporal_sim_std"]
            lo, hi = REAL_SPEECH_NORMS["encoder_temporal_sim_std"]
            s = _score_from_range(val, lo, hi)
            sub_scores.append(s)
            weights.append(1.5)
            if val < lo:
                anomalies.append(f"encoder_too_uniform ({val:.4f})")

        if "encoder_activation_std" in features:
            val = features["encoder_activation_std"]
            lo, hi = REAL_SPEECH_NORMS["encoder_activation_std"]
            s = _score_from_range(val, lo, hi)
            sub_scores.append(s)
            weights.append(1.0)

        # Modulation spectrum
        if "modulation_4hz_power" in features:
            val = features["modulation_4hz_power"]
            lo, hi = REAL_SPEECH_NORMS["modulation_4hz_power"]
            s = _score_from_range(val, lo, hi)
            sub_scores.append(s)
            weights.append(1.5)
            if val < lo:
                anomalies.append(f"missing_syllabic_modulation ({val:.4f})")

        if "modulation_30hz_power" in features:
            val = features["modulation_30hz_power"]
            lo, hi = REAL_SPEECH_NORMS["modulation_30hz_power"]
            s = _score_from_range(val, lo, hi)
            sub_scores.append(s)
            weights.append(1.0)
            if val < lo:
                anomalies.append(f"missing_phonetic_modulation ({val:.4f})")

        # Delta energy
        if "delta_energy_std" in features:
            val = features["delta_energy_std"]
            lo, hi = REAL_SPEECH_NORMS["delta_energy_std"]
            s = _score_from_range(val, lo, hi)
            sub_scores.append(s)
            weights.append(1.0)
            if val < lo:
                anomalies.append(f"flat_energy_dynamics ({val:.3f})")

        if not sub_scores:
            return 0.5, 0.3, ["insufficient_features"]

        weights_arr = np.array(weights)
        scores_arr = np.array(sub_scores)
        score = float(np.average(scores_arr, weights=weights_arr))

        # Confidence: more features + more agreement = higher confidence
        n_features = len(sub_scores)
        agreement = 1.0 - float(np.std(scores_arr))
        confidence = min(1.0, (n_features / 9.0) * 0.6 + agreement * 0.4)

        return score, confidence, anomalies

    def analyze(self, waveform: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
        """
        Full Whisper-based acoustic analysis.

        Args:
            waveform: 1D float32 array of audio samples.
            sr: Sample rate (will be resampled to 16kHz if different).

        Returns:
            dict with keys: score, confidence, features, anomalies
        """
        if not self._available:
            return {
                "score": 0.5,
                "confidence": 0.0,
                "features": {},
                "anomalies": ["whisper_model_unavailable"],
            }

        try:
            # Resample to 16kHz if needed
            if sr != 16000:
                try:
                    import librosa
                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
                except ImportError:
                    from scipy.signal import resample
                    target_len = int(len(waveform) * 16000 / sr)
                    waveform = resample(waveform, target_len).astype(np.float32)
                sr = 16000

            # Cap at 30 seconds
            max_samples = 30 * sr
            if len(waveform) > max_samples:
                waveform = waveform[:max_samples]

            features: Dict[str, float] = {}

            # Step 1: Compute log-mel spectrogram
            log_mel = self._compute_log_mel(waveform, sr)
            if log_mel is not None:
                mel_features = self._analyze_mel_spectrogram(log_mel)
                features.update(mel_features)

                # Step 2: Run encoder
                encoder_output = self._compute_encoder_output(log_mel)
                if encoder_output is not None:
                    enc_features = self._analyze_encoder_output(encoder_output)
                    features.update(enc_features)

            # Step 3: Score
            score, confidence, anomalies = self._score_features(features)

            return {
                "score": round(score, 4),
                "confidence": round(confidence, 4),
                "features": {k: round(v, 6) for k, v in features.items()},
                "anomalies": anomalies,
            }

        except Exception as e:
            logger.error("Whisper feature analysis failed: %s", e, exc_info=True)
            return {
                "score": 0.5,
                "confidence": 0.0,
                "features": {},
                "anomalies": [f"analysis_error: {e}"],
            }
