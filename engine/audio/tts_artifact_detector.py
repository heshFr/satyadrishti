"""
Modern TTS Artifact Detector
=============================
Targets artifacts that persist even in state-of-the-art neural TTS systems
(ElevenLabs, OpenAI TTS, XTTS, Bark, Tortoise, etc.).

Modern TTS has evolved past traditional vocoder artifacts. They simulate
breathing, natural prosody, formant transitions — fooling older heuristic
detectors. But fundamental physics of speech production cannot be perfectly
replicated. This module exploits 8 categories of residual artifacts:

1. Spectral Bandwidth Consistency — real speech bandwidth varies with phoneme,
   TTS bandwidth is unnaturally uniform
2. Silence & Room Tone Analysis — real recordings contain room noise/reverb,
   AI silence is digitally clean or uses templated noise
3. Spectral Envelope Smoothness — vocoders produce smoother spectral envelopes
   than the complex resonance patterns of a real vocal tract
4. Sub-band Temporal Correlation — real speech has complex cross-frequency-band
   correlations from physical resonance; TTS bands are more independent
5. Harmonic Structure Analysis — real voice harmonics have natural rolloff with
   per-harmonic jitter; TTS harmonics are too uniform or too sparse
6. Micro-Pause Regularity — natural micro-pauses (20-100ms) vary widely;
   TTS micro-pauses are suspiciously regular
7. Onset/Offset Transient Analysis — phoneme attack/release shapes differ
   between real vocal tract mechanics and neural upsampling
8. Pitch Micro-Dynamics — frame-level F0 has fractal-like variability in real
   speech; TTS pitch micro-dynamics show periodicity or excessive smoothness

Each sub-analyzer outputs a score 0-1 (higher = more TTS-like).
The final score is a reliability-weighted combination.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from scipy import signal as scipy_signal
    from scipy.stats import kurtosis, skew, entropy
    from scipy.ndimage import uniform_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

log = logging.getLogger(__name__)


class TTSArtifactDetector:
    """
    Detects artifacts specific to modern neural TTS systems.

    Args:
        verbose: Print per-layer scores during analysis
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.is_available = HAS_LIBROSA and HAS_SCIPY

    def analyze(self, waveform: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Run full 8-layer TTS artifact analysis.

        Args:
            waveform: Mono audio signal (float32, [-1, 1])
            sr: Sample rate in Hz

        Returns:
            Dict with score (0-1, higher = more TTS-like), confidence,
            layer_scores, anomalies, and detailed features.
        """
        if not self.is_available:
            return {
                "score": 0.5,
                "confidence": 0.0,
                "anomalies": [],
                "features": {"error": "librosa/scipy not available"},
            }

        duration = len(waveform) / sr
        if duration < 0.5:
            return {
                "score": 0.5,
                "confidence": 0.0,
                "anomalies": [],
                "features": {"error": "Audio too short", "duration": duration},
            }

        # Ensure float32 and mono
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)

        # Resample to 16kHz if needed (standardize analysis)
        if sr != 16000:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Pre-compute shared features
        stft = librosa.stft(waveform, n_fft=1024, hop_length=256)
        mag = np.abs(stft)
        mel_spec = librosa.feature.melspectrogram(
            y=waveform, sr=sr, n_mels=80, n_fft=1024, hop_length=256
        )
        mel_db = librosa.power_to_db(mel_spec + 1e-10, ref=np.max)

        # Run all 8 sub-analyzers
        layer_results = {}
        anomalies = []
        all_features = {}

        analyzers = [
            ("spectral_bandwidth", self._analyze_spectral_bandwidth, (mag, sr)),
            ("silence_room_tone", self._analyze_silence, (waveform, sr)),
            ("spectral_envelope", self._analyze_spectral_envelope, (mag, sr)),
            ("subband_correlation", self._analyze_subband_correlation, (mel_spec,)),
            ("harmonic_structure", self._analyze_harmonics, (waveform, sr)),
            ("micro_pause_regularity", self._analyze_micro_pauses, (waveform, sr)),
            ("onset_transients", self._analyze_onset_transients, (waveform, sr, mag)),
            ("pitch_micro_dynamics", self._analyze_pitch_dynamics, (waveform, sr)),
        ]

        for name, func, args in analyzers:
            try:
                score, conf, layer_anomalies, features = func(*args)
                layer_results[name] = {
                    "score": float(np.clip(score, 0, 1)),
                    "confidence": float(np.clip(conf, 0, 1)),
                }
                anomalies.extend(layer_anomalies)
                all_features.update({f"{name}_{k}": v for k, v in features.items()})

                if self.verbose:
                    print(f"  [TTS-Artifact] {name}: score={score:.3f}, "
                          f"conf={conf:.3f}, anomalies={layer_anomalies}")
            except Exception as e:
                log.warning("TTS artifact layer '%s' failed: %s", name, e)
                layer_results[name] = {"score": 0.5, "confidence": 0.0}

        # Weighted fusion of layer scores
        weights = {
            "spectral_bandwidth": 1.5,
            "silence_room_tone": 2.0,       # Very discriminating — AI silence is clean
            "spectral_envelope": 1.8,
            "subband_correlation": 1.5,
            "harmonic_structure": 1.3,
            "micro_pause_regularity": 1.2,
            "onset_transients": 1.4,
            "pitch_micro_dynamics": 1.8,     # Very discriminating — F0 micro-dynamics
        }

        weighted_sum = 0.0
        total_weight = 0.0
        n_confident = 0
        layer_scores = {}

        for name, result in layer_results.items():
            w = weights.get(name, 1.0)
            s = result["score"]
            c = result["confidence"]
            eff_w = w * max(0.1, c)  # Don't fully zero out even low-confidence layers
            weighted_sum += s * eff_w
            total_weight += eff_w
            layer_scores[name] = round(s, 4)
            if c > 0.3:
                n_confident += 1

        final_score = weighted_sum / total_weight if total_weight > 0 else 0.5

        # Overall confidence based on duration and layer coverage
        duration_conf = min(1.0, duration / 10.0)  # Full confidence at 10s
        coverage_conf = n_confident / len(analyzers)
        confidence = 0.5 * duration_conf + 0.5 * coverage_conf

        if self.verbose:
            print(f"  [TTS-Artifact] FINAL: score={final_score:.4f}, "
                  f"confidence={confidence:.3f}, anomalies={len(anomalies)}")

        return {
            "score": round(float(np.clip(final_score, 0, 1)), 4),
            "confidence": round(float(np.clip(confidence, 0, 1)), 4),
            "layer_scores": layer_scores,
            "anomalies": anomalies,
            "features": all_features,
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 1: Spectral Bandwidth Consistency
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_spectral_bandwidth(
        self, mag: np.ndarray, sr: int
    ) -> Tuple[float, float, List[str], Dict]:
        """
        Real speech: bandwidth varies dramatically per phoneme (/s/ = wideband,
        /m/ = narrowband). TTS bandwidth is more uniform because vocoders
        use fixed-width filterbanks.

        Checks:
        - Coefficient of variation of per-frame spectral bandwidth
        - Bandwidth range (max - min) across frames
        - Bandwidth kurtosis (real has heavier tails)
        """
        # Per-frame spectral bandwidth (weighted standard deviation of frequencies)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=(mag.shape[0] - 1) * 2)
        # Normalize magnitude per frame
        mag_norm = mag / (mag.sum(axis=0, keepdims=True) + 1e-10)
        # Weighted mean frequency per frame
        centroid = (freqs[:, None] * mag_norm).sum(axis=0)
        # Weighted std (bandwidth)
        bandwidth = np.sqrt(
            ((freqs[:, None] - centroid[None, :]) ** 2 * mag_norm).sum(axis=0)
        )

        # Filter out silence frames
        frame_energy = mag.sum(axis=0)
        energy_thresh = np.percentile(frame_energy, 20)
        voiced = bandwidth[frame_energy > energy_thresh]

        if len(voiced) < 10:
            return 0.5, 0.1, [], {"bandwidth_frames": len(voiced)}

        bw_mean = np.mean(voiced)
        bw_std = np.std(voiced)
        bw_cv = bw_std / (bw_mean + 1e-10)  # Coefficient of variation
        bw_range = np.ptp(voiced) / (bw_mean + 1e-10)  # Normalized range
        bw_kurtosis = float(kurtosis(voiced))

        anomalies = []
        score = 0.0

        # Real speech: CV typically 0.3-0.8, TTS: 0.1-0.3
        if bw_cv < 0.15:
            score += 0.4
            anomalies.append("very_uniform_spectral_bandwidth")
        elif bw_cv < 0.25:
            score += 0.2
            anomalies.append("uniform_spectral_bandwidth")

        # Real speech: range typically 0.8-2.0, TTS: 0.3-0.8
        if bw_range < 0.5:
            score += 0.3
            anomalies.append("narrow_bandwidth_range")
        elif bw_range < 0.8:
            score += 0.15

        # Real speech: kurtosis typically > 0.5 (heavy tails), TTS: < 0.5
        if bw_kurtosis < 0.0:
            score += 0.2
            anomalies.append("platykurtic_bandwidth")

        confidence = min(1.0, len(voiced) / 50)

        features = {
            "cv": round(bw_cv, 4),
            "range": round(bw_range, 4),
            "kurtosis": round(bw_kurtosis, 4),
            "mean": round(float(bw_mean), 2),
        }

        return min(1.0, score), confidence, anomalies, features

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 2: Silence & Room Tone Analysis
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_silence(
        self, waveform: np.ndarray, sr: int
    ) -> Tuple[float, float, List[str], Dict]:
        """
        Real recordings have room tone — a unique spectral fingerprint from
        the recording environment (HVAC, outside noise, room resonance).
        AI-generated audio has either:
        - Digital silence (zeros or very low amplitude noise)
        - Templated noise that's identical in every silence segment

        Checks:
        - Silence segment spectral variance (real varies, AI is identical)
        - Silence noise floor level (AI often too clean)
        - Silence spectral flatness (real room tone has colored spectrum)
        - Cross-segment silence correlation (AI reuses same noise pattern)
        """
        # Find silence segments (< 20th percentile RMS in 20ms frames)
        frame_len = int(0.02 * sr)
        hop = frame_len // 2
        n_frames = (len(waveform) - frame_len) // hop + 1

        if n_frames < 5:
            return 0.5, 0.0, [], {"silence_frames": 0}

        frame_rms = np.array([
            np.sqrt(np.mean(waveform[i * hop: i * hop + frame_len] ** 2))
            for i in range(n_frames)
        ])

        rms_thresh = np.percentile(frame_rms, 25)
        silence_mask = frame_rms < max(rms_thresh, 1e-5)
        silence_indices = np.where(silence_mask)[0]

        if len(silence_indices) < 5:
            return 0.5, 0.2, [], {"silence_frames": len(silence_indices)}

        # Extract silence segments
        silence_segments = []
        current_segment = [silence_indices[0]]
        for idx in silence_indices[1:]:
            if idx == current_segment[-1] + 1:
                current_segment.append(idx)
            else:
                if len(current_segment) >= 3:
                    silence_segments.append(current_segment)
                current_segment = [idx]
        if len(current_segment) >= 3:
            silence_segments.append(current_segment)

        anomalies = []
        score = 0.0

        if len(silence_segments) < 2:
            return 0.5, 0.15, [], {"silence_segments": len(silence_segments)}

        # Compute spectral fingerprints of silence segments
        silence_spectra = []
        silence_rms_values = []
        for seg in silence_segments[:20]:  # Cap at 20 segments
            start = seg[0] * hop
            end = min(seg[-1] * hop + frame_len, len(waveform))
            segment = waveform[start:end]
            if len(segment) < frame_len:
                continue

            # Spectral fingerprint (256-point FFT)
            n_fft = min(256, len(segment))
            spec = np.abs(np.fft.rfft(segment[:n_fft]))
            if spec.sum() > 0:
                spec = spec / (spec.sum() + 1e-10)
            silence_spectra.append(spec)
            silence_rms_values.append(np.sqrt(np.mean(segment ** 2)))

        if len(silence_spectra) < 2:
            return 0.5, 0.1, [], {"silence_segments": len(silence_segments)}

        # Pad spectra to same length
        max_len = max(len(s) for s in silence_spectra)
        silence_spectra = [
            np.pad(s, (0, max_len - len(s))) for s in silence_spectra
        ]
        silence_spectra = np.array(silence_spectra)

        # Check 1: Cross-segment correlation (AI reuses same noise)
        correlations = []
        for i in range(len(silence_spectra)):
            for j in range(i + 1, min(i + 5, len(silence_spectra))):
                corr = np.corrcoef(silence_spectra[i], silence_spectra[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        if correlations:
            mean_corr = np.mean(correlations)
            # Real room tone: moderate correlation 0.3-0.7 (same room but different moments)
            # AI silence: very high correlation >0.9 (identical template)
            # or very low <0.1 (random white noise per segment)
            if mean_corr > 0.92:
                score += 0.35
                anomalies.append("identical_silence_pattern")
            elif mean_corr > 0.85:
                score += 0.2
                anomalies.append("very_similar_silence")
            elif mean_corr < 0.05:
                score += 0.15
                anomalies.append("uncorrelated_silence_segments")

        # Check 2: Silence noise floor level
        mean_silence_rms = np.mean(silence_rms_values)
        if mean_silence_rms < 1e-5:
            # Digital silence — very suspicious (real mics always have some noise)
            score += 0.3
            anomalies.append("digital_silence")
        elif mean_silence_rms < 5e-5:
            score += 0.15
            anomalies.append("very_low_noise_floor")

        # Check 3: Spectral flatness of silence (real room tone is colored)
        silence_flatness_values = []
        for spec in silence_spectra:
            spec_pos = spec[spec > 0]
            if len(spec_pos) > 0:
                geo_mean = np.exp(np.mean(np.log(spec_pos + 1e-15)))
                arith_mean = np.mean(spec_pos)
                flatness = geo_mean / (arith_mean + 1e-15)
                silence_flatness_values.append(flatness)

        if silence_flatness_values:
            mean_flatness = np.mean(silence_flatness_values)
            # White noise: flatness ~1.0, real room tone: 0.1-0.6
            if mean_flatness > 0.85:
                score += 0.2
                anomalies.append("white_noise_silence")
            elif mean_flatness < 0.05:
                score += 0.15
                anomalies.append("too_colored_silence")

        # Check 4: RMS variation across silence segments
        if len(silence_rms_values) >= 3:
            rms_cv = np.std(silence_rms_values) / (np.mean(silence_rms_values) + 1e-10)
            # Real: RMS varies with distance to noise source — CV 0.2-1.0
            # AI: identical noise level — CV < 0.1
            if rms_cv < 0.05:
                score += 0.2
                anomalies.append("identical_silence_levels")
            elif rms_cv < 0.1:
                score += 0.1
        else:
            rms_cv = -1

        confidence = min(1.0, len(silence_segments) / 8)

        features = {
            "n_segments": len(silence_segments),
            "mean_cross_corr": round(float(np.mean(correlations)), 4) if correlations else 0,
            "mean_silence_rms": round(float(mean_silence_rms), 8),
            "mean_flatness": round(float(np.mean(silence_flatness_values)), 4) if silence_flatness_values else 0,
            "rms_cv": round(float(rms_cv), 4),
        }

        return min(1.0, score), confidence, anomalies, features

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 3: Spectral Envelope Smoothness
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_spectral_envelope(
        self, mag: np.ndarray, sr: int
    ) -> Tuple[float, float, List[str], Dict]:
        """
        Real vocal tract produces complex resonance with irregular peaks and
        anti-resonances (zeros). Neural vocoders smooth the spectral envelope
        because they learn averaged patterns, losing individual resonance detail.

        Checks:
        - Spectral envelope roughness (2nd derivative energy)
        - Per-frame spectral detail preservation (high-frequency fine structure)
        - Envelope consistency across frames (TTS is more self-consistent)
        """
        # Compute log-magnitude spectral envelope via cepstral smoothing
        n_freq, n_frames = mag.shape
        log_mag = np.log(mag + 1e-10)

        # Select voiced frames (higher energy)
        frame_energy = mag.sum(axis=0)
        energy_thresh = np.percentile(frame_energy, 40)
        voiced_idx = np.where(frame_energy > energy_thresh)[0]

        if len(voiced_idx) < 10:
            return 0.5, 0.1, [], {"voiced_frames": len(voiced_idx)}

        voiced_log_mag = log_mag[:, voiced_idx]

        # Cepstral smoothing: keep only first N cepstral coefficients
        # This gives us the smooth envelope; the remainder is the detail
        roughness_scores = []
        detail_scores = []

        for frame in voiced_log_mag.T[:100]:  # Cap at 100 frames
            # Cepstrum
            ceps = np.fft.irfft(frame)
            n_ceps = len(ceps)

            # Smooth envelope: keep first 30 cepstral coefficients
            smooth = np.copy(ceps)
            smooth[30:n_ceps - 30] = 0
            smooth_env = np.fft.rfft(smooth).real[:len(frame)]

            # Residual (spectral detail)
            residual = frame - smooth_env

            # Roughness: energy of 2nd derivative of smooth envelope
            if len(smooth_env) > 4:
                d2 = np.diff(smooth_env, n=2)
                roughness = np.sqrt(np.mean(d2 ** 2))
                roughness_scores.append(roughness)

            # Detail: energy of residual relative to envelope
            detail_energy = np.sqrt(np.mean(residual ** 2))
            envelope_energy = np.sqrt(np.mean(frame ** 2))
            if envelope_energy > 0:
                detail_ratio = detail_energy / envelope_energy
                detail_scores.append(detail_ratio)

        anomalies = []
        score = 0.0

        if roughness_scores:
            mean_roughness = np.mean(roughness_scores)
            roughness_cv = np.std(roughness_scores) / (mean_roughness + 1e-10)

            # Real speech: roughness varies widely (CV > 0.4)
            # TTS: smooth, consistent envelope (CV < 0.3)
            if roughness_cv < 0.2:
                score += 0.25
                anomalies.append("uniform_spectral_roughness")
            elif roughness_cv < 0.3:
                score += 0.12

            # Real speech: more roughness (complex resonances)
            # TTS: smoother envelopes
            if mean_roughness < 0.15:
                score += 0.2
                anomalies.append("smooth_spectral_envelope")
        else:
            mean_roughness = 0
            roughness_cv = 0

        if detail_scores:
            mean_detail = np.mean(detail_scores)
            # Real speech: more spectral detail (ratio > 0.15)
            # TTS: less detail (ratio < 0.10)
            if mean_detail < 0.08:
                score += 0.25
                anomalies.append("low_spectral_detail")
            elif mean_detail < 0.12:
                score += 0.12
        else:
            mean_detail = 0

        # Cross-frame envelope consistency
        if len(voiced_log_mag.T) >= 5:
            # Compute cosine similarity between consecutive frame envelopes
            env_sims = []
            for i in range(min(len(voiced_idx) - 1, 100)):
                a = voiced_log_mag[:, i]
                b = voiced_log_mag[:, i + 1]
                cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
                env_sims.append(cos_sim)

            if env_sims:
                sim_mean = np.mean(env_sims)
                sim_std = np.std(env_sims)
                # TTS: very consistent envelopes (high sim, low std)
                if sim_mean > 0.98 and sim_std < 0.01:
                    score += 0.2
                    anomalies.append("hyper_consistent_envelope")
                elif sim_mean > 0.96 and sim_std < 0.02:
                    score += 0.1
        else:
            sim_mean = 0
            sim_std = 0

        confidence = min(1.0, len(voiced_idx) / 40)

        features = {
            "mean_roughness": round(float(mean_roughness), 4),
            "roughness_cv": round(float(roughness_cv), 4),
            "mean_detail": round(float(mean_detail), 4),
            "envelope_sim_mean": round(float(sim_mean), 4) if isinstance(sim_mean, float) else 0,
        }

        return min(1.0, score), confidence, anomalies, features

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 4: Sub-band Temporal Correlation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_subband_correlation(
        self, mel_spec: np.ndarray,
    ) -> Tuple[float, float, List[str], Dict]:
        """
        In real speech, frequency bands are physically coupled through vocal
        tract resonance. When F1 energy changes, F2-F5 change in correlated
        ways dictated by articulatory physics. Neural TTS generates bands more
        independently, producing weaker or different cross-band correlations.

        Checks:
        - Average cross-band correlation matrix structure
        - Temporal lag in cross-band coupling
        - Band independence measure (how many bands behave independently)
        """
        mel_db = librosa.power_to_db(mel_spec + 1e-10)
        n_bands, n_frames = mel_db.shape

        if n_frames < 20:
            return 0.5, 0.1, [], {"n_frames": n_frames}

        # Group mel bands into 8 sub-bands
        n_groups = 8
        group_size = n_bands // n_groups
        subband_signals = []
        for g in range(n_groups):
            start = g * group_size
            end = start + group_size if g < n_groups - 1 else n_bands
            band_energy = mel_db[start:end, :].mean(axis=0)
            # Normalize
            band_energy = (band_energy - band_energy.mean()) / (band_energy.std() + 1e-10)
            subband_signals.append(band_energy)

        subband_signals = np.array(subband_signals)

        # Cross-correlation matrix (zero-lag)
        corr_matrix = np.corrcoef(subband_signals)
        # Get upper triangle (exclude diagonal)
        upper_tri = corr_matrix[np.triu_indices(n_groups, k=1)]
        upper_tri = upper_tri[~np.isnan(upper_tri)]

        if len(upper_tri) < 3:
            return 0.5, 0.1, [], {"valid_correlations": len(upper_tri)}

        mean_cross_corr = np.mean(np.abs(upper_tri))
        corr_std = np.std(upper_tri)

        # Adjacent band correlations (should be stronger in real speech)
        adjacent_corrs = []
        for i in range(n_groups - 1):
            adj_corr = corr_matrix[i, i + 1]
            if not np.isnan(adj_corr):
                adjacent_corrs.append(adj_corr)

        anomalies = []
        score = 0.0

        # Real speech: strong adjacent band correlation (>0.5) due to resonance
        # TTS: weaker or more uniform correlations
        if adjacent_corrs:
            mean_adj_corr = np.mean(adjacent_corrs)
            adj_corr_range = max(adjacent_corrs) - min(adjacent_corrs)

            if mean_adj_corr < 0.3:
                score += 0.25
                anomalies.append("weak_adjacent_band_coupling")
            elif mean_adj_corr < 0.4:
                score += 0.12

            # Real speech: adjacent correlations vary (different resonances)
            # TTS: more uniform
            if adj_corr_range < 0.15:
                score += 0.15
                anomalies.append("uniform_band_coupling")
        else:
            mean_adj_corr = 0
            adj_corr_range = 0

        # Check for bands that are too independent
        n_independent = sum(1 for c in upper_tri if abs(c) < 0.1)
        independence_ratio = n_independent / len(upper_tri)
        if independence_ratio > 0.6:
            score += 0.2
            anomalies.append("excessive_band_independence")
        elif independence_ratio > 0.4:
            score += 0.1

        # Check temporal cross-correlation at lag ±1-3
        # Real speech: adjacent bands have similar but time-shifted patterns
        lag_correlations = []
        for lag in [1, 2, 3]:
            for i in range(n_groups - 1):
                if len(subband_signals[i]) > lag + 5:
                    lagged_corr = np.corrcoef(
                        subband_signals[i][lag:],
                        subband_signals[i + 1][:-lag]
                    )[0, 1]
                    if not np.isnan(lagged_corr):
                        lag_correlations.append(lagged_corr)

        if lag_correlations:
            mean_lag_corr = np.mean(np.abs(lag_correlations))
            # Real speech: temporal coupling exists (>0.2)
            # TTS: less temporal coupling
            if mean_lag_corr < 0.1:
                score += 0.15
                anomalies.append("no_temporal_band_coupling")
        else:
            mean_lag_corr = 0

        confidence = min(1.0, n_frames / 60)

        features = {
            "mean_cross_corr": round(float(mean_cross_corr), 4),
            "mean_adj_corr": round(float(mean_adj_corr), 4),
            "adj_corr_range": round(float(adj_corr_range), 4),
            "independence_ratio": round(float(independence_ratio), 4),
            "mean_lag_corr": round(float(mean_lag_corr), 4),
        }

        return min(1.0, score), confidence, anomalies, features

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 5: Harmonic Structure Analysis
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_harmonics(
        self, waveform: np.ndarray, sr: int
    ) -> Tuple[float, float, List[str], Dict]:
        """
        Real voice harmonics have:
        - Natural rolloff (-12 to -18 dB/octave for modal voice)
        - Per-harmonic jitter (each harmonic varies independently)
        - Harmonic-to-noise distribution that changes with effort/register

        TTS harmonics:
        - Too uniform rolloff (learned average)
        - Harmonics vary together (correlation too high)
        - Missing higher harmonics (bandwidth limitation)
        """
        # Extract F0
        f0, voiced_flag, _ = librosa.pyin(
            waveform, fmin=65, fmax=600, sr=sr,
            frame_length=2048, hop_length=256
        )

        voiced_f0 = f0[~np.isnan(f0)]
        if len(voiced_f0) < 10:
            return 0.5, 0.1, [], {"voiced_frames": len(voiced_f0)}

        median_f0 = np.median(voiced_f0)

        # Analyze harmonic structure in voiced segments
        hop = 256
        frame_len = 2048
        harmonic_rolloffs = []
        harmonic_jitters = []
        n_harmonics_found = []

        for i in range(len(f0)):
            if np.isnan(f0[i]):
                continue

            start = i * hop
            end = start + frame_len
            if end > len(waveform):
                break

            frame = waveform[start:end]
            spec = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
            freqs = np.fft.rfftfreq(len(frame), 1.0 / sr)

            # Find harmonics (integer multiples of F0)
            current_f0 = f0[i]
            harmonic_amps = []
            max_harmonic = min(15, int(sr / 2 / current_f0))

            for h in range(1, max_harmonic + 1):
                target_freq = h * current_f0
                # Find nearest frequency bin
                bin_idx = np.argmin(np.abs(freqs - target_freq))
                # Take max in ±2 bins (allow slight deviation)
                lo = max(0, bin_idx - 2)
                hi = min(len(spec), bin_idx + 3)
                harmonic_amp = np.max(spec[lo:hi])
                harmonic_amps.append(harmonic_amp)

            if len(harmonic_amps) >= 5:
                # Harmonic rolloff: linear regression of log-amplitude vs harmonic number
                log_amps = np.log(np.array(harmonic_amps[:10]) + 1e-10)
                h_nums = np.arange(1, len(log_amps) + 1)
                if np.std(h_nums) > 0:
                    slope = np.polyfit(h_nums, log_amps, 1)[0]
                    harmonic_rolloffs.append(slope)

                # Per-harmonic jitter (relative amplitude variation from expected rolloff)
                expected = np.polyval(np.polyfit(h_nums, log_amps, 1), h_nums)
                residual = log_amps - expected
                harmonic_jitters.append(np.std(residual))

                n_harmonics_found.append(len(harmonic_amps))

        anomalies = []
        score = 0.0

        if harmonic_rolloffs:
            mean_rolloff = np.mean(harmonic_rolloffs)
            rolloff_cv = np.std(harmonic_rolloffs) / (abs(mean_rolloff) + 1e-10)

            # Real speech: rolloff varies with effort (-0.2 to -0.8)
            # TTS: more consistent rolloff (CV < 0.2)
            if rolloff_cv < 0.15:
                score += 0.2
                anomalies.append("uniform_harmonic_rolloff")
            elif rolloff_cv < 0.25:
                score += 0.1
        else:
            mean_rolloff = 0
            rolloff_cv = 0

        if harmonic_jitters:
            mean_jitter = np.mean(harmonic_jitters)
            # Real speech: per-harmonic jitter > 0.3 (irregular resonance)
            # TTS: smoother harmonics, jitter < 0.2
            if mean_jitter < 0.15:
                score += 0.25
                anomalies.append("smooth_harmonics")
            elif mean_jitter < 0.25:
                score += 0.12
        else:
            mean_jitter = 0

        if n_harmonics_found:
            mean_n_harmonics = np.mean(n_harmonics_found)
            # Real speech with effort can have 10+ harmonics
            # TTS often limited to 6-8
            if mean_n_harmonics < 5:
                score += 0.15
                anomalies.append("few_harmonics")
        else:
            mean_n_harmonics = 0

        confidence = min(1.0, len(harmonic_rolloffs) / 30)

        features = {
            "mean_rolloff": round(float(mean_rolloff), 4),
            "rolloff_cv": round(float(rolloff_cv), 4),
            "mean_harmonic_jitter": round(float(mean_jitter), 4),
            "mean_n_harmonics": round(float(mean_n_harmonics), 1),
            "median_f0": round(float(median_f0), 1),
        }

        return min(1.0, score), confidence, anomalies, features

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 6: Micro-Pause Regularity
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_micro_pauses(
        self, waveform: np.ndarray, sr: int
    ) -> Tuple[float, float, List[str], Dict]:
        """
        Natural speech has micro-pauses (20-200ms) at phrase boundaries,
        hesitations, and breathing moments. Their duration and spacing is
        highly variable (CV > 0.5). TTS produces more regular pauses because
        the prosody model learns averaged pause patterns.

        Checks:
        - Pause duration coefficient of variation
        - Pause interval regularity (autocorrelation)
        - Pause duration distribution shape
        """
        # Compute RMS energy in short frames
        frame_len = int(0.01 * sr)  # 10ms frames
        hop = frame_len // 2
        n_frames = (len(waveform) - frame_len) // hop + 1

        if n_frames < 20:
            return 0.5, 0.0, [], {"n_frames": n_frames}

        rms = np.array([
            np.sqrt(np.mean(waveform[i * hop: i * hop + frame_len] ** 2))
            for i in range(n_frames)
        ])

        # Adaptive threshold for silence vs speech
        rms_sorted = np.sort(rms)
        # Use valley in RMS histogram as threshold
        p20 = np.percentile(rms, 20)
        p80 = np.percentile(rms, 80)
        threshold = p20 + (p80 - p20) * 0.15

        is_silence = rms < max(threshold, 1e-5)

        # Find pause segments (contiguous silence)
        pauses = []
        in_pause = False
        pause_start = 0
        for i in range(len(is_silence)):
            if is_silence[i] and not in_pause:
                pause_start = i
                in_pause = True
            elif not is_silence[i] and in_pause:
                pause_duration_ms = (i - pause_start) * (hop / sr) * 1000
                if 15 < pause_duration_ms < 500:  # Micro-pauses: 15-500ms
                    pauses.append(pause_duration_ms)
                in_pause = False

        anomalies = []
        score = 0.0

        if len(pauses) < 3:
            return 0.5, 0.15, [], {"n_pauses": len(pauses)}

        pauses = np.array(pauses)
        pause_mean = np.mean(pauses)
        pause_std = np.std(pauses)
        pause_cv = pause_std / (pause_mean + 1e-10)

        # Check 1: Duration CV (real speech > 0.5, TTS < 0.3)
        if pause_cv < 0.2:
            score += 0.35
            anomalies.append("very_regular_pauses")
        elif pause_cv < 0.35:
            score += 0.18
            anomalies.append("regular_pauses")

        # Check 2: Pause interval regularity
        # Time between consecutive pauses
        if len(pauses) >= 4:
            # Use indices of pause starts for inter-pause intervals
            pause_intervals = np.diff(np.arange(len(pauses)))  # simplified
            # Actually compute inter-pause intervals from the audio
            # Using autocorrelation of the silence mask instead
            silence_signal = is_silence.astype(float)
            if len(silence_signal) > 10:
                acf = np.correlate(silence_signal - silence_signal.mean(),
                                   silence_signal - silence_signal.mean(),
                                   mode='full')
                acf = acf[len(acf) // 2:]
                if acf[0] > 0:
                    acf = acf / acf[0]
                    # Check for periodic peaks (regular pause pattern)
                    # Find peaks in ACF
                    peaks = []
                    for i in range(5, min(len(acf), 200)):
                        if acf[i] > acf[i - 1] and acf[i] > acf[i + 1] if i + 1 < len(acf) else True:
                            if acf[i] > 0.15:
                                peaks.append((i, acf[i]))
                    if peaks and peaks[0][1] > 0.3:
                        score += 0.2
                        anomalies.append("periodic_pause_pattern")

        # Check 3: Pause duration distribution
        pause_kurtosis = float(kurtosis(pauses)) if len(pauses) >= 5 else 0
        # Real speech: broad distribution (low kurtosis, many different pause lengths)
        # TTS: peaked distribution (high kurtosis, one dominant pause length)
        if pause_kurtosis > 3.0 and len(pauses) >= 5:
            score += 0.15
            anomalies.append("peaked_pause_distribution")

        confidence = min(1.0, len(pauses) / 10)

        features = {
            "n_pauses": len(pauses),
            "pause_cv": round(float(pause_cv), 4),
            "pause_mean_ms": round(float(pause_mean), 1),
            "pause_kurtosis": round(float(pause_kurtosis), 4),
        }

        return min(1.0, score), confidence, anomalies, features

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 7: Onset/Offset Transient Analysis
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_onset_transients(
        self, waveform: np.ndarray, sr: int, mag: np.ndarray
    ) -> Tuple[float, float, List[str], Dict]:
        """
        Real speech onsets (plosives like /p/, /t/, /k/) have sharp transients
        shaped by the physical release of articulatory closure. Neural vocoders
        must reconstruct these transients from mel-spectrograms, and the
        reconstruction introduces subtle artifacts:

        - Onset shape similarity (real onsets vary; vocoder onsets are templated)
        - Onset spectral spread (real plosives have broadband noise; vocoder
          onsets can be narrower)
        - Attack time variability (real varies with effort; TTS more consistent)
        """
        # Detect onsets
        onset_frames = librosa.onset.onset_detect(
            y=waveform, sr=sr, hop_length=256,
            backtrack=False, units='frames'
        )

        if len(onset_frames) < 5:
            return 0.5, 0.1, [], {"n_onsets": len(onset_frames)}

        # Extract onset characteristics
        onset_shapes = []
        onset_spectral_spreads = []
        attack_times = []

        for onset_frame in onset_frames:
            # Get waveform around onset (±20ms)
            onset_sample = onset_frame * 256
            window_samples = int(0.02 * sr)
            start = max(0, onset_sample - window_samples)
            end = min(len(waveform), onset_sample + window_samples)
            onset_region = waveform[start:end]

            if len(onset_region) < window_samples:
                continue

            # Onset shape (normalized envelope)
            envelope = np.abs(onset_region)
            smooth_env = uniform_filter1d(envelope, size=max(3, int(0.002 * sr)))
            if smooth_env.max() > 0:
                norm_env = smooth_env / smooth_env.max()
                onset_shapes.append(norm_env[:window_samples])

            # Spectral spread at onset
            if onset_frame < mag.shape[1]:
                onset_spec = mag[:, onset_frame]
                if onset_spec.sum() > 0:
                    onset_spec_norm = onset_spec / onset_spec.sum()
                    # Spectral spread = weighted std
                    freq_bins = np.arange(len(onset_spec_norm))
                    centroid = (freq_bins * onset_spec_norm).sum()
                    spread = np.sqrt(((freq_bins - centroid) ** 2 * onset_spec_norm).sum())
                    onset_spectral_spreads.append(spread)

            # Attack time (time from 10% to 90% of peak)
            if smooth_env.max() > 0:
                peak_idx = np.argmax(smooth_env)
                if peak_idx > 5:
                    pre_peak = smooth_env[:peak_idx + 1]
                    thresh_10 = 0.1 * smooth_env.max()
                    thresh_90 = 0.9 * smooth_env.max()
                    idx_10 = np.where(pre_peak >= thresh_10)[0]
                    idx_90 = np.where(pre_peak >= thresh_90)[0]
                    if len(idx_10) > 0 and len(idx_90) > 0:
                        attack_time_ms = (idx_90[0] - idx_10[0]) / sr * 1000
                        if 0.1 < attack_time_ms < 50:
                            attack_times.append(attack_time_ms)

        anomalies = []
        score = 0.0

        # Check 1: Onset shape similarity (template detection)
        if len(onset_shapes) >= 5:
            # Pad to same length and compute cross-correlations
            min_len = min(len(s) for s in onset_shapes)
            shapes = np.array([s[:min_len] for s in onset_shapes])

            # Average pairwise correlation
            shape_corrs = []
            for i in range(len(shapes)):
                for j in range(i + 1, min(i + 5, len(shapes))):
                    corr = np.corrcoef(shapes[i], shapes[j])[0, 1]
                    if not np.isnan(corr):
                        shape_corrs.append(corr)

            if shape_corrs:
                mean_shape_corr = np.mean(shape_corrs)
                # Real onsets: moderate similarity (0.3-0.7) — different phonemes
                # TTS onsets: higher similarity (>0.8) — vocoder template
                if mean_shape_corr > 0.85:
                    score += 0.3
                    anomalies.append("templated_onset_shapes")
                elif mean_shape_corr > 0.75:
                    score += 0.15
                    anomalies.append("similar_onset_shapes")
            else:
                mean_shape_corr = 0
        else:
            mean_shape_corr = 0

        # Check 2: Attack time variability
        if len(attack_times) >= 4:
            attack_cv = np.std(attack_times) / (np.mean(attack_times) + 1e-10)
            # Real speech: attack CV > 0.5 (different consonants, different effort)
            # TTS: attack CV < 0.3 (uniform reconstruction)
            if attack_cv < 0.2:
                score += 0.25
                anomalies.append("uniform_attack_times")
            elif attack_cv < 0.35:
                score += 0.12
        else:
            attack_cv = 0

        # Check 3: Onset spectral spread consistency
        if len(onset_spectral_spreads) >= 5:
            spread_cv = np.std(onset_spectral_spreads) / (np.mean(onset_spectral_spreads) + 1e-10)
            # Real speech: different consonants have very different spectral spread
            # TTS: more uniform
            if spread_cv < 0.2:
                score += 0.2
                anomalies.append("uniform_onset_spectra")
            elif spread_cv < 0.3:
                score += 0.1
        else:
            spread_cv = 0

        confidence = min(1.0, len(onset_shapes) / 15)

        features = {
            "n_onsets": len(onset_frames),
            "mean_shape_corr": round(float(mean_shape_corr), 4),
            "attack_cv": round(float(attack_cv), 4),
            "spread_cv": round(float(spread_cv), 4),
        }

        return min(1.0, score), confidence, anomalies, features

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 8: Pitch Micro-Dynamics
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_pitch_dynamics(
        self, waveform: np.ndarray, sr: int
    ) -> Tuple[float, float, List[str], Dict]:
        """
        Frame-level F0 in real speech has fractal-like variability — the
        variance of F0 changes depends on the time scale you measure at.
        TTS pitch contours are generated from text prosody models that
        produce smoother, more predictable F0 trajectories.

        Checks:
        - F0 delta (1st derivative) distribution shape
        - F0 delta-delta (2nd derivative) energy — real speech has more
        - F0 autocorrelation decay rate (TTS decays slower = more predictable)
        - F0 micro-jitter periodicity (TTS jitter is periodic, not random)
        - F0 multi-scale variability ratio (fractal dimension proxy)
        """
        # Extract F0
        f0, voiced_flag, _ = librosa.pyin(
            waveform, fmin=65, fmax=600, sr=sr,
            frame_length=1024, hop_length=128
        )

        # Get voiced F0 with indices
        voiced_indices = np.where(~np.isnan(f0))[0]
        if len(voiced_indices) < 20:
            return 0.5, 0.1, [], {"voiced_frames": len(voiced_indices)}

        # Get longest contiguous voiced segment
        voiced_f0 = f0[voiced_indices]

        # Find longest contiguous run
        diffs = np.diff(voiced_indices)
        segments = np.split(np.arange(len(voiced_indices)), np.where(diffs > 3)[0] + 1)
        longest_seg = max(segments, key=len)

        if len(longest_seg) < 15:
            return 0.5, 0.15, [], {"longest_voiced_segment": len(longest_seg)}

        f0_contiguous = voiced_f0[longest_seg]

        # Convert to cents relative to median (perceptual scale)
        median_f0 = np.median(f0_contiguous)
        f0_cents = 1200 * np.log2(f0_contiguous / (median_f0 + 1e-10) + 1e-10)

        anomalies = []
        score = 0.0

        # Check 1: F0 delta distribution
        f0_delta = np.diff(f0_cents)
        if len(f0_delta) > 5:
            delta_std = np.std(f0_delta)
            delta_kurtosis = float(kurtosis(f0_delta))

            # Real speech: varied deltas, moderate kurtosis (1-5)
            # TTS: smaller deltas OR too-perfect kurtosis
            if delta_std < 5.0:
                score += 0.2
                anomalies.append("smooth_pitch_contour")
            elif delta_std < 10.0:
                score += 0.08

            # TTS often has excess kurtosis (many small changes, few large)
            if delta_kurtosis > 8.0:
                score += 0.15
                anomalies.append("leptokurtic_pitch_deltas")
        else:
            delta_std = 0
            delta_kurtosis = 0

        # Check 2: F0 delta-delta energy (acceleration)
        if len(f0_delta) > 5:
            f0_dd = np.diff(f0_delta)
            dd_energy = np.sqrt(np.mean(f0_dd ** 2))
            # Real speech: higher delta-delta (more complex pitch movements)
            # TTS: smoother → lower delta-delta
            if dd_energy < 3.0:
                score += 0.15
                anomalies.append("low_pitch_acceleration")
        else:
            dd_energy = 0

        # Check 3: F0 autocorrelation decay
        if len(f0_cents) > 20:
            f0_centered = f0_cents - f0_cents.mean()
            acf = np.correlate(f0_centered, f0_centered, mode='full')
            acf = acf[len(acf) // 2:]
            if acf[0] > 0:
                acf = acf / acf[0]

                # Find lag where ACF drops below 0.5
                half_life = len(acf)
                for lag in range(1, min(len(acf), 50)):
                    if acf[lag] < 0.5:
                        half_life = lag
                        break

                # Real speech: ACF drops quickly (half_life 3-8 frames)
                # TTS: ACF drops slowly (half_life > 12) = more predictable
                if half_life > 15:
                    score += 0.2
                    anomalies.append("slow_pitch_decorrelation")
                elif half_life > 10:
                    score += 0.1
            else:
                half_life = 0
        else:
            half_life = 0

        # Check 4: Micro-jitter periodicity
        if len(f0_delta) > 20:
            # Check if jitter has periodic structure (TTS artifacts)
            jitter_fft = np.abs(np.fft.rfft(f0_delta - f0_delta.mean()))
            if len(jitter_fft) > 5:
                # Remove DC and very low frequency
                jitter_fft[0] = 0
                if jitter_fft.max() > 0:
                    jitter_fft = jitter_fft / jitter_fft.max()
                    # Check for dominant peak (periodic jitter)
                    peak_idx = np.argmax(jitter_fft[2:]) + 2
                    peak_val = jitter_fft[peak_idx]
                    mean_val = np.mean(jitter_fft[2:])

                    if peak_val > 3 * mean_val and peak_val > 0.4:
                        score += 0.2
                        anomalies.append("periodic_pitch_jitter")
                    elif peak_val > 2 * mean_val and peak_val > 0.3:
                        score += 0.1
                else:
                    peak_val = 0
            else:
                peak_val = 0
        else:
            peak_val = 0

        # Check 5: Multi-scale variability ratio (fractal-like property)
        if len(f0_cents) > 30:
            # Variance at different temporal scales
            scales = [1, 2, 4, 8, 16]
            variances = []
            for s in scales:
                if len(f0_cents) > s * 3:
                    # Compute variance of mean over non-overlapping windows of size s
                    n_windows = len(f0_cents) // s
                    windowed = f0_cents[:n_windows * s].reshape(n_windows, s)
                    window_means = windowed.mean(axis=1)
                    variances.append(np.var(window_means))

            if len(variances) >= 3:
                # In real speech, variance decreases with scale but not uniformly
                # (fractal behavior). In TTS, decrease is more uniform/predictable
                log_scales = np.log(scales[:len(variances)])
                log_vars = np.log(np.array(variances) + 1e-10)
                # Fit line in log-log space
                if np.std(log_scales) > 0:
                    slope, intercept = np.polyfit(log_scales, log_vars, 1)
                    residuals = log_vars - (slope * log_scales + intercept)
                    residual_energy = np.sqrt(np.mean(residuals ** 2))

                    # Real speech: irregular scaling (high residual > 0.5)
                    # TTS: clean scaling (low residual < 0.3)
                    if residual_energy < 0.2:
                        score += 0.15
                        anomalies.append("clean_pitch_scaling")
                else:
                    residual_energy = 0
            else:
                residual_energy = 0
        else:
            residual_energy = 0

        confidence = min(1.0, len(f0_contiguous) / 50)

        features = {
            "delta_std_cents": round(float(delta_std), 2),
            "delta_kurtosis": round(float(delta_kurtosis), 4),
            "dd_energy": round(float(dd_energy), 2),
            "acf_half_life": int(half_life),
            "jitter_periodicity": round(float(peak_val), 4) if isinstance(peak_val, float) else 0,
        }

        return min(1.0, score), confidence, anomalies, features
