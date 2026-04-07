"""
Spectral Continuity Analyzer
==============================
Detects AI-generated audio by analyzing spectral envelope evolution over time.

Key insight: Human speech has natural spectral transitions driven by
articulatory dynamics (tongue, lips, jaw movement). These transitions
follow physical constraints:

1. **Formant Transition Speed**: Formant frequencies can only change as
   fast as the articulators can move (~50-200 Hz/s for F1-F3).

2. **Spectral Tilt Consistency**: The overall spectral shape (tilt) of
   voiced speech reflects glottal source characteristics that are
   consistent within a speaker but vary naturally.

3. **Spectral Flux**: The rate of change of the spectral envelope over
   time follows patterns determined by the phonetic content. TTS systems
   often produce either too-smooth or too-abrupt transitions.

4. **Sub-band Energy Trajectories**: Energy in different frequency bands
   evolves according to phonological rules. AI audio may violate these.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


class SpectralContinuityAnalyzer:
    """Analyzes spectral envelope evolution for AI detection."""

    def __init__(self, sr: int = 16000):
        self.sr = sr

    def analyze(self, waveform: np.ndarray, sr: int = None) -> dict:
        """
        Analyze spectral continuity of audio.

        Args:
            waveform: Audio waveform (1D numpy array, float).
            sr: Sample rate (default 16000).

        Returns:
            {
                "score": float (0=real, 1=spoof),
                "confidence": float,
                "spectral_flux_score": float,
                "tilt_consistency_score": float,
                "subband_trajectory_score": float,
                "transition_smoothness_score": float,
                "anomalies": list[str],
            }
        """
        if sr is None:
            sr = self.sr

        if not HAS_LIBROSA:
            return self._fallback_result()

        if len(waveform) < sr * 0.5:  # Need at least 0.5s
            return self._fallback_result()

        # Ensure mono
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1)

        # Normalize
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val

        anomalies = []

        # Compute mel spectrogram
        n_fft = 1024
        hop = 256
        mel_spec = librosa.feature.melspectrogram(
            y=waveform, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=80
        )
        mel_db = librosa.power_to_db(mel_spec + 1e-10, ref=np.max)

        # 1. Spectral flux analysis
        flux_score = self._analyze_spectral_flux(mel_db)

        # 2. Spectral tilt consistency
        tilt_score = self._analyze_spectral_tilt(mel_db)

        # 3. Sub-band energy trajectories
        subband_score = self._analyze_subband_trajectories(mel_db)

        # 4. Transition smoothness
        transition_score = self._analyze_transition_smoothness(mel_db)

        # 5. Frame-to-frame correlation structure
        correlation_score = self._analyze_temporal_correlation(mel_db)

        if flux_score > 0.55:
            anomalies.append(f"Abnormal spectral flux pattern ({flux_score:.2f})")
        if tilt_score > 0.55:
            anomalies.append(f"Spectral tilt inconsistency ({tilt_score:.2f})")
        if subband_score > 0.55:
            anomalies.append(f"Sub-band trajectory anomaly ({subband_score:.2f})")
        if transition_score > 0.55:
            anomalies.append(f"Unnatural spectral transitions ({transition_score:.2f})")
        if correlation_score > 0.55:
            anomalies.append(f"Temporal correlation anomaly ({correlation_score:.2f})")

        # Weighted combination
        weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
        scores = np.array([flux_score, tilt_score, subband_score, transition_score, correlation_score])
        final_score = float(np.dot(scores, weights))

        confidence = 0.5 * (1.0 - float(np.std(scores))) + 0.3 * float(np.max(scores)) + 0.2 * min(1.0, len(anomalies) / 3)

        return {
            "score": float(np.clip(final_score, 0.0, 1.0)),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "spectral_flux_score": float(flux_score),
            "tilt_consistency_score": float(tilt_score),
            "subband_trajectory_score": float(subband_score),
            "transition_smoothness_score": float(transition_score),
            "temporal_correlation_score": float(correlation_score),
            "anomalies": anomalies,
        }

    def _analyze_spectral_flux(self, mel_db: np.ndarray) -> float:
        """
        Analyze spectral flux (rate of spectral change over time).

        Real speech: variable flux with clear peaks at phoneme boundaries
        and low flux during sustained vowels.
        AI speech: often too smooth (low flux variance) or too jittery.
        """
        # Frame-to-frame spectral change
        flux = np.sqrt(np.mean(np.diff(mel_db, axis=1) ** 2, axis=0))

        if len(flux) < 10:
            return 0.3

        # Statistics of spectral flux
        flux_mean = np.mean(flux)
        flux_std = np.std(flux)
        flux_cv = flux_std / (flux_mean + 1e-10)  # Coefficient of variation

        scores = []

        # Real speech: CV typically 0.4-0.9 (variable flux)
        # AI (too smooth): CV < 0.3
        # AI (too jittery): CV > 1.2
        if flux_cv < 0.25:
            scores.append(0.75)  # Too smooth
        elif flux_cv < 0.35:
            scores.append(0.5)
        elif flux_cv > 1.3:
            scores.append(0.65)  # Too jittery
        elif 0.35 <= flux_cv <= 1.0:
            scores.append(0.15)
        else:
            scores.append(0.3)

        # Check for periodicity in flux (TTS often has rhythmic patterns)
        if len(flux) > 30:
            flux_centered = flux - flux_mean
            autocorr = np.correlate(flux_centered, flux_centered, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)

            # Look for periodic peaks (TTS rhythmicity)
            if len(autocorr) > 20:
                peaks = []
                for i in range(5, len(autocorr) - 1):
                    if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
                        if autocorr[i] > 0.15:
                            peaks.append(autocorr[i])

                if peaks and max(peaks) > 0.30:
                    scores.append(0.65)  # Periodic flux = TTS artifact
                else:
                    scores.append(0.15)

        return float(np.mean(scores))

    def _analyze_spectral_tilt(self, mel_db: np.ndarray) -> float:
        """
        Analyze spectral tilt consistency over time.

        Spectral tilt reflects glottal source characteristics.
        Real speech: tilt varies naturally with vocal effort/emotion.
        AI speech: tilt may be too consistent or have unnatural patterns.
        """
        n_mels, n_frames = mel_db.shape

        # Compute spectral tilt per frame (slope of mel spectrum)
        tilts = []
        mel_indices = np.arange(n_mels).astype(float)
        for t in range(n_frames):
            frame = mel_db[:, t]
            if np.std(frame) < 0.1:
                continue
            # Linear fit: tilt = slope of spectrum
            coeffs = np.polyfit(mel_indices, frame, 1)
            tilts.append(coeffs[0])

        if len(tilts) < 10:
            return 0.3

        tilts = np.array(tilts)
        tilt_std = np.std(tilts)
        tilt_range = np.ptp(tilts)

        scores = []

        # Real speech: tilt_std typically 0.02-0.08
        # AI (too consistent tilt): < 0.015
        # AI (erratic tilt): > 0.12
        if tilt_std < 0.01:
            scores.append(0.75)  # Suspiciously consistent
        elif tilt_std < 0.02:
            scores.append(0.5)
        elif tilt_std > 0.15:
            scores.append(0.6)  # Erratic
        else:
            scores.append(0.15)

        # Tilt transition smoothness
        tilt_diff = np.diff(tilts)
        tilt_diff_std = np.std(tilt_diff)
        if tilt_diff_std < 0.005:
            scores.append(0.6)  # Too smooth transitions
        elif tilt_diff_std > 0.06:
            scores.append(0.5)  # Too abrupt
        else:
            scores.append(0.15)

        return float(np.mean(scores))

    def _analyze_subband_trajectories(self, mel_db: np.ndarray) -> float:
        """
        Analyze energy trajectories in different frequency bands.

        Real speech: band energies follow phonological patterns
        (vowels have specific formant band distributions).
        AI speech: may have uncorrelated or overly correlated band trajectories.
        """
        n_mels, n_frames = mel_db.shape
        if n_frames < 10:
            return 0.3

        # Split into 4 sub-bands
        n_bands = 4
        band_size = n_mels // n_bands
        band_energies = []

        for b in range(n_bands):
            start = b * band_size
            end = (b + 1) * band_size
            band_energy = np.mean(mel_db[start:end, :], axis=0)
            band_energies.append(band_energy)

        scores = []

        # Cross-band correlation
        # Real speech: moderate correlation between adjacent bands
        # AI: often too high (all bands move together) or too low
        for i in range(n_bands - 1):
            if np.std(band_energies[i]) < 0.1 or np.std(band_energies[i + 1]) < 0.1:
                continue
            corr = np.corrcoef(band_energies[i], band_energies[i + 1])[0, 1]
            if np.isnan(corr):
                continue

            # Real: correlation 0.3-0.8 for adjacent bands
            if abs(corr) > 0.95:
                scores.append(0.65)  # Too correlated
            elif abs(corr) < 0.1:
                scores.append(0.55)  # Too uncorrelated
            else:
                scores.append(0.15)

        # Low-to-high band energy ratio consistency
        if len(band_energies) >= 4:
            low_energy = band_energies[0]
            high_energy = band_energies[3]
            ratio = low_energy / (high_energy + 1e-10)
            ratio_cv = np.std(ratio) / (np.mean(ratio) + 1e-10)

            # Real: ratio varies (different phonemes have different formant energy)
            # AI: ratio may be too consistent
            if ratio_cv < 0.1:
                scores.append(0.6)
            elif ratio_cv < 0.2:
                scores.append(0.35)
            else:
                scores.append(0.15)

        return float(np.mean(scores)) if scores else 0.3

    def _analyze_transition_smoothness(self, mel_db: np.ndarray) -> float:
        """
        Analyze smoothness of spectral transitions.

        Real speech: transitions follow articulatory dynamics with
        varying speeds. AI: may have unnaturally uniform transitions.
        """
        n_mels, n_frames = mel_db.shape
        if n_frames < 20:
            return 0.3

        # Compute per-band transition velocities
        velocities = np.abs(np.diff(mel_db, axis=1))

        # Second-order: acceleration
        accelerations = np.abs(np.diff(velocities, axis=1))

        # Ratio of acceleration to velocity (jerk measure)
        mean_vel = np.mean(velocities, axis=0)
        mean_acc = np.mean(accelerations, axis=0)

        if len(mean_vel) < 5:
            return 0.3

        # Match lengths
        n = min(len(mean_vel), len(mean_acc))
        vel_trimmed = mean_vel[:n]
        acc_trimmed = mean_acc[:n]

        # Jerk ratio
        jerk_ratio = acc_trimmed / (vel_trimmed + 1e-10)
        jerk_mean = np.mean(jerk_ratio)
        jerk_std = np.std(jerk_ratio)

        scores = []

        # Real speech: variable jerk (natural acceleration/deceleration)
        # AI: often too uniform jerk
        if jerk_std < 0.3:
            scores.append(0.65)  # Too uniform
        elif jerk_std < 0.5:
            scores.append(0.35)
        else:
            scores.append(0.15)

        return float(np.mean(scores))

    def _analyze_temporal_correlation(self, mel_db: np.ndarray) -> float:
        """
        Analyze frame-to-frame correlation structure.

        Real speech: correlation decreases smoothly with temporal distance.
        AI: may have irregular correlation decay.
        """
        n_mels, n_frames = mel_db.shape
        if n_frames < 30:
            return 0.3

        # Compute correlation at different lags
        max_lag = min(20, n_frames // 3)
        correlations = []

        for lag in range(1, max_lag):
            frames_a = mel_db[:, :-lag].flatten()
            frames_b = mel_db[:, lag:].flatten()
            if np.std(frames_a) < 0.01 or np.std(frames_b) < 0.01:
                continue
            # Subsample for efficiency
            n = min(50000, len(frames_a))
            corr = np.corrcoef(frames_a[:n], frames_b[:n])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

        if len(correlations) < 5:
            return 0.3

        # Check for smooth decay
        corr_diff = np.diff(correlations)

        # Real: monotonically decreasing (mostly)
        # AI: may have non-monotonic patterns
        non_decreasing = np.sum(corr_diff > 0.02)  # Count increases
        fraction_increasing = non_decreasing / (len(corr_diff) + 1e-10)

        if fraction_increasing > 0.4:
            return 0.65  # Non-monotonic correlation decay
        elif fraction_increasing > 0.2:
            return 0.4
        else:
            return 0.15

    def _fallback_result(self) -> dict:
        return {
            "score": 0.5,
            "confidence": 0.0,
            "spectral_flux_score": 0.5,
            "tilt_consistency_score": 0.5,
            "subband_trajectory_score": 0.5,
            "transition_smoothness_score": 0.5,
            "temporal_correlation_score": 0.5,
            "anomalies": ["Insufficient data or missing librosa"],
        }
