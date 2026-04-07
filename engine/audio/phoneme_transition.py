"""
Phoneme Transition Analyzer
=============================
Detects AI-generated speech by analyzing coarticulation patterns at
phoneme boundaries.

Human speech is produced by physical articulators (tongue, lips, jaw,
velum) that have mass and inertia. This creates predictable coarticulation
effects where adjacent phonemes influence each other:

1. **Formant Transition Trajectories**: When transitioning between phonemes,
   formant frequencies follow smooth trajectories dictated by articulator
   kinematics. TTS systems often produce transitions that are either too
   abrupt (vocoder artifacts) or too smooth (over-smoothed neural TTS).

2. **Coarticulation Symmetry**: In natural speech, the influence of a
   phoneme on its neighbor is asymmetric (anticipatory > carryover).
   TTS systems may not model this asymmetry correctly.

3. **Transition Duration**: Natural transitions have durations determined
   by articulator biomechanics (typically 30-80ms for place changes).
   TTS may have different timing characteristics.

4. **Spectral Continuity at Boundaries**: Natural boundaries show specific
   spectral patterns. TTS systems using frame-based synthesis may show
   discontinuities at synthesis frame boundaries.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


class PhonemeTransitionAnalyzer:
    """Analyzes coarticulation patterns for AI speech detection."""

    def __init__(self, sr: int = 16000):
        self.sr = sr

    def analyze(self, waveform: np.ndarray, sr: int = None) -> dict:
        """
        Analyze phoneme transition patterns.

        Args:
            waveform: Audio waveform (1D float array).
            sr: Sample rate.

        Returns:
            dict with score, confidence, sub-scores, and anomalies.
        """
        if sr is None:
            sr = self.sr

        if not HAS_LIBROSA:
            return self._fallback_result()

        if len(waveform) < sr * 0.5:
            return self._fallback_result()

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1)

        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val

        anomalies = []

        # Compute STFT and MFCCs
        n_fft = 512
        hop = 160  # 10ms at 16kHz — fine resolution for transitions
        stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop)
        mag = np.abs(stft)
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=20, hop_length=hop)

        # 1. Formant transition analysis (via MFCC delta patterns)
        formant_score = self._analyze_formant_transitions(mfcc)

        # 2. Transition duration analysis
        duration_score = self._analyze_transition_durations(mfcc, sr, hop)

        # 3. Spectral boundary continuity
        boundary_score = self._analyze_boundary_continuity(mag)

        # 4. Delta-delta (acceleration) patterns
        acceleration_score = self._analyze_acceleration_patterns(mfcc)

        # 5. Voiced-unvoiced transition analysis
        vu_score = self._analyze_voiced_unvoiced_transitions(waveform, sr, hop, mag)

        if formant_score > 0.55:
            anomalies.append(f"Formant transition anomaly ({formant_score:.2f})")
        if duration_score > 0.55:
            anomalies.append(f"Transition duration anomaly ({duration_score:.2f})")
        if boundary_score > 0.55:
            anomalies.append(f"Spectral boundary discontinuity ({boundary_score:.2f})")
        if acceleration_score > 0.55:
            anomalies.append(f"Spectral acceleration anomaly ({acceleration_score:.2f})")
        if vu_score > 0.55:
            anomalies.append(f"Voiced-unvoiced transition anomaly ({vu_score:.2f})")

        weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
        scores = np.array([formant_score, duration_score, boundary_score, acceleration_score, vu_score])
        final_score = float(np.dot(scores, weights))

        confidence = (
            0.5 * (1.0 - float(np.std(scores)))
            + 0.3 * float(np.max(scores))
            + 0.2 * min(1.0, len(anomalies) / 3)
        )

        return {
            "score": float(np.clip(final_score, 0.0, 1.0)),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "formant_transition_score": float(formant_score),
            "transition_duration_score": float(duration_score),
            "boundary_continuity_score": float(boundary_score),
            "acceleration_score": float(acceleration_score),
            "voiced_unvoiced_score": float(vu_score),
            "anomalies": anomalies,
        }

    def _analyze_formant_transitions(self, mfcc: np.ndarray) -> float:
        """
        Analyze MFCC delta patterns as proxy for formant transitions.

        MFCCs 2-5 capture formant-like spectral shape. Their deltas
        reflect how formants move over time.

        Real speech: deltas have characteristic distributions driven
        by articulatory dynamics.
        AI speech: deltas may be too uniform or have wrong kurtosis.
        """
        if mfcc.shape[1] < 10:
            return 0.3

        # Compute deltas of formant-related MFCCs (2-5)
        formant_mfcc = mfcc[1:6, :]  # MFCCs 2-6
        deltas = np.diff(formant_mfcc, axis=1)

        scores = []

        # Delta kurtosis: real speech has heavy-tailed delta distributions
        # (many small changes + occasional large transitions)
        for i in range(min(4, deltas.shape[0])):
            d = deltas[i].flatten()
            kurt = self._kurtosis(d)
            # Real: kurtosis typically 3-15 (heavy tails)
            # AI: often < 2 (too Gaussian) or > 20 (too spiky)
            if kurt < 1.5:
                scores.append(0.7)
            elif kurt < 3.0:
                scores.append(0.4)
            elif kurt > 25:
                scores.append(0.55)
            else:
                scores.append(0.15)

        # Delta cross-correlation between adjacent MFCCs
        # Real: moderate correlation (articulators are coupled)
        # AI: may be too high (all formants move together) or too low
        for i in range(min(3, deltas.shape[0] - 1)):
            if np.std(deltas[i]) < 1e-5 or np.std(deltas[i + 1]) < 1e-5:
                continue
            corr = np.corrcoef(deltas[i], deltas[i + 1])[0, 1]
            if np.isnan(corr):
                continue
            if abs(corr) > 0.90:
                scores.append(0.6)
            elif abs(corr) < 0.05:
                scores.append(0.5)
            else:
                scores.append(0.15)

        return float(np.mean(scores)) if scores else 0.3

    def _analyze_transition_durations(self, mfcc: np.ndarray, sr: int, hop: int) -> float:
        """
        Analyze duration of spectral transitions.

        Natural transitions: 30-80ms for place changes, 10-30ms for voicing.
        AI: may have different timing characteristics.
        """
        if mfcc.shape[1] < 20:
            return 0.3

        # Use MFCC delta magnitude as transition indicator
        deltas = np.diff(mfcc[1:8, :], axis=1)
        delta_energy = np.sqrt(np.mean(deltas ** 2, axis=0))

        # Find transition peaks (high delta energy)
        mean_energy = np.mean(delta_energy)
        std_energy = np.std(delta_energy)
        threshold = mean_energy + 1.0 * std_energy

        in_transition = delta_energy > threshold
        frame_duration_ms = hop / sr * 1000  # ms per frame

        # Find transition durations
        transition_durations = []
        current_duration = 0
        for is_trans in in_transition:
            if is_trans:
                current_duration += frame_duration_ms
            else:
                if current_duration > 0:
                    transition_durations.append(current_duration)
                    current_duration = 0
        if current_duration > 0:
            transition_durations.append(current_duration)

        if len(transition_durations) < 3:
            return 0.3

        durations = np.array(transition_durations)
        mean_dur = np.mean(durations)
        std_dur = np.std(durations)
        cv_dur = std_dur / (mean_dur + 1e-10)

        scores = []

        # Natural transition durations: 20-100ms, CV ~ 0.4-0.9
        # AI: may have too uniform durations (low CV)
        if cv_dur < 0.25:
            scores.append(0.7)  # Suspiciously uniform
        elif cv_dur < 0.35:
            scores.append(0.45)
        elif cv_dur > 1.2:
            scores.append(0.5)  # Too variable
        else:
            scores.append(0.15)

        # Mean duration check
        if mean_dur < 15 or mean_dur > 120:
            scores.append(0.6)  # Outside natural range
        elif mean_dur < 25 or mean_dur > 90:
            scores.append(0.35)
        else:
            scores.append(0.15)

        return float(np.mean(scores))

    def _analyze_boundary_continuity(self, mag: np.ndarray) -> float:
        """
        Check for frame-level discontinuities in the spectrogram.

        Neural TTS using frame-based synthesis may produce subtle
        discontinuities at synthesis frame boundaries.
        """
        if mag.shape[1] < 20:
            return 0.3

        # Frame-to-frame spectral distance
        diffs = np.sqrt(np.mean(np.diff(mag, axis=1) ** 2, axis=0))

        if len(diffs) < 10:
            return 0.3

        # Look for periodic discontinuity patterns
        # TTS systems that process in chunks show periodic peaks
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)

        # Autocorrelation of frame differences
        centered = diffs - mean_diff
        n = len(centered)
        autocorr = np.correlate(centered, centered, mode='full')
        autocorr = autocorr[n - 1:]
        autocorr = autocorr / (autocorr[0] + 1e-10)

        scores = []

        # Check for periodic peaks in autocorrelation
        # TTS chunk sizes often correspond to ~50-200 frames
        max_peak = 0
        for lag in range(5, min(100, n // 2)):
            if autocorr[lag] > max_peak:
                max_peak = autocorr[lag]

        if max_peak > 0.3:
            scores.append(0.7)  # Periodic pattern in frame differences
        elif max_peak > 0.15:
            scores.append(0.4)
        else:
            scores.append(0.15)

        # Check for outlier discontinuities (isolated spikes)
        threshold = mean_diff + 3 * std_diff
        n_outliers = np.sum(diffs > threshold)
        outlier_ratio = n_outliers / len(diffs)

        # Real speech: sparse outliers at clear phoneme boundaries
        # AI: may have more uniformly distributed outliers
        if outlier_ratio > 0.08:
            scores.append(0.6)
        elif outlier_ratio < 0.005:
            scores.append(0.5)  # Suspiciously few discontinuities
        else:
            scores.append(0.15)

        return float(np.mean(scores))

    def _analyze_acceleration_patterns(self, mfcc: np.ndarray) -> float:
        """
        Analyze delta-delta (acceleration) patterns.

        In real speech, acceleration reflects articulatory dynamics
        with specific statistical properties.
        """
        if mfcc.shape[1] < 15:
            return 0.3

        # Compute delta-deltas
        deltas = np.diff(mfcc[1:8, :], axis=1)
        delta_deltas = np.diff(deltas, axis=1)

        scores = []

        # Delta-delta energy distribution
        dd_energy = np.sqrt(np.mean(delta_deltas ** 2, axis=0))

        if len(dd_energy) < 5:
            return 0.3

        dd_cv = np.std(dd_energy) / (np.mean(dd_energy) + 1e-10)

        # Real: CV 0.5-1.2 (variable acceleration)
        # AI: often too uniform acceleration
        if dd_cv < 0.3:
            scores.append(0.65)
        elif dd_cv < 0.45:
            scores.append(0.4)
        else:
            scores.append(0.15)

        # Ratio of delta-delta to delta energy
        d_energy = np.sqrt(np.mean(deltas ** 2, axis=0))
        if len(d_energy) > len(dd_energy):
            d_energy = d_energy[:len(dd_energy)]
        elif len(dd_energy) > len(d_energy):
            dd_energy = dd_energy[:len(d_energy)]

        ratio = np.mean(dd_energy) / (np.mean(d_energy) + 1e-10)

        # Real: ratio typically 0.4-0.8
        # AI: may differ
        if ratio < 0.25 or ratio > 1.0:
            scores.append(0.6)
        elif ratio < 0.35 or ratio > 0.85:
            scores.append(0.35)
        else:
            scores.append(0.15)

        return float(np.mean(scores))

    def _analyze_voiced_unvoiced_transitions(
        self, waveform: np.ndarray, sr: int, hop: int, mag: np.ndarray
    ) -> float:
        """
        Analyze transitions between voiced and unvoiced segments.

        Natural speech: gradual onset/offset of voicing with specific
        spectral characteristics.
        AI: may have abrupt or unnaturally smooth V/UV transitions.
        """
        # Simple energy-based voicing detection
        frame_length = hop * 2
        n_frames = len(waveform) // hop - 1

        if n_frames < 20:
            return 0.3

        energies = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop
            end = min(start + frame_length, len(waveform))
            frame = waveform[start:end]
            energies[i] = np.mean(frame ** 2)

        # Zero-crossing rate as voicing indicator
        zcr = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop
            end = min(start + frame_length, len(waveform))
            frame = waveform[start:end]
            if len(frame) > 1:
                zcr[i] = np.mean(np.abs(np.diff(np.sign(frame))) > 0)

        # Voiced: high energy, low ZCR
        # Unvoiced: low energy or high ZCR
        energy_threshold = np.mean(energies) * 0.3
        zcr_threshold = 0.15

        voiced = (energies > energy_threshold) & (zcr < zcr_threshold)

        # Find V/UV transitions
        transitions = np.diff(voiced.astype(int))
        onset_indices = np.where(transitions == 1)[0]
        offset_indices = np.where(transitions == -1)[0]

        scores = []

        if len(onset_indices) >= 3 and len(offset_indices) >= 3:
            # Analyze onset sharpness
            onset_sharpnesses = []
            for idx in onset_indices:
                if idx > 2 and idx < len(energies) - 2:
                    # Energy rise rate around onset
                    before = np.mean(energies[max(0, idx - 2):idx])
                    after = np.mean(energies[idx:min(len(energies), idx + 3)])
                    if before > 1e-8:
                        onset_sharpnesses.append(after / before)

            if len(onset_sharpnesses) >= 2:
                sharpness_cv = np.std(onset_sharpnesses) / (np.mean(onset_sharpnesses) + 1e-10)
                # Real: varied onset sharpness (CV > 0.3)
                # AI: uniform onsets (CV < 0.2)
                if sharpness_cv < 0.15:
                    scores.append(0.65)
                elif sharpness_cv < 0.25:
                    scores.append(0.4)
                else:
                    scores.append(0.15)

        return float(np.mean(scores)) if scores else 0.3

    @staticmethod
    def _kurtosis(arr: np.ndarray) -> float:
        n = len(arr)
        if n < 4:
            return 0.0
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-10:
            return 0.0
        m4 = np.mean((arr - mean) ** 4)
        return float(m4 / (std ** 4) - 3.0)

    def _fallback_result(self) -> dict:
        return {
            "score": 0.5,
            "confidence": 0.0,
            "formant_transition_score": 0.5,
            "transition_duration_score": 0.5,
            "boundary_continuity_score": 0.5,
            "acceleration_score": 0.5,
            "voiced_unvoiced_score": 0.5,
            "anomalies": ["Insufficient data or missing librosa"],
        }
