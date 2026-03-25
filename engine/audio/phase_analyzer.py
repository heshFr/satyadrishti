"""
Phase-Domain Forensics Analyzer
================================
Detects spoofing cues embedded in the phase domain that magnitude-only
analysis misses. Natural speech has smooth, correlated phase trajectories
across harmonics; vocoders and neural TTS introduce characteristic
discontinuities at synthesis frame boundaries.

Key features extracted:
- Group delay statistics (derivative of phase w.r.t. frequency)
- Instantaneous frequency variation (derivative of phase w.r.t. time)
- Cross-harmonic phase coherence
- Phase discontinuity rate
- Phase distribution entropy
"""

import logging
from typing import Dict, List

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import kurtosis as scipy_kurtosis

log = logging.getLogger("satyadrishti.phase_analyzer")

# Minimum audio duration in seconds for reliable analysis
MIN_DURATION_S = 0.1
# STFT parameters
DEFAULT_N_FFT = 1024
DEFAULT_HOP_LENGTH = 256
DEFAULT_WIN_LENGTH = 1024

# Thresholds for anomaly detection (calibrated on ASVspoof-style data)
THRESHOLDS = {
    "group_delay_std_high": 50.0,         # high std suggests vocoder artifacts
    "group_delay_kurtosis_low": 1.5,      # natural speech is moderately leptokurtic
    "group_delay_kurtosis_high": 30.0,    # extremely high kurtosis = artificial
    "if_std_low": 5.0,                    # too-stable IF is suspicious
    "phase_coherence_high": 0.95,         # unnaturally high coherence
    "phase_coherence_low": 0.1,           # very low coherence is also suspicious
    "discontinuity_rate_high": 50.0,      # discontinuities per second
    "phase_entropy_low": 1.5,             # too-ordered phase distribution
}


class PhaseAnalyzer:
    """
    Analyzes the phase domain of audio signals for deepfake detection.

    Vocoders (Griffin-Lim, WaveRNN, HiFi-GAN, etc.) reconstruct magnitude
    spectrograms but struggle with phase coherence. This analyzer detects
    those artifacts by examining:

    1. Group delay: d(phase)/d(frequency) — should be smooth for natural speech
    2. Instantaneous frequency: d(phase)/d(time) — temporal phase stability
    3. Phase coherence: cross-harmonic phase consistency at fundamental + harmonics
    4. Discontinuities: abrupt phase jumps at vocoder frame boundaries
    5. Phase entropy: distribution of phase values (uniform for noise, structured for speech)
    """

    def __init__(
        self,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        win_length: int = DEFAULT_WIN_LENGTH,
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def analyze(self, waveform: np.ndarray, sr: int = 16000) -> Dict:
        """
        Analyze phase characteristics of an audio waveform.

        Args:
            waveform: 1D numpy array (mono, float32/float64), expected 16kHz
            sr: Sample rate in Hz

        Returns:
            dict with keys: score, confidence, features, anomalies
        """
        # Validate input
        if waveform is None or len(waveform) == 0:
            log.warning("Empty waveform provided to PhaseAnalyzer")
            return self._empty_result("Empty waveform")

        # Ensure 1D mono
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1) if waveform.ndim == 2 else waveform.flatten()

        waveform = waveform.astype(np.float64)
        duration_s = len(waveform) / sr

        if duration_s < MIN_DURATION_S:
            log.warning(f"Audio too short for phase analysis: {duration_s:.3f}s (min {MIN_DURATION_S}s)")
            return self._empty_result("Audio too short")

        # Check for silence (RMS below threshold)
        rms = np.sqrt(np.mean(waveform**2))
        if rms < 1e-6:
            log.warning("Silent audio provided to PhaseAnalyzer")
            return self._empty_result("Silent audio")

        try:
            # Compute STFT with phase
            phase_matrix, magnitude_matrix = self._compute_stft(waveform, sr)

            if phase_matrix is None or phase_matrix.shape[1] < 3:
                return self._empty_result("Insufficient frames for phase analysis")

            # Extract features
            features = {}
            anomalies = []

            # 1. Group delay analysis
            gd_std, gd_kurtosis = self._analyze_group_delay(phase_matrix)
            features["group_delay_std"] = float(gd_std)
            features["group_delay_kurtosis"] = float(gd_kurtosis)

            # 2. Instantaneous frequency analysis
            if_std = self._analyze_instantaneous_frequency(phase_matrix, sr)
            features["instantaneous_freq_std"] = float(if_std)

            # 3. Phase coherence across harmonics
            phase_coh = self._analyze_phase_coherence(phase_matrix, magnitude_matrix, sr)
            features["phase_coherence"] = float(phase_coh)

            # 4. Phase discontinuity rate
            disc_rate = self._analyze_discontinuities(phase_matrix, sr)
            features["phase_discontinuity_rate"] = float(disc_rate)

            # 5. Phase entropy
            ph_entropy = self._analyze_phase_entropy(phase_matrix)
            features["phase_entropy"] = float(ph_entropy)

            # Detect anomalies
            anomalies = self._detect_anomalies(features)

            # Compute composite spoof score
            score, confidence = self._compute_score(features, duration_s)

            return {
                "score": float(np.clip(score, 0.0, 1.0)),
                "confidence": float(np.clip(confidence, 0.0, 1.0)),
                "features": features,
                "anomalies": anomalies,
            }

        except Exception as e:
            log.error(f"Phase analysis failed: {e}", exc_info=True)
            return self._empty_result(f"Analysis error: {str(e)}")

    def _compute_stft(self, waveform: np.ndarray, sr: int):
        """
        Compute STFT and return phase and magnitude matrices.

        Returns:
            phase_matrix: (n_freq, n_frames) unwrapped phase
            magnitude_matrix: (n_freq, n_frames) magnitude
        """
        # Use scipy for STFT to get complex output directly
        freqs, times, stft_complex = scipy_signal.stft(
            waveform,
            fs=sr,
            window="hann",
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            nfft=self.n_fft,
            return_onesided=True,
        )

        magnitude_matrix = np.abs(stft_complex)
        phase_matrix = np.angle(stft_complex)

        return phase_matrix, magnitude_matrix

    def _analyze_group_delay(self, phase_matrix: np.ndarray):
        """
        Compute group delay statistics.

        Group delay = -d(unwrapped_phase) / d(frequency_bin)
        Natural speech has relatively smooth group delay; vocoders produce
        spikes and irregularities.

        Returns:
            (std, kurtosis) of the group delay across frequency, averaged over frames
        """
        n_freq, n_frames = phase_matrix.shape

        if n_freq < 3:
            return 0.0, 0.0

        # Unwrap phase along frequency axis for each frame
        unwrapped = np.unwrap(phase_matrix, axis=0)

        # Group delay: negative derivative of phase w.r.t. frequency bin
        # Using central difference for interior points
        group_delay = -np.diff(unwrapped, axis=0)  # (n_freq-1, n_frames)

        # Compute per-frame statistics, then average
        frame_stds = np.std(group_delay, axis=0)
        mean_std = np.mean(frame_stds)

        # Kurtosis of the group delay distribution (pooled across all frames)
        gd_flat = group_delay.flatten()
        # Filter out extreme outliers before kurtosis (prevents inf)
        gd_clipped = gd_flat[np.isfinite(gd_flat)]
        if len(gd_clipped) < 4:
            return mean_std, 0.0

        # Clip extreme values to prevent numerical overflow
        p01, p99 = np.percentile(gd_clipped, [1, 99])
        gd_clipped = gd_clipped[(gd_clipped >= p01) & (gd_clipped <= p99)]

        if len(gd_clipped) < 4:
            return mean_std, 0.0

        gd_kurt = scipy_kurtosis(gd_clipped, fisher=True, nan_policy="omit")
        if not np.isfinite(gd_kurt):
            gd_kurt = 0.0

        return mean_std, gd_kurt

    def _analyze_instantaneous_frequency(self, phase_matrix: np.ndarray, sr: int):
        """
        Compute instantaneous frequency variation.

        IF = d(unwrapped_phase) / d(time) / (2*pi) in Hz
        Natural speech shows gradual IF changes; vocoders may show abrupt shifts
        at frame boundaries.

        Returns:
            Mean standard deviation of IF across frequency bins
        """
        n_freq, n_frames = phase_matrix.shape

        if n_frames < 3:
            return 0.0

        # Unwrap phase along time axis for each frequency bin
        unwrapped = np.unwrap(phase_matrix, axis=1)

        # Time derivative of phase (instantaneous frequency in radians/sample)
        dt = self.hop_length / sr  # time step between STFT frames
        d_phase_dt = np.diff(unwrapped, axis=1) / dt  # (n_freq, n_frames-1)

        # Convert to Hz
        inst_freq = d_phase_dt / (2.0 * np.pi)

        # Focus on voiced frequency range (80Hz - 4000Hz)
        freq_resolution = sr / self.n_fft
        low_bin = max(1, int(80.0 / freq_resolution))
        high_bin = min(n_freq, int(4000.0 / freq_resolution))

        if low_bin >= high_bin:
            return 0.0

        inst_freq_voiced = inst_freq[low_bin:high_bin, :]

        # Standard deviation of IF across time for each frequency bin
        if_std_per_bin = np.std(inst_freq_voiced, axis=1)
        mean_if_std = np.mean(if_std_per_bin)

        if not np.isfinite(mean_if_std):
            return 0.0

        return mean_if_std

    def _analyze_phase_coherence(
        self, phase_matrix: np.ndarray, magnitude_matrix: np.ndarray, sr: int
    ):
        """
        Measure cross-harmonic phase consistency.

        Natural speech has correlated phases at fundamental frequency and
        its harmonics (F0, 2*F0, 3*F0, ...). Vocoders often fail to maintain
        this relationship.

        Returns:
            Phase coherence score (0.0 = incoherent, 1.0 = perfectly coherent)
        """
        n_freq, n_frames = phase_matrix.shape
        freq_resolution = sr / self.n_fft

        if n_frames < 2:
            return 0.5  # neutral value when insufficient data

        coherence_values = []

        # Analyze each frame independently
        for t in range(n_frames):
            mag_frame = magnitude_matrix[:, t]

            # Estimate fundamental frequency from magnitude spectrum
            # Look in typical F0 range: 80-400 Hz
            low_bin = max(1, int(80.0 / freq_resolution))
            high_bin = min(n_freq - 1, int(400.0 / freq_resolution))

            if low_bin >= high_bin:
                continue

            search_region = mag_frame[low_bin:high_bin]
            if np.max(search_region) < 1e-8:
                continue  # skip silent/unvoiced frames

            f0_bin_local = np.argmax(search_region)
            f0_bin = f0_bin_local + low_bin

            if f0_bin < 1:
                continue

            # Collect phases at harmonics (F0, 2*F0, 3*F0, ...)
            harmonic_phases = []
            max_harmonics = 6
            for h in range(1, max_harmonics + 1):
                h_bin = f0_bin * h
                if h_bin >= n_freq:
                    break
                # Only include if harmonic has significant energy
                if mag_frame[h_bin] > np.max(mag_frame) * 0.01:
                    harmonic_phases.append(phase_matrix[h_bin, t])

            if len(harmonic_phases) >= 3:
                # Measure phase consistency using circular variance
                # Phase differences between consecutive harmonics should be consistent
                phases_arr = np.array(harmonic_phases)
                phase_diffs = np.diff(phases_arr)
                # Wrap to [-pi, pi]
                phase_diffs = (phase_diffs + np.pi) % (2 * np.pi) - np.pi

                # Circular consistency: mean resultant length of phase differences
                # 1.0 = perfectly consistent, 0.0 = random
                mean_cos = np.mean(np.cos(phase_diffs))
                mean_sin = np.mean(np.sin(phase_diffs))
                resultant_length = np.sqrt(mean_cos**2 + mean_sin**2)
                coherence_values.append(resultant_length)

        if not coherence_values:
            return 0.5  # neutral when no voiced frames detected

        return float(np.mean(coherence_values))

    def _analyze_discontinuities(self, phase_matrix: np.ndarray, sr: int):
        """
        Count phase discontinuities per second.

        Vocoders operating on fixed frame sizes produce periodic phase jumps
        at frame boundaries. Natural speech has smooth phase evolution.

        Returns:
            Discontinuities per second (averaged across frequency bins in voiced range)
        """
        n_freq, n_frames = phase_matrix.shape

        if n_frames < 3:
            return 0.0

        duration_s = (n_frames * self.hop_length) / sr
        if duration_s < MIN_DURATION_S:
            return 0.0

        freq_resolution = sr / self.n_fft
        low_bin = max(1, int(100.0 / freq_resolution))
        high_bin = min(n_freq, int(4000.0 / freq_resolution))

        if low_bin >= high_bin:
            return 0.0

        # Unwrap phase along time for each frequency bin
        unwrapped = np.unwrap(phase_matrix[low_bin:high_bin, :], axis=1)

        # Second derivative of phase (acceleration) — discontinuities appear as spikes
        phase_accel = np.diff(unwrapped, n=2, axis=1)  # (bins, frames-2)

        # A discontinuity is where the phase acceleration exceeds pi
        # (indicating an abrupt change that unwrapping couldn't smooth)
        threshold = np.pi
        discontinuity_count = np.sum(np.abs(phase_accel) > threshold, axis=1)  # per frequency bin

        # Average across frequency bins and normalize to per-second rate
        mean_disc_count = np.mean(discontinuity_count)
        disc_per_second = mean_disc_count / duration_s

        if not np.isfinite(disc_per_second):
            return 0.0

        return disc_per_second

    def _analyze_phase_entropy(self, phase_matrix: np.ndarray):
        """
        Compute entropy of the phase distribution.

        Natural speech has structured phase patterns; pure noise has
        uniformly distributed phase (maximum entropy). Vocoders may produce
        either too-structured (low entropy) or noise-like (high entropy) phase.

        Returns:
            Normalized entropy (0.0 = fully structured, ~1.0 = uniform/random)
        """
        n_freq, n_frames = phase_matrix.shape

        if n_freq * n_frames < 10:
            return 0.5

        # Flatten all phase values and wrap to [-pi, pi]
        all_phases = phase_matrix.flatten()
        all_phases = (all_phases + np.pi) % (2 * np.pi) - np.pi

        # Histogram of phase values
        n_bins = 64
        counts, _ = np.histogram(all_phases, bins=n_bins, range=(-np.pi, np.pi))

        # Normalize to probability distribution
        total = counts.sum()
        if total == 0:
            return 0.5

        probs = counts / total
        probs = probs[probs > 0]  # remove zero bins for log

        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs))

        # Normalize by maximum entropy (uniform distribution = log2(n_bins))
        max_entropy = np.log2(n_bins)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.5

        return float(normalized_entropy)

    def _detect_anomalies(self, features: Dict) -> List[str]:
        """Detect phase-domain anomalies based on feature thresholds."""
        anomalies = []

        gd_std = features["group_delay_std"]
        gd_kurt = features["group_delay_kurtosis"]
        if_std = features["instantaneous_freq_std"]
        phase_coh = features["phase_coherence"]
        disc_rate = features["phase_discontinuity_rate"]
        ph_entropy = features["phase_entropy"]

        if gd_std > THRESHOLDS["group_delay_std_high"]:
            anomalies.append(
                f"High group delay variance ({gd_std:.1f}) — possible vocoder artifacts"
            )

        if gd_kurt > THRESHOLDS["group_delay_kurtosis_high"]:
            anomalies.append(
                f"Extreme group delay kurtosis ({gd_kurt:.1f}) — non-natural phase distribution"
            )
        elif gd_kurt < THRESHOLDS["group_delay_kurtosis_low"]:
            anomalies.append(
                f"Low group delay kurtosis ({gd_kurt:.2f}) — unnaturally smooth phase"
            )

        if if_std < THRESHOLDS["if_std_low"]:
            anomalies.append(
                f"Low instantaneous frequency variation ({if_std:.2f} Hz) — possible synthesis artifacts"
            )

        if phase_coh > THRESHOLDS["phase_coherence_high"]:
            anomalies.append(
                f"Unnaturally high phase coherence ({phase_coh:.3f}) — possible copy-synthesis"
            )
        elif phase_coh < THRESHOLDS["phase_coherence_low"]:
            anomalies.append(
                f"Very low phase coherence ({phase_coh:.3f}) — phase reconstruction failure"
            )

        if disc_rate > THRESHOLDS["discontinuity_rate_high"]:
            anomalies.append(
                f"High phase discontinuity rate ({disc_rate:.1f}/s) — vocoder frame boundary artifacts"
            )

        if ph_entropy < THRESHOLDS["phase_entropy_low"] / np.log2(64):
            anomalies.append(
                f"Very low phase entropy ({ph_entropy:.3f}) — overly structured phase pattern"
            )

        return anomalies

    def _compute_score(self, features: Dict, duration_s: float):
        """
        Compute composite spoof score from phase features.

        Score: 0.0 = likely bonafide, 1.0 = likely spoof
        Confidence increases with duration and number of anomalies.
        """
        sub_scores = []
        weights = []

        # Group delay std: higher = more suspicious
        gd_std = features["group_delay_std"]
        gd_score = np.clip(gd_std / 100.0, 0.0, 1.0)  # normalize
        sub_scores.append(gd_score)
        weights.append(1.5)

        # Group delay kurtosis: very high or very low is suspicious
        gd_kurt = features["group_delay_kurtosis"]
        if gd_kurt > THRESHOLDS["group_delay_kurtosis_high"]:
            kurt_score = np.clip((gd_kurt - 10.0) / 40.0, 0.0, 1.0)
        elif gd_kurt < THRESHOLDS["group_delay_kurtosis_low"]:
            kurt_score = np.clip(1.0 - gd_kurt / THRESHOLDS["group_delay_kurtosis_low"], 0.0, 1.0)
        else:
            kurt_score = 0.0
        sub_scores.append(kurt_score)
        weights.append(1.0)

        # Instantaneous frequency: too-low std is suspicious
        if_std = features["instantaneous_freq_std"]
        if if_std < THRESHOLDS["if_std_low"]:
            if_score = np.clip(1.0 - if_std / THRESHOLDS["if_std_low"], 0.0, 1.0)
        else:
            if_score = 0.0
        sub_scores.append(if_score)
        weights.append(1.0)

        # Phase coherence: extremes are suspicious
        phase_coh = features["phase_coherence"]
        if phase_coh > THRESHOLDS["phase_coherence_high"]:
            coh_score = np.clip((phase_coh - 0.9) / 0.1, 0.0, 1.0)
        elif phase_coh < THRESHOLDS["phase_coherence_low"]:
            coh_score = np.clip(1.0 - phase_coh / THRESHOLDS["phase_coherence_low"], 0.0, 1.0)
        else:
            coh_score = 0.0
        sub_scores.append(coh_score)
        weights.append(1.2)

        # Discontinuity rate: higher = more suspicious
        disc_rate = features["phase_discontinuity_rate"]
        disc_score = np.clip(disc_rate / 100.0, 0.0, 1.0)
        sub_scores.append(disc_score)
        weights.append(1.5)

        # Phase entropy: very low = suspicious
        ph_entropy = features["phase_entropy"]
        if ph_entropy < 0.3:
            ent_score = np.clip(1.0 - ph_entropy / 0.3, 0.0, 1.0)
        else:
            ent_score = 0.0
        sub_scores.append(ent_score)
        weights.append(0.8)

        # Weighted average
        weights = np.array(weights)
        sub_scores = np.array(sub_scores)
        score = np.sum(sub_scores * weights) / np.sum(weights)

        # Confidence based on duration and signal quality
        # Longer audio = more reliable analysis
        duration_conf = np.clip(duration_s / 3.0, 0.3, 1.0)  # reaches max at 3s
        confidence = duration_conf * 0.8  # base confidence from duration

        # Boost confidence if features are clearly anomalous or clearly normal
        score_extremity = abs(score - 0.5) * 2.0  # 0.0 at midpoint, 1.0 at extremes
        confidence += score_extremity * 0.2

        return float(score), float(np.clip(confidence, 0.0, 1.0))

    @staticmethod
    def _empty_result(reason: str) -> Dict:
        """Return a neutral result when analysis cannot be performed."""
        return {
            "score": 0.5,
            "confidence": 0.0,
            "features": {
                "group_delay_std": 0.0,
                "group_delay_kurtosis": 0.0,
                "instantaneous_freq_std": 0.0,
                "phase_coherence": 0.0,
                "phase_discontinuity_rate": 0.0,
                "phase_entropy": 0.0,
            },
            "anomalies": [f"Analysis skipped: {reason}"],
        }
