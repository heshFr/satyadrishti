"""
Formant-Based Forensics Analyzer
==================================
Detects deepfake audio by analyzing formant characteristics. Deepfake models
frequently fail to accurately replicate formant transitions at consonant-vowel
boundaries, produce inconsistent vocal tract lengths across an utterance, and
compress the vowel space relative to natural speech.

Key features extracted:
- F1, F2, F3 statistics (mean, std)
- Vowel space area in the F1-F2 plane
- Formant bandwidth
- Vocal tract length consistency across frames
- Formant transition rate
- F1-F2 correlation
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal

log = logging.getLogger("satyadrishti.formant_analyzer")

# Minimum audio duration in seconds for reliable analysis
MIN_DURATION_S = 0.1
# Pre-emphasis coefficient
PRE_EMPHASIS_COEFF = 0.97
# Speed of sound in cm/s (for VTL estimation)
SPEED_OF_SOUND_CM = 34300.0
# Frame parameters for LPC analysis
FRAME_DURATION_MS = 25
FRAME_SHIFT_MS = 10

# Formant frequency ranges (Hz)
F1_RANGE = (200.0, 900.0)
F2_RANGE = (600.0, 2800.0)
F3_RANGE = (1800.0, 3500.0)

# Thresholds for anomaly detection
THRESHOLDS = {
    "vtl_consistency_low": 0.7,           # inconsistent VTL across frames
    "vowel_space_small": 50000.0,         # narrow vowel space (Hz^2)
    "formant_transition_low": 0.5,        # too-smooth transitions (Hz/ms)
    "formant_transition_high": 50.0,      # unnaturally fast transitions
    "f1_f2_correlation_high": 0.85,       # too-correlated (artificial coupling)
    "f1_f2_correlation_low": -0.1,        # anti-correlated (unusual)
    "bandwidth_narrow": 30.0,             # unnaturally narrow bandwidths
    "f1_std_low": 15.0,                   # too-uniform F1
    "f2_std_low": 30.0,                   # too-uniform F2
}


def _levinson_durbin(r: np.ndarray, order: int) -> Optional[np.ndarray]:
    """
    Levinson-Durbin recursion for LPC coefficient estimation.

    Args:
        r: Autocorrelation sequence of length >= order+1
        order: LPC order

    Returns:
        LPC coefficients (length order+1, first element is 1.0) or None on failure
    """
    if len(r) < order + 1:
        return None

    if r[0] == 0.0:
        return None

    a = np.zeros(order + 1)
    a[0] = 1.0
    error = r[0]

    for i in range(1, order + 1):
        # Compute reflection coefficient
        acc = 0.0
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k = -(r[i] + acc) / error

        if abs(k) >= 1.0:
            # Unstable — return what we have so far
            return a if i > 1 else None

        # Update coefficients
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + k * a[i - j]
        a_new[i] = k
        a = a_new

        error *= (1.0 - k * k)
        if error <= 0:
            return a if i > 1 else None

    return a


class FormantAnalyzer:
    """
    Analyzes formant characteristics of audio signals for deepfake detection.

    LPC (Linear Predictive Coding) is used to model the vocal tract as an
    all-pole filter. The roots of the LPC polynomial correspond to formant
    frequencies and bandwidths.

    Detection heuristics:
    - AI voices often show too-smooth formant transitions at consonant-vowel
      boundaries (natural speech has rapid, non-linear transitions)
    - Inconsistent vocal tract length (VTL) across frames suggests splicing
      or synthesis artifacts
    - Narrow vowel space in the F1-F2 plane indicates reduced articulatory
      range (common in TTS)
    """

    def __init__(
        self,
        frame_duration_ms: int = FRAME_DURATION_MS,
        frame_shift_ms: int = FRAME_SHIFT_MS,
    ):
        self.frame_duration_ms = frame_duration_ms
        self.frame_shift_ms = frame_shift_ms

    def analyze(self, waveform: np.ndarray, sr: int = 16000) -> Dict:
        """
        Analyze formant characteristics of an audio waveform.

        Args:
            waveform: 1D numpy array (mono, float32/float64), expected 16kHz
            sr: Sample rate in Hz

        Returns:
            dict with keys: score, confidence, features, anomalies
        """
        # Validate input
        if waveform is None or len(waveform) == 0:
            log.warning("Empty waveform provided to FormantAnalyzer")
            return self._empty_result("Empty waveform")

        # Ensure 1D mono
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1) if waveform.ndim == 2 else waveform.flatten()

        waveform = waveform.astype(np.float64)
        duration_s = len(waveform) / sr

        if duration_s < MIN_DURATION_S:
            log.warning(f"Audio too short for formant analysis: {duration_s:.3f}s (min {MIN_DURATION_S}s)")
            return self._empty_result("Audio too short")

        # Check for silence
        rms = np.sqrt(np.mean(waveform**2))
        if rms < 1e-6:
            log.warning("Silent audio provided to FormantAnalyzer")
            return self._empty_result("Silent audio")

        try:
            # Extract formants for all frames
            formant_tracks, bandwidth_tracks = self._extract_formants(waveform, sr)

            if formant_tracks is None or len(formant_tracks) < 3:
                return self._empty_result("Insufficient voiced frames for formant analysis")

            # Compute features
            features = {}
            anomalies = []

            # F1, F2, F3 statistics
            f1_values = formant_tracks[:, 0]
            f2_values = formant_tracks[:, 1]
            f3_values = formant_tracks[:, 2]

            features["f1_mean"] = float(np.mean(f1_values))
            features["f1_std"] = float(np.std(f1_values))
            features["f2_mean"] = float(np.mean(f2_values))
            features["f2_std"] = float(np.std(f2_values))
            features["f3_mean"] = float(np.mean(f3_values))
            features["f3_std"] = float(np.std(f3_values))

            # Vowel space area
            vsa = self._compute_vowel_space_area(f1_values, f2_values)
            features["vowel_space_area"] = float(vsa)

            # Formant bandwidth
            bw_mean = self._compute_bandwidth_mean(bandwidth_tracks)
            features["formant_bandwidth_mean"] = float(bw_mean)

            # Vocal tract length consistency
            vtl_consistency = self._compute_vtl_consistency(f1_values)
            features["vtl_consistency"] = float(vtl_consistency)

            # Formant transition rate
            transition_rate = self._compute_transition_rate(formant_tracks, sr)
            features["formant_transition_rate"] = float(transition_rate)

            # F1-F2 correlation
            f1_f2_corr = self._compute_f1_f2_correlation(f1_values, f2_values)
            features["f1_f2_correlation"] = float(f1_f2_corr)

            # Detect anomalies
            anomalies = self._detect_anomalies(features)

            # Compute composite score
            score, confidence = self._compute_score(features, duration_s, len(formant_tracks))

            return {
                "score": float(np.clip(score, 0.0, 1.0)),
                "confidence": float(np.clip(confidence, 0.0, 1.0)),
                "features": features,
                "anomalies": anomalies,
            }

        except Exception as e:
            log.error(f"Formant analysis failed: {e}", exc_info=True)
            return self._empty_result(f"Analysis error: {str(e)}")

    def _extract_formants(
        self, waveform: np.ndarray, sr: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract formant tracks (F1, F2, F3) and bandwidths via LPC analysis.

        Steps:
        1. Pre-emphasis filter to boost high frequencies
        2. Frame the signal with windowing
        3. LPC analysis per frame (order = 2 + sr/1000)
        4. Find roots of LPC polynomial
        5. Convert to frequencies and bandwidths
        6. Select F1, F2, F3 by expected frequency ranges

        Returns:
            (formant_tracks, bandwidth_tracks): each (n_voiced_frames, 3)
            or (None, None) on failure
        """
        # Pre-emphasis
        emphasized = np.append(waveform[0], waveform[1:] - PRE_EMPHASIS_COEFF * waveform[:-1])

        # Frame parameters
        frame_length = int(sr * self.frame_duration_ms / 1000)
        frame_shift = int(sr * self.frame_shift_ms / 1000)

        if frame_length > len(emphasized):
            frame_length = len(emphasized)

        # LPC order: 2 + sr/1000 (= 18 for 16kHz)
        lpc_order = 2 + int(sr / 1000)

        # Hamming window
        window = np.hamming(frame_length)

        formant_list = []
        bandwidth_list = []

        n_frames = max(1, (len(emphasized) - frame_length) // frame_shift + 1)

        for i in range(n_frames):
            start = i * frame_shift
            end = start + frame_length
            if end > len(emphasized):
                break

            frame = emphasized[start:end] * window

            # Skip low-energy frames (likely unvoiced or silence)
            frame_energy = np.sum(frame**2) / frame_length
            if frame_energy < 1e-8:
                continue

            # Compute autocorrelation
            autocorr = np.correlate(frame, frame, mode="full")
            autocorr = autocorr[len(frame) - 1:]  # keep non-negative lags

            if len(autocorr) < lpc_order + 1:
                continue

            # LPC via Levinson-Durbin
            lpc_coeffs = _levinson_durbin(autocorr[:lpc_order + 1], lpc_order)
            if lpc_coeffs is None:
                continue

            # Find roots of LPC polynomial
            try:
                roots = np.roots(lpc_coeffs)
            except Exception:
                continue

            # Keep only roots with positive imaginary part (each conjugate pair = one formant)
            roots = roots[np.imag(roots) > 0]

            if len(roots) == 0:
                continue

            # Convert roots to frequencies and bandwidths
            angles = np.angle(roots)
            freqs = angles * (sr / (2.0 * np.pi))
            bandwidths = -0.5 * (sr / (2.0 * np.pi)) * np.log(np.abs(roots) + 1e-12)

            # Filter: keep only roots with reasonable bandwidth (< 400 Hz)
            # and positive frequency
            valid_mask = (freqs > 50) & (bandwidths > 0) & (bandwidths < 400)
            freqs = freqs[valid_mask]
            bandwidths = bandwidths[valid_mask]

            if len(freqs) == 0:
                continue

            # Sort by frequency
            sort_idx = np.argsort(freqs)
            freqs = freqs[sort_idx]
            bandwidths = bandwidths[sort_idx]

            # Select F1, F2, F3 by range
            f1, bw1 = self._select_formant(freqs, bandwidths, *F1_RANGE)
            f2, bw2 = self._select_formant(freqs, bandwidths, *F2_RANGE)
            f3, bw3 = self._select_formant(freqs, bandwidths, *F3_RANGE)

            # All three formants must be found for a valid frame
            if f1 is not None and f2 is not None and f3 is not None:
                # Ensure ordering F1 < F2 < F3
                if f1 < f2 < f3:
                    formant_list.append([f1, f2, f3])
                    bandwidth_list.append([bw1, bw2, bw3])

        if len(formant_list) < 3:
            return None, None

        return np.array(formant_list), np.array(bandwidth_list)

    @staticmethod
    def _select_formant(
        freqs: np.ndarray, bandwidths: np.ndarray, low: float, high: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """Select the strongest formant candidate within a frequency range."""
        mask = (freqs >= low) & (freqs <= high)
        candidates = freqs[mask]
        candidate_bw = bandwidths[mask]

        if len(candidates) == 0:
            return None, None

        # Prefer the candidate with the narrowest bandwidth (strongest resonance)
        best_idx = np.argmin(candidate_bw)
        return float(candidates[best_idx]), float(candidate_bw[best_idx])

    def _compute_vowel_space_area(self, f1_values: np.ndarray, f2_values: np.ndarray) -> float:
        """
        Estimate vowel space area from the F1-F2 distribution.

        The vowel space is approximated as the area of the triangle formed by
        extreme vowels in the F1-F2 plane: [i] (high F2, low F1), [a] (high F1,
        mid F2), [u] (low F1, low F2). We estimate these from the range of
        observed formant values.

        A narrow vowel space indicates reduced articulatory precision, which is
        common in synthesized speech.

        Returns:
            Area in Hz^2 (higher = more natural articulatory range)
        """
        if len(f1_values) < 3 or len(f2_values) < 3:
            return 0.0

        # Estimate corner vowels from distribution
        # [i]-like: low F1, high F2 (use 10th percentile F1, 90th percentile F2)
        # [a]-like: high F1, mid F2 (use 90th percentile F1, median F2)
        # [u]-like: low F1, low F2 (use 10th percentile F1, 10th percentile F2)
        f1_p10 = np.percentile(f1_values, 10)
        f1_p90 = np.percentile(f1_values, 90)
        f2_p10 = np.percentile(f2_values, 10)
        f2_p50 = np.percentile(f2_values, 50)
        f2_p90 = np.percentile(f2_values, 90)

        # Triangle vertices in F1-F2 space
        i_vowel = (f1_p10, f2_p90)
        a_vowel = (f1_p90, f2_p50)
        u_vowel = (f1_p10, f2_p10)

        # Triangle area via cross product: 0.5 * |det([[x2-x1, x3-x1], [y2-y1, y3-y1]])|
        area = 0.5 * abs(
            (a_vowel[0] - i_vowel[0]) * (u_vowel[1] - i_vowel[1])
            - (u_vowel[0] - i_vowel[0]) * (a_vowel[1] - i_vowel[1])
        )

        return area

    def _compute_bandwidth_mean(self, bandwidth_tracks: np.ndarray) -> float:
        """Compute mean formant bandwidth across all frames and formants."""
        if bandwidth_tracks is None or len(bandwidth_tracks) == 0:
            return 0.0
        return float(np.mean(bandwidth_tracks))

    def _compute_vtl_consistency(self, f1_values: np.ndarray) -> float:
        """
        Estimate vocal tract length (VTL) consistency across frames.

        VTL = speed_of_sound / (4 * F1) for a simple quarter-wave tube model.
        Natural speakers maintain a consistent VTL across an utterance (it's an
        anatomical property). Synthetic speech from models that don't maintain
        consistent speaker characteristics will show VTL variation.

        Returns:
            Consistency score: 0.0 = highly inconsistent, 1.0 = perfectly consistent
        """
        if len(f1_values) < 3:
            return 0.5

        # Filter out extreme F1 values that would give unrealistic VTL
        valid_f1 = f1_values[(f1_values > 150) & (f1_values < 900)]
        if len(valid_f1) < 3:
            return 0.5

        # Estimate VTL for each frame (in cm)
        vtl_values = SPEED_OF_SOUND_CM / (4.0 * valid_f1)

        # Coefficient of variation (CV) — lower CV = more consistent
        vtl_mean = np.mean(vtl_values)
        if vtl_mean < 1e-6:
            return 0.5

        vtl_cv = np.std(vtl_values) / vtl_mean

        # Map CV to consistency score: CV=0 -> 1.0, CV=0.5 -> 0.0
        consistency = np.clip(1.0 - 2.0 * vtl_cv, 0.0, 1.0)

        return float(consistency)

    def _compute_transition_rate(self, formant_tracks: np.ndarray, sr: int) -> float:
        """
        Compute average formant transition rate (Hz/ms).

        Natural speech has characteristic transition rates at consonant-vowel
        boundaries. AI voices may have too-smooth (slow) or too-abrupt (fast)
        transitions.

        Returns:
            Average transition rate in Hz/ms across F1, F2, F3
        """
        n_frames = len(formant_tracks)
        if n_frames < 2:
            return 0.0

        # Time step between frames in milliseconds
        dt_ms = self.frame_shift_ms

        # Compute frame-to-frame differences for each formant
        total_rate = 0.0
        for f_idx in range(3):  # F1, F2, F3
            track = formant_tracks[:, f_idx]
            diffs = np.abs(np.diff(track))  # Hz per frame step
            rates = diffs / dt_ms  # Hz per ms
            total_rate += np.mean(rates)

        return float(total_rate / 3.0)  # average across F1, F2, F3

    def _compute_f1_f2_correlation(self, f1_values: np.ndarray, f2_values: np.ndarray) -> float:
        """
        Compute Pearson correlation between F1 and F2 tracks.

        Natural speech shows a moderate correlation between F1 and F2 due to
        articulatory coupling. Very high or very low correlation is suspicious.

        Returns:
            Pearson correlation coefficient (-1.0 to 1.0)
        """
        if len(f1_values) < 3 or len(f2_values) < 3:
            return 0.0

        # Check for constant values (would produce NaN correlation)
        if np.std(f1_values) < 1e-6 or np.std(f2_values) < 1e-6:
            return 0.0

        corr_matrix = np.corrcoef(f1_values, f2_values)
        corr = corr_matrix[0, 1]

        if not np.isfinite(corr):
            return 0.0

        return float(corr)

    def _detect_anomalies(self, features: Dict) -> List[str]:
        """Detect formant-based anomalies indicating possible synthesis."""
        anomalies = []

        vtl = features["vtl_consistency"]
        vsa = features["vowel_space_area"]
        trans_rate = features["formant_transition_rate"]
        f1_f2_corr = features["f1_f2_correlation"]
        bw_mean = features["formant_bandwidth_mean"]
        f1_std = features["f1_std"]
        f2_std = features["f2_std"]

        if vtl < THRESHOLDS["vtl_consistency_low"]:
            anomalies.append(
                f"Inconsistent vocal tract length ({vtl:.2f}) — possible speaker synthesis artifacts"
            )

        if vsa < THRESHOLDS["vowel_space_small"]:
            anomalies.append(
                f"Narrow vowel space area ({vsa:.0f} Hz^2) — reduced articulatory range"
            )

        if trans_rate < THRESHOLDS["formant_transition_low"]:
            anomalies.append(
                f"Very slow formant transitions ({trans_rate:.2f} Hz/ms) — unnaturally smooth speech"
            )
        elif trans_rate > THRESHOLDS["formant_transition_high"]:
            anomalies.append(
                f"Very fast formant transitions ({trans_rate:.1f} Hz/ms) — possible splicing artifacts"
            )

        if f1_f2_corr > THRESHOLDS["f1_f2_correlation_high"]:
            anomalies.append(
                f"High F1-F2 correlation ({f1_f2_corr:.2f}) — possible artificial formant coupling"
            )
        elif f1_f2_corr < THRESHOLDS["f1_f2_correlation_low"]:
            anomalies.append(
                f"Unusual F1-F2 anti-correlation ({f1_f2_corr:.2f}) — abnormal articulatory pattern"
            )

        if bw_mean < THRESHOLDS["bandwidth_narrow"]:
            anomalies.append(
                f"Very narrow formant bandwidths ({bw_mean:.1f} Hz) — unnaturally sharp resonances"
            )

        if f1_std < THRESHOLDS["f1_std_low"]:
            anomalies.append(
                f"Low F1 variation ({f1_std:.1f} Hz) — unnaturally uniform first formant"
            )

        if f2_std < THRESHOLDS["f2_std_low"]:
            anomalies.append(
                f"Low F2 variation ({f2_std:.1f} Hz) — unnaturally uniform second formant"
            )

        return anomalies

    def _compute_score(self, features: Dict, duration_s: float, n_voiced_frames: int):
        """
        Compute composite spoof score from formant features.

        Score: 0.0 = likely bonafide, 1.0 = likely spoof
        Confidence depends on duration, number of voiced frames, and score extremity.
        """
        sub_scores = []
        weights = []

        # VTL consistency: lower = more suspicious
        vtl = features["vtl_consistency"]
        vtl_score = np.clip(1.0 - vtl, 0.0, 1.0)
        sub_scores.append(vtl_score)
        weights.append(2.0)

        # Vowel space area: smaller = more suspicious
        vsa = features["vowel_space_area"]
        vsa_threshold = THRESHOLDS["vowel_space_small"]
        if vsa < vsa_threshold:
            vsa_score = np.clip(1.0 - vsa / vsa_threshold, 0.0, 1.0)
        else:
            vsa_score = 0.0
        sub_scores.append(vsa_score)
        weights.append(1.5)

        # Formant transition rate: extremes are suspicious
        trans_rate = features["formant_transition_rate"]
        if trans_rate < THRESHOLDS["formant_transition_low"]:
            trans_score = np.clip(
                1.0 - trans_rate / THRESHOLDS["formant_transition_low"], 0.0, 1.0
            )
        elif trans_rate > THRESHOLDS["formant_transition_high"]:
            trans_score = np.clip(
                (trans_rate - THRESHOLDS["formant_transition_high"]) / 50.0, 0.0, 1.0
            )
        else:
            trans_score = 0.0
        sub_scores.append(trans_score)
        weights.append(1.5)

        # F1-F2 correlation: high or very negative is suspicious
        f1_f2_corr = features["f1_f2_correlation"]
        if f1_f2_corr > THRESHOLDS["f1_f2_correlation_high"]:
            corr_score = np.clip(
                (f1_f2_corr - THRESHOLDS["f1_f2_correlation_high"]) / 0.15, 0.0, 1.0
            )
        elif f1_f2_corr < THRESHOLDS["f1_f2_correlation_low"]:
            corr_score = np.clip(
                (THRESHOLDS["f1_f2_correlation_low"] - f1_f2_corr) / 0.3, 0.0, 1.0
            )
        else:
            corr_score = 0.0
        sub_scores.append(corr_score)
        weights.append(1.0)

        # Bandwidth: narrow bandwidths are suspicious
        bw_mean = features["formant_bandwidth_mean"]
        if bw_mean < THRESHOLDS["bandwidth_narrow"]:
            bw_score = np.clip(1.0 - bw_mean / THRESHOLDS["bandwidth_narrow"], 0.0, 1.0)
        else:
            bw_score = 0.0
        sub_scores.append(bw_score)
        weights.append(0.8)

        # F1/F2 std: low variation is suspicious
        f1_std = features["f1_std"]
        f2_std = features["f2_std"]
        if f1_std < THRESHOLDS["f1_std_low"]:
            f1_std_score = np.clip(1.0 - f1_std / THRESHOLDS["f1_std_low"], 0.0, 1.0)
        else:
            f1_std_score = 0.0
        if f2_std < THRESHOLDS["f2_std_low"]:
            f2_std_score = np.clip(1.0 - f2_std / THRESHOLDS["f2_std_low"], 0.0, 1.0)
        else:
            f2_std_score = 0.0
        std_score = (f1_std_score + f2_std_score) / 2.0
        sub_scores.append(std_score)
        weights.append(1.2)

        # Weighted average
        weights = np.array(weights)
        sub_scores = np.array(sub_scores)
        score = np.sum(sub_scores * weights) / np.sum(weights)

        # Confidence based on duration, number of voiced frames
        duration_conf = np.clip(duration_s / 3.0, 0.3, 1.0)
        frame_conf = np.clip(n_voiced_frames / 50.0, 0.3, 1.0)
        base_confidence = min(duration_conf, frame_conf) * 0.75

        # Boost confidence if score is clearly anomalous or clearly normal
        score_extremity = abs(score - 0.5) * 2.0
        confidence = base_confidence + score_extremity * 0.25

        return float(score), float(np.clip(confidence, 0.0, 1.0))

    @staticmethod
    def _empty_result(reason: str) -> Dict:
        """Return a neutral result when analysis cannot be performed."""
        return {
            "score": 0.5,
            "confidence": 0.0,
            "features": {
                "f1_mean": 0.0,
                "f1_std": 0.0,
                "f2_mean": 0.0,
                "f2_std": 0.0,
                "f3_mean": 0.0,
                "f3_std": 0.0,
                "vowel_space_area": 0.0,
                "formant_bandwidth_mean": 0.0,
                "vtl_consistency": 0.0,
                "formant_transition_rate": 0.0,
                "f1_f2_correlation": 0.0,
            },
            "anomalies": [f"Analysis skipped: {reason}"],
        }
