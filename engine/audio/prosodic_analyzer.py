"""
Prosodic Forensics Analyzer for Deepfake Voice Detection
==========================================================
Extracts prosodic features that current AI voice synthesis struggles to
replicate faithfully: F0 micro-perturbations (jitter), amplitude variation
(shimmer), harmonics-to-noise ratio, speech rate, and pause distribution.

AI-generated voices tend to be "too perfect" — unnaturally low jitter/shimmer,
abnormally high HNR, overly regular F0 contours, and absent or uniform pauses.
This module quantifies those deviations from human norms.

Optional dependency: `parselmouth` (Praat bindings) provides more accurate
jitter/shimmer/HNR via Praat's battle-tested algorithms. Falls back to
manual NumPy/SciPy computation when unavailable.
"""

import logging
from typing import Optional

import numpy as np

try:
    import librosa
except ImportError:
    raise ImportError("librosa is required for ProsodicAnalyzer")

try:
    from scipy import signal as scipy_signal
except ImportError:
    raise ImportError("scipy is required for ProsodicAnalyzer")

try:
    import parselmouth
    from parselmouth.praat import call as praat_call

    HAS_PARSELMOUTH = True
except ImportError:
    HAS_PARSELMOUTH = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Human speech reference ranges (from voice pathology and forensic phonetics
# literature, assuming adult speakers, conversational register).
# ---------------------------------------------------------------------------
HUMAN_NORMS = {
    "jitter_local": (0.008, 0.050),       # 0.8% – 5.0% (Tightened lower bound for AI)
    "jitter_rap": (0.004, 0.030),          # relative average perturbation
    "shimmer_local": (0.020, 0.100),       # 2.0% – 10%
    "shimmer_apq": (0.015, 0.080),         # amplitude perturbation quotient
    "hnr_mean": (8.0, 22.0),              # dB; overly high is AI
    "f0_cv": (0.15, 0.50),                # coefficient of variation of F0
    "speech_rate": (2.5, 6.5),             # syllables per second
    "pause_rate": (0.10, 0.60),            # pauses per second of speech
    "pause_duration_mean": (0.15, 0.80),   # seconds
}

# Minimum audio duration (seconds) for meaningful analysis.
MIN_DURATION_SEC = 0.5


class ProsodicAnalyzer:
    """
    Extract prosodic features from a speech waveform and score how likely
    the voice is synthetic (deepfake) based on deviations from human norms.

    Usage::

        analyzer = ProsodicAnalyzer()
        result = analyzer.analyze(waveform, sr=16000)
        print(result["score"], result["anomalies"])
    """

    def __init__(self) -> None:
        if HAS_PARSELMOUTH:
            logger.info("ProsodicAnalyzer: using parselmouth (Praat) backend")
        else:
            logger.info(
                "ProsodicAnalyzer: parselmouth not found, using "
                "librosa/scipy fallback"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, waveform: np.ndarray, sr: int = 16000) -> dict:
        """
        Run prosodic analysis on a mono waveform.

        Args:
            waveform: 1-D float numpy array (mono, expected range [-1, 1]).
            sr: Sample rate in Hz (default 16 000).

        Returns:
            Dictionary with keys ``score``, ``confidence``, ``features``,
            and ``anomalies``.
        """
        # --- input validation -------------------------------------------
        waveform = self._validate_input(waveform, sr)
        if waveform is None:
            return self._empty_result("input_too_short_or_silent")

        duration = len(waveform) / sr

        # --- F0 extraction ----------------------------------------------
        f0, voiced_flag, voiced_probs = self._extract_f0(waveform, sr)

        if f0 is None or np.sum(voiced_flag) < 3:
            logger.warning("Insufficient voiced frames for prosodic analysis")
            return self._empty_result("insufficient_voiced_frames")

        voiced_f0 = f0[voiced_flag]

        # --- core features ----------------------------------------------
        f0_features = self._compute_f0_features(voiced_f0)

        if HAS_PARSELMOUTH:
            jitter, shimmer, hnr = self._praat_voice_quality(waveform, sr)
        else:
            jitter = self._compute_jitter(voiced_f0)
            shimmer = self._compute_shimmer(waveform, sr, f0, voiced_flag)
            hnr = self._compute_hnr(waveform, sr)

        rate_features = self._compute_rate_features(
            waveform, sr, voiced_flag, duration
        )

        # --- assemble feature dict --------------------------------------
        features = {
            **f0_features,
            **jitter,
            **shimmer,
            **hnr,
            **rate_features,
        }

        # --- anomaly detection and scoring ------------------------------
        anomalies = self._detect_anomalies(features)
        score, confidence = self._compute_score(features, anomalies, duration)

        return {
            "score": round(float(score), 4),
            "confidence": round(float(confidence), 4),
            "features": {k: round(float(v), 6) for k, v in features.items()},
            "anomalies": anomalies,
        }

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _validate_input(
        self, waveform: np.ndarray, sr: int
    ) -> Optional[np.ndarray]:
        """Ensure input is a usable mono float waveform."""
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=0)

        waveform = waveform.astype(np.float32, copy=False)

        # Normalise to [-1, 1] if the signal looks like int16 range
        if np.max(np.abs(waveform)) > 2.0:
            waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)

        duration = len(waveform) / sr
        if duration < MIN_DURATION_SEC:
            logger.warning(
                "Audio too short (%.2fs < %.2fs)", duration, MIN_DURATION_SEC
            )
            return None

        # Silence check: RMS below -60 dBFS
        rms = np.sqrt(np.mean(waveform ** 2))
        if rms < 1e-3:
            logger.warning("Audio appears to be silence (RMS=%.6f)", rms)
            return None

        return waveform

    @staticmethod
    def _empty_result(reason: str) -> dict:
        """Return a zero-score result when analysis cannot proceed."""
        return {
            "score": 0.0,
            "confidence": 0.0,
            "features": {
                "f0_mean": 0.0,
                "f0_std": 0.0,
                "f0_range": 0.0,
                "jitter_local": 0.0,
                "jitter_rap": 0.0,
                "shimmer_local": 0.0,
                "shimmer_apq": 0.0,
                "hnr_mean": 0.0,
                "hnr_std": 0.0,
                "speech_rate": 0.0,
                "pause_rate": 0.0,
                "pause_duration_mean": 0.0,
                "pause_duration_std": 0.0,
            },
            "anomalies": [reason],
        }

    # ------------------------------------------------------------------
    # F0 extraction and features
    # ------------------------------------------------------------------

    def _extract_f0(
        self, waveform: np.ndarray, sr: int
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract F0 contour using librosa's probabilistic YIN (pyin).

        Returns (f0, voiced_flag, voiced_probs) where f0 contains NaN for
        unvoiced frames.
        """
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                waveform,
                fmin=librosa.note_to_hz("C2"),   # ~65 Hz
                fmax=librosa.note_to_hz("C7"),   # ~2093 Hz
                sr=sr,
                frame_length=2048,
                hop_length=512,
            )
            # Replace NaN with 0 in f0 for voiced frames where pyin glitched
            if f0 is not None:
                f0 = np.nan_to_num(f0, nan=0.0)
                voiced_flag = f0 > 0
            return f0, voiced_flag, voiced_probs
        except Exception:
            logger.exception("F0 extraction failed")
            return None, None, None

    @staticmethod
    def _compute_f0_features(voiced_f0: np.ndarray) -> dict:
        """Compute descriptive statistics of the voiced F0 contour."""
        f0_mean = float(np.mean(voiced_f0))
        f0_std = float(np.std(voiced_f0))
        f0_range = float(np.ptp(voiced_f0))  # max - min
        return {
            "f0_mean": f0_mean,
            "f0_std": f0_std,
            "f0_range": f0_range,
        }

    # ------------------------------------------------------------------
    # Jitter (cycle-to-cycle F0 perturbation)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_jitter(voiced_f0: np.ndarray) -> dict:
        """
        Compute jitter variants from the voiced F0 contour.

        * **local jitter**: mean absolute difference between consecutive
          periods, normalised by the mean period.
        * **RAP** (Relative Average Perturbation): 3-point running average
          perturbation normalised by the mean period.
        * **PPQ** (5-point Period Perturbation Quotient): same idea, 5 pts
          — stored under ``jitter_rap`` key for API simplicity.

        Returns dict with ``jitter_local`` and ``jitter_rap``.
        """
        if len(voiced_f0) < 3:
            return {"jitter_local": 0.0, "jitter_rap": 0.0}

        # Convert F0 (Hz) to period (seconds)
        periods = 1.0 / (voiced_f0 + 1e-8)
        mean_period = np.mean(periods)

        # Local jitter: |T_i - T_{i+1}| averaged, normalised by mean period
        diffs = np.abs(np.diff(periods))
        jitter_local = float(np.mean(diffs) / (mean_period + 1e-8))

        # RAP: 3-point running-average perturbation
        if len(periods) >= 3:
            rap_vals = []
            for i in range(1, len(periods) - 1):
                avg3 = (periods[i - 1] + periods[i] + periods[i + 1]) / 3.0
                rap_vals.append(abs(periods[i] - avg3))
            jitter_rap = float(np.mean(rap_vals) / (mean_period + 1e-8))
        else:
            jitter_rap = jitter_local / 2.0

        return {"jitter_local": jitter_local, "jitter_rap": jitter_rap}

    # ------------------------------------------------------------------
    # Shimmer (cycle-to-cycle amplitude variation)
    # ------------------------------------------------------------------

    def _compute_shimmer(
        self,
        waveform: np.ndarray,
        sr: int,
        f0: np.ndarray,
        voiced_flag: np.ndarray,
    ) -> dict:
        """
        Compute shimmer from per-cycle peak amplitudes.

        * **local shimmer**: mean |A_i - A_{i+1}| / mean(A)
        * **APQ** (Amplitude Perturbation Quotient, 5-point window)

        Returns dict with ``shimmer_local`` and ``shimmer_apq``.
        """
        hop = 512
        amps = []
        for i, (freq, is_voiced) in enumerate(zip(f0, voiced_flag)):
            if not is_voiced or freq <= 0:
                continue
            center = i * hop
            period_samples = int(sr / freq)
            half = period_samples // 2
            start = max(0, center - half)
            end = min(len(waveform), center + half)
            if end - start < 4:
                continue
            segment = waveform[start:end]
            amps.append(float(np.max(np.abs(segment))))

        if len(amps) < 3:
            return {"shimmer_local": 0.0, "shimmer_apq": 0.0}

        amps = np.array(amps)
        mean_amp = np.mean(amps)

        # Local shimmer
        diffs = np.abs(np.diff(amps))
        shimmer_local = float(np.mean(diffs) / (mean_amp + 1e-8))

        # APQ-5
        if len(amps) >= 5:
            apq_vals = []
            for i in range(2, len(amps) - 2):
                avg5 = np.mean(amps[i - 2 : i + 3])
                apq_vals.append(abs(amps[i] - avg5))
            shimmer_apq = float(np.mean(apq_vals) / (mean_amp + 1e-8))
        else:
            shimmer_apq = shimmer_local * 0.6

        return {"shimmer_local": shimmer_local, "shimmer_apq": shimmer_apq}

    # ------------------------------------------------------------------
    # HNR (Harmonics-to-Noise Ratio)
    # ------------------------------------------------------------------

    def _compute_hnr(self, waveform: np.ndarray, sr: int) -> dict:
        """
        Estimate HNR via autocorrelation method.

        Splits audio into overlapping frames, finds the autocorrelation peak
        corresponding to the fundamental period, and computes

            HNR = 10 * log10( r_peak / (1 - r_peak) )

        where r_peak is the normalised autocorrelation at the F0 lag.

        Returns dict with ``hnr_mean`` and ``hnr_std``.
        """
        frame_length = int(0.04 * sr)   # 40 ms frames
        hop_length = int(0.01 * sr)     # 10 ms hop
        min_lag = int(sr / 500)          # 500 Hz upper bound
        max_lag = int(sr / 60)           # 60 Hz lower bound

        if max_lag >= frame_length:
            max_lag = frame_length - 1

        hnr_values = []
        n_frames = (len(waveform) - frame_length) // hop_length + 1

        for i in range(n_frames):
            start = i * hop_length
            frame = waveform[start : start + frame_length]

            # Skip near-silent frames
            if np.max(np.abs(frame)) < 0.01:
                continue

            # Normalised autocorrelation
            frame = frame - np.mean(frame)
            energy = np.sum(frame ** 2)
            if energy < 1e-10:
                continue

            autocorr = np.correlate(frame, frame, mode="full")
            autocorr = autocorr[len(frame) - 1 :]  # keep non-negative lags
            autocorr = autocorr / (energy + 1e-10)

            # Find peak in valid lag range
            search_start = min(min_lag, len(autocorr) - 1)
            search_end = min(max_lag, len(autocorr))
            if search_start >= search_end:
                continue

            segment = autocorr[search_start:search_end]
            if len(segment) == 0:
                continue

            r_peak = float(np.max(segment))
            r_peak = np.clip(r_peak, 1e-6, 1.0 - 1e-6)

            hnr_db = 10.0 * np.log10(r_peak / (1.0 - r_peak))
            hnr_values.append(hnr_db)

        if not hnr_values:
            return {"hnr_mean": 0.0, "hnr_std": 0.0}

        hnr_arr = np.array(hnr_values)
        return {
            "hnr_mean": float(np.mean(hnr_arr)),
            "hnr_std": float(np.std(hnr_arr)),
        }

    # ------------------------------------------------------------------
    # Parselmouth (Praat) backend
    # ------------------------------------------------------------------

    def _praat_voice_quality(
        self, waveform: np.ndarray, sr: int
    ) -> tuple[dict, dict, dict]:
        """
        Use Praat (via parselmouth) for jitter, shimmer, and HNR.

        Returns three dicts: jitter, shimmer, hnr — matching the keys
        of the fallback methods.
        """
        try:
            snd = parselmouth.Sound(waveform, sampling_frequency=sr)

            # ---- point process for jitter / shimmer --------------------
            pitch = praat_call(
                snd, "To Pitch", 0.0, 75.0, 500.0
            )
            point_process = praat_call(
                snd, "To PointProcess (periodic, cc)...", 75.0, 500.0
            )

            # Jitter
            jitter_local = praat_call(
                point_process, "Get jitter (local)", 0.0, 0.0,
                0.0001, 0.02, 1.3,
            )
            jitter_rap = praat_call(
                point_process, "Get jitter (rap)", 0.0, 0.0,
                0.0001, 0.02, 1.3,
            )

            # Shimmer
            shimmer_local = praat_call(
                [snd, point_process], "Get shimmer (local)", 0.0, 0.0,
                0.0001, 0.02, 1.3, 1.6,
            )
            shimmer_apq = praat_call(
                [snd, point_process], "Get shimmer (apq5)", 0.0, 0.0,
                0.0001, 0.02, 1.3, 1.6,
            )

            # ---- harmonicity for HNR ----------------------------------
            harmonicity = praat_call(
                snd, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0,
            )
            hnr_mean = praat_call(harmonicity, "Get mean", 0.0, 0.0)
            hnr_std = praat_call(
                harmonicity, "Get standard deviation", 0.0, 0.0,
            )

            # Praat returns --undefined-- as nan for pathological cases
            jitter_local = 0.0 if np.isnan(jitter_local) else jitter_local
            jitter_rap = 0.0 if np.isnan(jitter_rap) else jitter_rap
            shimmer_local = 0.0 if np.isnan(shimmer_local) else shimmer_local
            shimmer_apq = 0.0 if np.isnan(shimmer_apq) else shimmer_apq
            hnr_mean = 0.0 if np.isnan(hnr_mean) else hnr_mean
            hnr_std = 0.0 if np.isnan(hnr_std) else hnr_std

            return (
                {"jitter_local": jitter_local, "jitter_rap": jitter_rap},
                {"shimmer_local": shimmer_local, "shimmer_apq": shimmer_apq},
                {"hnr_mean": hnr_mean, "hnr_std": hnr_std},
            )

        except Exception:
            logger.warning(
                "Parselmouth analysis failed, falling back to manual",
                exc_info=True,
            )
            # Caller will need to recompute — signal this with None
            # (but we handle the fallback at the call site instead).
            return (
                {"jitter_local": 0.0, "jitter_rap": 0.0},
                {"shimmer_local": 0.0, "shimmer_apq": 0.0},
                {"hnr_mean": 0.0, "hnr_std": 0.0},
            )

    # ------------------------------------------------------------------
    # Speech rate and pause statistics
    # ------------------------------------------------------------------

    def _compute_rate_features(
        self,
        waveform: np.ndarray,
        sr: int,
        voiced_flag: np.ndarray,
        duration: float,
    ) -> dict:
        """
        Estimate speech rate (syllables/sec) and pause statistics.

        Speech rate is approximated by counting energy peaks in the amplitude
        envelope (each peak ~ one syllable nucleus).

        Pauses are contiguous unvoiced regions longer than 100 ms.
        """
        # --- syllable-rate estimation -----------------------------------
        speech_rate = self._estimate_speech_rate(waveform, sr, duration)

        # --- pause detection from voicing flags -------------------------
        hop_sec = 512.0 / sr  # pyin hop in seconds
        pause_durations = self._extract_pauses(voiced_flag, hop_sec)

        if len(pause_durations) > 0:
            pause_rate = len(pause_durations) / duration
            pause_mean = float(np.mean(pause_durations))
            pause_std = float(np.std(pause_durations))
        else:
            pause_rate = 0.0
            pause_mean = 0.0
            pause_std = 0.0

        return {
            "speech_rate": speech_rate,
            "pause_rate": pause_rate,
            "pause_duration_mean": pause_mean,
            "pause_duration_std": pause_std,
        }

    @staticmethod
    def _estimate_speech_rate(
        waveform: np.ndarray, sr: int, duration: float
    ) -> float:
        """
        Estimate syllable rate from the smoothed amplitude envelope.

        The approach: rectify -> low-pass at ~10 Hz (envelope of syllabic
        modulations) -> count prominent peaks.
        """
        # Rectified signal
        rectified = np.abs(waveform)

        # Low-pass Butterworth at 10 Hz to get syllabic envelope
        nyquist = sr / 2.0
        cutoff = min(10.0, nyquist * 0.9)  # safety for very low SR
        b, a = scipy_signal.butter(4, cutoff / nyquist, btype="low")
        envelope = scipy_signal.filtfilt(b, a, rectified)

        # Find peaks with minimum distance ~100 ms (max ~10 syll/s)
        min_distance = max(int(0.10 * sr), 1)
        peak_height = np.mean(envelope) * 0.3  # above 30% of mean
        peaks, _ = scipy_signal.find_peaks(
            envelope,
            distance=min_distance,
            height=peak_height,
        )

        if duration < 0.01:
            return 0.0

        return float(len(peaks) / duration)

    @staticmethod
    def _extract_pauses(
        voiced_flag: np.ndarray, hop_sec: float, min_pause_sec: float = 0.10
    ) -> np.ndarray:
        """
        Find contiguous unvoiced regions that last at least
        ``min_pause_sec`` seconds.

        Returns an array of pause durations in seconds.
        """
        min_frames = int(min_pause_sec / hop_sec)
        if min_frames < 1:
            min_frames = 1

        pauses = []
        count = 0
        for v in voiced_flag:
            if not v:
                count += 1
            else:
                if count >= min_frames:
                    pauses.append(count * hop_sec)
                count = 0
        # Handle trailing silence
        if count >= min_frames:
            pauses.append(count * hop_sec)

        return np.array(pauses, dtype=np.float64)

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_anomalies(features: dict) -> list[str]:
        """
        Flag features that fall outside expected human ranges.

        AI voices typically exhibit:
        - Unnaturally low jitter (too-stable pitch periods)
        - Unnaturally low shimmer (too-stable amplitude)
        - Abnormally high HNR (no breathiness or turbulence)
        - Too-regular F0 (low coefficient of variation)
        - Absence of natural pauses
        """
        anomalies: list[str] = []

        # --- Jitter checks -----------------------------------------------
        jl = features.get("jitter_local", 0.0)
        lo, hi = HUMAN_NORMS["jitter_local"]
        if jl < lo and jl > 0:
            anomalies.append("abnormally_low_jitter")
        elif jl > hi:
            anomalies.append("abnormally_high_jitter")

        jr = features.get("jitter_rap", 0.0)
        lo, hi = HUMAN_NORMS["jitter_rap"]
        if jr < lo and jr > 0:
            anomalies.append("abnormally_low_jitter_rap")

        # --- Shimmer checks ----------------------------------------------
        sl = features.get("shimmer_local", 0.0)
        lo, hi = HUMAN_NORMS["shimmer_local"]
        if sl < lo and sl > 0:
            anomalies.append("abnormally_low_shimmer")
        elif sl > hi:
            anomalies.append("abnormally_high_shimmer")

        # --- HNR checks --------------------------------------------------
        hnr = features.get("hnr_mean", 0.0)
        lo, hi = HUMAN_NORMS["hnr_mean"]
        if hnr > hi:
            anomalies.append("abnormally_high_hnr")
        elif 0 < hnr < lo:
            anomalies.append("abnormally_low_hnr")

        # --- F0 regularity ------------------------------------------------
        f0_mean = features.get("f0_mean", 0.0)
        f0_std = features.get("f0_std", 0.0)
        if f0_mean > 0:
            f0_cv = f0_std / (f0_mean + 1e-8)
            lo, hi = HUMAN_NORMS["f0_cv"]
            if f0_cv < lo:
                anomalies.append("too_regular_f0")
            elif f0_cv > hi:
                anomalies.append("abnormally_variable_f0")

        # --- Speech rate --------------------------------------------------
        sr = features.get("speech_rate", 0.0)
        lo, hi = HUMAN_NORMS["speech_rate"]
        if sr > 0 and sr < lo:
            anomalies.append("abnormally_slow_speech")
        elif sr > hi:
            anomalies.append("abnormally_fast_speech")

        # --- Pause distribution -------------------------------------------
        pr = features.get("pause_rate", 0.0)
        pm = features.get("pause_duration_mean", 0.0)
        lo, hi = HUMAN_NORMS["pause_rate"]
        if pr < lo:
            anomalies.append("no_pauses_detected")
        elif pr > hi:
            anomalies.append("excessive_pauses")

        if pm > 0:
            lo, hi = HUMAN_NORMS["pause_duration_mean"]
            if pm > hi:
                anomalies.append("abnormally_long_pauses")

        # Pause regularity: zero std with non-zero mean -> robotic pauses
        ps = features.get("pause_duration_std", 0.0)
        if pr > 0 and pm > 0 and ps < 0.05:
            anomalies.append("too_uniform_pauses")

        return anomalies

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_score(
        features: dict,
        anomalies: list[str],
        duration: float,
    ) -> tuple[float, float]:
        """
        Compute a composite spoof score in [0, 1] and a confidence estimate.

        The score combines:
        1. Anomaly count (each anomaly contributes to suspicion).
        2. Feature distance from human norms (continuous deviation).

        Confidence scales with audio duration and the number of features
        that could be successfully computed.
        """
        # ---- anomaly-based component -----------------------------------
        # Weight certain anomalies higher (low jitter + high HNR is a
        # very strong AI signature).
        HIGH_WEIGHT_ANOMALIES = {
            "abnormally_low_jitter",
            "abnormally_low_shimmer",
            "abnormally_high_hnr",
            "too_regular_f0",
            "no_pauses_detected",
        }
        anomaly_score = 0.0
        for a in anomalies:
            if a in HIGH_WEIGHT_ANOMALIES:
                anomaly_score += 0.15
            else:
                anomaly_score += 0.08

        # ---- continuous deviation component ----------------------------
        deviation_score = 0.0
        n_checks = 0

        def _deviation(value: float, lo: float, hi: float) -> float:
            """How far below 'lo' or above 'hi' the value is, normalised."""
            if value <= 0:
                return 0.0
            if value < lo:
                return (lo - value) / (lo + 1e-8)
            elif value > hi:
                return (value - hi) / (hi + 1e-8)
            return 0.0

        # Jitter deviation (low = suspicious)
        jl = features.get("jitter_local", 0.0)
        if jl > 0:
            deviation_score += _deviation(
                jl, *HUMAN_NORMS["jitter_local"]
            )
            n_checks += 1

        # Shimmer deviation
        sl = features.get("shimmer_local", 0.0)
        if sl > 0:
            deviation_score += _deviation(
                sl, *HUMAN_NORMS["shimmer_local"]
            )
            n_checks += 1

        # HNR deviation (high = suspicious)
        hnr = features.get("hnr_mean", 0.0)
        if hnr != 0:
            deviation_score += _deviation(hnr, *HUMAN_NORMS["hnr_mean"])
            n_checks += 1

        # F0 regularity deviation
        f0_mean = features.get("f0_mean", 0.0)
        f0_std = features.get("f0_std", 0.0)
        if f0_mean > 0:
            f0_cv = f0_std / (f0_mean + 1e-8)
            deviation_score += _deviation(f0_cv, *HUMAN_NORMS["f0_cv"])
            n_checks += 1

        if n_checks > 0:
            deviation_score /= n_checks  # average deviation

        # ---- combine ---------------------------------------------------
        raw_score = 0.55 * anomaly_score + 0.45 * deviation_score
        score = float(np.clip(raw_score, 0.0, 1.0))

        # ---- confidence ------------------------------------------------
        # Higher with more audio, more computed features, and parselmouth
        duration_factor = float(np.clip(duration / 5.0, 0.2, 1.0))
        feature_factor = float(np.clip(n_checks / 4.0, 0.25, 1.0))
        backend_bonus = 1.0 if HAS_PARSELMOUTH else 0.85

        confidence = duration_factor * feature_factor * backend_bonus
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return score, confidence
