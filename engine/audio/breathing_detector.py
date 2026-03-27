"""
Biological Signal Analyzer — Breathing Pattern Detection
==========================================================
Detects presence/absence and patterns of breathing in speech audio.

AI-generated voices frequently lack natural respiratory artifacts:
inhalation before phrases, exhalation during pauses, and the subtle
spectral footprint of airflow through the vocal tract. This module
exploits that gap by segmenting audio into speech/silence/breath,
then scoring how closely the breathing pattern matches a live speaker.

Key heuristics:
  - Humans breathe 12-20 times per minute during conversational speech
  - Breathing occurs at phrase boundaries, not mid-word
  - Inhalation precedes speech onset; exhalation fills pauses
  - Natural breath timing shows moderate variability (CV > 0.2)
  - Absence of any breathing in >10 s of speech is a strong spoof signal
"""

import logging
from typing import Optional

import librosa
import numpy as np
from scipy import signal as sp_signal
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_AUDIO_DURATION = 1.0          # seconds — below this we bail out
_FRAME_LENGTH = 512                # samples per energy frame @ 16 kHz ≈ 32 ms
_HOP_LENGTH = 256                  # hop between frames ≈ 16 ms
_BREATH_FREQ_LO = 100.0           # Hz — lower edge of breathing band
_BREATH_FREQ_HI = 2000.0          # Hz — upper edge
_MIN_BREATH_DURATION = 0.08       # seconds — shorter than this isn't a breath
_MAX_BREATH_DURATION = 1.5        # seconds — longer is a sustained pause, not breath
_MIN_PAUSE_FOR_BREATH = 0.05      # seconds — minimum silence gap to even check
_NORMAL_BREATH_RATE_LO = 12.0     # breaths / minute (resting lower bound)
_NORMAL_BREATH_RATE_HI = 20.0     # breaths / minute (conversational upper bound)
_SPEECH_GAP_SPOOF_THRESHOLD = 10.0  # seconds of speech with zero breaths → suspicious
_REGULARITY_SUSPICIOUS_CV = 0.1   # coefficient of variation below this → too regular


class BreathingDetector:
    """Analyze breathing patterns in speech audio for deepfake detection."""

    def __init__(self) -> None:
        logger.info("BreathingDetector initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, waveform: np.ndarray, sr: int = 16000) -> dict:
        """
        Analyze breathing in a mono waveform.

        Parameters
        ----------
        waveform : np.ndarray
            1-D float array (mono, expected range [-1, 1]).
        sr : int
            Sample rate in Hz (default 16 000).

        Returns
        -------
        dict  with keys score, confidence, features, anomalies,
              breath_timestamps.
        """
        # --- Ensure mono float32 ------------------------------------------
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)

        duration = len(waveform) / sr
        logger.debug("Analyzing %.2f s of audio (sr=%d)", duration, sr)

        # --- Edge case: too short -----------------------------------------
        if duration < _MIN_AUDIO_DURATION:
            logger.warning("Audio too short (%.2f s) for breathing analysis", duration)
            return self._empty_result(
                anomalies=["audio_too_short"],
                confidence=0.0,
            )

        # --- Step 1: Energy-based VAD -------------------------------------
        speech_mask, energy = self._compute_vad(waveform, sr)

        speech_fraction = speech_mask.mean()
        if speech_fraction < 0.05:
            logger.warning("Almost no speech detected (%.1f%%)", speech_fraction * 100)
            return self._empty_result(
                anomalies=["insufficient_speech"],
                confidence=0.1,
            )

        # --- Step 2 & 3: Identify silence segments, classify as breath ----
        silence_segments = self._find_silence_segments(speech_mask, sr)
        breath_segments = self._classify_segments(
            waveform, sr, silence_segments,
        )

        # --- Step 4: Analyze breathing pattern ----------------------------
        features = self._compute_features(
            breath_segments, speech_mask, waveform, sr, duration,
        )

        # --- Step 5: Score anomalies --------------------------------------
        anomalies, score = self._score(features, speech_mask, sr, duration)

        # Confidence depends on how much speech we have to judge
        confidence = self._estimate_confidence(duration, speech_fraction, features)

        breath_timestamps = [
            (round(s, 4), round(e, 4)) for s, e in breath_segments
        ]

        return {
            "score": round(float(np.clip(score, 0.0, 1.0)), 4),
            "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 4),
            "features": features,
            "anomalies": anomalies,
            "breath_timestamps": breath_timestamps,
        }

    # ------------------------------------------------------------------
    # Internal: VAD
    # ------------------------------------------------------------------

    def _compute_vad(
        self, waveform: np.ndarray, sr: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a per-frame boolean speech mask and RMS energy array."""
        # RMS energy per frame
        energy = librosa.feature.rms(
            y=waveform, frame_length=_FRAME_LENGTH, hop_length=_HOP_LENGTH
        )[0]

        # Adaptive threshold: use a percentile of the energy distribution.
        # Noise floor ≈ 20th percentile, speech threshold a bit above.
        if energy.max() < 1e-8:
            return np.zeros(len(energy), dtype=bool), energy

        noise_floor = np.percentile(energy, 20)
        speech_threshold = noise_floor + 0.35 * (np.percentile(energy, 80) - noise_floor)

        raw_mask = energy > speech_threshold

        # Smooth with a small median filter to remove isolated blips
        smoothed = sp_signal.medfilt(raw_mask.astype(np.float64), kernel_size=7) > 0.5

        return smoothed.astype(bool), energy

    # ------------------------------------------------------------------
    # Internal: Silence segment extraction
    # ------------------------------------------------------------------

    def _find_silence_segments(
        self, speech_mask: np.ndarray, sr: int
    ) -> list[tuple[float, float]]:
        """
        Find contiguous silence (non-speech) segments.

        Returns list of (start_sec, end_sec).
        """
        frame_dur = _HOP_LENGTH / sr
        segments: list[tuple[float, float]] = []

        in_silence = False
        start_frame = 0

        for i, is_speech in enumerate(speech_mask):
            if not is_speech and not in_silence:
                in_silence = True
                start_frame = i
            elif is_speech and in_silence:
                in_silence = False
                seg_start = start_frame * frame_dur
                seg_end = i * frame_dur
                if (seg_end - seg_start) >= _MIN_PAUSE_FOR_BREATH:
                    segments.append((seg_start, seg_end))

        # Handle trailing silence
        if in_silence:
            seg_start = start_frame * frame_dur
            seg_end = len(speech_mask) * frame_dur
            if (seg_end - seg_start) >= _MIN_PAUSE_FOR_BREATH:
                segments.append((seg_start, seg_end))

        return segments

    # ------------------------------------------------------------------
    # Internal: Classify silence as breath / pause / true-silence
    # ------------------------------------------------------------------

    def _classify_segments(
        self,
        waveform: np.ndarray,
        sr: int,
        silence_segments: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """
        For each silence segment, decide if it contains a breath.

        Breathing has:
          - Low but non-negligible energy
          - Broad spectral spread (noise-like)
          - Energy concentrated in 100-2000 Hz
        Pure silence has energy ≈ 0 across all bands.
        """
        breaths: list[tuple[float, float]] = []

        for seg_start, seg_end in silence_segments:
            seg_dur = seg_end - seg_start
            if seg_dur < _MIN_BREATH_DURATION or seg_dur > _MAX_BREATH_DURATION:
                continue

            start_sample = int(seg_start * sr)
            end_sample = int(seg_end * sr)
            segment = waveform[start_sample:end_sample]

            if len(segment) < 256:
                continue

            if self._is_breath_like(segment, sr):
                breaths.append((seg_start, seg_end))

        return breaths

    def _is_breath_like(self, segment: np.ndarray, sr: int) -> bool:
        """Check if a short audio segment has breath-like spectral properties."""
        # Energy check — breathing is quiet but not silent
        rms = np.sqrt(np.mean(segment ** 2))
        if rms < 1e-6:
            return False  # true silence

        # Compute power spectrum
        n_fft = min(1024, len(segment))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        spectrum = np.abs(np.fft.rfft(segment[:n_fft])) ** 2

        if spectrum.sum() < 1e-12:
            return False

        # Proportion of energy in breath band (100-2000 Hz)
        breath_mask = (freqs >= _BREATH_FREQ_LO) & (freqs <= _BREATH_FREQ_HI)
        breath_energy = spectrum[breath_mask].sum()
        total_energy = spectrum.sum()
        breath_ratio = breath_energy / total_energy

        # Spectral flatness — breathing is noise-like (flat spectrum)
        spectral_flatness = self._spectral_flatness(spectrum[breath_mask])

        # Breathing: energy in band ≳ 40 %, fairly flat spectrum
        is_breath = breath_ratio > 0.35 and spectral_flatness > 0.01
        return is_breath

    @staticmethod
    def _spectral_flatness(power_spectrum: np.ndarray) -> float:
        """Geometric mean / arithmetic mean of power spectrum (Wiener entropy)."""
        ps = power_spectrum[power_spectrum > 0]
        if len(ps) < 2:
            return 0.0
        log_mean = np.mean(np.log(ps + 1e-20))
        arith_mean = np.mean(ps)
        if arith_mean < 1e-20:
            return 0.0
        return float(np.exp(log_mean) / arith_mean)

    # ------------------------------------------------------------------
    # Internal: Feature computation
    # ------------------------------------------------------------------

    def _compute_features(
        self,
        breath_segments: list[tuple[float, float]],
        speech_mask: np.ndarray,
        waveform: np.ndarray,
        sr: int,
        duration: float,
    ) -> dict:
        """Compute the full feature dictionary."""
        breath_count = len(breath_segments)

        # Durations
        if breath_count > 0:
            durations = np.array([e - s for s, e in breath_segments])
            mean_breath_duration = float(durations.mean())
            total_breath_time = float(durations.sum())
        else:
            mean_breath_duration = 0.0
            total_breath_time = 0.0

        breath_to_speech_ratio = total_breath_time / max(duration, 1e-6)

        # Breath rate
        speech_duration_minutes = (speech_mask.sum() * _HOP_LENGTH / sr) / 60.0
        if speech_duration_minutes > 0.01:
            breath_rate = breath_count / speech_duration_minutes
        else:
            breath_rate = 0.0

        # Inter-breath intervals & periodicity
        if breath_count >= 2:
            centers = np.array([(s + e) / 2 for s, e in breath_segments])
            ibis = np.diff(centers)
            ibi_cv = float(np.std(ibis) / np.mean(ibis)) if np.mean(ibis) > 0 else 0.0
            # Periodicity: 1 - normalised std (capped at 1)
            breath_periodicity = float(np.clip(1.0 - ibi_cv, 0.0, 1.0))
        else:
            ibi_cv = 0.0
            breath_periodicity = 0.0

        # Inhalation detection — look for a breath segment just before
        # a speech onset (within 200 ms before speech starts)
        has_inhalation = self._detect_inhalation(breath_segments, speech_mask, sr)

        # Phonation onset sharpness — how abruptly energy rises at the
        # first speech onset after silence
        phonation_onset_sharpness = self._onset_sharpness(waveform, speech_mask, sr)

        return {
            "breath_count": breath_count,
            "breath_rate_per_minute": round(breath_rate, 2),
            "breath_periodicity": round(breath_periodicity, 4),
            "mean_breath_duration": round(mean_breath_duration, 4),
            "breath_to_speech_ratio": round(breath_to_speech_ratio, 4),
            "has_inhalation": has_inhalation,
            "inter_breath_interval_cv": round(ibi_cv, 4),
            "phonation_onset_sharpness": round(phonation_onset_sharpness, 4),
        }

    def _detect_inhalation(
        self,
        breath_segments: list[tuple[float, float]],
        speech_mask: np.ndarray,
        sr: int,
    ) -> bool:
        """
        Check if any breath segment occurs just before a speech onset.

        An inhalation is a breath that ends within 200 ms before a
        transition from non-speech to speech.
        """
        if not breath_segments:
            return False

        frame_dur = _HOP_LENGTH / sr
        onset_tolerance = 0.2  # seconds

        # Find speech onset times
        onsets: list[float] = []
        for i in range(1, len(speech_mask)):
            if speech_mask[i] and not speech_mask[i - 1]:
                onsets.append(i * frame_dur)

        if not onsets:
            return False

        for _, breath_end in breath_segments:
            for onset_time in onsets:
                gap = onset_time - breath_end
                if 0 <= gap <= onset_tolerance:
                    return True

        return False

    def _onset_sharpness(
        self,
        waveform: np.ndarray,
        speech_mask: np.ndarray,
        sr: int,
    ) -> float:
        """
        Measure how abruptly phonation starts after silence.

        Returns a value in [0, 1] where 1 = very sharp/abrupt,
        0 = gentle ramp-up.  Synthetic speech often has sharper onsets
        because it lacks the gradual airflow build-up of real speech.
        """
        frame_dur = _HOP_LENGTH / sr
        rms = librosa.feature.rms(
            y=waveform, frame_length=_FRAME_LENGTH, hop_length=_HOP_LENGTH
        )[0]

        # Find first substantial speech onset
        onset_frames: list[int] = []
        for i in range(1, len(speech_mask)):
            if speech_mask[i] and not speech_mask[i - 1]:
                onset_frames.append(i)

        if not onset_frames:
            return 0.5  # neutral

        sharpness_values: list[float] = []
        for onset_idx in onset_frames[:10]:  # check up to 10 onsets
            # Look at a ±5 frame window around onset
            lo = max(0, onset_idx - 3)
            hi = min(len(rms), onset_idx + 5)
            window = rms[lo:hi]
            if len(window) < 4:
                continue

            # Sharpness = max gradient in the window, normalised
            gradient = np.diff(window)
            max_grad = np.max(np.abs(gradient))
            # Normalise by local RMS peak
            local_peak = window.max()
            if local_peak > 1e-8:
                sharpness_values.append(float(np.clip(max_grad / local_peak, 0.0, 1.0)))

        if not sharpness_values:
            return 0.5

        return float(np.mean(sharpness_values))

    # ------------------------------------------------------------------
    # Internal: Scoring
    # ------------------------------------------------------------------

    def _score(
        self,
        features: dict,
        speech_mask: np.ndarray,
        sr: int,
        duration: float,
    ) -> tuple[list[str], float]:
        """
        Score how likely the audio is spoofed based on breathing features.

        Returns (anomalies_list, score_0_to_1).
        """
        anomalies: list[str] = []
        penalties: list[float] = []  # each in [0, 1]

        breath_count = features["breath_count"]
        breath_rate = features["breath_rate_per_minute"]
        ibi_cv = features["inter_breath_interval_cv"]
        has_inhalation = features["has_inhalation"]
        periodicity = features["breath_periodicity"]
        onset_sharpness = features["phonation_onset_sharpness"]
        speech_seconds = float(speech_mask.sum()) * _HOP_LENGTH / sr

        # ---- Anomaly 1: No breathing in extended speech ------------------
        if speech_seconds > _SPEECH_GAP_SPOOF_THRESHOLD and breath_count == 0:
            anomalies.append("no_breathing_detected")
            penalties.append(0.85)
        elif speech_seconds > 5.0 and breath_count == 0:
            anomalies.append("no_breathing_short_clip")
            penalties.append(0.45)

        # ---- Anomaly 2: Breath rate outside normal range -----------------
        if breath_count >= 2:
            if breath_rate < _NORMAL_BREATH_RATE_LO * 0.5:
                anomalies.append("breath_rate_too_low")
                deficit = (_NORMAL_BREATH_RATE_LO - breath_rate) / _NORMAL_BREATH_RATE_LO
                penalties.append(float(np.clip(deficit * 0.5, 0.0, 0.6)))
            elif breath_rate > _NORMAL_BREATH_RATE_HI * 2.0:
                anomalies.append("breath_rate_too_high")
                excess = (breath_rate - _NORMAL_BREATH_RATE_HI) / _NORMAL_BREATH_RATE_HI
                penalties.append(float(np.clip(excess * 0.3, 0.0, 0.4)))

        # ---- Anomaly 3: Too-regular breathing (machine-like) -------------
        if breath_count >= 3 and ibi_cv < _REGULARITY_SUSPICIOUS_CV:
            anomalies.append("breathing_too_regular")
            regularity_penalty = (1.0 - ibi_cv / _REGULARITY_SUSPICIOUS_CV) * 0.5
            penalties.append(float(np.clip(regularity_penalty, 0.0, 0.5)))

        # ---- Anomaly 4: No inhalation before speech onsets ---------------
        if speech_seconds > 4.0 and not has_inhalation and breath_count > 0:
            anomalies.append("no_pre_speech_inhalation")
            penalties.append(0.65)  # Heavy penalty: TTS often drops pre-onset breath

        if speech_seconds > 4.0 and not has_inhalation and breath_count == 0:
            # Already penalised for no breathing, but this confirms synthetic generation
            anomalies.append("no_inhalation_at_all")
            penalties.append(0.85)  # Catastrophic biological failure

        # ---- Anomaly 5: Onset sharpness (TTS often very sharp) -----------
        if onset_sharpness > 0.85:
            anomalies.append("abrupt_phonation_onset")
            penalties.append(float((onset_sharpness - 0.85) / 0.15 * 0.3))

        # ---- Combine penalties (soft-OR: 1 - product of (1-p)) -----------
        if penalties:
            score = 1.0 - float(np.prod([1.0 - p for p in penalties]))
        else:
            # No anomalies → give a small base score reflecting minor
            # uncertainty (breathing detection isn't perfect).
            score = 0.05

        # ---- Bonus: strong bonafide signals lower the score --------------
        if (
            breath_count >= 2
            and _NORMAL_BREATH_RATE_LO <= breath_rate <= _NORMAL_BREATH_RATE_HI
            and ibi_cv > 0.2
            and has_inhalation
        ):
            # Everything looks human — push score toward 0
            score *= 0.3

        return anomalies, score

    # ------------------------------------------------------------------
    # Internal: Confidence estimation
    # ------------------------------------------------------------------

    def _estimate_confidence(
        self,
        duration: float,
        speech_fraction: float,
        features: dict,
    ) -> float:
        """
        Estimate how much we should trust the breathing analysis.

        Confidence grows with audio length and speech content.
        """
        # Base confidence from duration (sigmoid-ish ramp: full at ~30 s)
        dur_conf = float(np.clip(duration / 30.0, 0.0, 1.0))

        # Speech fraction contribution — need a good mix of speech & pauses
        speech_conf = 1.0 - abs(speech_fraction - 0.6) * 1.5
        speech_conf = float(np.clip(speech_conf, 0.2, 1.0))

        # More breaths detected → more confident in the pattern analysis
        bc = features["breath_count"]
        breath_conf = float(np.clip(bc / 5.0, 0.0, 1.0))

        # Weighted combination
        confidence = 0.4 * dur_conf + 0.3 * speech_conf + 0.3 * breath_conf
        return confidence

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result(
        anomalies: Optional[list[str]] = None,
        confidence: float = 0.0,
    ) -> dict:
        return {
            "score": 0.5,
            "confidence": round(confidence, 4),
            "features": {
                "breath_count": 0,
                "breath_rate_per_minute": 0.0,
                "breath_periodicity": 0.0,
                "mean_breath_duration": 0.0,
                "breath_to_speech_ratio": 0.0,
                "has_inhalation": False,
                "inter_breath_interval_cv": 0.0,
                "phonation_onset_sharpness": 0.0,
            },
            "anomalies": anomalies or [],
            "breath_timestamps": [],
        }
