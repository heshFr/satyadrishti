"""
Codec Artifact Detector
========================
Identifies the audio codec used in a recording and adjusts deepfake
detection strategy accordingly. Different codecs introduce specific
artifacts that can mask or mimic deepfake indicators.

WHY THIS MATTERS:
━━━━━━━━━━━━━━━━━
Voice calls traverse complex codec chains:
  Phone call: mic → AAC/Opus → network → G.711/AMR → network → decode
  WhatsApp:   mic → Opus @16kHz → internet → decode
  Zoom:       mic → Opus @48kHz → internet → decode
  Landline:   mic → G.711 μ-law @8kHz → PSTN → decode

Each codec:
  - Cuts off frequencies above its bandwidth (G.711: 3.4kHz, AMR-WB: 7kHz)
  - Introduces quantization noise
  - Smooths spectral detail (lossy compression)
  - May add comfort noise during silence

These artifacts affect deepfake detectors:
  - Frequency analysis becomes unreliable above codec cutoff
  - Phase coherence is destroyed by lossy codecs
  - Breathing detection is harder under G.711
  - Formant analysis is less accurate at 8kHz narrowband

DETECTION METHOD:
━━━━━━━━━━━━━━━━━
1. Spectral Cutoff Detection
   - Compute power spectral density
   - Find the frequency where power drops sharply (codec bandwidth limit)
   - G.711: ~3.4kHz, AMR-NB: ~3.4kHz, AMR-WB: ~7kHz, Opus: ~20kHz

2. Quantization Noise Analysis
   - Measure noise floor uniformity (codecs produce flat noise floor)
   - Detect quantization step artifacts in amplitude histogram

3. Comfort Noise Detection
   - G.711 and some codecs insert artificial comfort noise during silence
   - Detect: suspiciously uniform noise in silent segments

4. Codec Fingerprinting
   - Each codec leaves a characteristic spectral signature
   - G.711 μ-law: characteristic quantization curve
   - Opus: specific spectral shaping above bandwidth
   - AMR: distinctive spectral holes at band boundaries

OUTPUT:
━━━━━━━
  {
      "codec": "opus" | "g711_ulaw" | "g711_alaw" | "amr_nb" | "amr_wb" |
               "aac" | "mp3" | "pcm" | "unknown",
      "bandwidth_hz": float,
      "quality_tier": "narrowband" | "wideband" | "fullband",
      "comfort_noise_detected": bool,
      "codec_confidence": float,
      "reliability_adjustments": { analyzer_name: weight_multiplier },
      "score": float (0-1, how much codec degrades detection reliability),
      "details": { ... }
  }
"""

import logging
from typing import Dict, Any, Tuple, List, Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    from scipy import signal as scipy_signal
    from scipy.stats import entropy as scipy_entropy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ── Codec bandwidth signatures ──
CODEC_PROFILES = {
    "g711_ulaw": {
        "cutoff_hz": (3200, 3600),
        "sample_rate": 8000,
        "quality": "narrowband",
        "description": "G.711 μ-law (PSTN/landline)",
    },
    "g711_alaw": {
        "cutoff_hz": (3200, 3600),
        "sample_rate": 8000,
        "quality": "narrowband",
        "description": "G.711 A-law (European PSTN)",
    },
    "amr_nb": {
        "cutoff_hz": (3200, 3500),
        "sample_rate": 8000,
        "quality": "narrowband",
        "description": "AMR Narrowband (GSM mobile)",
    },
    "amr_wb": {
        "cutoff_hz": (6800, 7200),
        "sample_rate": 16000,
        "quality": "wideband",
        "description": "AMR Wideband (HD Voice/VoLTE)",
    },
    "opus_voip": {
        "cutoff_hz": (7500, 8500),
        "sample_rate": 16000,
        "quality": "wideband",
        "description": "Opus VoIP mode (WhatsApp/Telegram)",
    },
    "opus_fullband": {
        "cutoff_hz": (19000, 22000),
        "sample_rate": 48000,
        "quality": "fullband",
        "description": "Opus fullband (Zoom/Discord)",
    },
    "aac": {
        "cutoff_hz": (15000, 20000),
        "sample_rate": 44100,
        "quality": "fullband",
        "description": "AAC (iOS/FaceTime)",
    },
    "mp3": {
        "cutoff_hz": (15000, 19000),
        "sample_rate": 44100,
        "quality": "fullband",
        "description": "MP3 (recordings)",
    },
    "pcm": {
        "cutoff_hz": (20000, 24000),
        "sample_rate": 48000,
        "quality": "fullband",
        "description": "Uncompressed PCM/WAV",
    },
}

# Reliability adjustments: how much to trust each analyzer under each codec quality
RELIABILITY_ADJUSTMENTS = {
    "narrowband": {
        "ast": 0.85,            # neural detector somewhat robust
        "ssl": 0.70,            # SSL features degraded at 8kHz
        "whisper_features": 0.60,  # Whisper trained on wider bandwidth
        "prosodic": 0.80,       # F0/jitter still measurable
        "breathing": 0.50,      # breathing hard to detect at narrowband
        "phase": 0.40,          # phase coherence destroyed by codec
        "formant": 0.70,        # F1-F3 still in band, F4+ lost
        "temporal": 0.85,       # embedding stability still valid
        "tts_artifacts": 0.55,  # many TTS cues above 3.4kHz are lost
    },
    "wideband": {
        "ast": 0.95,
        "ssl": 0.90,
        "whisper_features": 0.85,
        "prosodic": 0.90,
        "breathing": 0.75,
        "phase": 0.70,
        "formant": 0.90,
        "temporal": 0.90,
        "tts_artifacts": 0.80,
    },
    "fullband": {
        "ast": 1.0,
        "ssl": 1.0,
        "whisper_features": 1.0,
        "prosodic": 1.0,
        "breathing": 1.0,
        "phase": 1.0,
        "formant": 1.0,
        "temporal": 1.0,
        "tts_artifacts": 1.0,
    },
}


class CodecDetector:
    """
    Identifies audio codec from spectral characteristics and provides
    reliability adjustments for downstream deepfake detectors.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.is_available = HAS_SCIPY

    def analyze(self, waveform: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Analyze audio to identify codec and compute reliability adjustments.

        Args:
            waveform: Mono audio signal (float32, [-1, 1])
            sr: Sample rate in Hz

        Returns:
            Dict with codec identification and reliability adjustments.
        """
        if not self.is_available:
            return self._default_result()

        details = {}

        # Step 1: Compute power spectral density
        bandwidth_hz, spectral_features = self._estimate_bandwidth(waveform, sr)
        details["spectral"] = spectral_features

        # Step 2: Detect comfort noise
        comfort_noise = self._detect_comfort_noise(waveform, sr)
        details["comfort_noise"] = comfort_noise

        # Step 3: Quantization analysis
        quantization = self._analyze_quantization(waveform)
        details["quantization"] = quantization

        # Step 4: Identify codec
        codec, codec_confidence = self._identify_codec(
            bandwidth_hz, sr, comfort_noise, quantization
        )

        # Step 5: Determine quality tier
        quality_tier = CODEC_PROFILES.get(codec, {}).get("quality", "fullband")
        if quality_tier == "fullband" and bandwidth_hz < 8000:
            quality_tier = "wideband"
        if bandwidth_hz < 4000:
            quality_tier = "narrowband"

        # Step 6: Get reliability adjustments
        reliability = dict(RELIABILITY_ADJUSTMENTS.get(quality_tier, RELIABILITY_ADJUSTMENTS["fullband"]))

        # Step 7: Compute degradation score (how much codec hurts detection)
        # Higher = more degradation = less reliable detection
        mean_reliability = float(np.mean(list(reliability.values())))
        degradation_score = 1.0 - mean_reliability

        codec_description = CODEC_PROFILES.get(codec, {}).get("description", "Unknown codec")

        return {
            "codec": codec,
            "codec_description": codec_description,
            "bandwidth_hz": round(bandwidth_hz, 0),
            "quality_tier": quality_tier,
            "comfort_noise_detected": comfort_noise["detected"],
            "codec_confidence": round(codec_confidence, 4),
            "reliability_adjustments": {k: round(v, 3) for k, v in reliability.items()},
            "score": round(degradation_score, 4),
            "confidence": round(codec_confidence, 4),
            "details": details,
        }

    def _estimate_bandwidth(
        self, waveform: np.ndarray, sr: int
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Estimate effective audio bandwidth from spectral density.

        Finds the frequency where power drops below -40dB relative to peak.
        """
        nperseg = min(4096, len(waveform))
        freqs, psd = scipy_signal.welch(
            waveform, fs=sr, nperseg=nperseg, noverlap=nperseg // 2
        )

        # Convert to dB
        psd_db = 10 * np.log10(psd + 1e-20)
        peak_db = np.max(psd_db)

        # Find cutoff: where power drops 40dB below peak
        threshold_db = peak_db - 40.0
        above_threshold = psd_db > threshold_db

        # Find highest frequency still above threshold
        if np.any(above_threshold):
            last_idx = np.where(above_threshold)[0][-1]
            bandwidth_hz = float(freqs[last_idx])
        else:
            bandwidth_hz = float(freqs[-1])

        # Also find -20dB point for narrower estimate
        threshold_20 = peak_db - 20.0
        above_20 = psd_db > threshold_20
        if np.any(above_20):
            last_20_idx = np.where(above_20)[0][-1]
            bandwidth_20_hz = float(freqs[last_20_idx])
        else:
            bandwidth_20_hz = bandwidth_hz

        # Spectral rolloff (frequency below which 85% of energy is concentrated)
        cumulative_energy = np.cumsum(psd)
        total_energy = cumulative_energy[-1]
        rolloff_idx = np.searchsorted(cumulative_energy, total_energy * 0.85)
        rolloff_hz = float(freqs[min(rolloff_idx, len(freqs) - 1)])

        # Spectral centroid
        centroid = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-10))

        return bandwidth_hz, {
            "bandwidth_40db": round(bandwidth_hz, 0),
            "bandwidth_20db": round(bandwidth_20_hz, 0),
            "spectral_rolloff_85": round(rolloff_hz, 0),
            "spectral_centroid": round(centroid, 0),
            "peak_db": round(peak_db, 1),
            "sample_rate": sr,
            "nyquist": sr // 2,
        }

    def _detect_comfort_noise(
        self, waveform: np.ndarray, sr: int
    ) -> Dict[str, Any]:
        """
        Detect artificial comfort noise inserted during silence.

        Comfort noise has: uniform spectral content, constant amplitude,
        suspiciously flat noise floor.
        """
        # Find silent segments (below -40dBFS)
        frame_len = int(0.02 * sr)  # 20ms frames
        n_frames = len(waveform) // frame_len
        frame_energies = np.array([
            np.sqrt(np.mean(waveform[i*frame_len:(i+1)*frame_len]**2))
            for i in range(n_frames)
        ])

        # Threshold for silence: below 1% of max RMS
        silence_threshold = np.max(frame_energies) * 0.01
        silent_frames = frame_energies < silence_threshold

        if np.sum(silent_frames) < 3:
            return {"detected": False, "silent_frames": 0, "reason": "No silent segments found"}

        # Analyze noise in silent frames
        silent_segments = []
        for i in range(n_frames):
            if silent_frames[i]:
                segment = waveform[i*frame_len:(i+1)*frame_len]
                silent_segments.append(segment)

        if len(silent_segments) < 2:
            return {"detected": False, "silent_frames": int(np.sum(silent_frames)), "reason": "Insufficient silence"}

        # Check if silent segments have suspiciously uniform noise
        silent_rmses = [float(np.sqrt(np.mean(s**2))) for s in silent_segments]
        rms_cv = float(np.std(silent_rmses) / (np.mean(silent_rmses) + 1e-10))

        # Check spectral flatness of silent segments
        flatnesses = []
        for seg in silent_segments[:10]:
            if len(seg) > 256:
                spectrum = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
                if np.any(spectrum > 0):
                    geometric_mean = np.exp(np.mean(np.log(spectrum + 1e-20)))
                    arithmetic_mean = np.mean(spectrum)
                    flatness = geometric_mean / (arithmetic_mean + 1e-10)
                    flatnesses.append(float(flatness))

        mean_flatness = float(np.mean(flatnesses)) if flatnesses else 0.0

        # Comfort noise: low RMS variation + high spectral flatness
        is_comfort_noise = rms_cv < 0.3 and mean_flatness > 0.5

        return {
            "detected": is_comfort_noise,
            "silent_frames": int(np.sum(silent_frames)),
            "rms_cv": round(rms_cv, 4),
            "spectral_flatness": round(mean_flatness, 4),
        }

    def _analyze_quantization(self, waveform: np.ndarray) -> Dict[str, Any]:
        """
        Analyze amplitude quantization artifacts.

        G.711 μ-law has characteristic non-linear quantization.
        """
        # Amplitude histogram
        hist, bin_edges = np.histogram(waveform, bins=256, range=(-1, 1))
        hist_normalized = hist / (hist.sum() + 1e-10)

        # Entropy of amplitude distribution (lower = more quantized)
        amp_entropy = float(scipy_entropy(hist_normalized + 1e-10))

        # Check for quantization steps (gaps in histogram)
        zero_bins = np.sum(hist == 0)
        zero_ratio = zero_bins / len(hist)

        # μ-law signature: more bins near zero, fewer at extremes
        center_mass = float(np.sum(hist[96:160]) / (np.sum(hist) + 1e-10))

        # Quantization detected if: low entropy + many empty bins
        is_quantized = amp_entropy < 4.0 and zero_ratio > 0.1

        return {
            "amplitude_entropy": round(amp_entropy, 4),
            "zero_bin_ratio": round(zero_ratio, 4),
            "center_mass_ratio": round(center_mass, 4),
            "quantization_detected": is_quantized,
        }

    def _identify_codec(
        self,
        bandwidth_hz: float,
        sample_rate: int,
        comfort_noise: Dict,
        quantization: Dict,
    ) -> Tuple[str, float]:
        """
        Identify the most likely codec from spectral and quantization features.
        """
        scores = {}

        for codec_name, profile in CODEC_PROFILES.items():
            score = 0.0
            cutoff_low, cutoff_high = profile["cutoff_hz"]

            # Bandwidth match
            if cutoff_low <= bandwidth_hz <= cutoff_high:
                score += 0.5
            elif abs(bandwidth_hz - (cutoff_low + cutoff_high) / 2) < 1000:
                score += 0.3
            elif abs(bandwidth_hz - (cutoff_low + cutoff_high) / 2) < 2000:
                score += 0.1

            # Sample rate match
            if sample_rate == profile["sample_rate"]:
                score += 0.2
            elif sample_rate >= profile["sample_rate"]:
                score += 0.1

            # Codec-specific signatures
            if codec_name in ("g711_ulaw", "g711_alaw"):
                if quantization["quantization_detected"]:
                    score += 0.2
                if comfort_noise["detected"]:
                    score += 0.1

            if codec_name.startswith("amr"):
                if bandwidth_hz < 4000 and codec_name == "amr_nb":
                    score += 0.15
                elif 6500 < bandwidth_hz < 7500 and codec_name == "amr_wb":
                    score += 0.15

            scores[codec_name] = score

        # Best match
        best_codec = max(scores, key=scores.get)
        best_score = scores[best_codec]

        # Normalize confidence
        confidence = min(1.0, best_score / 0.7) if best_score > 0.2 else 0.3

        # Default to PCM if no strong match
        if best_score < 0.2:
            best_codec = "pcm"
            confidence = 0.5

        return best_codec, confidence

    def _default_result(self) -> Dict[str, Any]:
        """Default result when scipy is not available."""
        return {
            "codec": "unknown",
            "codec_description": "Unknown (scipy unavailable)",
            "bandwidth_hz": 0,
            "quality_tier": "fullband",
            "comfort_noise_detected": False,
            "codec_confidence": 0.0,
            "reliability_adjustments": {k: 1.0 for k in RELIABILITY_ADJUSTMENTS["fullband"]},
            "score": 0.0,
            "confidence": 0.0,
            "details": {"error": "scipy not installed"},
        }
