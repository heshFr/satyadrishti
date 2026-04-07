"""
Room Acoustics Consistency Analyzer
=====================================
Detects AI-generated audio by analyzing room impulse response (RIR)
consistency across the recording.

Key insight: Real audio recorded in a physical space has a consistent
room impulse response determined by room geometry, materials, and
microphone position. This creates:

1. **Consistent Reverberation**: The reverb tail (RT60) is the same
   throughout the recording. AI audio may have inconsistent or
   missing reverb characteristics.

2. **Early Reflection Pattern**: The pattern of early reflections is
   fixed by room geometry. AI audio generated without room simulation
   lacks these or has synthetic/inconsistent patterns.

3. **Direct-to-Reverberant Ratio (DRR)**: In real recordings, the DRR
   is relatively consistent unless the speaker moves. AI audio may have
   unnaturally high DRR (dry sound) or inconsistent DRR.

4. **Background Noise Floor**: Real recordings have characteristic
   background noise shaped by the room. AI audio often has either
   no background noise or synthetic noise that doesn't match room acoustics.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


class RoomAcousticsAnalyzer:
    """Analyzes room acoustics consistency for AI audio detection."""

    def __init__(self, sr: int = 16000):
        self.sr = sr

    def analyze(self, waveform: np.ndarray, sr: int = None) -> dict:
        """
        Analyze room acoustics consistency.

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

        if len(waveform) < sr:  # Need at least 1s
            return self._fallback_result()

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1)

        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val

        anomalies = []

        # 1. Reverberation consistency across segments
        reverb_score = self._analyze_reverb_consistency(waveform, sr)

        # 2. Background noise floor analysis
        noise_score = self._analyze_noise_floor(waveform, sr)

        # 3. Direct-to-reverberant ratio consistency
        drr_score = self._analyze_drr_consistency(waveform, sr)

        # 4. Room mode analysis (resonance frequencies)
        room_mode_score = self._analyze_room_modes(waveform, sr)

        # 5. Silence segment analysis
        silence_score = self._analyze_silence_segments(waveform, sr)

        if reverb_score > 0.55:
            anomalies.append(f"Reverberation inconsistency ({reverb_score:.2f})")
        if noise_score > 0.55:
            anomalies.append(f"Background noise anomaly ({noise_score:.2f})")
        if drr_score > 0.55:
            anomalies.append(f"DRR consistency anomaly ({drr_score:.2f})")
        if room_mode_score > 0.55:
            anomalies.append(f"Room resonance anomaly ({room_mode_score:.2f})")
        if silence_score > 0.55:
            anomalies.append(f"Silence segment anomaly ({silence_score:.2f})")

        weights = np.array([0.25, 0.25, 0.20, 0.15, 0.15])
        scores = np.array([reverb_score, noise_score, drr_score, room_mode_score, silence_score])
        final_score = float(np.dot(scores, weights))

        confidence = (
            0.5 * (1.0 - float(np.std(scores)))
            + 0.3 * float(np.max(scores))
            + 0.2 * min(1.0, len(anomalies) / 3)
        )

        return {
            "score": float(np.clip(final_score, 0.0, 1.0)),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "reverb_consistency_score": float(reverb_score),
            "noise_floor_score": float(noise_score),
            "drr_consistency_score": float(drr_score),
            "room_mode_score": float(room_mode_score),
            "silence_analysis_score": float(silence_score),
            "anomalies": anomalies,
        }

    def _analyze_reverb_consistency(self, waveform: np.ndarray, sr: int) -> float:
        """
        Check if reverberation characteristics are consistent across
        different segments of the recording.
        """
        # Divide into segments
        seg_duration = int(sr * 2.0)  # 2-second segments
        n_segments = len(waveform) // seg_duration
        if n_segments < 2:
            return 0.3

        decay_rates = []
        for i in range(min(n_segments, 5)):
            segment = waveform[i * seg_duration:(i + 1) * seg_duration]
            decay = self._estimate_reverb_decay(segment, sr)
            if decay is not None:
                decay_rates.append(decay)

        if len(decay_rates) < 2:
            return 0.3

        # Consistent reverb: low variance in decay rates
        decay_std = np.std(decay_rates)
        decay_mean = np.mean(decay_rates)
        decay_cv = decay_std / (abs(decay_mean) + 1e-10)

        scores = []

        # Real recordings: consistent reverb (CV < 0.3)
        # AI: inconsistent (CV > 0.5) or no reverb at all
        if decay_cv > 0.6:
            scores.append(0.7)
        elif decay_cv > 0.4:
            scores.append(0.45)
        else:
            scores.append(0.15)

        # Check if reverb is present at all
        # Very low decay rate = no reverb = likely AI or studio recording
        if abs(decay_mean) < 0.01:
            scores.append(0.55)  # Suspiciously dry
        else:
            scores.append(0.15)

        return float(np.mean(scores))

    def _estimate_reverb_decay(self, segment: np.ndarray, sr: int):
        """
        Estimate reverberation decay rate from a segment.
        Uses energy decay curve (EDC) from the Schroeder method.
        """
        # Find high-energy regions (speech) followed by decay
        frame_size = int(sr * 0.02)  # 20ms frames
        hop = frame_size // 2
        n_frames = (len(segment) - frame_size) // hop

        if n_frames < 10:
            return None

        energies = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop
            frame = segment[start:start + frame_size]
            energies[i] = np.mean(frame ** 2)

        if np.max(energies) < 1e-8:
            return None

        # Find speech offset points (energy drops)
        mean_energy = np.mean(energies)
        above_threshold = energies > mean_energy * 2

        offsets = []
        for i in range(1, len(above_threshold)):
            if above_threshold[i - 1] and not above_threshold[i]:
                offsets.append(i)

        if not offsets:
            return None

        # Measure decay rate after the first clear offset
        decay_rates = []
        for offset_idx in offsets[:3]:  # Check first 3 offsets
            if offset_idx + 10 >= len(energies):
                continue
            # Measure energy decay over next 10 frames
            decay_window = energies[offset_idx:offset_idx + 10]
            if decay_window[0] < 1e-10:
                continue
            # Log energy decay rate
            log_decay = np.log(decay_window + 1e-10)
            if len(log_decay) > 2:
                slope = np.polyfit(np.arange(len(log_decay)), log_decay, 1)[0]
                decay_rates.append(slope)

        if decay_rates:
            return float(np.mean(decay_rates))
        return None

    def _analyze_noise_floor(self, waveform: np.ndarray, sr: int) -> float:
        """
        Analyze background noise characteristics.

        Real recordings: shaped noise floor from room acoustics.
        AI audio: often perfectly clean or with synthetic noise.
        """
        # Find low-energy (silence) segments
        frame_size = int(sr * 0.03)  # 30ms frames
        hop = frame_size
        n_frames = len(waveform) // hop

        if n_frames < 10:
            return 0.3

        frame_energies = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop
            end = min(start + frame_size, len(waveform))
            frame_energies[i] = np.mean(waveform[start:end] ** 2)

        # Find lowest-energy frames (background noise)
        sorted_indices = np.argsort(frame_energies)
        n_noise_frames = max(3, n_frames // 10)
        noise_indices = sorted_indices[:n_noise_frames]

        # Collect noise segments
        noise_samples = []
        for idx in noise_indices:
            start = idx * hop
            end = min(start + frame_size, len(waveform))
            noise_samples.extend(waveform[start:end])

        if len(noise_samples) < 500:
            return 0.3

        noise = np.array(noise_samples)
        noise_level = np.std(noise)

        scores = []

        # Very low noise floor is suspicious (AI often produces clean audio)
        if noise_level < 1e-5:
            scores.append(0.7)  # Suspiciously clean
        elif noise_level < 5e-4:
            scores.append(0.45)
        else:
            scores.append(0.15)

        # Analyze noise spectrum shape
        if len(noise) >= 512:
            noise_fft = np.abs(np.fft.rfft(noise[:512]))
            if noise_fft.sum() > 1e-10:
                noise_fft_norm = noise_fft / noise_fft.sum()
                # Entropy of noise spectrum
                noise_entropy = -np.sum(noise_fft_norm[noise_fft_norm > 0] *
                                        np.log2(noise_fft_norm[noise_fft_norm > 0] + 1e-10))
                max_entropy = np.log2(len(noise_fft_norm))
                normalized_entropy = noise_entropy / max_entropy

                # Real room noise: shaped spectrum (lower entropy)
                # White/synthetic noise: higher entropy (flatter spectrum)
                if normalized_entropy > 0.92:
                    scores.append(0.65)  # Suspiciously flat (white noise)
                elif normalized_entropy < 0.5:
                    scores.append(0.5)  # Suspiciously shaped
                else:
                    scores.append(0.15)

        # Noise consistency across silence segments
        noise_energies = []
        for idx in noise_indices:
            start = idx * hop
            end = min(start + frame_size, len(waveform))
            noise_energies.append(np.std(waveform[start:end]))

        if len(noise_energies) > 2:
            noise_energy_cv = np.std(noise_energies) / (np.mean(noise_energies) + 1e-10)
            # Real: relatively consistent noise floor (CV < 0.5)
            # AI: may have varying noise (CV > 0.8)
            if noise_energy_cv > 0.9:
                scores.append(0.6)
            elif noise_energy_cv > 0.6:
                scores.append(0.35)
            else:
                scores.append(0.15)

        return float(np.mean(scores))

    def _analyze_drr_consistency(self, waveform: np.ndarray, sr: int) -> float:
        """
        Analyze Direct-to-Reverberant Ratio consistency.

        Estimates DRR from spectral characteristics across segments.
        """
        seg_duration = int(sr * 1.5)
        n_segments = len(waveform) // seg_duration
        if n_segments < 3:
            return 0.3

        spectral_centroids = []
        for i in range(min(n_segments, 6)):
            segment = waveform[i * seg_duration:(i + 1) * seg_duration]
            # Spectral centroid as proxy for DRR
            # Higher centroid = more direct sound; lower = more reverberant
            fft = np.abs(np.fft.rfft(segment))
            freqs = np.fft.rfftfreq(len(segment), 1.0 / sr)
            if fft.sum() > 1e-10:
                centroid = np.sum(freqs * fft) / np.sum(fft)
                spectral_centroids.append(centroid)

        if len(spectral_centroids) < 3:
            return 0.3

        centroid_cv = np.std(spectral_centroids) / (np.mean(spectral_centroids) + 1e-10)

        # Real (stationary speaker): CV < 0.15
        # AI (inconsistent acoustics): CV > 0.25
        if centroid_cv > 0.30:
            return 0.65
        elif centroid_cv > 0.20:
            return 0.4
        elif centroid_cv < 0.02:
            return 0.55  # Suspiciously consistent (synthetic)
        else:
            return 0.15

    def _analyze_room_modes(self, waveform: np.ndarray, sr: int) -> float:
        """
        Check for consistent room resonance modes.

        Real rooms have specific resonance frequencies determined by
        room dimensions. These appear as peaks in the long-term spectrum.
        """
        # Long-term average spectrum
        n_fft = 2048
        hop = n_fft // 2

        if len(waveform) < n_fft * 3:
            return 0.3

        # Compute average power spectrum
        n_frames = (len(waveform) - n_fft) // hop
        avg_spectrum = np.zeros(n_fft // 2 + 1)

        for i in range(n_frames):
            start = i * hop
            frame = waveform[start:start + n_fft]
            window = np.hanning(n_fft)
            fft = np.abs(np.fft.rfft(frame * window)) ** 2
            avg_spectrum += fft

        avg_spectrum /= (n_frames + 1e-10)

        if avg_spectrum.max() < 1e-10:
            return 0.3

        # Normalize
        avg_spectrum = avg_spectrum / avg_spectrum.max()

        # Smoothed version for finding peaks
        kernel_size = 11
        smoothed = np.convolve(avg_spectrum, np.ones(kernel_size) / kernel_size, mode='same')

        # Find spectral peaks (room modes)
        peaks = []
        for i in range(2, len(smoothed) - 2):
            if (smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]
                    and smoothed[i] > smoothed[i - 2] and smoothed[i] > smoothed[i + 2]):
                if smoothed[i] > 0.1:
                    peaks.append(smoothed[i])

        scores = []

        # Real rooms: moderate number of peaks (5-20 prominent room modes)
        # AI (no room): very smooth spectrum (0-2 peaks)
        # AI (synthetic room): may have different peak distribution
        n_peaks = len(peaks)
        if n_peaks < 3:
            scores.append(0.6)  # Too smooth — no room modes
        elif n_peaks > 30:
            scores.append(0.5)  # Too many peaks — unusual
        else:
            scores.append(0.15)

        # Peak height distribution
        if len(peaks) >= 3:
            peak_cv = np.std(peaks) / (np.mean(peaks) + 1e-10)
            # Real rooms: moderate CV (different modes have different strengths)
            # AI: may be too uniform
            if peak_cv < 0.1:
                scores.append(0.55)
            else:
                scores.append(0.15)

        return float(np.mean(scores))

    def _analyze_silence_segments(self, waveform: np.ndarray, sr: int) -> float:
        """
        Analyze what silence sounds like in the recording.

        Real recordings: silence has room tone (filtered noise).
        AI: silence may be digital silence (zeros) or white noise.
        """
        frame_size = int(sr * 0.05)  # 50ms frames
        hop = frame_size
        n_frames = len(waveform) // hop

        if n_frames < 10:
            return 0.3

        # Find silence frames
        energies = []
        for i in range(n_frames):
            start = i * hop
            end = min(start + frame_size, len(waveform))
            energies.append(np.mean(waveform[start:end] ** 2))

        energies = np.array(energies)
        threshold = np.percentile(energies, 15)
        silence_mask = energies < threshold

        # Collect silence samples
        silence_frames = []
        for i in range(n_frames):
            if silence_mask[i]:
                start = i * hop
                end = min(start + frame_size, len(waveform))
                silence_frames.append(waveform[start:end])

        if len(silence_frames) < 3:
            return 0.3

        scores = []

        # Check if silence is digital zeros
        all_silence = np.concatenate(silence_frames)
        zero_ratio = np.mean(np.abs(all_silence) < 1e-8)

        if zero_ratio > 0.95:
            scores.append(0.75)  # Digital silence — highly suspicious
        elif zero_ratio > 0.8:
            scores.append(0.5)
        else:
            scores.append(0.15)

        # Check silence spectral shape consistency
        silence_spectra = []
        for frame in silence_frames[:10]:
            if len(frame) >= 256:
                spec = np.abs(np.fft.rfft(frame[:256]))
                if spec.sum() > 1e-10:
                    silence_spectra.append(spec / spec.sum())

        if len(silence_spectra) >= 3:
            # Pairwise spectral distance between silence segments
            distances = []
            for i in range(len(silence_spectra)):
                for j in range(i + 1, len(silence_spectra)):
                    dist = np.sqrt(np.mean((silence_spectra[i] - silence_spectra[j]) ** 2))
                    distances.append(dist)

            mean_dist = np.mean(distances)
            # Real: consistent room tone (low distance between silence spectra)
            # AI: may have varying silence characteristics
            if mean_dist > 0.05:
                scores.append(0.6)
            elif mean_dist > 0.02:
                scores.append(0.35)
            else:
                scores.append(0.15)

        return float(np.mean(scores))

    def _fallback_result(self) -> dict:
        return {
            "score": 0.5,
            "confidence": 0.0,
            "reverb_consistency_score": 0.5,
            "noise_floor_score": 0.5,
            "drr_consistency_score": 0.5,
            "room_mode_score": 0.5,
            "silence_analysis_score": 0.5,
            "anomalies": ["Insufficient data or missing librosa"],
        }
