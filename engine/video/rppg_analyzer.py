"""
Remote Photoplethysmography (rPPG) Analyzer
=============================================
Detects blood pulse signal in facial skin color variations to distinguish
real human video from AI-generated video.

THEORETICAL FOUNDATION:
━━━━━━━━━━━━━━━━━━━━━━━
The human cardiovascular system pumps blood in rhythmic cycles (60-100 BPM
at rest). Each heartbeat causes a transient increase in blood volume in
peripheral capillaries, which changes the optical absorption of skin tissue.
This is the same principle behind pulse oximeters and medical PPG sensors.

In video, these blood volume changes manifest as sub-pixel color variations
(primarily in the green channel, where hemoglobin has peak absorption
differential). These variations are:
  - ALWAYS present in real human video (30-240 BPM range)
  - Typically 0.1-0.5% of total pixel intensity
  - Strongest in forehead, cheeks, and exposed skin areas
  - Detectable even through moderate compression

WHY AI GENERATORS CANNOT FAKE THIS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current AI video generators (Sora, Runway, Kling, Pika, etc.) model visual
appearance but NOT physiological processes. They don't simulate:
  - Blood flow dynamics
  - Cardiac rhythm
  - Hemoglobin absorption spectra
  - Autonomic nervous system effects on skin color

This makes rPPG analysis arguably the single most powerful signal for
detecting AI-generated video containing human faces.

ALGORITHM — CHROM (Chrominance-based rPPG):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
De Haan & Jeanne (2013) "Robust Pulse Rate From Chrominance-Based rPPG"

1. Detect face ROI across frames (forehead + cheek regions)
2. Extract spatially-averaged RGB signals from skin ROI
3. Normalize RGB channels (divide by temporal mean)
4. Apply CHROM color transformation:
   S1 = 3R - 2G
   S2 = 1.5R + G - 1.5B
   α = σ(S1) / σ(S2)
   pulse_signal = S1 - α × S2
5. Bandpass filter to cardiac range (0.7-4.0 Hz = 42-240 BPM)
6. Compute Power Spectral Density (PSD) via Welch's method
7. Find dominant peak in cardiac band
8. Compute Signal-to-Noise Ratio (SNR):
   SNR = power_at_peak / average_power_outside_peak

SCORING:
━━━━━━━━
  - SNR > 4.0 dB → Strong pulse detected → REAL video (score < 0.2)
  - SNR 2.0-4.0  → Weak pulse detected → Likely REAL (score 0.2-0.4)
  - SNR 1.0-2.0  → Ambiguous → Inconclusive (score 0.4-0.6)
  - SNR < 1.0    → No pulse detected → Likely AI (score 0.6-0.8)
  - SNR < 0.3    → Flat spectrum → Definitely AI (score 0.8-1.0)

ADDITIONAL METRICS:
━━━━━━━━━━━━━━━━━━━
  - Heart Rate Variability (HRV): real heartbeats have natural variability
  - Multi-ROI consistency: forehead and cheek should show same frequency
  - Signal quality index: motion artifact contamination level
  - Harmonic structure: real pulse has harmonics (2nd, 3rd) of fundamental
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, List, Optional


class RPPGAnalyzer:
    """
    Remote Photoplethysmography analyzer for detecting physiological
    signals in video to distinguish real humans from AI-generated faces.
    """

    # Cardiac frequency band (Hz)
    CARDIAC_LOW = 0.7    # 42 BPM (extreme low)
    CARDIAC_HIGH = 4.0   # 240 BPM (extreme high)

    # Typical resting heart rate band
    RESTING_LOW = 0.83   # 50 BPM
    RESTING_HIGH = 2.0   # 120 BPM

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        # Load face detection cascade
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_path)

        # Eye cascade for better face alignment and forehead estimation
        eye_haar = cv2.data.haarcascades + "haarcascade_eye.xml"
        self.eye_cascade = cv2.CascadeClassifier(eye_haar)

    def analyze(self, video_path: str, max_seconds: float = 15.0) -> Dict[str, Any]:
        """
        Run full rPPG analysis on a video file.

        Args:
            video_path: Path to video file
            max_seconds: Maximum duration to analyze (longer = more accurate)

        Returns:
            Dict with:
              - ai_probability: float 0-1 (1 = definitely AI, no pulse)
              - has_pulse: bool
              - estimated_bpm: float or None
              - snr_db: float (signal-to-noise ratio in dB)
              - metrics: detailed analysis metrics
        """
        # Extract frames and face ROIs
        frames_rgb, fps, face_signals = self._extract_face_signals(
            video_path, max_seconds
        )

        if face_signals is None or len(face_signals) < 30:
            return {
                "ai_probability": 0.5,
                "has_pulse": None,
                "estimated_bpm": None,
                "snr_db": 0.0,
                "metrics": {"error": "Insufficient face frames for rPPG analysis",
                            "frames_with_face": len(face_signals) if face_signals is not None else 0},
            }

        # Run CHROM algorithm on the extracted RGB signals
        pulse_signal, chrom_metrics = self._chrom_rppg(face_signals, fps)

        if pulse_signal is None:
            return {
                "ai_probability": 0.5,
                "has_pulse": None,
                "estimated_bpm": None,
                "snr_db": 0.0,
                "metrics": {"error": "CHROM extraction failed", **chrom_metrics},
            }

        # Analyze the pulse signal
        spectral_result = self._analyze_pulse_spectrum(pulse_signal, fps)

        # Multi-ROI validation (if we have forehead + cheek separately)
        multi_roi_score = self._multi_roi_validation(face_signals, fps)

        # Harmonic analysis
        harmonic_score = self._analyze_harmonics(pulse_signal, fps, spectral_result)

        # Signal quality assessment
        quality = self._assess_signal_quality(pulse_signal, face_signals)

        # ─── Compute Final AI Probability ───
        snr = spectral_result["snr_db"]
        peak_power_ratio = spectral_result["peak_power_ratio"]
        has_clear_peak = spectral_result["has_clear_peak"]

        # Primary score from SNR
        if snr > 5.0:
            snr_score = 0.05   # Very strong pulse → definitely real
        elif snr > 3.0:
            snr_score = 0.15   # Strong pulse → real
        elif snr > 2.0:
            snr_score = 0.30   # Moderate pulse → likely real
        elif snr > 1.0:
            snr_score = 0.50   # Weak signal → inconclusive
        elif snr > 0.5:
            snr_score = 0.70   # Very weak → likely AI
        elif snr > 0.2:
            snr_score = 0.85   # Almost nothing → probably AI
        else:
            snr_score = 0.95   # Flat spectrum → definitely AI

        # Harmonic bonus (real pulse has harmonics)
        if harmonic_score > 0.5:
            snr_score = snr_score * 0.8  # Reduce AI probability (harmonics = real)

        # Multi-ROI consistency bonus
        if multi_roi_score < 0.3:
            snr_score = snr_score * 0.85  # Consistent across ROIs = real

        # Quality penalty (motion artifacts can mask real pulse)
        if quality["signal_quality"] < 0.3:
            # Bad quality → don't trust the absence of signal
            snr_score = 0.5 + (snr_score - 0.5) * 0.5  # Pull toward 0.5

        ai_probability = float(np.clip(snr_score, 0, 1))

        estimated_bpm = spectral_result.get("estimated_bpm")
        has_pulse = snr > 1.5 and has_clear_peak

        if self.verbose:
            print(f"  [rPPG] SNR={snr:.2f}dB, BPM={estimated_bpm}, "
                  f"has_pulse={has_pulse}, ai_prob={ai_probability:.3f}")

        return {
            "ai_probability": round(ai_probability, 4),
            "has_pulse": has_pulse,
            "estimated_bpm": round(estimated_bpm, 1) if estimated_bpm else None,
            "snr_db": round(snr, 2),
            "metrics": {
                "snr_db": round(snr, 3),
                "peak_power_ratio": round(peak_power_ratio, 4),
                "has_clear_peak": has_clear_peak,
                "estimated_bpm": round(estimated_bpm, 1) if estimated_bpm else None,
                "harmonic_score": round(harmonic_score, 4),
                "multi_roi_consistency": round(multi_roi_score, 4),
                "signal_quality": round(quality["signal_quality"], 4),
                "motion_contamination": round(quality["motion_contamination"], 4),
                "frames_analyzed": len(face_signals),
                "fps": round(fps, 2),
                "duration_seconds": round(len(face_signals) / fps, 1),
                **chrom_metrics,
            },
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Face Detection & Signal Extraction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _extract_face_signals(
        self, video_path: str, max_seconds: float
    ) -> Tuple[Optional[List], float, Optional[np.ndarray]]:
        """
        Extract per-frame RGB signals from face skin regions.

        For each frame:
        1. Detect face bounding box
        2. Extract forehead ROI (upper 30% of face, inner 60% width)
        3. Extract left cheek and right cheek ROIs
        4. Compute mean RGB for each ROI

        Returns:
            (frames_list, fps, face_signals)
            face_signals shape: (N_frames, 3, 3) — 3 ROIs × 3 channels
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, 0.0, None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = int(min(total, max_seconds * fps))

        # For rPPG we need CONSECUTIVE frames (not sampled) for temporal continuity
        # Read from the start, up to max_frames
        face_rgb_signals = []  # (N, 3, 3) — [forehead, left_cheek, right_cheek] × [R, G, B]

        # Track stable face bbox using exponential moving average
        stable_bbox = None
        ema_alpha = 0.3

        frame_idx = 0
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1

            # Detect face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )

            if len(faces) > 0:
                # Use largest face
                bbox = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = bbox

                # Smooth bbox with EMA for stability
                if stable_bbox is not None:
                    stable_bbox = (
                        int(stable_bbox[0] * (1 - ema_alpha) + x * ema_alpha),
                        int(stable_bbox[1] * (1 - ema_alpha) + y * ema_alpha),
                        int(stable_bbox[2] * (1 - ema_alpha) + w * ema_alpha),
                        int(stable_bbox[3] * (1 - ema_alpha) + h * ema_alpha),
                    )
                else:
                    stable_bbox = (x, y, w, h)
            elif stable_bbox is None:
                # No face found yet — skip frame
                continue

            # Extract ROIs from stable bbox
            x, y, w, h = stable_bbox
            fh, fw = frame.shape[:2]

            # ROI 1: Forehead (upper 30% of face, inner 60% width)
            forehead_y1 = max(0, y + int(h * 0.05))
            forehead_y2 = max(0, y + int(h * 0.30))
            forehead_x1 = max(0, x + int(w * 0.20))
            forehead_x2 = min(fw, x + int(w * 0.80))

            # ROI 2: Left cheek (40-70% height, 10-35% width)
            lcheek_y1 = max(0, y + int(h * 0.40))
            lcheek_y2 = min(fh, y + int(h * 0.70))
            lcheek_x1 = max(0, x + int(w * 0.10))
            lcheek_x2 = max(0, x + int(w * 0.35))

            # ROI 3: Right cheek (40-70% height, 65-90% width)
            rcheek_y1 = max(0, y + int(h * 0.40))
            rcheek_y2 = min(fh, y + int(h * 0.70))
            rcheek_x1 = min(fw, x + int(w * 0.65))
            rcheek_x2 = min(fw, x + int(w * 0.90))

            # Extract mean RGB from each ROI (convert BGR to RGB)
            rois = [
                frame[forehead_y1:forehead_y2, forehead_x1:forehead_x2],
                frame[lcheek_y1:lcheek_y2, lcheek_x1:lcheek_x2],
                frame[rcheek_y1:rcheek_y2, rcheek_x1:rcheek_x2],
            ]

            roi_means = []
            valid = True
            for roi in rois:
                if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
                    valid = False
                    break
                # BGR to RGB mean
                mean_bgr = roi.mean(axis=(0, 1))
                roi_means.append([mean_bgr[2], mean_bgr[1], mean_bgr[0]])  # R, G, B

            if valid:
                face_rgb_signals.append(roi_means)

        cap.release()

        if len(face_rgb_signals) < 30:
            return None, fps, np.array(face_rgb_signals) if face_rgb_signals else None

        return None, fps, np.array(face_rgb_signals)  # (N, 3, 3)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CHROM Algorithm
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _chrom_rppg(
        self, face_signals: np.ndarray, fps: float
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Apply CHROM (Chrominance-based) rPPG algorithm.

        De Haan & Jeanne (2013):
        1. Normalize RGB channels by temporal mean
        2. Build chrominance signals: S1 = 3R-2G, S2 = 1.5R+G-1.5B
        3. Combine: pulse = S1 - α*S2 where α = σ(S1)/σ(S2)
        4. Bandpass filter to cardiac range

        Args:
            face_signals: (N, 3, 3) array — 3 ROIs × RGB
            fps: video frame rate

        Returns:
            (pulse_signal, metrics_dict)
        """
        # Average across ROIs for the primary signal
        # (N, 3) — mean R, G, B across all face ROIs
        rgb_mean = face_signals.mean(axis=1)  # (N, 3)

        R = rgb_mean[:, 0].astype(np.float64)
        G = rgb_mean[:, 1].astype(np.float64)
        B = rgb_mean[:, 2].astype(np.float64)

        # Normalize by temporal mean (removes DC component, keeps AC variation)
        R_norm = R / (R.mean() + 1e-6)
        G_norm = G / (G.mean() + 1e-6)
        B_norm = B / (B.mean() + 1e-6)

        # CHROM chrominance signals
        S1 = 3.0 * R_norm - 2.0 * G_norm
        S2 = 1.5 * R_norm + G_norm - 1.5 * B_norm

        # Adaptive combination
        std_S1 = np.std(S1) + 1e-10
        std_S2 = np.std(S2) + 1e-10
        alpha = std_S1 / std_S2

        pulse_raw = S1 - alpha * S2

        # Detrend (remove slow baseline drift using polynomial fit)
        pulse_detrended = self._detrend(pulse_raw, order=5)

        # Bandpass filter to cardiac frequency range
        pulse_filtered = self._bandpass_filter(
            pulse_detrended, fps,
            low_freq=self.CARDIAC_LOW,
            high_freq=self.CARDIAC_HIGH,
        )

        if pulse_filtered is None:
            return None, {"chrom_alpha": round(alpha, 4)}

        # Also extract simple green channel signal for comparison
        green_signal = self._bandpass_filter(
            self._detrend(G_norm, order=5),
            fps, self.CARDIAC_LOW, self.CARDIAC_HIGH,
        )

        metrics = {
            "chrom_alpha": round(alpha, 4),
            "raw_signal_std": round(float(np.std(pulse_raw)), 6),
            "filtered_signal_std": round(float(np.std(pulse_filtered)), 6),
            "green_channel_std": round(float(np.std(green_signal)), 6) if green_signal is not None else 0,
            "signal_length": len(pulse_filtered),
        }

        return pulse_filtered, metrics

    def _detrend(self, signal: np.ndarray, order: int = 5) -> np.ndarray:
        """Remove slow baseline drift using polynomial detrending."""
        n = len(signal)
        x = np.arange(n)
        try:
            coeffs = np.polyfit(x, signal, order)
            trend = np.polyval(coeffs, x)
            return signal - trend
        except Exception:
            return signal - np.mean(signal)

    def _bandpass_filter(
        self, signal: np.ndarray, fps: float,
        low_freq: float, high_freq: float,
        order: int = 4,
    ) -> Optional[np.ndarray]:
        """
        Apply Butterworth bandpass filter.

        Args:
            signal: 1D signal
            fps: sampling rate
            low_freq: low cutoff (Hz)
            high_freq: high cutoff (Hz)
            order: filter order
        """
        try:
            from scipy.signal import butter, filtfilt

            nyquist = fps / 2.0
            if high_freq >= nyquist:
                high_freq = nyquist * 0.95

            if low_freq >= high_freq:
                return signal

            low = low_freq / nyquist
            high = high_freq / nyquist

            b, a = butter(order, [low, high], btype='band')
            filtered = filtfilt(b, a, signal)
            return filtered

        except ImportError:
            # Fallback: simple moving average subtraction (crude bandpass)
            kernel_low = max(3, int(fps / high_freq))
            kernel_high = max(kernel_low + 2, int(fps / low_freq))

            smoothed_low = np.convolve(signal, np.ones(kernel_low) / kernel_low, mode='same')
            smoothed_high = np.convolve(signal, np.ones(kernel_high) / kernel_high, mode='same')

            return smoothed_low - smoothed_high

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Spectral Analysis
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_pulse_spectrum(
        self, pulse_signal: np.ndarray, fps: float
    ) -> Dict[str, Any]:
        """
        Analyze the power spectral density of the pulse signal.

        Computes:
        - Dominant frequency and corresponding BPM
        - Signal-to-Noise Ratio (SNR) in dB
        - Peak prominence (how much the peak stands above the noise floor)
        - Spectral entropy (flat = noise, peaked = real pulse)
        """
        try:
            from scipy.signal import welch
            nperseg = min(len(pulse_signal), int(fps * 8))  # 8-second windows
            freqs, psd = welch(pulse_signal, fs=fps, nperseg=nperseg,
                               noverlap=nperseg // 2)
        except ImportError:
            # Manual FFT-based PSD
            n = len(pulse_signal)
            windowed = pulse_signal * np.hanning(n)
            fft_result = np.fft.rfft(windowed)
            psd = np.abs(fft_result) ** 2 / n
            freqs = np.fft.rfftfreq(n, d=1.0 / fps)

        # Focus on cardiac frequency band
        cardiac_mask = (freqs >= self.CARDIAC_LOW) & (freqs <= self.CARDIAC_HIGH)
        resting_mask = (freqs >= self.RESTING_LOW) & (freqs <= self.RESTING_HIGH)

        if not np.any(cardiac_mask):
            return {
                "snr_db": 0.0,
                "peak_power_ratio": 0.0,
                "has_clear_peak": False,
                "estimated_bpm": None,
                "spectral_entropy": 1.0,
            }

        cardiac_psd = psd[cardiac_mask]
        cardiac_freqs = freqs[cardiac_mask]

        # Find dominant peak in cardiac band
        peak_idx = np.argmax(cardiac_psd)
        peak_freq = cardiac_freqs[peak_idx]
        peak_power = cardiac_psd[peak_idx]
        estimated_bpm = peak_freq * 60.0

        # SNR: peak power vs average noise power
        # Exclude a ±0.15 Hz window around the peak for noise estimation
        peak_band = np.abs(cardiac_freqs - peak_freq) < 0.15
        noise_psd = cardiac_psd[~peak_band]
        noise_mean = noise_psd.mean() if len(noise_psd) > 0 else 1e-10

        snr_linear = peak_power / (noise_mean + 1e-10)
        snr_db = 10.0 * np.log10(snr_linear + 1e-10)

        # Peak power ratio (peak / total power in cardiac band)
        total_cardiac_power = cardiac_psd.sum() + 1e-10
        peak_power_ratio = peak_power / total_cardiac_power

        # Spectral entropy (normalized) — flat spectrum = high entropy = noise
        cardiac_psd_norm = cardiac_psd / (cardiac_psd.sum() + 1e-10)
        spectral_entropy = -np.sum(
            cardiac_psd_norm * np.log2(cardiac_psd_norm + 1e-10)
        )
        max_entropy = np.log2(len(cardiac_psd))
        norm_entropy = spectral_entropy / (max_entropy + 1e-10)

        # Is there a clear peak? (peak must be significantly above noise)
        has_clear_peak = snr_db > 1.5 and peak_power_ratio > 0.08

        # Check if BPM is in physiologically plausible range
        bpm_plausible = 40 < estimated_bpm < 200

        return {
            "snr_db": float(snr_db),
            "peak_power_ratio": float(peak_power_ratio),
            "has_clear_peak": has_clear_peak and bpm_plausible,
            "estimated_bpm": float(estimated_bpm) if bpm_plausible else None,
            "peak_freq_hz": float(peak_freq),
            "noise_floor_power": float(noise_mean),
            "spectral_entropy": float(norm_entropy),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Multi-ROI Validation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _multi_roi_validation(
        self, face_signals: np.ndarray, fps: float
    ) -> float:
        """
        Validate pulse signal consistency across different face ROIs.

        Real pulse: forehead and cheeks should show the same dominant frequency
        (they share the same blood supply). Disagreement suggests noise/AI.

        Returns inconsistency score (0 = perfectly consistent, 1 = completely different).
        """
        if face_signals.shape[1] < 2:
            return 0.5

        dominant_freqs = []

        for roi_idx in range(face_signals.shape[1]):
            # Green channel for this ROI
            green = face_signals[:, roi_idx, 1].astype(np.float64)
            green_norm = green / (green.mean() + 1e-6)
            green_detrended = self._detrend(green_norm, order=5)
            green_filtered = self._bandpass_filter(
                green_detrended, fps, self.CARDIAC_LOW, self.CARDIAC_HIGH
            )

            if green_filtered is None:
                continue

            # Find dominant frequency
            try:
                from scipy.signal import welch
                nperseg = min(len(green_filtered), int(fps * 6))
                freqs, psd = welch(green_filtered, fs=fps, nperseg=nperseg)
            except ImportError:
                n = len(green_filtered)
                fft_r = np.fft.rfft(green_filtered * np.hanning(n))
                psd = np.abs(fft_r) ** 2 / n
                freqs = np.fft.rfftfreq(n, d=1.0 / fps)

            cardiac_mask = (freqs >= self.CARDIAC_LOW) & (freqs <= self.CARDIAC_HIGH)
            if np.any(cardiac_mask):
                cardiac_freqs = freqs[cardiac_mask]
                cardiac_psd = psd[cardiac_mask]
                peak_freq = cardiac_freqs[np.argmax(cardiac_psd)]
                dominant_freqs.append(peak_freq)

        if len(dominant_freqs) < 2:
            return 0.5

        # Compute pairwise frequency differences
        diffs = []
        for i in range(len(dominant_freqs)):
            for j in range(i + 1, len(dominant_freqs)):
                diffs.append(abs(dominant_freqs[i] - dominant_freqs[j]))

        avg_diff = float(np.mean(diffs))

        # Normalize: 0 Hz diff = perfect consistency, >0.5 Hz = very inconsistent
        inconsistency = np.clip(avg_diff / 0.5, 0, 1)

        return float(inconsistency)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Harmonic Analysis
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_harmonics(
        self, pulse_signal: np.ndarray, fps: float,
        spectral_result: Dict,
    ) -> float:
        """
        Check for harmonic structure in the pulse signal.

        Real cardiac signals have harmonics: if fundamental is at f0,
        there should be energy at 2*f0, 3*f0 (from the non-sinusoidal
        waveform of the blood volume pulse). AI-generated content
        will not have this harmonic structure.

        Returns harmonic_score (0 = no harmonics, 1 = strong harmonics).
        """
        peak_freq = spectral_result.get("peak_freq_hz")
        if peak_freq is None or peak_freq < self.CARDIAC_LOW:
            return 0.0

        # Compute FFT
        n = len(pulse_signal)
        windowed = pulse_signal * np.hanning(n)
        fft_result = np.abs(np.fft.rfft(windowed)) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / fps)

        if len(freqs) < 2:
            return 0.0

        freq_resolution = freqs[1] - freqs[0]

        # Find power at fundamental and harmonics
        def power_at_freq(target_freq, bandwidth=0.2):
            mask = np.abs(freqs - target_freq) < bandwidth
            if np.any(mask):
                return float(np.max(fft_result[mask]))
            return 0.0

        fundamental_power = power_at_freq(peak_freq)
        harmonic_2_power = power_at_freq(peak_freq * 2)
        harmonic_3_power = power_at_freq(peak_freq * 3)

        # Noise floor
        cardiac_mask = (freqs >= self.CARDIAC_LOW) & (freqs <= self.CARDIAC_HIGH * 3)
        noise_floor = float(np.median(fft_result[cardiac_mask])) + 1e-10

        # Harmonic ratios (relative to noise floor)
        h2_ratio = harmonic_2_power / noise_floor
        h3_ratio = harmonic_3_power / noise_floor

        # Score: harmonics should be present but weaker than fundamental
        # h2 typically 20-50% of fundamental power
        h2_score = 1.0 if h2_ratio > 2.0 else h2_ratio / 2.0
        h3_score = 1.0 if h3_ratio > 1.5 else h3_ratio / 1.5

        harmonic_score = (h2_score * 0.6 + h3_score * 0.4)

        return float(np.clip(harmonic_score, 0, 1))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Signal Quality Assessment
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _assess_signal_quality(
        self, pulse_signal: np.ndarray, face_signals: np.ndarray
    ) -> Dict[str, float]:
        """
        Assess the quality of the rPPG signal extraction.

        High motion → unreliable signal → can't distinguish AI from real
        Good quality → absence of pulse is strong AI evidence

        Metrics:
        - Signal stationarity (real pulse is quasi-stationary)
        - Motion artifact level (from ROI position changes)
        - Signal amplitude (too high = motion artifact, too low = noise)
        """
        # Signal stationarity: divide into segments, compare power
        n_segments = 4
        seg_len = len(pulse_signal) // n_segments
        segment_powers = []
        for i in range(n_segments):
            seg = pulse_signal[i * seg_len:(i + 1) * seg_len]
            segment_powers.append(float(np.var(seg)))

        power_cv = float(np.std(segment_powers) / (np.mean(segment_powers) + 1e-10))
        stationarity = max(0.0, 1.0 - power_cv)

        # Motion from ROI signal changes
        roi_green = face_signals[:, :, 1].mean(axis=1)  # average green across ROIs
        roi_diff = np.diff(roi_green)
        motion_level = float(np.std(roi_diff) / (np.mean(roi_green) + 1e-6))
        motion_contamination = min(1.0, motion_level * 10)

        # Overall quality
        signal_quality = stationarity * (1.0 - motion_contamination * 0.5)

        return {
            "signal_quality": float(np.clip(signal_quality, 0, 1)),
            "stationarity": round(stationarity, 4),
            "motion_contamination": round(float(motion_contamination), 4),
        }
