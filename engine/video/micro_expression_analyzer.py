"""
Physiological Micro-Expression Timing Analyzer
=================================================
Detects AI-generated video by analyzing the timing and dynamics of
involuntary facial micro-movements, especially eye blinks.

THEORETICAL FOUNDATION:
━━━━━━━━━━━━━━━━━━━━━━━
Human facial expressions follow precise physiological constraints:

EYE BLINKS:
  - Average rate: 15-20 blinks per minute (adults)
  - Duration: 100-400ms (typically 150-300ms)
  - Velocity profile: ASYMMETRIC — fast close (~75ms), slow open (~150ms)
  - Inter-blink interval: quasi-periodic with natural variability
  - Bilateral: both eyes blink simultaneously (with <5ms offset)

These properties are hardwired into the neuromuscular system and are
extremely difficult for AI generators to replicate because:
  - AI doesn't model the orbicularis oculi muscle dynamics
  - AI blinks tend to be symmetric (same speed close and open)
  - AI blink timing is often too regular or too irregular
  - AI may generate asymmetric (one-eye) blinks unnaturally
  - AI blink durations may fall outside the physiological range

MICRO-EXPRESSIONS:
  - Duration: 40-200ms (involuntary, cannot be consciously controlled)
  - Follow FACS (Facial Action Coding System) patterns
  - Smooth onset/offset with specific acceleration profiles
  - AI: may produce unnatural timing, abrupt onset, or impossible combinations

ANALYSIS LAYERS:
━━━━━━━━━━━━━━━━
1. Blink Detection & Timing
   - Track Eye Aspect Ratio (EAR) across frames
   - Detect blinks as EAR dips
   - Measure blink duration, rate, and velocity profile

2. Blink Dynamics Analysis
   - Close/open velocity asymmetry (fast close, slow open = real)
   - Duration distribution (100-400ms = real)
   - Blink completeness (real blinks fully close)

3. Blink Rhythm Analysis
   - Inter-blink interval distribution
   - Regularity (too regular = AI, natural = slight variability)
   - Rate plausibility (15-20/min at rest)

4. Micro-Movement Analysis
   - Track small facial movements between blinks
   - Real faces have continuous micro-movements (never truly still)
   - AI faces may be unnaturally still between expressions

5. Bilateral Symmetry
   - Both eyes should blink together
   - Measure temporal offset between left/right eye closure
   - AI may produce asynchronous blinks
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, List, Optional


class MicroExpressionAnalyzer:
    """
    Analyzes physiological micro-expression timing (especially blinks)
    to distinguish real human video from AI-generated video.
    """

    # Physiological constants
    MIN_BLINK_DURATION_MS = 80
    MAX_BLINK_DURATION_MS = 500
    TYPICAL_BLINK_RATE = 17.0  # blinks per minute (adult resting)
    MIN_BLINK_RATE = 8.0
    MAX_BLINK_RATE = 30.0

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        # Face and eye cascades
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        self.left_eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_lefteye_2splits.xml"
        )
        self.right_eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_righteye_2splits.xml"
        )

    def analyze(self, video_path: str, max_seconds: float = 15.0) -> Dict[str, Any]:
        """
        Run full micro-expression analysis.

        Returns dict with ai_probability and detailed blink/expression metrics.
        """
        # Extract eye signals
        eye_data = self._extract_eye_signals(video_path, max_seconds)

        if eye_data is None:
            return {
                "ai_probability": 0.5,
                "metrics": {"error": "Could not extract eye data (no face/eyes detected)"},
            }

        ear_signal = eye_data["ear_signal"]
        left_ear = eye_data["left_ear"]
        right_ear = eye_data["right_ear"]
        fps = eye_data["fps"]
        face_movement = eye_data["face_movement"]

        if len(ear_signal) < int(fps * 2):
            return {
                "ai_probability": 0.5,
                "metrics": {"error": "Insufficient eye tracking data",
                            "frames": len(ear_signal)},
            }

        # Layer 1: Blink Detection & Timing
        blinks = self._detect_blinks(ear_signal, fps)
        blink_score, blink_metrics = self._analyze_blink_timing(blinks, fps, len(ear_signal))

        # Layer 2: Blink Dynamics
        dynamics_score, dynamics_metrics = self._analyze_blink_dynamics(
            ear_signal, blinks, fps
        )

        # Layer 3: Blink Rhythm
        rhythm_score, rhythm_metrics = self._analyze_blink_rhythm(blinks, fps)

        # Layer 4: Micro-Movement Analysis
        micro_score, micro_metrics = self._analyze_micro_movements(
            ear_signal, face_movement, fps
        )

        # Layer 5: Bilateral Symmetry
        bilateral_score, bilateral_metrics = self._analyze_bilateral_symmetry(
            left_ear, right_ear, fps
        )

        # Weighted ensemble
        weights = {
            "blink_timing": (blink_score, 1.5),
            "blink_dynamics": (dynamics_score, 1.3),
            "blink_rhythm": (rhythm_score, 1.2),
            "micro_movements": (micro_score, 0.8),
            "bilateral_symmetry": (bilateral_score, 1.0),
        }

        total_w = sum(w for _, w in weights.values())
        ai_probability = sum(s * w for s, w in weights.values()) / total_w
        ai_probability = float(np.clip(ai_probability, 0, 1))

        if self.verbose:
            for name, (score, _) in weights.items():
                print(f"  [MicroExpr] {name}: {score:.4f}")

        return {
            "ai_probability": round(ai_probability, 4),
            "layer_scores": {
                name: round(score, 4) for name, (score, _) in weights.items()
            },
            "metrics": {
                "n_blinks": len(blinks),
                "duration_seconds": round(len(ear_signal) / fps, 1),
                "fps": round(fps, 1),
                "blink": blink_metrics,
                "dynamics": dynamics_metrics,
                "rhythm": rhythm_metrics,
                "micro": micro_metrics,
                "bilateral": bilateral_metrics,
            },
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Eye Signal Extraction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _compute_eye_aspect_ratio(self, eye_roi: np.ndarray) -> float:
        """
        Compute a proxy for Eye Aspect Ratio (EAR) from an eye ROI.

        Without facial landmarks, we approximate EAR using:
        - Vertical gradient energy (eye opening) / horizontal gradient energy (eye width)
        - When eyes are open: high vertical gradients (upper/lower lid edges)
        - When eyes are closed: low vertical gradients (flat eyelid)
        """
        if eye_roi.size == 0 or eye_roi.shape[0] < 3 or eye_roi.shape[1] < 3:
            return 0.5

        # Convert to grayscale if needed
        if len(eye_roi.shape) == 3:
            gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = eye_roi

        # Compute gradients
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

        vertical_energy = float(np.mean(np.abs(gy)))
        horizontal_energy = float(np.mean(np.abs(gx))) + 1e-6

        # EAR proxy: vertical/horizontal gradient ratio
        # Open eyes: high ratio (strong horizontal lid edges)
        # Closed eyes: low ratio (flat)
        ear_proxy = vertical_energy / horizontal_energy

        return ear_proxy

    def _extract_eye_signals(
        self, video_path: str, max_seconds: float
    ) -> Optional[Dict]:
        """
        Extract eye aspect ratio signals from video.

        Returns dict with ear_signal, left_ear, right_ear, fps, face_movement.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = int(min(total, max_seconds * fps))

        ear_signals = []
        left_ears = []
        right_ears = []
        face_movements = []

        stable_face = None
        prev_face_center = None
        ema_alpha = 0.3

        for frame_idx in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect face
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )

            if len(faces) > 0:
                bbox = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = bbox

                if stable_face is not None:
                    stable_face = (
                        int(stable_face[0] * (1 - ema_alpha) + x * ema_alpha),
                        int(stable_face[1] * (1 - ema_alpha) + y * ema_alpha),
                        int(stable_face[2] * (1 - ema_alpha) + w * ema_alpha),
                        int(stable_face[3] * (1 - ema_alpha) + h * ema_alpha),
                    )
                else:
                    stable_face = (x, y, w, h)

            if stable_face is None:
                continue

            x, y, w, h = stable_face
            fh, fw = frame.shape[:2]

            # Track face movement
            face_center = (x + w // 2, y + h // 2)
            if prev_face_center is not None:
                movement = np.sqrt(
                    (face_center[0] - prev_face_center[0]) ** 2 +
                    (face_center[1] - prev_face_center[1]) ** 2
                )
                face_movements.append(float(movement))
            prev_face_center = face_center

            # Extract eye regions from face
            # Upper half of face, split left/right
            eye_y1 = max(0, y + int(h * 0.15))
            eye_y2 = max(0, y + int(h * 0.45))

            left_x1 = max(0, x + int(w * 0.05))
            left_x2 = max(0, x + int(w * 0.45))
            right_x1 = min(fw, x + int(w * 0.55))
            right_x2 = min(fw, x + int(w * 0.95))

            left_eye_roi = gray[eye_y1:eye_y2, left_x1:left_x2]
            right_eye_roi = gray[eye_y1:eye_y2, right_x1:right_x2]

            # Compute EAR for each eye
            left_ear = self._compute_eye_aspect_ratio(left_eye_roi)
            right_ear = self._compute_eye_aspect_ratio(right_eye_roi)
            avg_ear = (left_ear + right_ear) / 2.0

            ear_signals.append(avg_ear)
            left_ears.append(left_ear)
            right_ears.append(right_ear)

        cap.release()

        if len(ear_signals) < int(fps * 2):
            return None

        return {
            "ear_signal": np.array(ear_signals),
            "left_ear": np.array(left_ears),
            "right_ear": np.array(right_ears),
            "fps": fps,
            "face_movement": np.array(face_movements) if face_movements else np.zeros(1),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Blink Detection
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _detect_blinks(
        self, ear_signal: np.ndarray, fps: float
    ) -> List[Dict]:
        """
        Detect blinks as dips in the EAR signal.

        Returns list of blink dicts with:
          - start_frame, end_frame, peak_frame (minimum EAR)
          - duration_ms
          - ear_drop (how much EAR decreased)
        """
        # Smooth the signal slightly to reduce noise
        kernel_size = max(3, int(fps * 0.02))
        if kernel_size % 2 == 0:
            kernel_size += 1
        ear_smooth = np.convolve(
            ear_signal,
            np.ones(kernel_size) / kernel_size,
            mode='same',
        )

        # Adaptive threshold: blink is when EAR drops below moving baseline
        window = int(fps * 1.0)  # 1-second window
        blinks = []
        min_blink_frames = max(2, int(fps * self.MIN_BLINK_DURATION_MS / 1000))
        max_blink_frames = int(fps * self.MAX_BLINK_DURATION_MS / 1000)

        # Compute local baseline (moving median)
        baseline = np.array([
            np.median(ear_smooth[max(0, i - window):i + window])
            for i in range(len(ear_smooth))
        ])

        # Blink threshold: EAR drops below 70% of baseline
        threshold_ratio = 0.70
        below_threshold = ear_smooth < (baseline * threshold_ratio)

        # Find contiguous regions below threshold (blink events)
        in_blink = False
        blink_start = 0

        for i in range(len(below_threshold)):
            if below_threshold[i] and not in_blink:
                blink_start = i
                in_blink = True
            elif not below_threshold[i] and in_blink:
                blink_end = i
                duration = blink_end - blink_start

                if min_blink_frames <= duration <= max_blink_frames:
                    peak_frame = blink_start + np.argmin(ear_smooth[blink_start:blink_end])
                    ear_drop = float(baseline[peak_frame] - ear_smooth[peak_frame])

                    blinks.append({
                        "start": blink_start,
                        "end": blink_end,
                        "peak": peak_frame,
                        "duration_frames": duration,
                        "duration_ms": duration / fps * 1000,
                        "ear_drop": ear_drop,
                    })

                in_blink = False

        return blinks

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 1: Blink Timing Analysis
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_blink_timing(
        self, blinks: List[Dict], fps: float, n_frames: int
    ) -> Tuple[float, Dict]:
        """
        Analyze blink rate and duration plausibility.

        Real: 15-20 blinks/min, 100-400ms duration
        AI: abnormal rate, abnormal duration
        """
        duration_seconds = n_frames / fps

        if duration_seconds < 3:
            return 0.5, {"error": "video too short for blink analysis"}

        blink_rate = len(blinks) / (duration_seconds / 60.0)  # blinks per minute

        # Rate plausibility
        if self.MIN_BLINK_RATE <= blink_rate <= self.MAX_BLINK_RATE:
            rate_score = 0.1  # Normal rate → likely real
        elif blink_rate < 3:
            rate_score = 0.8  # Very few blinks → suspicious (AI may not generate them)
        elif blink_rate == 0:
            rate_score = 0.95  # No blinks in >5 seconds → very suspicious
        else:
            # Abnormal rate
            rate_score = 0.5 + min(0.3, abs(blink_rate - self.TYPICAL_BLINK_RATE) / 30)

        # Duration plausibility
        if blinks:
            durations = [b["duration_ms"] for b in blinks]
            avg_duration = float(np.mean(durations))
            std_duration = float(np.std(durations))

            # Check if durations are in physiological range
            in_range = sum(1 for d in durations
                         if self.MIN_BLINK_DURATION_MS <= d <= self.MAX_BLINK_DURATION_MS)
            range_ratio = in_range / len(durations)

            duration_score = np.clip((1.0 - range_ratio) * 1.5, 0, 1)
        else:
            avg_duration = 0
            std_duration = 0
            range_ratio = 0
            duration_score = 0.7 if duration_seconds > 5 else 0.5

        combined = rate_score * 0.55 + duration_score * 0.45

        metrics = {
            "blink_rate_per_min": round(blink_rate, 1),
            "avg_duration_ms": round(avg_duration, 1),
            "std_duration_ms": round(std_duration, 1),
            "in_range_ratio": round(range_ratio, 4) if blinks else 0,
            "rate_plausible": self.MIN_BLINK_RATE <= blink_rate <= self.MAX_BLINK_RATE,
        }

        return float(np.clip(combined, 0, 1)), metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 2: Blink Dynamics
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_blink_dynamics(
        self, ear_signal: np.ndarray, blinks: List[Dict], fps: float
    ) -> Tuple[float, Dict]:
        """
        Analyze the velocity profile of blinks.

        Real blinks have ASYMMETRIC velocity:
        - Closing phase: ~75ms (fast) — orbicularis oculi contracts rapidly
        - Opening phase: ~150ms (slow) — levator palpebrae relaxes gradually
        - Ratio close/open duration: typically 0.3-0.6

        AI blinks tend to be SYMMETRIC (same speed close and open)
        because generators don't model muscle dynamics.
        """
        if len(blinks) < 2:
            return 0.5, {"error": "too few blinks for dynamics analysis"}

        asymmetry_ratios = []
        close_velocities = []
        open_velocities = []

        for blink in blinks:
            start = blink["start"]
            peak = blink["peak"]
            end = blink["end"]

            close_duration = peak - start
            open_duration = end - peak

            if close_duration < 1 or open_duration < 1:
                continue

            # Asymmetry ratio: close_duration / open_duration
            # Real: 0.3-0.6 (fast close, slow open)
            asymmetry = close_duration / open_duration
            asymmetry_ratios.append(asymmetry)

            # Closing velocity (EAR drop per frame)
            close_vel = blink["ear_drop"] / close_duration
            close_velocities.append(close_vel)

            # Opening velocity (EAR rise per frame)
            ear_at_end = ear_signal[min(end, len(ear_signal) - 1)]
            ear_at_peak = ear_signal[min(peak, len(ear_signal) - 1)]
            open_vel = (ear_at_end - ear_at_peak) / open_duration
            open_velocities.append(open_vel)

        if not asymmetry_ratios:
            return 0.5, {"error": "no valid blink dynamics"}

        avg_asymmetry = float(np.mean(asymmetry_ratios))
        std_asymmetry = float(np.std(asymmetry_ratios))

        # Check if asymmetry is in physiological range (0.3-0.6)
        # Symmetric blinks (ratio ~1.0) are suspicious
        asym_plausible = 0.2 < avg_asymmetry < 0.7

        # Score
        # Asymmetry near 1.0 (symmetric) = AI
        # Asymmetry 0.3-0.6 = real
        if 0.25 <= avg_asymmetry <= 0.65:
            asym_score = 0.1  # Physiological range
        elif 0.7 <= avg_asymmetry <= 1.3:
            asym_score = 0.7  # Too symmetric — AI-like
        else:
            asym_score = 0.5  # Unusual but inconclusive

        # High asymmetry variance = inconsistent dynamics = AI
        var_score = np.clip((std_asymmetry - 0.10) / 0.30, 0, 0.5)

        # Velocity ratio check
        if close_velocities and open_velocities:
            avg_close_vel = float(np.mean(close_velocities))
            avg_open_vel = float(np.mean(open_velocities))
            vel_ratio = avg_close_vel / (avg_open_vel + 1e-8)
            # Real: close is 1.5-3x faster than open
            if 1.3 <= vel_ratio <= 3.5:
                vel_score = 0.1
            else:
                vel_score = 0.6
        else:
            vel_ratio = 0
            vel_score = 0.5

        combined = asym_score * 0.45 + var_score * 0.25 + vel_score * 0.30

        metrics = {
            "avg_asymmetry_ratio": round(avg_asymmetry, 4),
            "std_asymmetry_ratio": round(std_asymmetry, 4),
            "asymmetry_plausible": asym_plausible,
            "close_open_velocity_ratio": round(vel_ratio, 4),
            "n_valid_blinks": len(asymmetry_ratios),
        }

        return float(np.clip(combined, 0, 1)), metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 3: Blink Rhythm
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_blink_rhythm(
        self, blinks: List[Dict], fps: float
    ) -> Tuple[float, Dict]:
        """
        Analyze the rhythm/regularity of blink timing.

        Real blink timing: quasi-periodic with natural variability
        - Inter-blink interval (IBI): 2-6 seconds typically
        - Coefficient of variation of IBI: 0.3-0.7 (moderate variability)

        AI blinks may be:
        - Too regular (CV < 0.2 — metronomic)
        - Too irregular (CV > 0.9 — random)
        - Absent entirely
        """
        if len(blinks) < 3:
            return 0.5, {"error": "too few blinks for rhythm analysis",
                         "n_blinks": len(blinks)}

        # Inter-blink intervals
        ibis = []
        for i in range(len(blinks) - 1):
            ibi = (blinks[i + 1]["start"] - blinks[i]["end"]) / fps
            ibis.append(ibi)

        ibis = np.array(ibis)

        avg_ibi = float(np.mean(ibis))
        std_ibi = float(np.std(ibis))
        cv_ibi = std_ibi / (avg_ibi + 1e-6)

        # Check regularity
        # Too regular (CV < 0.15): suspicious — real blinks have natural variability
        # Natural range (CV 0.25-0.65): normal physiological variability
        # Too irregular (CV > 0.85): suspicious — may be random noise, not real blinks
        if 0.20 <= cv_ibi <= 0.70:
            regularity_score = 0.1  # Natural variability
        elif cv_ibi < 0.15:
            regularity_score = 0.6  # Too regular (metronomic) — AI
        elif cv_ibi > 0.90:
            regularity_score = 0.7  # Too irregular — AI
        else:
            regularity_score = 0.3  # Borderline

        # IBI plausibility: 1-8 seconds for most blinks
        in_range = sum(1 for ibi in ibis if 1.0 <= ibi <= 8.0)
        ibi_range_ratio = in_range / len(ibis)
        ibi_score = np.clip((1.0 - ibi_range_ratio), 0, 0.6)

        combined = regularity_score * 0.60 + ibi_score * 0.40

        metrics = {
            "avg_ibi_seconds": round(avg_ibi, 2),
            "std_ibi_seconds": round(std_ibi, 2),
            "cv_ibi": round(cv_ibi, 4),
            "ibi_in_range_ratio": round(ibi_range_ratio, 4),
            "regularity": "too_regular" if cv_ibi < 0.15 else
                         "natural" if cv_ibi < 0.70 else "too_irregular",
        }

        return float(np.clip(combined, 0, 1)), metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 4: Micro-Movement Analysis
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_micro_movements(
        self, ear_signal: np.ndarray, face_movement: np.ndarray,
        fps: float,
    ) -> Tuple[float, Dict]:
        """
        Analyze continuous micro-movements between blinks.

        Real human faces are NEVER perfectly still. Even when "resting",
        there are continuous micro-movements from:
        - Breathing (slight facial displacement)
        - Muscle tone fluctuations
        - Eye micro-saccades
        - Emotional processing micro-expressions

        AI-generated faces may be unnaturally still between deliberate
        expressions, or may have unnaturally uniform micro-movement.
        """
        # Micro-movements in EAR signal (eye region)
        ear_diffs = np.abs(np.diff(ear_signal))

        # Between-blink periods: remove blink events (large EAR changes)
        large_change_threshold = np.percentile(ear_diffs, 90)
        micro_diffs = ear_diffs[ear_diffs < large_change_threshold]

        if len(micro_diffs) < 10:
            return 0.5, {"error": "insufficient data"}

        # Micro-movement statistics
        micro_mean = float(np.mean(micro_diffs))
        micro_std = float(np.std(micro_diffs))
        micro_cv = micro_std / (micro_mean + 1e-8)

        # Check for "dead zones" — periods of near-zero movement
        stillness_threshold = np.percentile(micro_diffs, 10)
        n_very_still = np.sum(micro_diffs < stillness_threshold * 0.5)
        still_ratio = float(n_very_still / len(micro_diffs))

        # Face movement analysis
        if len(face_movement) > 5:
            face_micro = face_movement[face_movement < np.percentile(face_movement, 80)]
            face_micro_std = float(np.std(face_micro)) if len(face_micro) > 2 else 0
            face_micro_mean = float(np.mean(face_micro)) if len(face_micro) > 2 else 0
        else:
            face_micro_std = 0
            face_micro_mean = 0

        # Score
        # Real faces have moderate micro-movement (not too still, not too jittery)
        # micro_mean: real ~0.005-0.02, AI too_low (<0.003) or too_high (>0.03)
        if micro_mean < 0.002:
            movement_score = 0.7  # Unnaturally still
        elif micro_mean > 0.04:
            movement_score = 0.6  # Unnaturally jittery
        else:
            movement_score = 0.15  # Normal micro-movement

        # still_ratio: real ~0.05-0.20, AI ~0.30-0.60+ (too still)
        still_score = np.clip((still_ratio - 0.15) / 0.30, 0, 0.5)

        # micro_cv: real ~0.4-0.8, AI too_low (<0.3 = uniform jitter) or too_high
        if micro_cv < 0.25:
            cv_score = 0.5  # Suspiciously uniform
        elif micro_cv > 1.0:
            cv_score = 0.4  # Too variable
        else:
            cv_score = 0.1  # Natural variability

        combined = movement_score * 0.40 + still_score * 0.30 + cv_score * 0.30

        metrics = {
            "micro_movement_mean": round(micro_mean, 6),
            "micro_movement_std": round(micro_std, 6),
            "micro_movement_cv": round(micro_cv, 4),
            "stillness_ratio": round(still_ratio, 4),
            "face_micro_movement": round(face_micro_mean, 4),
        }

        return float(np.clip(combined, 0, 1)), metrics

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Layer 5: Bilateral Symmetry
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _analyze_bilateral_symmetry(
        self, left_ear: np.ndarray, right_ear: np.ndarray,
        fps: float,
    ) -> Tuple[float, Dict]:
        """
        Check bilateral symmetry of eye movements.

        Real humans: both eyes blink simultaneously (within 5ms offset).
        Left and right EAR signals should be highly correlated.
        AI: may generate asymmetric eye movements, lower correlation.
        """
        if len(left_ear) < 10:
            return 0.5, {"error": "insufficient data"}

        # Correlation between left and right eye signals
        correlation = float(np.corrcoef(left_ear, right_ear)[0, 1])
        if np.isnan(correlation):
            correlation = 0.5

        # Difference signal (should be near zero for real)
        diff_signal = np.abs(left_ear - right_ear)
        avg_diff = float(np.mean(diff_signal))
        max_diff = float(np.max(diff_signal))

        # Normalized difference (relative to signal magnitude)
        signal_mag = float(np.mean(np.abs(left_ear) + np.abs(right_ear))) / 2 + 1e-8
        norm_diff = avg_diff / signal_mag

        # Score
        # correlation: real ~0.85-0.98, AI ~0.5-0.85
        corr_score = np.clip((0.85 - correlation) / 0.30, 0, 1)

        # norm_diff: real ~0.02-0.08, AI ~0.08-0.25+
        diff_score = np.clip((norm_diff - 0.04) / 0.15, 0, 1)

        combined = corr_score * 0.55 + diff_score * 0.45

        metrics = {
            "bilateral_correlation": round(correlation, 4),
            "avg_lr_difference": round(avg_diff, 6),
            "max_lr_difference": round(max_diff, 6),
            "normalized_difference": round(norm_diff, 4),
        }

        return float(np.clip(combined, 0, 1)), metrics
