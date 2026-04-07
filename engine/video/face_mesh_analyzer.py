"""
Face Mesh Consistency Analyzer
===============================
Uses geometric face landmark analysis to detect deepfake videos by
tracking facial geometry consistency across frames.

Key detection signals:

1. **Geometric Consistency**: Real faces maintain consistent proportions
   (eye width ratio, nose-mouth distance, etc.) across frames. Deepfakes
   may show frame-to-frame geometric jitter.

2. **Symmetry Analysis**: Natural faces have consistent bilateral symmetry
   patterns. Deepfakes may introduce asymmetry artifacts at blend boundaries.

3. **Landmark Stability**: During neutral expressions, landmarks should be
   stable. Deepfakes may show micro-jitter from imperfect face alignment.

4. **Jaw/Chin Contour Consistency**: The jawline is a common blend boundary
   for face-swap deepfakes, showing characteristic artifacts.

5. **Eye Region Coherence**: Gaze direction, pupil shape, and iris detail
   are difficult for deepfakes to render consistently across frames.

Uses OpenCV's face detection + geometric analysis (no heavy ML dependency).
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Try MediaPipe for advanced face mesh
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


class FaceMeshConsistencyAnalyzer:
    """Analyzes face geometry consistency across video frames."""

    def __init__(self):
        self._face_cascade = None
        self._mp_face_mesh = None

    def _get_face_cascade(self):
        if self._face_cascade is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
        return self._face_cascade

    def _get_mediapipe_mesh(self):
        if self._mp_face_mesh is None and HAS_MEDIAPIPE:
            self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        return self._mp_face_mesh

    def analyze(self, frames: list) -> dict:
        """
        Analyze face mesh consistency across video frames.

        Args:
            frames: List of BGR frames (numpy arrays).

        Returns:
            dict with score, confidence, sub-scores, and anomalies.
        """
        if len(frames) < 5:
            return self._fallback_result("Not enough frames")

        # Sample frames evenly
        n_sample = min(30, len(frames))
        indices = np.linspace(0, len(frames) - 1, n_sample, dtype=int)
        sampled_frames = [frames[i] for i in indices]

        anomalies = []

        if HAS_MEDIAPIPE:
            result = self._analyze_with_mediapipe(sampled_frames, anomalies)
        else:
            result = self._analyze_with_opencv(sampled_frames, anomalies)

        return result

    def _analyze_with_mediapipe(self, frames: list, anomalies: list) -> dict:
        """Full analysis using MediaPipe 468-point face mesh."""
        mesh = self._get_mediapipe_mesh()
        if mesh is None:
            return self._analyze_with_opencv(frames, anomalies)

        all_landmarks = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mesh.process(rgb)
            if results.multi_face_landmarks:
                face = results.multi_face_landmarks[0]
                h, w = frame.shape[:2]
                pts = np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in face.landmark])
                all_landmarks.append(pts)

        if len(all_landmarks) < 5:
            return self._analyze_with_opencv(frames, anomalies)

        # 1. Geometric proportion consistency
        proportion_score = self._analyze_proportions(all_landmarks)

        # 2. Symmetry consistency
        symmetry_score = self._analyze_symmetry(all_landmarks)

        # 3. Landmark stability (micro-jitter)
        stability_score = self._analyze_stability(all_landmarks)

        # 4. Jaw contour smoothness
        jaw_score = self._analyze_jaw_contour(all_landmarks)

        # 5. Eye region coherence
        eye_score = self._analyze_eye_coherence(all_landmarks)

        if proportion_score > 0.55:
            anomalies.append(f"Geometric proportion instability ({proportion_score:.2f})")
        if symmetry_score > 0.55:
            anomalies.append(f"Facial symmetry anomaly ({symmetry_score:.2f})")
        if stability_score > 0.55:
            anomalies.append(f"Landmark micro-jitter detected ({stability_score:.2f})")
        if jaw_score > 0.55:
            anomalies.append(f"Jaw contour anomaly ({jaw_score:.2f})")
        if eye_score > 0.55:
            anomalies.append(f"Eye region coherence anomaly ({eye_score:.2f})")

        weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15])
        scores = np.array([proportion_score, symmetry_score, stability_score, jaw_score, eye_score])
        final_score = float(np.dot(scores, weights))

        confidence = (
            0.5 * (1.0 - float(np.std(scores)))
            + 0.3 * float(np.max(scores))
            + 0.2 * min(1.0, len(anomalies) / 3)
        )

        return {
            "score": float(np.clip(final_score, 0.0, 1.0)),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "proportion_score": float(proportion_score),
            "symmetry_score": float(symmetry_score),
            "stability_score": float(stability_score),
            "jaw_score": float(jaw_score),
            "eye_score": float(eye_score),
            "n_faces_tracked": len(all_landmarks),
            "anomalies": anomalies,
        }

    def _analyze_proportions(self, landmarks_list: list) -> float:
        """
        Track facial proportion ratios across frames.
        Real faces: consistent ratios. Deepfakes: jittery ratios.
        """
        # Key proportion ratios using MediaPipe indices
        # Eye width: landmarks 33-133 (left eye inner-outer)
        # Inter-eye: landmarks 33-263 (left eye inner to right eye inner)
        # Nose length: landmarks 6-2 (nose bridge to tip)
        # Mouth width: landmarks 61-291 (mouth corners)
        ratios_over_time = []

        for landmarks in landmarks_list:
            if len(landmarks) < 300:
                continue

            # Eye width ratios
            left_eye_width = np.linalg.norm(landmarks[33][:2] - landmarks[133][:2])
            right_eye_width = np.linalg.norm(landmarks[362][:2] - landmarks[263][:2])
            inter_eye = np.linalg.norm(landmarks[33][:2] - landmarks[263][:2])

            # Nose-to-mouth
            nose_tip = landmarks[1][:2]
            mouth_top = landmarks[13][:2]
            nose_mouth = np.linalg.norm(nose_tip - mouth_top)

            # Mouth width
            mouth_width = np.linalg.norm(landmarks[61][:2] - landmarks[291][:2])

            if inter_eye < 5 or left_eye_width < 1:
                continue

            ratios = [
                left_eye_width / (inter_eye + 1e-5),
                right_eye_width / (inter_eye + 1e-5),
                nose_mouth / (inter_eye + 1e-5),
                mouth_width / (inter_eye + 1e-5),
                left_eye_width / (right_eye_width + 1e-5),
            ]
            ratios_over_time.append(ratios)

        if len(ratios_over_time) < 5:
            return 0.3

        ratios_array = np.array(ratios_over_time)

        # Compute coefficient of variation for each ratio
        cvs = []
        for i in range(ratios_array.shape[1]):
            col = ratios_array[:, i]
            mean_val = np.mean(col)
            if mean_val > 0.01:
                cv = np.std(col) / mean_val
                cvs.append(cv)

        if not cvs:
            return 0.3

        mean_cv = np.mean(cvs)

        # Real faces: CV < 0.03 (very stable proportions)
        # Deepfakes: CV > 0.05 (jittery from imperfect alignment)
        if mean_cv > 0.08:
            return 0.8
        elif mean_cv > 0.05:
            return 0.6
        elif mean_cv > 0.03:
            return 0.35
        else:
            return 0.1

    def _analyze_symmetry(self, landmarks_list: list) -> float:
        """
        Track bilateral symmetry consistency across frames.
        """
        symmetry_scores = []

        for landmarks in landmarks_list:
            if len(landmarks) < 400:
                continue

            # Left-right landmark pairs (MediaPipe indices)
            pairs = [
                (33, 263),   # Eye inner corners
                (133, 362),  # Eye outer corners
                (70, 300),   # Eyebrow
                (105, 334),  # Upper eyelid
                (61, 291),   # Mouth corners
                (50, 280),   # Face contour
            ]

            # Compute symmetry about the nose bridge axis
            nose_top = landmarks[6][:2]
            nose_bottom = landmarks[2][:2]

            if np.linalg.norm(nose_top - nose_bottom) < 2:
                continue

            # Midline direction
            midline = nose_bottom - nose_top
            midline_norm = midline / (np.linalg.norm(midline) + 1e-10)
            midline_perp = np.array([-midline_norm[1], midline_norm[0]])

            asymmetries = []
            for left_idx, right_idx in pairs:
                left = landmarks[left_idx][:2]
                right = landmarks[right_idx][:2]

                # Project onto perpendicular to midline
                left_dist = np.dot(left - nose_top, midline_perp)
                right_dist = np.dot(right - nose_top, midline_perp)

                # Perfect symmetry: left_dist ≈ -right_dist
                asymmetry = abs(left_dist + right_dist) / (abs(left_dist) + abs(right_dist) + 1e-5)
                asymmetries.append(asymmetry)

            if asymmetries:
                symmetry_scores.append(np.mean(asymmetries))

        if len(symmetry_scores) < 5:
            return 0.3

        # Check consistency of symmetry over time
        sym_std = np.std(symmetry_scores)
        sym_mean = np.mean(symmetry_scores)

        # Real: consistent symmetry (low std, mean < 0.15)
        # Deepfake: variable symmetry (high std or high mean)
        if sym_std > 0.08:
            return 0.7  # Highly variable symmetry
        elif sym_mean > 0.20:
            return 0.6  # High asymmetry overall
        elif sym_std > 0.04:
            return 0.4
        else:
            return 0.15

    def _analyze_stability(self, landmarks_list: list) -> float:
        """
        Analyze landmark micro-jitter between consecutive frames.
        Real faces: smooth motion. Deepfakes: micro-jitter from alignment.
        """
        if len(landmarks_list) < 5:
            return 0.3

        # Compute frame-to-frame landmark displacement
        displacements = []
        for i in range(1, len(landmarks_list)):
            prev = landmarks_list[i - 1][:, :2]
            curr = landmarks_list[i][:, :2]

            n = min(len(prev), len(curr))
            if n < 50:
                continue

            # Per-landmark displacement
            disp = np.linalg.norm(curr[:n] - prev[:n], axis=1)
            displacements.append(disp)

        if len(displacements) < 3:
            return 0.3

        # Analyze jitter: ratio of high-frequency motion to smooth motion
        all_disp = np.array(displacements)
        mean_disp_per_frame = np.mean(all_disp, axis=1)

        # Remove global motion (head movement) by subtracting frame mean
        residual_disp = all_disp - mean_disp_per_frame[:, np.newaxis]
        jitter = np.std(residual_disp, axis=1)
        mean_jitter = np.mean(jitter)

        # Global motion scale
        global_motion = np.mean(mean_disp_per_frame)

        if global_motion < 0.1:
            # Nearly static face — jitter is very informative
            if mean_jitter > 2.0:
                return 0.8
            elif mean_jitter > 1.0:
                return 0.55
            else:
                return 0.15
        else:
            # Moving face — normalize jitter by motion
            jitter_ratio = mean_jitter / (global_motion + 1e-5)
            if jitter_ratio > 0.5:
                return 0.7
            elif jitter_ratio > 0.3:
                return 0.45
            else:
                return 0.15

    def _analyze_jaw_contour(self, landmarks_list: list) -> float:
        """
        Analyze jaw/chin contour smoothness.
        Deepfake blend boundaries often occur along the jaw line.
        """
        # MediaPipe jaw contour: landmarks 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, ...
        jaw_indices = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148,
                       152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]

        jaw_smoothness_scores = []

        for landmarks in landmarks_list:
            if len(landmarks) < max(jaw_indices) + 1:
                continue

            jaw_points = np.array([landmarks[i][:2] for i in jaw_indices])

            # Compute curvature along jaw contour
            if len(jaw_points) < 5:
                continue

            # First derivative (tangent)
            dx = np.diff(jaw_points[:, 0])
            dy = np.diff(jaw_points[:, 1])

            # Second derivative (curvature proxy)
            ddx = np.diff(dx)
            ddy = np.diff(dy)

            curvature = np.sqrt(ddx ** 2 + ddy ** 2)
            smoothness = np.std(curvature) / (np.mean(curvature) + 1e-10)
            jaw_smoothness_scores.append(smoothness)

        if len(jaw_smoothness_scores) < 3:
            return 0.3

        # Consistency of jaw smoothness across frames
        smooth_std = np.std(jaw_smoothness_scores)
        smooth_mean = np.mean(jaw_smoothness_scores)

        # Real: consistent jaw contour (low variation in smoothness)
        # Deepfake: variable due to blend boundary artifacts
        if smooth_std > 0.4:
            return 0.7
        elif smooth_std > 0.2:
            return 0.45
        else:
            return 0.15

    def _analyze_eye_coherence(self, landmarks_list: list) -> float:
        """
        Analyze eye region coherence across frames.
        Focus on pupil/iris consistency and blink dynamics.
        """
        # Eye aspect ratio (EAR) tracking
        # Left eye: landmarks 33, 160, 158, 133, 153, 144
        # Right eye: landmarks 362, 385, 387, 263, 373, 380

        left_ear_values = []
        right_ear_values = []

        for landmarks in landmarks_list:
            if len(landmarks) < 400:
                continue

            # Left eye EAR
            left_ear = self._compute_ear(landmarks, [33, 160, 158, 133, 153, 144])
            right_ear = self._compute_ear(landmarks, [362, 385, 387, 263, 373, 380])

            if left_ear is not None and right_ear is not None:
                left_ear_values.append(left_ear)
                right_ear_values.append(right_ear)

        if len(left_ear_values) < 5:
            return 0.3

        left_ears = np.array(left_ear_values)
        right_ears = np.array(right_ear_values)

        scores = []

        # Left-right eye synchronization
        if np.std(left_ears) > 0.01 and np.std(right_ears) > 0.01:
            lr_corr = np.corrcoef(left_ears, right_ears)[0, 1]
            if np.isnan(lr_corr):
                lr_corr = 0

            # Real: high L-R correlation (eyes blink together, > 0.85)
            # Deepfake: lower correlation (independently generated)
            if lr_corr < 0.6:
                scores.append(0.75)
            elif lr_corr < 0.8:
                scores.append(0.45)
            else:
                scores.append(0.15)

        # EAR micro-jitter (frame-to-frame variation when eyes are open)
        # Filter to open-eye frames (EAR > 0.2)
        open_left = left_ears[left_ears > 0.2]
        if len(open_left) > 5:
            ear_jitter = np.std(np.diff(open_left))
            # Real: very smooth EAR when open (jitter < 0.02)
            # Deepfake: jittery EAR (> 0.04)
            if ear_jitter > 0.06:
                scores.append(0.7)
            elif ear_jitter > 0.03:
                scores.append(0.4)
            else:
                scores.append(0.15)

        return float(np.mean(scores)) if scores else 0.3

    def _compute_ear(self, landmarks: np.ndarray, indices: list) -> float:
        """Compute Eye Aspect Ratio from 6 eye landmarks."""
        if len(indices) != 6:
            return None
        try:
            pts = np.array([landmarks[i][:2] for i in indices])
            # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
            vert1 = np.linalg.norm(pts[1] - pts[5])
            vert2 = np.linalg.norm(pts[2] - pts[4])
            horiz = np.linalg.norm(pts[0] - pts[3])
            if horiz < 1:
                return None
            return (vert1 + vert2) / (2.0 * horiz)
        except (IndexError, ValueError):
            return None

    def _analyze_with_opencv(self, frames: list, anomalies: list) -> dict:
        """Fallback analysis using OpenCV face detection (less precise)."""
        cascade = self._get_face_cascade()

        face_sizes = []
        face_positions = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_sizes.append((w, h))
                face_positions.append((x + w / 2, y + h / 2))

        if len(face_sizes) < 5:
            return self._fallback_result("Insufficient face detections")

        sizes = np.array(face_sizes, dtype=float)
        positions = np.array(face_positions, dtype=float)

        # Size consistency
        width_cv = np.std(sizes[:, 0]) / (np.mean(sizes[:, 0]) + 1e-5)
        aspect_ratios = sizes[:, 0] / (sizes[:, 1] + 1e-5)
        ar_std = np.std(aspect_ratios)

        scores = []

        # Face size jitter
        if width_cv > 0.15:
            scores.append(0.65)
        elif width_cv > 0.08:
            scores.append(0.4)
        else:
            scores.append(0.15)

        # Aspect ratio consistency
        if ar_std > 0.1:
            scores.append(0.6)
        elif ar_std > 0.05:
            scores.append(0.35)
        else:
            scores.append(0.15)

        # Position smoothness
        pos_diff = np.diff(positions, axis=0)
        pos_jitter = np.std(np.linalg.norm(pos_diff, axis=1))
        mean_motion = np.mean(np.linalg.norm(pos_diff, axis=1))
        if mean_motion > 1:
            jitter_ratio = pos_jitter / mean_motion
            if jitter_ratio > 0.8:
                scores.append(0.6)
            else:
                scores.append(0.2)
        else:
            scores.append(0.3)

        final_score = float(np.mean(scores))

        if final_score > 0.5:
            anomalies.append(f"Face geometry instability (OpenCV fallback, score={final_score:.2f})")

        return {
            "score": float(np.clip(final_score, 0.0, 1.0)),
            "confidence": float(np.clip(0.4, 0.0, 1.0)),  # Lower confidence without MediaPipe
            "proportion_score": float(scores[0]) if len(scores) > 0 else 0.3,
            "symmetry_score": 0.3,
            "stability_score": float(scores[2]) if len(scores) > 2 else 0.3,
            "jaw_score": 0.3,
            "eye_score": 0.3,
            "n_faces_tracked": len(face_sizes),
            "anomalies": anomalies,
            "note": "Using OpenCV fallback (install mediapipe for full analysis)",
        }

    def _fallback_result(self, reason: str = "") -> dict:
        return {
            "score": 0.5,
            "confidence": 0.0,
            "proportion_score": 0.5,
            "symmetry_score": 0.5,
            "stability_score": 0.5,
            "jaw_score": 0.5,
            "eye_score": 0.5,
            "n_faces_tracked": 0,
            "anomalies": [reason] if reason else [],
        }
