"""
Pupil Light Reflex Analyzer
=============================
Detects deepfake videos by analyzing pupil behavior, which is one of
the hardest physiological signals for AI to replicate correctly.

Detection signals:

1. **Pupil Size Consistency**: Both pupils should be the same size
   (consensual light reflex). Deepfakes may have asymmetric pupils.

2. **Light Response**: When lighting changes, pupils should constrict
   (bright) or dilate (dark) with specific dynamics (latency ~200ms,
   constriction speed ~1mm/s). AI videos rarely model this.

3. **Pupil Shape**: Real pupils are circular. Deepfake artifacts may
   create irregular pupil shapes.

4. **Corneal Reflection Consistency**: Real eyes show consistent
   reflections of light sources. Deepfakes often have inconsistent
   or missing reflections.

5. **Iris Texture Stability**: Real iris patterns are unique and
   stable across frames. Deepfakes may show flickering iris texture.

Uses OpenCV for eye region analysis (no heavy ML dependency).
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


class PupilLightReflexAnalyzer:
    """Analyzes pupil behavior for deepfake detection."""

    def __init__(self):
        self._mp_face_mesh = None

    def _get_mesh(self):
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
        Analyze pupil light reflex and eye consistency across frames.

        Args:
            frames: List of BGR video frames.

        Returns:
            dict with score, confidence, sub-scores, and anomalies.
        """
        if len(frames) < 8:
            return self._fallback_result("Not enough frames")

        # Sample frames
        n_sample = min(40, len(frames))
        indices = np.linspace(0, len(frames) - 1, n_sample, dtype=int)
        sampled = [frames[i] for i in indices]

        anomalies = []

        # Extract eye regions across frames
        eye_data = self._extract_eye_regions(sampled)

        if len(eye_data) < 5:
            return self._fallback_result("Insufficient eye detections")

        # 1. Pupil symmetry analysis
        symmetry_score = self._analyze_pupil_symmetry(eye_data)

        # 2. Light response analysis
        light_score = self._analyze_light_response(eye_data, sampled)

        # 3. Corneal reflection consistency
        reflection_score = self._analyze_reflections(eye_data)

        # 4. Iris texture stability
        iris_score = self._analyze_iris_stability(eye_data)

        # 5. Pupil roundness
        roundness_score = self._analyze_pupil_roundness(eye_data)

        if symmetry_score > 0.55:
            anomalies.append(f"Pupil asymmetry detected ({symmetry_score:.2f})")
        if light_score > 0.55:
            anomalies.append(f"Abnormal pupil light response ({light_score:.2f})")
        if reflection_score > 0.55:
            anomalies.append(f"Corneal reflection inconsistency ({reflection_score:.2f})")
        if iris_score > 0.55:
            anomalies.append(f"Iris texture instability ({iris_score:.2f})")
        if roundness_score > 0.55:
            anomalies.append(f"Pupil shape anomaly ({roundness_score:.2f})")

        weights = np.array([0.20, 0.25, 0.20, 0.20, 0.15])
        scores = np.array([symmetry_score, light_score, reflection_score, iris_score, roundness_score])
        final_score = float(np.dot(scores, weights))

        confidence = (
            0.5 * (1.0 - float(np.std(scores)))
            + 0.3 * float(np.max(scores))
            + 0.2 * min(1.0, len(anomalies) / 3)
        )

        return {
            "score": float(np.clip(final_score, 0.0, 1.0)),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "symmetry_score": float(symmetry_score),
            "light_response_score": float(light_score),
            "reflection_score": float(reflection_score),
            "iris_stability_score": float(iris_score),
            "roundness_score": float(roundness_score),
            "n_frames_with_eyes": len(eye_data),
            "anomalies": anomalies,
        }

    def _extract_eye_regions(self, frames: list) -> list:
        """Extract eye region data from each frame."""
        eye_data = []
        mesh = self._get_mesh() if HAS_MEDIAPIPE else None

        for frame in frames:
            h, w = frame.shape[:2]

            if mesh is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = mesh.process(rgb)

                if results.multi_face_landmarks:
                    face = results.multi_face_landmarks[0]
                    landmarks = np.array([(lm.x * w, lm.y * h) for lm in face.landmark])

                    # Left eye region: landmarks around left eye
                    left_eye_pts = landmarks[[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]]
                    right_eye_pts = landmarks[[362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]]

                    # Left iris: landmarks 468-472
                    # Right iris: landmarks 473-477
                    left_iris = landmarks[[468, 469, 470, 471, 472]] if len(landmarks) > 472 else None
                    right_iris = landmarks[[473, 474, 475, 476, 477]] if len(landmarks) > 477 else None

                    # Extract eye region crops
                    left_crop = self._crop_eye_region(frame, left_eye_pts)
                    right_crop = self._crop_eye_region(frame, right_eye_pts)

                    # Overall frame brightness
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    brightness = np.mean(gray)

                    eye_data.append({
                        "left_eye_pts": left_eye_pts,
                        "right_eye_pts": right_eye_pts,
                        "left_iris": left_iris,
                        "right_iris": right_iris,
                        "left_crop": left_crop,
                        "right_crop": right_crop,
                        "brightness": brightness,
                    })
            else:
                # Fallback: use Haar cascade for eye detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                eye_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_eye.xml"
                )
                eyes = eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))
                if len(eyes) >= 2:
                    # Sort by x to get left/right
                    eyes_sorted = sorted(eyes, key=lambda e: e[0])
                    left_rect = eyes_sorted[0]
                    right_rect = eyes_sorted[1]

                    lx, ly, lw, lh = left_rect
                    rx, ry, rw, rh = right_rect

                    left_crop = frame[ly:ly + lh, lx:lx + lw]
                    right_crop = frame[ry:ry + rh, rx:rx + rw]

                    eye_data.append({
                        "left_eye_pts": None,
                        "right_eye_pts": None,
                        "left_iris": None,
                        "right_iris": None,
                        "left_crop": left_crop,
                        "right_crop": right_crop,
                        "brightness": np.mean(gray),
                        "left_size": (lw, lh),
                        "right_size": (rw, rh),
                    })

        return eye_data

    def _crop_eye_region(self, frame: np.ndarray, eye_pts: np.ndarray) -> np.ndarray:
        """Crop eye region from frame using landmarks."""
        if eye_pts is None or len(eye_pts) < 4:
            return None

        x_min = max(0, int(np.min(eye_pts[:, 0])) - 5)
        x_max = min(frame.shape[1], int(np.max(eye_pts[:, 0])) + 5)
        y_min = max(0, int(np.min(eye_pts[:, 1])) - 5)
        y_max = min(frame.shape[0], int(np.max(eye_pts[:, 1])) + 5)

        if x_max - x_min < 10 or y_max - y_min < 5:
            return None

        return frame[y_min:y_max, x_min:x_max].copy()

    def _analyze_pupil_symmetry(self, eye_data: list) -> float:
        """
        Analyze bilateral pupil symmetry across frames.
        Both pupils should be approximately the same size.
        """
        asymmetry_values = []

        for data in eye_data:
            left_iris = data.get("left_iris")
            right_iris = data.get("right_iris")

            if left_iris is not None and right_iris is not None:
                # Estimate pupil size from iris landmarks
                left_size = self._estimate_iris_diameter(left_iris)
                right_size = self._estimate_iris_diameter(right_iris)

                if left_size > 2 and right_size > 2:
                    asymmetry = abs(left_size - right_size) / ((left_size + right_size) / 2)
                    asymmetry_values.append(asymmetry)
            else:
                # Fallback: use crop size ratio
                left_crop = data.get("left_crop")
                right_crop = data.get("right_crop")
                if left_crop is not None and right_crop is not None:
                    left_dark = self._estimate_pupil_from_crop(left_crop)
                    right_dark = self._estimate_pupil_from_crop(right_crop)
                    if left_dark > 0 and right_dark > 0:
                        asymmetry = abs(left_dark - right_dark) / ((left_dark + right_dark) / 2)
                        asymmetry_values.append(asymmetry)

        if len(asymmetry_values) < 3:
            return 0.3

        mean_asymmetry = np.mean(asymmetry_values)
        std_asymmetry = np.std(asymmetry_values)

        # Real: asymmetry < 0.10 (pupils are similar size)
        # Deepfake: asymmetry > 0.15 or highly variable
        if mean_asymmetry > 0.25:
            return 0.8
        elif mean_asymmetry > 0.15:
            return 0.55
        elif std_asymmetry > 0.15:
            return 0.5  # Inconsistent asymmetry
        else:
            return 0.15

    def _analyze_light_response(self, eye_data: list, frames: list) -> float:
        """
        Check if pupil size correlates with ambient brightness.
        Real pupils: smaller in bright light, larger in dark.
        """
        pupil_sizes = []
        brightnesses = []

        for data in eye_data:
            brightness = data.get("brightness", 128)

            # Get pupil size estimate
            size = None
            left_iris = data.get("left_iris")
            if left_iris is not None:
                size = self._estimate_iris_diameter(left_iris)
            elif data.get("left_crop") is not None:
                size = self._estimate_pupil_from_crop(data["left_crop"])

            if size is not None and size > 0:
                pupil_sizes.append(size)
                brightnesses.append(brightness)

        if len(pupil_sizes) < 5:
            return 0.3

        pupil_sizes = np.array(pupil_sizes)
        brightnesses = np.array(brightnesses)

        # Check correlation: should be NEGATIVE (brighter → smaller pupil)
        if np.std(brightnesses) < 1 or np.std(pupil_sizes) < 0.1:
            # Not enough variation to test
            return 0.35

        corr = np.corrcoef(brightnesses, pupil_sizes)[0, 1]
        if np.isnan(corr):
            return 0.35

        # Real eyes: correlation typically -0.2 to -0.7
        # AI: near-zero or positive correlation
        if corr > 0.1:
            return 0.75  # Wrong direction — suspicious
        elif corr > -0.1:
            return 0.55  # No response to light
        elif corr > -0.2:
            return 0.35
        else:
            return 0.1  # Normal light response

    def _analyze_reflections(self, eye_data: list) -> float:
        """
        Analyze corneal reflection consistency.
        Real eyes have consistent specular reflections from light sources.
        """
        reflection_positions = []

        for data in eye_data:
            for side in ["left_crop", "right_crop"]:
                crop = data.get(side)
                if crop is None or crop.size < 100:
                    continue

                # Find bright spots (reflections) in eye crop
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
                threshold = max(200, np.percentile(gray_crop, 98))
                bright_mask = gray_crop > threshold

                if bright_mask.sum() > 0:
                    # Center of mass of bright pixels
                    ys, xs = np.where(bright_mask)
                    center = (np.mean(xs), np.mean(ys))
                    # Normalize by crop size
                    h, w = gray_crop.shape
                    norm_center = (center[0] / w, center[1] / h)
                    reflection_positions.append(norm_center)

        if len(reflection_positions) < 5:
            return 0.4  # Can't determine reflection consistency

        positions = np.array(reflection_positions)
        pos_std_x = np.std(positions[:, 0])
        pos_std_y = np.std(positions[:, 1])

        # Real: consistent reflection position (std < 0.1)
        # Deepfake: variable reflection position (std > 0.2)
        avg_std = (pos_std_x + pos_std_y) / 2
        if avg_std > 0.25:
            return 0.75
        elif avg_std > 0.15:
            return 0.5
        elif avg_std > 0.08:
            return 0.3
        else:
            return 0.1

    def _analyze_iris_stability(self, eye_data: list) -> float:
        """
        Check if iris texture is consistent across frames.
        Real iris: stable unique pattern.
        Deepfake: may flicker or change.
        """
        iris_features = []

        for data in eye_data:
            crop = data.get("left_crop")
            if crop is None or crop.size < 200:
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

            # Resize to standard size
            try:
                resized = cv2.resize(gray, (32, 16))
            except Exception:
                continue

            # Flatten as feature vector
            iris_features.append(resized.flatten().astype(float))

        if len(iris_features) < 5:
            return 0.3

        features = np.array(iris_features)

        # Normalize
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms < 1e-5] = 1.0
        features_norm = features / norms

        # Pairwise cosine similarity between consecutive frames
        similarities = []
        for i in range(1, len(features_norm)):
            sim = np.dot(features_norm[i - 1], features_norm[i])
            similarities.append(sim)

        if len(similarities) < 3:
            return 0.3

        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)

        # Real iris: high consistent similarity (> 0.90, std < 0.05)
        # Deepfake: lower or more variable similarity
        if mean_sim < 0.75:
            return 0.8  # Iris texture changing significantly
        elif mean_sim < 0.85:
            return 0.55
        elif std_sim > 0.10:
            return 0.5  # Inconsistent stability
        else:
            return 0.15

    def _analyze_pupil_roundness(self, eye_data: list) -> float:
        """Check if detected pupil shapes are consistently circular."""
        roundness_values = []

        for data in eye_data:
            for side in ["left_crop", "right_crop"]:
                crop = data.get(side)
                if crop is None or crop.size < 200:
                    continue

                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

                # Find dark region (pupil)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest)
                    perimeter = cv2.arcLength(largest, True)

                    if perimeter > 0 and area > 20:
                        # Circularity: 4π * area / perimeter²
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        roundness_values.append(circularity)

        if len(roundness_values) < 5:
            return 0.3

        mean_round = np.mean(roundness_values)
        std_round = np.std(roundness_values)

        # Real pupils: circularity > 0.7, consistent
        # Deepfake: lower or more variable circularity
        scores = []
        if mean_round < 0.5:
            scores.append(0.7)
        elif mean_round < 0.65:
            scores.append(0.45)
        else:
            scores.append(0.15)

        if std_round > 0.20:
            scores.append(0.6)
        elif std_round > 0.10:
            scores.append(0.35)
        else:
            scores.append(0.15)

        return float(np.mean(scores))

    def _estimate_iris_diameter(self, iris_pts: np.ndarray) -> float:
        """Estimate iris diameter from MediaPipe iris landmarks."""
        if iris_pts is None or len(iris_pts) < 4:
            return 0
        center = iris_pts[0]
        radii = np.linalg.norm(iris_pts[1:] - center, axis=1)
        return float(np.mean(radii) * 2)

    def _estimate_pupil_from_crop(self, crop: np.ndarray) -> float:
        """Estimate pupil size from eye crop using dark region detection."""
        if crop is None or crop.size < 50:
            return 0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        # Dark region = pupil
        dark_threshold = max(30, np.percentile(gray, 15))
        dark_pixels = np.sum(gray < dark_threshold)
        total_pixels = gray.size

        if total_pixels < 10:
            return 0

        # Effective diameter
        return float(np.sqrt(dark_pixels / np.pi) * 2)

    def _fallback_result(self, reason: str = "") -> dict:
        return {
            "score": 0.5,
            "confidence": 0.0,
            "symmetry_score": 0.5,
            "light_response_score": 0.5,
            "reflection_score": 0.5,
            "iris_stability_score": 0.5,
            "roundness_score": 0.5,
            "n_frames_with_eyes": 0,
            "anomalies": [reason] if reason else [],
        }
