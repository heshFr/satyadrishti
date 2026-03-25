import cv2
import numpy as np
import logging
from typing import Dict, Tuple, List
from skimage.metrics import structural_similarity as ssim
import os

logger = logging.getLogger(__name__)

class FaceForensicsDetector:
    def __init__(self):
        """
        Initializes the Face-Specific Detection Module.
        Uses OpenCV Haar Cascades to extract precise facial bounding boxes
        and eyes. We target the left and right eyes to perform
        Micro-Consistency reflection analysis (SSIM).
        """
        cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
        haar_model = os.path.join(cv2_base_dir, 'data', 'haarcascade_frontalface_default.xml')
        eye_model = os.path.join(cv2_base_dir, 'data', 'haarcascade_eye.xml')
        
        self.face_cascade = cv2.CascadeClassifier(haar_model)
        self.eye_cascade = cv2.CascadeClassifier(eye_model)

    def analyze_faces(self, image: np.ndarray) -> Tuple[float, Dict]:
        """
        Extracts faces and computes a deepfake probability based on micro-anomalies.
        Primary metric: Eye Reflection Structural Similarity (SSIM).
        AI struggles to make ambient reflections identical across both corneas.
        """
        if image is None:
            return 0.0, {"status": "error", "reason": "invalid_image"}
            
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return 0.0, {"status": "no_faces", "num_faces": 0}
            
            face_scores = []
            eye_ssim_scores = []
            
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                
                # Detect eyes in the face region
                eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
                
                # We need exactly 2 eyes to compare reflections
                if len(eyes) >= 2:
                    # Sort eyes by X coordinate to get left and right
                    eyes = sorted(eyes, key=lambda e: e[0])
                    
                    left_eye_rect = eyes[0]
                    right_eye_rect = eyes[-1] # Usually the furthest right
                    
                    # Extract eye pixels
                    ex1, ey1, ew1, eh1 = left_eye_rect
                    ex2, ey2, ew2, eh2 = right_eye_rect
                    
                    # Crop tightly to the center (pupil/iris area)
                    # Removing the whites and eyelids to focus strictly on reflection glints
                    cy1, cx1 = ey1 + eh1//2, ex1 + ew1//2
                    cy2, cx2 = ey2 + eh2//2, ex2 + ew2//2
                    
                    r1 = min(ew1, eh1) // 3
                    r2 = min(ew2, eh2) // 3
                    
                    left_eye_iris = roi_gray[cy1-r1:cy1+r1, cx1-r1:cx1+r1]
                    right_eye_iris = roi_gray[cy2-r2:cy2+r2, cx2-r2:cx2+r2]
                    
                    if left_eye_iris.size == 0 or right_eye_iris.size == 0:
                        continue
                        
                    # Normalize sizes to perfectly compare them
                    target_size = (32, 32)
                    left_iris_norm = cv2.resize(left_eye_iris, target_size)
                    right_iris_norm = cv2.resize(right_eye_iris, target_size)
                    
                    # Apply strong histogram equalization to pop the white glossy reflections
                    left_iris_norm = cv2.equalizeHist(left_iris_norm)
                    right_iris_norm = cv2.equalizeHist(right_iris_norm)
                    
                    # Structural Similarity Index (SSIM)
                    # Real human eyes reflect the exact same room/lights (high SSIM -> ~0.6+)
                    # AI generated eyes construct arbitrary independent reflections (low SSIM -> ~0.2)
                    score, _ = ssim(left_iris_norm, right_iris_norm, full=True)
                    eye_ssim_scores.append(score)
                    
                    if score < 0.20:
                        face_scores.append(0.85)  # Definitely AI mismatched reflections
                    elif score < 0.35:
                        face_scores.append(0.60)  # Suspiciously asynchronous reflections
                    elif score < 0.50:
                        face_scores.append(0.35)  # Moderate
                    else:
                        face_scores.append(0.08)  # Identical reflections, likely authentic
                else:
                    # Fallback to Laplacian skin variance if eyes are hidden
                    laplacian_var = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
                    if laplacian_var < 30.0:
                        face_scores.append(0.55)
                    elif laplacian_var < 60.0:
                        face_scores.append(0.30)
                    else:
                        face_scores.append(0.10)

            if not face_scores:
                return 0.0, {"status": "no_valid_eyes_or_faces", "num_faces": len(faces)}

            # If we never found eyes (only used Laplacian fallback), the scores
            # are less reliable — dampen them to prevent false positives
            if not eye_ssim_scores and face_scores:
                face_scores = [s * 0.6 for s in face_scores]

            max_fake = max(face_scores)
            avg_ssim = sum(eye_ssim_scores) / len(eye_ssim_scores) if eye_ssim_scores else 1.0
            
            return max_fake, {
                "status": "success",
                "num_faces": len(faces),
                "face_scores": face_scores,
                "avg_eye_ssim": round(avg_ssim, 3),
                "max_score": max_fake
            }

        except Exception as e:
            logger.error(f"Face Forensics Error: {e}")
            return 0.0, {"status": "error", "reason": str(e)}
