"""
Two-Stream Video Deepfake Detector
====================================
Combines spatial ViT-B/16 (frame-level artifact detection) with
temporal R3D-18 (inter-frame inconsistency detection) for robust
video deepfake detection.

Inference pipeline:
  1. Extract frames from video via OpenCV
  2. Detect and crop faces using Haar cascade
  3. Run spatial stream on individual face crops
  4. Run temporal stream on 16-frame face-cropped clips
  5. Late-fuse predictions (weighted average)
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.amp import autocast

from .spatial_vit import DeepfakeViT
from .temporal_r3d import TemporalR3D

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TwoStreamDetector:
    """
    Two-stream video deepfake detector combining spatial and temporal analysis.

    Spatial stream (ViT-B/16): detects per-frame blending artifacts
    Temporal stream (R3D-18): detects inter-frame temporal inconsistencies

    Args:
        spatial_ckpt: Path to spatial ViT v2 checkpoint
        temporal_ckpt: Path to temporal R3D v2 checkpoint
        device: torch device
        spatial_weight: Weight for spatial stream in late fusion (default 0.4)
        temporal_weight: Weight for temporal stream in late fusion (default 0.6)
    """

    def __init__(
        self,
        spatial_ckpt: str,
        temporal_ckpt: str,
        device: torch.device = None,
        spatial_weight: float = 0.4,
        temporal_weight: float = 0.6,
    ):
        self.device = device or torch.device("cpu")
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.use_amp = self.device.type == "cuda"

        # Face detection via OpenCV Haar cascade
        haar_path = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        self.face_cascade = cv2.CascadeClassifier(haar_path)

        # Load spatial ViT-B/16
        self.spatial = DeepfakeViT(num_classes=2, pretrained=False)
        ckpt = torch.load(spatial_ckpt, map_location=self.device, weights_only=False)
        self.spatial.load_state_dict(ckpt["model_state_dict"])
        self.spatial.to(self.device).eval()
        print(f"[Video] Spatial ViT-B/16 loaded from {spatial_ckpt}")

        # Load temporal R3D-18
        self.temporal = TemporalR3D(num_classes=2, pretrained=False)
        ckpt = torch.load(temporal_ckpt, map_location=self.device, weights_only=False)
        self.temporal.load_state_dict(ckpt["model_state_dict"])
        self.temporal.to(self.device).eval()
        print(f"[Video] Temporal R3D-18 loaded from {temporal_ckpt}")

        # Cached spatial transform — matches training pipeline exactly
        from torchvision import transforms
        self._spatial_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def detect_face(self, frame: np.ndarray):
        """Detect the largest face in a frame. Returns (x, y, w, h) or None."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return None
        # Return the largest face by area
        return max(faces, key=lambda f: f[2] * f[3])

    def crop_face(self, frame: np.ndarray, bbox=None, target_size: int = 224) -> np.ndarray:
        """
        Crop face from frame with padding. Falls back to center crop if no face detected.

        Args:
            frame: BGR image (H, W, 3)
            bbox: (x, y, w, h) face bounding box, or None for center crop
            target_size: output size

        Returns:
            RGB face crop resized to (target_size, target_size, 3) uint8
        """
        h, w = frame.shape[:2]

        if bbox is not None:
            x, y, fw, fh = bbox
            # Add 30% padding around the face
            pad_x = int(fw * 0.3)
            pad_y = int(fh * 0.3)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + fw + pad_x)
            y2 = min(h, y + fh + pad_y)
            crop = frame[y1:y2, x1:x2]
        else:
            # Center crop: take central 70% of the frame
            crop_size = int(min(h, w) * 0.7)
            cy, cx = h // 2, w // 2
            y1 = cy - crop_size // 2
            x1 = cx - crop_size // 2
            crop = frame[y1:y1 + crop_size, x1:x1 + crop_size]

        crop = cv2.resize(crop, (target_size, target_size))
        # Convert BGR to RGB
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return crop

    def extract_frames(self, video_path: str, max_frames: int = 32):
        """
        Extract evenly-spaced frames from a video, detect and crop faces.

        Returns:
            List of face-cropped RGB frames as uint8 numpy arrays (224, 224, 3)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        n_sample = min(max_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, n_sample, dtype=int)

        # First pass: detect faces to find a stable bounding box
        raw_frames = []
        face_bboxes = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            raw_frames.append(frame)
            face_bboxes.append(self.detect_face(frame))

        cap.release()

        if not raw_frames:
            return []

        # Use median face bbox across frames for stable cropping
        valid_bboxes = [b for b in face_bboxes if b is not None]
        if len(valid_bboxes) >= len(raw_frames) * 0.3:
            # Use median bbox for stability
            median_bbox = np.median(np.array(valid_bboxes), axis=0).astype(int)
        else:
            median_bbox = None

        # Crop faces from all frames
        cropped = []
        for i, frame in enumerate(raw_frames):
            # Prefer per-frame bbox, fall back to median, then center crop
            bbox = face_bboxes[i] if face_bboxes[i] is not None else median_bbox
            crop = self.crop_face(frame, bbox)
            cropped.append(crop)

        return cropped

    def _normalize_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Normalize a single RGB uint8 frame using the EXACT same pipeline
        as training (torchvision.transforms) to avoid precision mismatches.
        """
        from PIL import Image
        return self._spatial_transform(Image.fromarray(frame))

    def _normalize_clip(self, frames: list) -> torch.Tensor:
        """Normalize a list of 16 RGB uint8 frames to a tensor for temporal model."""
        clip = np.stack(frames)  # (16, 224, 224, 3)
        clip = clip.astype(np.float32) / 255.0
        clip = (clip - IMAGENET_MEAN) / IMAGENET_STD
        clip = clip.transpose(3, 0, 1, 2)  # (3, 16, 224, 224)
        return torch.from_numpy(clip).float()

    @torch.no_grad()
    def predict(self, video_path: str, max_frames: int = 32) -> dict:
        """
        Run full two-stream deepfake detection on a video.

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to sample

        Returns:
            dict with fake_probability, spatial/temporal breakdown, verdict
        """
        frames = self.extract_frames(video_path, max_frames)

        if not frames:
            return {
                "fake_probability": 0.5,
                "spatial_fake_prob": 0.5,
                "temporal_fake_prob": 0.5,
                "is_deepfake": False,
                "confidence": 0.0,
                "error": "Could not extract frames from video",
            }

        # --- Spatial predictions (per-frame) ---
        spatial_probs = []
        batch_size = 4
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_tensor = torch.stack(
                [self._normalize_frame(f) for f in batch_frames]
            ).to(self.device)

            if self.use_amp:
                with autocast(device_type="cuda"):
                    logits = self.spatial(batch_tensor)
            else:
                logits = self.spatial(batch_tensor)

            probs = torch.softmax(logits, dim=1)[:, 1]  # P(fake)
            spatial_probs.extend(probs.cpu().numpy().tolist())

        # --- Temporal predictions (16-frame clips) ---
        temporal_probs = []
        if len(frames) >= 16:
            stride = max(1, (len(frames) - 16) // 3 + 1)
            for i in range(0, len(frames) - 15, stride):
                clip_frames = frames[i:i + 16]
                clip_tensor = self._normalize_clip(clip_frames).unsqueeze(0).to(self.device)

                if self.use_amp:
                    with autocast(device_type="cuda"):
                        logits = self.temporal(clip_tensor)
                else:
                    logits = self.temporal(clip_tensor)

                prob = torch.softmax(logits, dim=1)[0, 1].item()
                temporal_probs.append(prob)

        # --- Late fusion ---
        spatial_avg = float(np.mean(spatial_probs))

        if temporal_probs:
            temporal_avg = float(np.mean(temporal_probs))
            combined = (
                spatial_avg * self.spatial_weight
                + temporal_avg * self.temporal_weight
            )
        else:
            temporal_avg = spatial_avg
            combined = spatial_avg

        is_fake = combined > 0.5
        confidence = combined * 100 if is_fake else (1 - combined) * 100

        return {
            "fake_probability": round(combined, 4),
            "spatial_fake_prob": round(spatial_avg, 4),
            "temporal_fake_prob": round(temporal_avg, 4),
            "is_deepfake": is_fake,
            "confidence": round(confidence, 1),
            "num_frames_analyzed": len(frames),
            "num_clips_analyzed": len(temporal_probs),
        }

    @torch.no_grad()
    def extract_embedding(self, video_path: str) -> torch.Tensor:
        """
        Extract 768d video embedding for the fusion network.
        Uses the spatial ViT backbone averaged across sampled frames.

        Returns:
            (1, 768) tensor
        """
        frames = self.extract_frames(video_path, max_frames=8)

        if not frames:
            return torch.zeros(1, self.spatial.embed_dim, device=self.device)

        embeddings = []
        for frame in frames:
            tensor = self._normalize_frame(frame).unsqueeze(0).to(self.device)
            if self.use_amp:
                with autocast(device_type="cuda"):
                    emb = self.spatial.extract_embedding(tensor)
            else:
                emb = self.spatial.extract_embedding(tensor)
            embeddings.append(emb)

        # Average-pool across frames → single 768d embedding
        return torch.stack(embeddings).mean(dim=0)
