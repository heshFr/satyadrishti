"""
FF++ Video Preprocessing — Face Extraction Pipeline
=====================================================
Extracts face-cropped frames from FaceForensics++ videos for
spatial (ViT) and temporal (X3D) training.

Usage:
    python scripts/preprocess_ff_videos.py
    python scripts/preprocess_ff_videos.py --frames_per_video 30 --clips_per_video 4
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions
from mediapipe.tasks.python import BaseOptions
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "datasets" / "deepfake"
OUTPUT_DIR = DATA_DIR / "frames"

# Deterministic splits
RANDOM_SEED = 42


def discover_videos(data_dir: Path) -> dict:
    """Find all real/fake videos from both zip extract and scripted downloads."""
    videos = {"real": [], "fake": []}

    # Source 1: New zip extract (FF++/real/, FF++/fake/)
    ff_new = data_dir / "ff_new" / "FF++"
    if ff_new.exists():
        for f in sorted((ff_new / "real").glob("*.mp4")):
            videos["real"].append(f)
        for f in sorted((ff_new / "fake").glob("*.mp4")):
            videos["fake"].append(f)

    # Source 2: Scripted downloads (original_sequences, manipulated_sequences)
    orig_dir = data_dir / "original_sequences" / "youtube" / "c23" / "videos"
    if orig_dir.exists():
        for f in sorted(orig_dir.glob("*.mp4")):
            videos["real"].append(f)

    manip_dir = data_dir / "manipulated_sequences"
    if manip_dir.exists():
        for method in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
            vid_dir = manip_dir / method / "c23" / "videos"
            if vid_dir.exists():
                for f in sorted(vid_dir.glob("*.mp4")):
                    videos["fake"].append(f)

    print(f"Discovered: {len(videos['real'])} real, {len(videos['fake'])} fake videos")
    return videos


def balance_videos(videos: dict, max_per_class: int = None) -> dict:
    """Balance real/fake counts. Subsample the larger class."""
    n_real = len(videos["real"])
    n_fake = len(videos["fake"])
    n_min = min(n_real, n_fake)

    if max_per_class:
        n_min = min(n_min, max_per_class)

    rng = random.Random(RANDOM_SEED)
    balanced = {
        "real": sorted(rng.sample(videos["real"], n_min), key=lambda p: p.name),
        "fake": sorted(rng.sample(videos["fake"], n_min), key=lambda p: p.name),
    }
    print(f"Balanced: {n_min} real, {n_min} fake videos")
    return balanced


def split_videos(videos: dict, train_ratio=0.8, val_ratio=0.1):
    """Split videos into train/val/test by video (not frame) to prevent leakage."""
    rng = random.Random(RANDOM_SEED)
    splits = {"train": [], "val": [], "test": []}

    for label, paths in videos.items():
        shuffled = list(paths)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        for p in shuffled[:n_train]:
            splits["train"].append((p, 0 if label == "real" else 1))
        for p in shuffled[n_train:n_train + n_val]:
            splits["val"].append((p, 0 if label == "real" else 1))
        for p in shuffled[n_train + n_val:]:
            splits["test"].append((p, 0 if label == "real" else 1))

    for s in splits:
        rng.shuffle(splits[s])

    print(f"Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    return splits


class FaceExtractor:
    """MediaPipe tasks API face detector and cropper."""

    def __init__(self, margin: float = 0.3, target_size: int = 224):
        self.margin = margin
        self.target_size = target_size
        model_path = str(PROJECT_ROOT / "models" / "blaze_face_short_range.tflite")
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            min_detection_confidence=0.5,
        )
        self.detector = FaceDetector.create_from_options(options)

    def extract_face(self, frame: np.ndarray) -> np.ndarray | None:
        """Detect and crop the largest face with margin. Returns None if no face."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.detector.detect(mp_image)

        if not results.detections:
            return None

        # Take the largest detection
        det = max(results.detections, key=lambda d: d.bounding_box.width)
        bb = det.bounding_box

        # bb has origin_x, origin_y, width, height in pixels
        cx = bb.origin_x + bb.width // 2
        cy = bb.origin_y + bb.height // 2
        face_w = int(bb.width * (1 + self.margin))
        face_h = int(bb.height * (1 + self.margin))
        side = max(face_w, face_h)

        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, x1 + side)
        y2 = min(h, y1 + side)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None

        face = cv2.resize(face, (self.target_size, self.target_size))
        return face

    def close(self):
        self.detector.close()


def extract_frames_from_video(
    video_path: Path,
    face_extractor: FaceExtractor,
    n_frames: int = 30,
) -> list[np.ndarray]:
    """Extract n evenly-spaced face-cropped frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < n_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int).tolist()

    faces = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        face = face_extractor.extract_face(frame)
        if face is not None:
            faces.append(face)

    cap.release()
    return faces


def extract_clips_from_video(
    video_path: Path,
    face_extractor: FaceExtractor,
    clip_length: int = 16,
    n_clips: int = 4,
) -> list[np.ndarray]:
    """Extract n clips of clip_length consecutive face-cropped frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < clip_length:
        cap.release()
        return []

    # Evenly space clip start points
    max_start = total_frames - clip_length
    if max_start <= 0:
        starts = [0]
    elif n_clips == 1:
        starts = [max_start // 2]
    else:
        starts = np.linspace(0, max_start, n_clips, dtype=int).tolist()

    clips = []
    for start in starts:
        clip_frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        valid = True
        for _ in range(clip_length):
            ret, frame = cap.read()
            if not ret:
                valid = False
                break
            face = face_extractor.extract_face(frame)
            if face is None:
                valid = False
                break
            clip_frames.append(face)

        if valid and len(clip_frames) == clip_length:
            # Stack: (clip_length, H, W, 3)
            clips.append(np.stack(clip_frames))

    cap.release()
    return clips


def preprocess(args):
    print("=" * 60)
    print("Satya Drishti — FF++ Video Preprocessing")
    print("=" * 60)

    # 1. Discover videos
    videos = discover_videos(DATA_DIR)
    videos = balance_videos(videos, max_per_class=args.max_videos)
    splits = split_videos(videos)

    # 2. Setup
    face_extractor = FaceExtractor(margin=args.margin, target_size=224)
    manifest = {"train": [], "val": [], "test": []}

    for split_name, video_list in splits.items():
        frames_dir = OUTPUT_DIR / split_name / "spatial"
        clips_dir = OUTPUT_DIR / split_name / "temporal"
        frames_dir.mkdir(parents=True, exist_ok=True)
        clips_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Processing {split_name} ({len(video_list)} videos) ---")

        for video_path, label in tqdm(video_list, desc=split_name):
            vid_id = video_path.stem

            # Spatial: individual face frames
            faces = extract_frames_from_video(video_path, face_extractor, args.frames_per_video)
            for i, face in enumerate(faces):
                fname = f"{vid_id}_f{i:03d}.jpg"
                fpath = frames_dir / fname
                cv2.imwrite(str(fpath), face, [cv2.IMWRITE_JPEG_QUALITY, 95])
                manifest[split_name].append({
                    "path": str(fpath.relative_to(PROJECT_ROOT)),
                    "type": "spatial",
                    "label": label,
                    "video_id": vid_id,
                })

            # Temporal: 16-frame clips saved as .npy
            clips = extract_clips_from_video(
                video_path, face_extractor, clip_length=16, n_clips=args.clips_per_video
            )
            for i, clip in enumerate(clips):
                fname = f"{vid_id}_clip{i:02d}.npy"
                fpath = clips_dir / fname
                np.save(str(fpath), clip)
                manifest[split_name].append({
                    "path": str(fpath.relative_to(PROJECT_ROOT)),
                    "type": "temporal",
                    "label": label,
                    "video_id": vid_id,
                })

    face_extractor.close()

    # 3. Save manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Stats
    for split_name in ["train", "val", "test"]:
        spatial = [e for e in manifest[split_name] if e["type"] == "spatial"]
        temporal = [e for e in manifest[split_name] if e["type"] == "temporal"]
        n_real_s = sum(1 for e in spatial if e["label"] == 0)
        n_fake_s = sum(1 for e in spatial if e["label"] == 1)
        n_real_t = sum(1 for e in temporal if e["label"] == 0)
        n_fake_t = sum(1 for e in temporal if e["label"] == 1)
        print(f"\n{split_name}:")
        print(f"  Spatial frames: {len(spatial)} (real={n_real_s}, fake={n_fake_s})")
        print(f"  Temporal clips:  {len(temporal)} (real={n_real_t}, fake={n_fake_t})")

    print(f"\nManifest saved to: {manifest_path}")
    print("[DONE] Preprocessing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess FF++ videos for training")
    parser.add_argument("--frames_per_video", type=int, default=30, help="Spatial frames per video")
    parser.add_argument("--clips_per_video", type=int, default=4, help="Temporal clips (16 frames each) per video")
    parser.add_argument("--margin", type=float, default=0.3, help="Face crop margin ratio")
    parser.add_argument("--max_videos", type=int, default=None, help="Max videos per class (for testing)")
    args = parser.parse_args()

    preprocess(args)
