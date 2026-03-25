"""
Satya Drishti — Unified Video Preprocessing Pipeline
======================================================
Discovers and preprocesses videos from all available datasets:
  - FaceForensics++ (4 manipulation methods)
  - Celeb-DF (celebrity deepfakes)
  - DeeperForensics (end-to-end face swaps)

Extracts face-cropped frames (spatial) and 16-frame clips (temporal)
with identity-aware train/val/test splits to prevent data leakage.

Usage:
    python scripts/preprocess_all_videos.py
    python scripts/preprocess_all_videos.py --max_videos 500 --frames_per_video 20
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
OUTPUT_DIR = DATA_DIR / "frames_v2"

RANDOM_SEED = 42


def discover_all_videos(data_dir: Path) -> dict:
    """
    Discover videos from all datasets with source tracking.
    Returns dict with 'real' and 'fake' lists of (path, source, identity) tuples.
    Identity is used for leak-free splitting.
    """
    videos = {"real": [], "fake": []}

    # 1. FaceForensics++ — existing FF++ data
    ff_new = data_dir / "ff_new" / "FF++"
    if ff_new.exists():
        for f in sorted((ff_new / "real").glob("*.mp4")):
            vid_id = f.stem.split("_")[0]  # identity from filename
            videos["real"].append((f, "ff++", f"ff_{vid_id}"))
        for f in sorted((ff_new / "fake").glob("*.mp4")):
            vid_id = f.stem.split("_")[0]
            videos["fake"].append((f, "ff++", f"ff_{vid_id}"))

    # FF++ scripted downloads
    orig_dir = data_dir / "original_sequences" / "youtube" / "c23" / "videos"
    if orig_dir.exists():
        for f in sorted(orig_dir.glob("*.mp4")):
            videos["real"].append((f, "ff++", f"ff_{f.stem}"))

    manip_dir = data_dir / "manipulated_sequences"
    if manip_dir.exists():
        for method in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
            vid_dir = manip_dir / method / "c23" / "videos"
            if vid_dir.exists():
                for f in sorted(vid_dir.glob("*.mp4")):
                    videos["fake"].append((f, f"ff++_{method}", f"ff_{f.stem.split('_')[0]}"))

    # 2. Celeb-DF
    celeb_dir = data_dir / "celeb_df"
    for subdir in ["Celeb-real", "YouTube-real"]:
        d = celeb_dir / subdir
        if d.exists():
            for f in sorted(d.glob("*.mp4")):
                # Identity from filename: id0_0000.mp4 -> id0
                vid_id = f.stem.rsplit("_", 1)[0]
                videos["real"].append((f, "celeb_df", f"celeb_{vid_id}"))

    celeb_synth = celeb_dir / "Celeb-synthesis"
    if celeb_synth.exists():
        for f in sorted(celeb_synth.glob("*.mp4")):
            vid_id = f.stem.rsplit("_", 1)[0]
            videos["fake"].append((f, "celeb_df", f"celeb_{vid_id}"))

    # 3. DeeperForensics
    df_manip = data_dir / "deeper_forensics" / "manipulated_videos" / "end_to_end"
    if df_manip.exists():
        for f in sorted(df_manip.glob("*.mp4")):
            # 711_M007.mp4 -> identity M007
            parts = f.stem.split("_")
            identity = parts[1] if len(parts) > 1 else f.stem
            videos["fake"].append((f, "deeper_forensics", f"df_{identity}"))

    df_source = data_dir / "deeper_forensics" / "source_videos"
    if df_source.exists():
        for f in sorted(df_source.rglob("*.mp4")):
            # Extract person ID from path: source_videos/M004/...
            parts = f.relative_to(df_source).parts
            identity = parts[0] if parts else f.stem
            videos["real"].append((f, "deeper_forensics", f"df_{identity}"))

    print(f"\nDiscovered videos:")
    # Count by source
    sources_real = {}
    sources_fake = {}
    for _, src, _ in videos["real"]:
        sources_real[src] = sources_real.get(src, 0) + 1
    for _, src, _ in videos["fake"]:
        sources_fake[src] = sources_fake.get(src, 0) + 1

    print(f"  Real ({len(videos['real'])} total):")
    for src, count in sorted(sources_real.items()):
        print(f"    {src}: {count}")
    print(f"  Fake ({len(videos['fake'])} total):")
    for src, count in sorted(sources_fake.items()):
        print(f"    {src}: {count}")

    return videos


def balance_and_sample(videos: dict, max_per_class: int = None) -> dict:
    """
    Balance real/fake and ensure source diversity.
    Samples proportionally from each source to maintain diversity.
    """
    rng = random.Random(RANDOM_SEED)

    n_real = len(videos["real"])
    n_fake = len(videos["fake"])
    n_target = min(n_real, n_fake)

    if max_per_class and max_per_class < n_target:
        n_target = max_per_class

    balanced = {}
    for label in ["real", "fake"]:
        items = list(videos[label])
        rng.shuffle(items)

        if len(items) > n_target:
            # Proportional sampling from each source for diversity
            sources = {}
            for item in items:
                src = item[1]
                sources.setdefault(src, []).append(item)

            sampled = []
            total_available = sum(len(v) for v in sources.values())
            for src, src_items in sources.items():
                # Proportional allocation with minimum guarantee
                n_from_src = max(10, int(n_target * len(src_items) / total_available))
                n_from_src = min(n_from_src, len(src_items))
                rng.shuffle(src_items)
                sampled.extend(src_items[:n_from_src])

            # Fill remaining quota
            remaining = [item for item in items if item not in sampled]
            rng.shuffle(remaining)
            while len(sampled) < n_target and remaining:
                sampled.append(remaining.pop())

            balanced[label] = sampled[:n_target]
        else:
            balanced[label] = items

    print(f"\nBalanced: {len(balanced['real'])} real, {len(balanced['fake'])} fake")
    return balanced


def split_by_identity(videos: dict, train_ratio=0.8, val_ratio=0.1):
    """
    Split by identity to prevent data leakage.
    All videos of the same identity go into the same split.
    """
    rng = random.Random(RANDOM_SEED)
    splits = {"train": [], "val": [], "test": []}

    for label_name, label_int in [("real", 0), ("fake", 1)]:
        # Group by identity
        identity_groups = {}
        for path, source, identity in videos[label_name]:
            identity_groups.setdefault(identity, []).append((path, source))

        identities = list(identity_groups.keys())
        rng.shuffle(identities)

        n = len(identities)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_ids = set(identities[:n_train])
        val_ids = set(identities[n_train:n_train + n_val])
        test_ids = set(identities[n_train + n_val:])

        for identity, items in identity_groups.items():
            if identity in train_ids:
                split = "train"
            elif identity in val_ids:
                split = "val"
            else:
                split = "test"

            for path, source in items:
                splits[split].append((path, label_int, source, identity))

    for s in splits:
        rng.shuffle(splits[s])

    print(f"\nSplits (identity-aware, no leakage):")
    for s in ["train", "val", "test"]:
        n_real = sum(1 for _, l, _, _ in splits[s] if l == 0)
        n_fake = sum(1 for _, l, _, _ in splits[s] if l == 1)
        print(f"  {s}: {len(splits[s])} videos (real={n_real}, fake={n_fake})")

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
        """Detect and crop the largest face with margin."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.detector.detect(mp_image)

        if not results.detections:
            return None

        det = max(results.detections, key=lambda d: d.bounding_box.width)
        bb = det.bounding_box

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
    n_frames: int = 20,
) -> list[np.ndarray]:
    """Extract n evenly-spaced face-cropped frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 5:
        cap.release()
        return []

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
    n_clips: int = 3,
) -> list[np.ndarray]:
    """Extract n clips of clip_length consecutive face-cropped frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < clip_length + 5:
        cap.release()
        return []

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
            clips.append(np.stack(clip_frames))

    cap.release()
    return clips


def preprocess(args):
    print("=" * 60)
    print("Satya Drishti — Unified Video Preprocessing")
    print("=" * 60)

    # 1. Discover all videos
    videos = discover_all_videos(DATA_DIR)

    if len(videos["real"]) == 0 or len(videos["fake"]) == 0:
        print("ERROR: No videos found. Check dataset paths.")
        sys.exit(1)

    # 2. Balance and sample
    videos = balance_and_sample(videos, max_per_class=args.max_videos)

    # 3. Split by identity (no leakage)
    splits = split_by_identity(videos)

    # 4. Setup face extractor
    face_extractor = FaceExtractor(margin=args.margin, target_size=224)
    manifest = {"train": [], "val": [], "test": []}

    total_frames_saved = 0
    total_clips_saved = 0
    failed_videos = 0

    for split_name in ["train", "val", "test"]:
        video_list = splits[split_name]
        frames_dir = OUTPUT_DIR / split_name / "spatial"
        clips_dir = OUTPUT_DIR / split_name / "temporal"
        frames_dir.mkdir(parents=True, exist_ok=True)
        clips_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Processing {split_name} ({len(video_list)} videos) ---")

        for video_path, label, source, identity in tqdm(video_list, desc=split_name):
            # Create unique vid_id from source + filename
            vid_id = f"{source}_{video_path.stem}"
            # Sanitize for filesystem
            vid_id = vid_id.replace("/", "_").replace("\\", "_").replace(" ", "_")

            try:
                # Spatial: individual face frames
                faces = extract_frames_from_video(
                    video_path, face_extractor, args.frames_per_video
                )

                if len(faces) < 3:
                    failed_videos += 1
                    continue

                for i, face in enumerate(faces):
                    fname = f"{vid_id}_f{i:03d}.jpg"
                    fpath = frames_dir / fname
                    cv2.imwrite(str(fpath), face, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    manifest[split_name].append({
                        "path": str(fpath.relative_to(PROJECT_ROOT)),
                        "type": "spatial",
                        "label": label,
                        "video_id": vid_id,
                        "source": source,
                    })
                    total_frames_saved += 1

                # Temporal: 16-frame clips
                clips = extract_clips_from_video(
                    video_path, face_extractor,
                    clip_length=16, n_clips=args.clips_per_video,
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
                        "source": source,
                    })
                    total_clips_saved += 1

            except Exception as e:
                failed_videos += 1
                continue

    face_extractor.close()

    # 5. Save manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # 6. Summary
    print("\n" + "=" * 60)
    print("Preprocessing Summary")
    print("=" * 60)
    print(f"Total spatial frames: {total_frames_saved}")
    print(f"Total temporal clips: {total_clips_saved}")
    print(f"Failed/skipped videos: {failed_videos}")

    for split_name in ["train", "val", "test"]:
        spatial = [e for e in manifest[split_name] if e["type"] == "spatial"]
        temporal = [e for e in manifest[split_name] if e["type"] == "temporal"]
        n_real_s = sum(1 for e in spatial if e["label"] == 0)
        n_fake_s = sum(1 for e in spatial if e["label"] == 1)
        n_real_t = sum(1 for e in temporal if e["label"] == 0)
        n_fake_t = sum(1 for e in temporal if e["label"] == 1)

        # Source distribution
        src_counts = {}
        for e in manifest[split_name]:
            src_counts[e["source"]] = src_counts.get(e["source"], 0) + 1

        print(f"\n{split_name}:")
        print(f"  Spatial: {len(spatial)} (real={n_real_s}, fake={n_fake_s})")
        print(f"  Temporal: {len(temporal)} (real={n_real_t}, fake={n_fake_t})")
        print(f"  Sources: {src_counts}")

    print(f"\nManifest: {manifest_path}")
    print("[DONE] Preprocessing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess all video datasets")
    parser.add_argument("--frames_per_video", type=int, default=20,
                        help="Spatial frames to extract per video")
    parser.add_argument("--clips_per_video", type=int, default=3,
                        help="16-frame temporal clips per video")
    parser.add_argument("--margin", type=float, default=0.3,
                        help="Face crop margin ratio")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Max videos per class (None=use all)")
    args = parser.parse_args()

    preprocess(args)
