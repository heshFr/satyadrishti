"""
Satya Drishti — Dataset Preparation for Deepfake Detection
==========================================================
Prepares training data for the EfficientNet-B4 deepfake detector.

Supports two dataset formats:
1. FaceForensics++ — video-based dataset with real and manipulated clips
2. DFDC (DeepFake Detection Challenge) — Facebook's large-scale dataset

Pipeline:
1. Extract frames from videos at a configurable FPS
2. Detect and crop faces using MTCNN
3. Save face crops organized by label (real / fake)
4. Generate a JSON manifest compatible with the training script

Usage:
    python scripts/prepare_dataset.py --source ff++ --input_dir /path/to/faceforensics --output_dir datasets/deepfake
    python scripts/prepare_dataset.py --source dfdc --input_dir /path/to/dfdc --output_dir datasets/deepfake
    python scripts/prepare_dataset.py --source folder --input_dir /path/to/images --output_dir datasets/deepfake
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from facenet_pytorch import MTCNN
    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False
    print("[Warning] facenet-pytorch not installed. Face extraction will be skipped.")
    print("         Install with: pip install facenet-pytorch")


def extract_frames_from_video(video_path: str, fps: int = 1) -> list:
    """Extract frames from a video file at the given FPS."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [Skip] Could not open: {video_path}")
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        cap.release()
        return []

    frame_interval = max(1, int(video_fps / fps))
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frames.append(frame)
        frame_idx += 1

    cap.release()
    return frames


def extract_face(image: np.ndarray, mtcnn, margin: float = 0.3, min_size: int = 60, output_size: int = 380):
    """
    Detect and crop the largest face in the image using MTCNN.
    Returns the face crop as a PIL Image, or None if no face found.
    """
    if mtcnn is None:
        # No face detection — return resized whole image
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return img.resize((output_size, output_size))

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    boxes, probs = mtcnn.detect(pil_img)

    if boxes is None or len(boxes) == 0:
        return None

    # Use highest-confidence face
    best_idx = np.argmax(probs)
    box = boxes[best_idx]
    x1, y1, x2, y2 = box

    # Check minimum size
    w, h = x2 - x1, y2 - y1
    if w < min_size or h < min_size:
        return None

    # Add margin
    margin_w, margin_h = w * margin, h * margin
    x1 = max(0, int(x1 - margin_w))
    y1 = max(0, int(y1 - margin_h))
    x2 = min(pil_img.width, int(x2 + margin_w))
    y2 = min(pil_img.height, int(y2 + margin_h))

    face_crop = pil_img.crop((x1, y1, x2, y2))
    face_crop = face_crop.resize((output_size, output_size))
    return face_crop


def process_faceforensics(input_dir: str, output_dir: str, mtcnn, cfg: dict) -> list:
    """
    Process FaceForensics++ dataset structure:
    input_dir/
      original_sequences/youtube/c23/videos/*.mp4   (real)
      manipulated_sequences/*/c23/videos/*.mp4       (fake)
    """
    manifest = []
    real_dir = os.path.join(output_dir, "real")
    fake_dir = os.path.join(output_dir, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    # Real videos
    real_video_dir = os.path.join(input_dir, "original_sequences", "youtube", "c23", "videos")
    if os.path.isdir(real_video_dir):
        print(f"[FF++] Processing real videos from {real_video_dir}...")
        manifest += _process_video_dir(real_video_dir, real_dir, label=0, mtcnn=mtcnn, cfg=cfg)

    # Fake videos (all manipulation methods)
    manip_root = os.path.join(input_dir, "manipulated_sequences")
    if os.path.isdir(manip_root):
        for method in os.listdir(manip_root):
            video_dir = os.path.join(manip_root, method, "c23", "videos")
            if os.path.isdir(video_dir):
                print(f"[FF++] Processing fake videos ({method}) from {video_dir}...")
                manifest += _process_video_dir(video_dir, fake_dir, label=1, mtcnn=mtcnn, cfg=cfg, prefix=method)

    return manifest


def process_dfdc(input_dir: str, output_dir: str, mtcnn, cfg: dict) -> list:
    """
    Process DFDC dataset structure:
    input_dir/
      dfdc_train_part_*/
        metadata.json
        *.mp4
    """
    manifest = []
    real_dir = os.path.join(output_dir, "real")
    fake_dir = os.path.join(output_dir, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    for part_dir in sorted(Path(input_dir).glob("dfdc_train_part_*")):
        meta_path = part_dir / "metadata.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            metadata = json.load(f)

        print(f"[DFDC] Processing {part_dir.name}...")

        for video_name, info in metadata.items():
            video_path = str(part_dir / video_name)
            if not os.path.exists(video_path):
                continue

            label = 1 if info["label"] == "FAKE" else 0
            out_dir = fake_dir if label == 1 else real_dir

            frames = extract_frames_from_video(video_path, fps=1)
            for i, frame in enumerate(frames[:cfg.get("max_frames_per_video", 10)]):
                face = extract_face(frame, mtcnn,
                                    margin=cfg.get("margin", 0.3),
                                    min_size=cfg.get("min_face_size", 60),
                                    output_size=cfg.get("output_size", 380))
                if face is None:
                    continue

                fname = f"{part_dir.name}_{Path(video_name).stem}_f{i:04d}.jpg"
                save_path = os.path.join(out_dir, fname)
                face.save(save_path, quality=95)
                manifest.append({"path": save_path, "label": label})

    return manifest


def process_folder(input_dir: str, output_dir: str, mtcnn, cfg: dict) -> list:
    """
    Process a simple folder structure:
    input_dir/
      real/  (images)
      fake/  (images)
    """
    manifest = []
    out_real = os.path.join(output_dir, "real")
    out_fake = os.path.join(output_dir, "fake")
    os.makedirs(out_real, exist_ok=True)
    os.makedirs(out_fake, exist_ok=True)

    for label, folder in [(0, "real"), (1, "fake")]:
        src_dir = os.path.join(input_dir, folder)
        dst_dir = out_real if label == 0 else out_fake
        if not os.path.isdir(src_dir):
            print(f"[Folder] Skipping missing dir: {src_dir}")
            continue

        print(f"[Folder] Processing {src_dir}...")
        for img_name in os.listdir(src_dir):
            img_path = os.path.join(src_dir, img_name)
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue

            image = cv2.imread(img_path)
            if image is None:
                continue

            face = extract_face(image, mtcnn,
                                margin=cfg.get("margin", 0.3),
                                min_size=cfg.get("min_face_size", 60),
                                output_size=cfg.get("output_size", 380))
            if face is None:
                # Fall back to resized whole image if no face detected
                face = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                face = face.resize((cfg.get("output_size", 380), cfg.get("output_size", 380)))

            save_path = os.path.join(dst_dir, img_name)
            face.save(save_path, quality=95)
            manifest.append({"path": save_path, "label": label})

    return manifest


def _process_video_dir(video_dir, out_dir, label, mtcnn, cfg, prefix=""):
    """Helper to extract face crops from all videos in a directory."""
    entries = []
    for video_file in sorted(os.listdir(video_dir)):
        if not video_file.endswith((".mp4", ".avi")):
            continue

        video_path = os.path.join(video_dir, video_file)
        frames = extract_frames_from_video(video_path, fps=1)

        for i, frame in enumerate(frames[:cfg.get("max_frames_per_video", 10)]):
            face = extract_face(frame, mtcnn,
                                margin=cfg.get("margin", 0.3),
                                min_size=cfg.get("min_face_size", 60),
                                output_size=cfg.get("output_size", 380))
            if face is None:
                continue

            stem = Path(video_file).stem
            fname = f"{prefix}_{stem}_f{i:04d}.jpg" if prefix else f"{stem}_f{i:04d}.jpg"
            save_path = os.path.join(out_dir, fname)
            face.save(save_path, quality=95)
            entries.append({"path": save_path, "label": label})

    return entries


def main():
    parser = argparse.ArgumentParser(description="Prepare deepfake detection dataset")
    parser.add_argument("--source", choices=["ff++", "dfdc", "folder"], required=True,
                        help="Dataset source format")
    parser.add_argument("--input_dir", required=True, help="Path to source dataset")
    parser.add_argument("--output_dir", default="datasets/deepfake", help="Output directory")
    parser.add_argument("--no_face_extract", action="store_true",
                        help="Skip face extraction (use whole images)")
    parser.add_argument("--max_frames_per_video", type=int, default=10,
                        help="Max frames to extract per video")
    parser.add_argument("--output_size", type=int, default=380,
                        help="Output face crop size")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize MTCNN
    mtcnn = None
    if HAS_MTCNN and not args.no_face_extract:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mtcnn = MTCNN(keep_all=False, device=device, min_face_size=60)
        print(f"[MTCNN] Face detector initialized on {device}")
    elif args.no_face_extract:
        print("[Info] Face extraction disabled — using whole images.")
    else:
        print("[Warning] MTCNN unavailable — using whole images instead of face crops.")

    cfg = {
        "margin": 0.3,
        "min_face_size": 60,
        "output_size": args.output_size,
        "max_frames_per_video": args.max_frames_per_video,
    }

    # Process dataset
    if args.source == "ff++":
        manifest = process_faceforensics(args.input_dir, args.output_dir, mtcnn, cfg)
    elif args.source == "dfdc":
        manifest = process_dfdc(args.input_dir, args.output_dir, mtcnn, cfg)
    else:
        manifest = process_folder(args.input_dir, args.output_dir, mtcnn, cfg)

    # Save manifest
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    real_count = sum(1 for e in manifest if e["label"] == 0)
    fake_count = sum(1 for e in manifest if e["label"] == 1)

    print(f"\n[Done] Manifest saved to {manifest_path}")
    print(f"[Done] Total: {len(manifest)} images — {real_count} real, {fake_count} fake")


if __name__ == "__main__":
    main()
