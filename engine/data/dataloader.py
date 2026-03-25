"""
Multimodal DataLoader — Synchronous Audio/Video/Text Loading
==============================================================
Unified PyTorch Dataset and DataLoader for simultaneously loading
aligned audio waveforms, video frame sequences, and text transcripts
for multimodal deepfake & coercion training.
"""

import os
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import cv2
from PIL import Image
from torchvision import transforms


class MultimodalDeepfakeDataset(Dataset):
    """
    Unified dataset that loads aligned (audio, video, text) triplets.

    Expected directory structure:
        dataset_root/
        ├── manifest.json       # maps sample_id → {audio, video, text, label}
        ├── audio/              # .wav or .flac files
        ├── video/              # .mp4 files or frame directories
        └── transcripts/        # .txt files

    manifest.json format:
        [
            {
                "id": "sample_001",
                "audio": "audio/sample_001.wav",
                "video": "video/sample_001.mp4",
                "transcript": "transcripts/sample_001.txt",
                "label": 0
            },
            ...
        ]

    Labels:
        0 = real (safe)
        1 = deepfake_only
        2 = coercion_only
        3 = deepfake + coercion

    Args:
        root_dir: Path to dataset root
        split: "train", "val", or "test"
        audio_max_length: Max audio samples (zero-pad or truncate)
        video_num_frames: Number of video frames to sample
        video_frame_size: Target frame resolution
        sample_rate: Audio sample rate
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        audio_max_length: int = 64000,  # 4 seconds at 16kHz
        video_num_frames: int = 16,
        video_frame_size: int = 224,
        sample_rate: int = 16000,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.audio_max_length = audio_max_length
        self.video_num_frames = video_num_frames
        self.sample_rate = sample_rate

        # Load manifest
        manifest_path = self.root_dir / f"{split}_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                self.samples = json.load(f)
        else:
            self.samples = []

        # Video frame transforms
        self.frame_transform = transforms.Compose([
            transforms.Resize((video_frame_size, video_frame_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio waveform."""
        full_path = self.root_dir / audio_path
        waveform, sr = torchaudio.load(str(full_path))

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad or truncate
        if waveform.shape[1] > self.audio_max_length:
            waveform = waveform[:, : self.audio_max_length]
        elif waveform.shape[1] < self.audio_max_length:
            pad_length = self.audio_max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        return waveform  # (1, audio_max_length)

    def _load_video_frames(self, video_path: str) -> torch.Tensor:
        """Load and uniformly sample video frames."""
        full_path = str(self.root_dir / video_path)
        cap = cv2.VideoCapture(full_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # Return zeroed frames as fallback
            return torch.zeros(3, self.video_num_frames, 224, 224)

        # Uniformly sample frame indices
        indices = np.linspace(0, total_frames - 1, self.video_num_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_tensor = self.frame_transform(frame_pil)
                frames.append(frame_tensor)
            else:
                frames.append(torch.zeros(3, 224, 224))

        cap.release()

        # Stack: (num_frames, 3, H, W) → (3, num_frames, H, W) for 3D CNNs
        frames_tensor = torch.stack(frames, dim=0)  # (T, 3, H, W)
        frames_tensor = frames_tensor.permute(1, 0, 2, 3)  # (3, T, H, W)

        return frames_tensor

    def _load_transcript(self, transcript_path: str) -> str:
        """Load text transcript."""
        full_path = self.root_dir / transcript_path
        if full_path.exists():
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        return ""

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a multimodal sample dict:
            - audio: (1, audio_max_length) waveform tensor
            - video: (3, T, H, W) video clip tensor
            - transcript: raw text string
            - label: integer class label
            - sample_id: unique identifier
        """
        sample = self.samples[idx]

        audio = self._load_audio(sample["audio"])
        video = self._load_video_frames(sample["video"])
        transcript = self._load_transcript(sample["transcript"])
        label = sample["label"]

        return {
            "audio": audio,
            "video": video,
            "transcript": transcript,
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": sample["id"],
        }


def create_dataloader(
    root_dir: str,
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
    **dataset_kwargs,
) -> DataLoader:
    """
    Factory function to create a multimodal DataLoader.

    Args:
        root_dir: Dataset root directory
        split: "train", "val", or "test"
        batch_size: Batch size
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle (defaults to True for train)
    """
    dataset = MultimodalDeepfakeDataset(root_dir, split, **dataset_kwargs)

    if shuffle is None:
        shuffle = split == "train"

    # Custom collate to handle variable-length transcripts
    def collate_fn(batch):
        return {
            "audio": torch.stack([s["audio"] for s in batch]),
            "video": torch.stack([s["video"] for s in batch]),
            "transcript": [s["transcript"] for s in batch],
            "label": torch.stack([s["label"] for s in batch]),
            "sample_id": [s["sample_id"] for s in batch],
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=split == "train",
    )
