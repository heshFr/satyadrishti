"""
Video Quality Analyzer
======================
Assesses video quality metrics to improve deepfake detection accuracy.
Low-quality videos (heavy compression, motion blur, low resolution) can
cause false positives — this module quantifies quality so the detection
pipeline can adjust thresholds and confidence accordingly.

Metrics computed:
  - Resolution, FPS, codec, duration
  - Estimated bitrate (file size / duration, or ffprobe if available)
  - Compression artifact score (Laplacian variance + DCT block boundaries)
  - Motion blur score (Laplacian variance + optical flow)
  - Composite quality tier: high / medium / low / very_low
"""

import os
import cv2
import numpy as np
import shutil
import subprocess
from typing import Dict, Any, List, Optional


class VideoQualityAnalyzer:
    """Analyzes video quality to calibrate deepfake detection confidence."""

    def __init__(self, sample_frames: int = 10):
        self.sample_frames = sample_frames

    def analyze(self, video_path: str) -> Dict[str, Any]:
        """
        Run full quality assessment on a video file.

        Returns dict with resolution, bitrate, compression_score,
        motion_blur_score, quality_tier, quality_score, etc.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return self._default_metrics("Could not open video")

        # Basic properties
        props = self._get_video_properties(cap)
        if props["total_frames"] <= 0:
            cap.release()
            return self._default_metrics("Video has no frames")

        # Sample frames for analysis
        frames = self._sample_frames(cap, props["total_frames"])
        cap.release()

        if len(frames) < 2:
            return self._default_metrics("Could not extract enough frames")

        # Bitrate estimation
        bitrate = self._estimate_bitrate(video_path, props["duration_s"])

        # Compression artifacts
        compression_score, laplacian_mean, laplacian_std = self._compute_compression_score(frames)

        # Motion blur
        motion_blur_score = self._compute_motion_blur_score(frames)

        # Composite quality score and tier
        metrics = {
            "width": props["width"],
            "height": props["height"],
            "fps": props["fps"],
            "duration_s": round(props["duration_s"], 2),
            "total_frames": props["total_frames"],
            "codec": props["codec"],
            "estimated_bitrate_kbps": round(bitrate, 1),
            "compression_score": round(compression_score, 3),
            "motion_blur_score": round(motion_blur_score, 3),
            "laplacian_variance_mean": round(laplacian_mean, 1),
            "laplacian_variance_std": round(laplacian_std, 1),
        }

        metrics["quality_score"] = self._compute_quality_score(metrics)
        metrics["quality_tier"] = self._compute_quality_tier(metrics)

        return metrics

    def _get_video_properties(self, cap: cv2.VideoCapture) -> Dict[str, Any]:
        """Extract basic video properties from OpenCV VideoCapture."""
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_frames / fps if fps > 0 else 0.0

        # Decode codec fourcc
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]).strip()

        return {
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration_s": duration_s,
            "codec": codec or "unknown",
        }

    def _estimate_bitrate(self, video_path: str, duration_s: float) -> float:
        """
        Estimate video bitrate in kbps.
        Tries ffprobe first (precise), falls back to file_size / duration.
        """
        # Try ffprobe for accurate bitrate
        if shutil.which("ffprobe"):
            try:
                result = subprocess.run(
                    [
                        "ffprobe", "-v", "quiet",
                        "-show_entries", "format=bit_rate",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        video_path,
                    ],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip().isdigit():
                    return int(result.stdout.strip()) / 1000.0  # bps -> kbps
            except Exception:
                pass

        # Fallback: file size / duration
        if duration_s > 0:
            file_size_bytes = os.path.getsize(video_path)
            return (file_size_bytes * 8) / (duration_s * 1000)  # kbps

        return 0.0

    def _sample_frames(self, cap: cv2.VideoCapture, total_frames: int) -> List[np.ndarray]:
        """Sample evenly-spaced frames from the video."""
        n_sample = min(self.sample_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, n_sample, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)

        return frames

    def _compute_compression_score(self, frames: List[np.ndarray]) -> tuple:
        """
        Detect compression artifacts via:
        1. Laplacian variance (blurriness from compression)
        2. DCT block boundary detection (8x8 blocking artifacts from H.264/JPEG)

        Returns: (compression_score, laplacian_mean, laplacian_std)
        """
        laplacian_vars = []
        block_scores = []

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Laplacian variance — lower = blurrier (more compressed)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            laplacian_vars.append(lap_var)

            # DCT block boundary detection
            block_score = self._detect_block_artifacts(gray)
            block_scores.append(block_score)

        lap_mean = float(np.mean(laplacian_vars))
        lap_std = float(np.std(laplacian_vars))
        block_mean = float(np.mean(block_scores))

        # Combine: low Laplacian + high block artifacts = heavy compression
        # Laplacian < 50 on a decent resolution frame = very compressed
        # Laplacian 50-200 = moderately compressed
        # Laplacian > 200 = clean
        if lap_mean < 30:
            lap_score = 1.0
        elif lap_mean < 80:
            lap_score = 0.7
        elif lap_mean < 200:
            lap_score = 0.3
        else:
            lap_score = 0.0

        compression_score = lap_score * 0.6 + block_mean * 0.4
        return compression_score, lap_mean, lap_std

    def _detect_block_artifacts(self, gray: np.ndarray) -> float:
        """
        Detect 8x8 DCT block boundary artifacts from H.264/JPEG compression.
        Computes gradient magnitude at 8-pixel intervals vs overall.
        Ratio > 1.3 indicates visible blocking.
        """
        h, w = gray.shape
        if h < 64 or w < 64:
            return 0.0

        # Compute horizontal and vertical gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Overall gradient magnitude
        overall_energy = np.mean(np.abs(grad_x)) + np.mean(np.abs(grad_y))
        if overall_energy < 1e-6:
            return 0.0

        # Gradient at 8-pixel column boundaries (vertical edges from blocking)
        boundary_cols = list(range(8, w - 1, 8))
        if not boundary_cols:
            return 0.0
        boundary_energy_x = np.mean(np.abs(grad_x[:, boundary_cols]))

        # Gradient at 8-pixel row boundaries (horizontal edges from blocking)
        boundary_rows = list(range(8, h - 1, 8))
        if not boundary_rows:
            return 0.0
        boundary_energy_y = np.mean(np.abs(grad_y[boundary_rows, :]))

        boundary_energy = (boundary_energy_x + boundary_energy_y) / 2
        ratio = boundary_energy / (overall_energy / 2 + 1e-6)

        # Normalize: ratio ~ 1.0 means uniform (no blocking), > 1.3 means blocking
        if ratio > 1.5:
            return 1.0
        elif ratio > 1.3:
            return 0.6
        elif ratio > 1.1:
            return 0.3
        return 0.0

    def _compute_motion_blur_score(self, frames: List[np.ndarray]) -> float:
        """
        Detect motion blur by combining:
        1. Per-frame Laplacian variance (low = blurry)
        2. Frame-to-frame optical flow magnitude (high flow + low sharpness = motion blur)
        """
        if len(frames) < 2:
            return 0.0

        blur_indicators = []

        for i in range(len(frames) - 1):
            gray_curr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray_next = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)

            # Resize for faster optical flow computation
            scale = 320.0 / max(gray_curr.shape)
            if scale < 1.0:
                size = (int(gray_curr.shape[1] * scale), int(gray_curr.shape[0] * scale))
                gray_curr_small = cv2.resize(gray_curr, size)
                gray_next_small = cv2.resize(gray_next, size)
            else:
                gray_curr_small = gray_curr
                gray_next_small = gray_next

            # Laplacian variance of current frame
            lap_var = cv2.Laplacian(gray_curr, cv2.CV_64F).var()

            # Optical flow magnitude between frames
            flow = cv2.calcOpticalFlowFarneback(
                gray_curr_small, gray_next_small,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flow_magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            mean_flow = float(np.mean(flow_magnitude))

            # High motion + low sharpness = motion blur
            # mean_flow > 3 pixels + lap_var < 80 = likely motion blur
            if mean_flow > 3.0 and lap_var < 80:
                blur_indicators.append(1.0)
            elif mean_flow > 2.0 and lap_var < 50:
                blur_indicators.append(0.7)
            elif mean_flow > 5.0:
                blur_indicators.append(0.5)
            else:
                blur_indicators.append(0.0)

        return float(np.mean(blur_indicators)) if blur_indicators else 0.0

    def _compute_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Compute composite quality score from 0.0 (worst) to 1.0 (best)."""
        score = 0.0

        # Resolution contribution (0-0.3)
        pixels = metrics["width"] * metrics["height"]
        if pixels >= 1920 * 1080:
            score += 0.3
        elif pixels >= 1280 * 720:
            score += 0.2
        elif pixels >= 640 * 480:
            score += 0.1

        # Bitrate contribution (0-0.3)
        bitrate = metrics["estimated_bitrate_kbps"]
        if bitrate >= 5000:
            score += 0.3
        elif bitrate >= 2000:
            score += 0.2
        elif bitrate >= 1000:
            score += 0.1

        # Low compression contribution (0-0.2)
        comp = metrics["compression_score"]
        score += 0.2 * (1 - comp)

        # Low motion blur contribution (0-0.2)
        blur = metrics["motion_blur_score"]
        score += 0.2 * (1 - blur)

        return round(min(1.0, max(0.0, score)), 3)

    def _compute_quality_tier(self, metrics: Dict[str, Any]) -> str:
        """Classify video into quality tiers based on composite metrics."""
        qs = metrics["quality_score"]
        resolution = min(metrics["width"], metrics["height"])

        if qs >= 0.7 and resolution >= 720:
            return "high"
        elif qs >= 0.45 and resolution >= 480:
            return "medium"
        elif qs >= 0.25:
            return "low"
        else:
            return "very_low"

    def _default_metrics(self, error: str) -> Dict[str, Any]:
        """Return default metrics when video cannot be analyzed."""
        return {
            "width": 0,
            "height": 0,
            "fps": 0.0,
            "duration_s": 0.0,
            "total_frames": 0,
            "codec": "unknown",
            "estimated_bitrate_kbps": 0.0,
            "compression_score": 0.5,
            "motion_blur_score": 0.5,
            "laplacian_variance_mean": 0.0,
            "laplacian_variance_std": 0.0,
            "quality_score": 0.0,
            "quality_tier": "very_low",
            "error": error,
        }
