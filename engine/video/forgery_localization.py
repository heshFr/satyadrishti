"""
Video Forgery Localization
============================
Generates manipulation heatmaps showing WHERE in each frame the forgery
is detected, not just WHETHER a video is fake.

THEORETICAL FOUNDATION:
━━━━━━━━━━━━━━━━━━━━━━━
Standard deepfake detectors output a single binary decision (real/fake).
But for practical forensic use, we need to SHOW the evidence:

1. Attention-Based Localization (ViT)
   - Vision Transformers compute attention maps over 16x16 patches
   - Patches that the model "looks at" most for its decision reveal
     manipulation regions
   - The CLS token's attention to spatial patches = importance map
   - Averaging across attention heads gives a robust saliency map

2. Error Level Analysis (per-frame)
   - Resave frame at known quality, compute difference
   - Manipulated regions show different error levels
   - Frame-level ELA localizes edits within each frame

3. Noise Level Mapping
   - Extract high-frequency noise per frame region
   - Manipulated regions have inconsistent noise levels
   - Compute per-block noise variance, flag anomalous blocks

4. Face Region Focus
   - For face-swap deepfakes, the manipulation is on the face
   - Compare face region features vs background features
   - Boundary artifacts around face = strong indicator

OUTPUT:
━━━━━━━
  {
      "heatmaps": [ base64-encoded heatmap images per frame ],
      "manipulation_regions": [ { "frame": int, "bbox": [...], "confidence": float } ],
      "face_boundary_score": float,
      "attention_concentration": float,
      "score": float,
  }
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ForgeryLocalizer:
    """
    Generates manipulation heatmaps for video frames using multiple
    localization techniques.

    Can work with or without a ViT model:
    - With ViT: attention-based localization (most accurate)
    - Without ViT: noise + ELA localization (still useful)
    """

    def __init__(
        self,
        vit_model=None,
        block_size: int = 16,
        verbose: bool = False,
    ):
        """
        Args:
            vit_model: Optional ViT model for attention extraction.
                       Must support get_attention_maps() or have
                       accessible attention weights.
            block_size: Block size for noise/ELA analysis
            verbose: Print detailed analysis info
        """
        self.vit_model = vit_model
        self.block_size = block_size
        self.verbose = verbose

    def analyze_frame(
        self, frame: np.ndarray, return_heatmap_image: bool = True
    ) -> Dict[str, Any]:
        """
        Generate forgery localization for a single video frame.

        Args:
            frame: BGR frame (OpenCV format)
            return_heatmap_image: If True, include base64 heatmap in result

        Returns:
            Dict with localization results and optional heatmap image.
        """
        if frame is None or frame.size == 0:
            return self._empty_result()

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Layer 1: Noise inconsistency map
        noise_map = self._compute_noise_map(gray)

        # Layer 2: Error level analysis map
        ela_map = self._compute_ela_map(frame)

        # Layer 3: Face boundary analysis
        face_result = self._analyze_face_boundary(frame, gray)

        # Layer 4: Attention map (if ViT available)
        attention_map = self._compute_attention_map(frame) if self.vit_model else None

        # Combine maps into final heatmap
        combined_heatmap = self._combine_heatmaps(
            noise_map, ela_map, face_result.get("boundary_map"), attention_map, (h, w)
        )

        # Detect manipulation regions from combined heatmap
        regions = self._extract_regions(combined_heatmap, threshold=0.5)

        # Attention concentration (how focused the manipulation is)
        if combined_heatmap is not None and combined_heatmap.size > 0:
            sorted_vals = np.sort(combined_heatmap.flatten())[::-1]
            top_20_pct = sorted_vals[:max(1, len(sorted_vals) // 5)]
            concentration = float(np.sum(top_20_pct) / (np.sum(sorted_vals) + 1e-10))
        else:
            concentration = 0.0

        # Overall manipulation score
        scores = [
            float(np.max(noise_map)) if noise_map is not None else 0.0,
            float(np.max(ela_map)) if ela_map is not None else 0.0,
            face_result.get("boundary_score", 0.0),
        ]
        if attention_map is not None:
            scores.append(float(np.max(attention_map)))

        overall_score = float(np.mean(scores))

        result = {
            "manipulation_regions": regions,
            "face_boundary_score": round(face_result.get("boundary_score", 0.0), 4),
            "attention_concentration": round(concentration, 4),
            "noise_anomaly_max": round(float(np.max(noise_map)) if noise_map is not None else 0.0, 4),
            "ela_anomaly_max": round(float(np.max(ela_map)) if ela_map is not None else 0.0, 4),
            "score": round(overall_score, 4),
            "has_attention_map": attention_map is not None,
        }

        # Generate heatmap image
        if return_heatmap_image and combined_heatmap is not None:
            heatmap_image = self._render_heatmap(frame, combined_heatmap)
            result["heatmap_image"] = heatmap_image

        return result

    def analyze_video_frames(
        self,
        frames: List[np.ndarray],
        sample_rate: int = 4,
    ) -> Dict[str, Any]:
        """
        Analyze multiple video frames and return per-frame + aggregate results.

        Args:
            frames: List of BGR frames
            sample_rate: Analyze every Nth frame

        Returns:
            Dict with per-frame results and aggregate statistics.
        """
        per_frame = []
        all_scores = []

        for i in range(0, len(frames), sample_rate):
            frame = frames[i]
            result = self.analyze_frame(frame, return_heatmap_image=False)
            result["frame_index"] = i
            per_frame.append(result)
            all_scores.append(result["score"])

        if not per_frame:
            return self._empty_result()

        # Temporal consistency of manipulation regions
        # If manipulation appears in same region across frames = face-swap
        # If manipulation location varies = AI-generated
        region_consistency = self._compute_region_consistency(per_frame)

        # Aggregate
        return {
            "per_frame_results": per_frame[:20],  # cap for response size
            "frames_analyzed": len(per_frame),
            "mean_score": round(float(np.mean(all_scores)), 4),
            "max_score": round(float(np.max(all_scores)), 4),
            "region_consistency": round(region_consistency, 4),
            "manipulation_type_hint": (
                "face_swap" if region_consistency > 0.6 else
                "ai_generated" if region_consistency < 0.3 and np.mean(all_scores) > 0.4 else
                "unknown"
            ),
            "score": round(float(np.mean(all_scores)), 4),
        }

    def _compute_noise_map(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute per-block noise variance map.

        Blocks with significantly different noise = manipulated.
        Returns normalized anomaly map (0-1).
        """
        h, w = gray.shape
        bs = self.block_size

        n_rows = h // bs
        n_cols = w // bs

        if n_rows < 2 or n_cols < 2:
            return None

        # High-pass filter to extract noise
        noise = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)

        # Per-block noise variance
        variance_map = np.zeros((n_rows, n_cols))
        for i in range(n_rows):
            for j in range(n_cols):
                block = noise[i*bs:(i+1)*bs, j*bs:(j+1)*bs]
                variance_map[i, j] = np.var(block)

        # Compute z-scores (how different each block is from global)
        global_mean = np.mean(variance_map)
        global_std = np.std(variance_map)

        if global_std < 1e-10:
            return np.zeros((n_rows, n_cols))

        z_map = np.abs(variance_map - global_mean) / global_std

        # Normalize to 0-1
        anomaly_map = np.clip(z_map / 4.0, 0, 1)  # z>4 = definitely anomalous

        # Upscale to frame resolution
        anomaly_full = cv2.resize(
            anomaly_map.astype(np.float32), (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

        return anomaly_full

    def _compute_ela_map(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Error Level Analysis per pixel.

        Resave at known quality and compute difference — manipulated regions
        show different error levels.
        """
        try:
            # Encode at quality 95, then decode
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            _, encoded = cv2.imencode('.jpg', frame, encode_param)
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

            if decoded is None:
                return None

            # Absolute difference
            diff = cv2.absdiff(frame, decoded).astype(np.float32)

            # Convert to grayscale difference
            ela_gray = np.mean(diff, axis=2) if len(diff.shape) == 3 else diff

            # Normalize: scale to show anomalies
            # Most real regions have similar ELA; manipulated regions differ
            ela_mean = np.mean(ela_gray)
            ela_std = np.std(ela_gray)

            if ela_std < 1e-10:
                return np.zeros_like(ela_gray)

            # Z-score based anomaly
            ela_anomaly = np.abs(ela_gray - ela_mean) / (ela_std + 1e-10)
            ela_anomaly = np.clip(ela_anomaly / 3.0, 0, 1)

            return ela_anomaly

        except Exception as e:
            log.debug("ELA computation failed: %s", e)
            return None

    def _analyze_face_boundary(
        self, frame: np.ndarray, gray: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze face region boundary for manipulation artifacts.

        Face-swap deepfakes have artifacts at the face-background boundary.
        """
        # Detect face using Haar cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))

        if len(faces) == 0:
            return {"boundary_score": 0.0, "face_detected": False, "boundary_map": None}

        h, w = gray.shape
        boundary_map = np.zeros((h, w), dtype=np.float32)
        boundary_scores = []

        for (fx, fy, fw, fh) in faces:
            # Expand face region slightly
            margin = int(max(fw, fh) * 0.15)
            fx1 = max(0, fx - margin)
            fy1 = max(0, fy - margin)
            fx2 = min(w, fx + fw + margin)
            fy2 = min(h, fy + fh + margin)

            # Create boundary strip (10px around face rectangle)
            strip_width = max(5, int(fw * 0.05))

            # Inner boundary
            inner_mask = np.zeros((h, w), dtype=bool)
            inner_mask[fy1:fy2, fx1:fx1+strip_width] = True
            inner_mask[fy1:fy2, fx2-strip_width:fx2] = True
            inner_mask[fy1:fy1+strip_width, fx1:fx2] = True
            inner_mask[fy2-strip_width:fy2, fx1:fx2] = True

            # Outer boundary
            ofx1 = max(0, fx1 - strip_width)
            ofy1 = max(0, fy1 - strip_width)
            ofx2 = min(w, fx2 + strip_width)
            ofy2 = min(h, fy2 + strip_width)

            outer_mask = np.zeros((h, w), dtype=bool)
            outer_mask[ofy1:ofy2, ofx1:ofx1+strip_width] = True
            outer_mask[ofy1:ofy2, ofx2-strip_width:ofx2] = True
            outer_mask[ofy1:ofy1+strip_width, ofx1:ofx2] = True
            outer_mask[ofy2-strip_width:ofy2, ofx1:ofx2] = True

            # Gradient at boundary (sharp transition = face-swap artifact)
            grad_x = cv2.Sobel(gray.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)

            inner_gradient = float(np.mean(grad_mag[inner_mask])) if np.any(inner_mask) else 0
            outer_gradient = float(np.mean(grad_mag[outer_mask])) if np.any(outer_mask) else 0
            global_gradient = float(np.mean(grad_mag))

            # High boundary gradient relative to global = possible face-swap
            if global_gradient > 0:
                boundary_ratio = max(inner_gradient, outer_gradient) / (global_gradient + 1e-10)
                boundary_score = float(np.clip((boundary_ratio - 1.0) * 0.5, 0, 1))
            else:
                boundary_score = 0.0

            boundary_scores.append(boundary_score)

            # Add to boundary map
            boundary_map[inner_mask] = boundary_score
            boundary_map[outer_mask] = boundary_score * 0.7

            # Also check noise difference between face and background
            face_region = gray[fy:fy+fh, fx:fx+fw]
            face_noise = np.var(cv2.Laplacian(face_region.astype(np.float64), cv2.CV_64F))
            bg_noise = np.var(cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F))

            noise_ratio = face_noise / (bg_noise + 1e-10)
            if noise_ratio < 0.5 or noise_ratio > 2.0:
                boundary_scores[-1] = min(1.0, boundary_scores[-1] + 0.2)

        mean_boundary_score = float(np.mean(boundary_scores)) if boundary_scores else 0.0

        return {
            "boundary_score": mean_boundary_score,
            "face_detected": True,
            "face_count": len(faces),
            "boundary_map": boundary_map,
        }

    def _compute_attention_map(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract attention map from ViT model.

        Uses CLS token attention to spatial patches as saliency.
        """
        if self.vit_model is None or not HAS_TORCH:
            return None

        try:
            # Preprocess frame for ViT
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
            ])

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = transform(frame_rgb).unsqueeze(0)

            # Try to get attention weights
            if hasattr(self.vit_model, 'get_attention_maps'):
                attention = self.vit_model.get_attention_maps(input_tensor)
            else:
                # Generic ViT: register hooks on attention layers
                attention = self._extract_vit_attention(input_tensor)

            if attention is None:
                return None

            # Reshape attention to spatial grid
            # ViT-B/16 on 224x224: 14x14 = 196 patches
            n_patches = attention.shape[-1]
            grid_size = int(np.sqrt(n_patches))

            if grid_size * grid_size != n_patches:
                return None

            # CLS token's attention to spatial patches (last layer, mean across heads)
            attn_map = attention.reshape(grid_size, grid_size)

            # Normalize to 0-1
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-10)

            # Upscale to frame resolution
            h, w = frame.shape[:2]
            attn_full = cv2.resize(attn_map.astype(np.float32), (w, h),
                                   interpolation=cv2.INTER_LINEAR)

            return attn_full

        except Exception as e:
            log.debug("Attention map extraction failed: %s", e)
            return None

    def _extract_vit_attention(self, input_tensor) -> Optional[np.ndarray]:
        """Extract attention from a generic ViT model using hooks."""
        if not HAS_TORCH:
            return None

        attention_weights = []

        def hook_fn(module, input, output):
            if hasattr(output, 'shape') and len(output.shape) >= 3:
                attention_weights.append(output.detach().cpu().numpy())

        hooks = []
        # Try to find attention modules
        for name, module in self.vit_model.named_modules():
            if 'attn' in name.lower() and 'drop' not in name.lower():
                hooks.append(module.register_forward_hook(hook_fn))

        if not hooks:
            return None

        try:
            with torch.no_grad():
                self.vit_model(input_tensor)
        except Exception:
            pass
        finally:
            for h in hooks:
                h.remove()

        if not attention_weights:
            return None

        # Use last layer attention, CLS token (index 0) to spatial patches
        last_attn = attention_weights[-1]
        if len(last_attn.shape) == 4:
            # (batch, heads, seq, seq) — CLS attention to patches
            cls_attn = last_attn[0, :, 0, 1:]  # exclude CLS-to-CLS
            mean_attn = np.mean(cls_attn, axis=0)
        elif len(last_attn.shape) == 3:
            mean_attn = last_attn[0, 0, 1:]
        else:
            return None

        return mean_attn

    def _combine_heatmaps(
        self,
        noise_map: Optional[np.ndarray],
        ela_map: Optional[np.ndarray],
        face_boundary_map: Optional[np.ndarray],
        attention_map: Optional[np.ndarray],
        target_size: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """
        Combine multiple localization maps into a single heatmap.
        """
        h, w = target_size
        maps = []
        weights = []

        if noise_map is not None:
            resized = cv2.resize(noise_map.astype(np.float32), (w, h))
            maps.append(resized)
            weights.append(0.25)

        if ela_map is not None:
            resized = cv2.resize(ela_map.astype(np.float32), (w, h))
            maps.append(resized)
            weights.append(0.25)

        if face_boundary_map is not None and np.any(face_boundary_map > 0):
            resized = cv2.resize(face_boundary_map.astype(np.float32), (w, h))
            maps.append(resized)
            weights.append(0.30)

        if attention_map is not None:
            resized = cv2.resize(attention_map.astype(np.float32), (w, h))
            maps.append(resized)
            weights.append(0.40)

        if not maps:
            return np.zeros((h, w), dtype=np.float32)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Weighted combination
        combined = np.zeros((h, w), dtype=np.float32)
        for m, weight in zip(maps, weights):
            combined += m * weight

        # Normalize to 0-1
        if combined.max() > 0:
            combined = combined / combined.max()

        return combined

    def _extract_regions(
        self, heatmap: Optional[np.ndarray], threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Extract manipulation regions from heatmap using contour detection.
        """
        if heatmap is None:
            return []

        # Threshold
        binary = (heatmap > threshold).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # skip tiny regions
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Average heatmap value in this region
            region_mask = np.zeros_like(heatmap, dtype=np.uint8)
            cv2.drawContours(region_mask, [contour], -1, 1, -1)
            region_score = float(np.mean(heatmap[region_mask > 0]))

            regions.append({
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "area": int(area),
                "confidence": round(region_score, 4),
            })

        # Sort by confidence
        regions.sort(key=lambda r: r["confidence"], reverse=True)
        return regions[:10]

    def _render_heatmap(
        self, frame: np.ndarray, heatmap: np.ndarray
    ) -> np.ndarray:
        """
        Render heatmap overlay on the original frame.

        Returns BGR image with red-green overlay (red = manipulated).
        """
        h, w = frame.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Convert to colormap
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Blend with original
        alpha = 0.4
        overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)

        return overlay

    def _compute_region_consistency(
        self, per_frame: List[Dict[str, Any]]
    ) -> float:
        """
        Compute how consistent manipulation regions are across frames.

        High consistency = face-swap (always in same region)
        Low consistency = AI-generated (artifacts move around)
        """
        all_regions = []
        for result in per_frame:
            regions = result.get("manipulation_regions", [])
            if regions:
                # Use center of first (highest confidence) region
                bbox = regions[0]["bbox"]
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                all_regions.append((cx, cy))

        if len(all_regions) < 2:
            return 0.5

        # Compute variance of region centers
        centers = np.array(all_regions)
        x_std = np.std(centers[:, 0])
        y_std = np.std(centers[:, 1])

        # Normalize by frame size (assume ~500px as reference)
        spread = np.sqrt(x_std**2 + y_std**2) / 500.0

        # Low spread = high consistency = face-swap
        consistency = max(0.0, 1.0 - spread * 3.0)

        return float(consistency)

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "manipulation_regions": [],
            "face_boundary_score": 0.0,
            "attention_concentration": 0.0,
            "noise_anomaly_max": 0.0,
            "ela_anomaly_max": 0.0,
            "score": 0.0,
            "has_attention_map": False,
        }
