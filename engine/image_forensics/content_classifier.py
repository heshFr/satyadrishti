"""
CLIP Content-Type Classifier (Layer 0)
=======================================
Zero-shot content classification using CLIP to distinguish between:
  - photograph: Real camera-captured images of real scenes
  - cartoon_2d: 2D animated content (Tom & Jerry, SpongeBob, etc.)
  - anime: Japanese animation style (Naruto, Attack on Titan, etc.)
  - stop_motion: Stop-motion animation (Shaun the Sheep, Coraline, etc.)
  - cgi_3d: 3D CGI renders (Pixar, DreamWorks, etc.)
  - illustration: Digital paintings, concept art, hand-drawn art
  - screenshot_ui: App screenshots, UI mockups, terminal output
  - meme_composite: Memes, collages, text-overlaid images

This runs BEFORE any deepfake detection. When artistic content is detected,
the pipeline skips checks that assume photographic input (face geometry,
GAN frequency, ELA, noise patterns) and returns an "artistic-content" verdict
instead of a misleading "ai-generated" false positive.

Uses the CLIP model already loaded by CLIPDetector — no extra VRAM cost.
"""

import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── Content categories with carefully tuned CLIP prompts ──
# Each category has multiple prompts for robustness. CLIP compares
# the image embedding against all prompts and averages per-category.

CONTENT_CATEGORIES = {
    "photograph": [
        "a real photograph taken with a camera",
        "a photo of a real scene captured by a person",
        "a natural photograph with realistic lighting and depth of field",
        "a candid photo of real people or real objects",
    ],
    "cartoon_2d": [
        "a frame from a 2D cartoon or animated TV show",
        "a cartoon drawing with flat colors and outlines",
        "a frame from an animated cartoon like SpongeBob or Tom and Jerry",
        "a colorful 2D animated character with bold outlines",
    ],
    "anime": [
        "a frame from a Japanese anime show or manga",
        "an anime illustration with large eyes and stylized hair",
        "a scene from a Japanese animated series",
        "anime-style artwork with cel shading and vibrant colors",
    ],
    "stop_motion": [
        "a frame from a stop-motion clay animation film",
        "a claymation scene with physical clay or puppet characters",
        "a stop-motion animated scene like Wallace and Gromit or Shaun the Sheep",
        "a frame from a puppet animation with real physical miniature sets",
    ],
    "cgi_3d": [
        "a 3D computer-generated animated scene from a Pixar or DreamWorks movie",
        "a CGI rendered character with 3D lighting and textures",
        "a frame from a 3D animated film with realistic rendering",
        "a computer-generated 3D scene with ray-traced lighting",
    ],
    "illustration": [
        "a digital painting or hand-drawn illustration",
        "an artistic illustration or concept art painting",
        "a watercolor or oil painting artwork",
        "a hand-drawn sketch or digital art piece",
    ],
    "screenshot_ui": [
        "a screenshot of a computer application or website",
        "a user interface screenshot with buttons and text",
        "a screenshot of a mobile app or desktop software",
        "a terminal or code editor screenshot",
    ],
    "meme_composite": [
        "an internet meme with text overlay on an image",
        "a collage or composite image with multiple photos combined",
        "a meme format image with caption text",
        "a photoshopped or edited composite image with added text",
    ],
}

# Categories that are considered "artistic" — pipeline should skip
# photo-specific deepfake checks for these
ARTISTIC_CATEGORIES = {
    "cartoon_2d", "anime", "stop_motion", "cgi_3d", "illustration"
}

# "Hybrid" categories — run pipeline but with adjustments
HYBRID_CATEGORIES = {
    "meme_composite", "screenshot_ui"
}

# Human-readable display names
CATEGORY_DISPLAY_NAMES = {
    "photograph": "Photograph",
    "cartoon_2d": "2D Cartoon / Animation",
    "anime": "Anime",
    "stop_motion": "Stop-Motion Animation",
    "cgi_3d": "3D CGI / Animated Film",
    "illustration": "Illustration / Digital Art",
    "screenshot_ui": "Screenshot / UI",
    "meme_composite": "Meme / Composite",
}


class ContentClassifier:
    """
    CLIP-based zero-shot content type classifier.

    Reuses the CLIP model from CLIPDetector to avoid loading a second copy.
    Falls back to heuristic classification if CLIP isn't available.
    """

    # Minimum confidence to declare artistic content (prevents misclassifying
    # slightly stylized photos as cartoons)
    ARTISTIC_CONFIDENCE_THRESHOLD = 0.55

    # Minimum margin over photograph score to declare artistic
    # (the artistic category must beat photograph by at least this much)
    ARTISTIC_MARGIN_THRESHOLD = 0.10

    def __init__(self, clip_detector=None):
        """
        Args:
            clip_detector: An existing CLIPDetector instance to reuse.
                          If None, falls back to heuristic-only classification.
        """
        self.clip_detector = clip_detector
        self._prompt_features = None  # Cached text embeddings

    @torch.no_grad()
    def _encode_prompts(self):
        """Pre-compute and cache text embeddings for all category prompts."""
        if self._prompt_features is not None:
            return self._prompt_features

        if not self.clip_detector:
            return None

        model = self.clip_detector.model
        processor = self.clip_detector.processor
        device = self.clip_detector.device

        category_embeddings = {}
        for category, prompts in CONTENT_CATEGORIES.items():
            inputs = processor(text=prompts, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()
                      if k in ("input_ids", "attention_mask")}
            text_out = model.text_model(**inputs)
            text_features = model.text_projection(text_out.pooler_output)
            # L2 normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # Average across prompts for this category
            category_embeddings[category] = text_features.mean(dim=0)

        self._prompt_features = category_embeddings
        return category_embeddings

    @torch.no_grad()
    def classify(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Classify the content type of an image.

        Args:
            image: BGR numpy array (OpenCV format)

        Returns:
            {
                "content_type": str,         # e.g. "cartoon_2d", "photograph"
                "content_type_display": str,  # e.g. "2D Cartoon / Animation"
                "confidence": float,          # 0-1 confidence in the classification
                "is_artistic": bool,          # True if content is artistic (not photo)
                "is_hybrid": bool,            # True if meme/screenshot
                "category_scores": dict,      # Per-category similarity scores
                "photo_score": float,         # How photographic the image is
                "artistic_margin": float,     # Margin of top artistic over photograph
                "method": str,                # "clip_zero_shot" or "heuristic"
            }
        """
        if self.clip_detector and HAS_TORCH:
            return self._classify_clip(image)
        return self._classify_heuristic(image)

    @torch.no_grad()
    def _classify_clip(self, image: np.ndarray) -> Dict[str, Any]:
        """CLIP-based zero-shot classification."""
        category_embeddings = self._encode_prompts()
        if category_embeddings is None:
            return self._classify_heuristic(image)

        # Get image embedding (reuses CLIPDetector's extract_embedding)
        image_features = self.clip_detector.extract_embedding(image)  # (1, 768)
        image_features = image_features.squeeze(0)  # (768,)

        # Compute cosine similarity to each category
        scores = {}
        for category, text_emb in category_embeddings.items():
            sim = torch.dot(image_features, text_emb).item()
            scores[category] = sim

        # Softmax over similarities to get probabilities
        score_tensor = torch.tensor(list(scores.values()))
        # Temperature scaling — lower temperature = more decisive
        temperature = 0.07
        probs = torch.softmax(score_tensor / temperature, dim=0)
        category_names = list(scores.keys())
        prob_dict = {name: round(float(p), 4) for name, p in zip(category_names, probs)}

        # Find top category
        top_idx = probs.argmax().item()
        top_category = category_names[top_idx]
        top_confidence = float(probs[top_idx])

        photo_score = prob_dict.get("photograph", 0.0)

        # Artistic margin: how much the top artistic category beats photograph
        artistic_scores = {k: v for k, v in prob_dict.items() if k in ARTISTIC_CATEGORIES}
        top_artistic = max(artistic_scores.values()) if artistic_scores else 0.0
        top_artistic_name = max(artistic_scores, key=artistic_scores.get) if artistic_scores else None
        artistic_margin = top_artistic - photo_score

        # Decision: is this artistic content?
        is_artistic = (
            top_category in ARTISTIC_CATEGORIES
            and top_confidence >= self.ARTISTIC_CONFIDENCE_THRESHOLD
            and artistic_margin >= self.ARTISTIC_MARGIN_THRESHOLD
        )

        is_hybrid = top_category in HYBRID_CATEGORIES and top_confidence >= 0.45

        # If artistic isn't confident enough, check if heuristic analysis agrees
        if not is_artistic and top_artistic_name and artistic_margin > 0.03:
            heuristic = self._classify_heuristic(image)
            if heuristic["is_artistic"] and heuristic["confidence"] > 0.6:
                # Heuristic confirms — trust it
                is_artistic = True
                top_category = top_artistic_name
                top_confidence = top_artistic

        return {
            "content_type": top_category,
            "content_type_display": CATEGORY_DISPLAY_NAMES.get(top_category, top_category),
            "confidence": round(top_confidence, 4),
            "is_artistic": is_artistic,
            "is_hybrid": is_hybrid,
            "category_scores": prob_dict,
            "photo_score": photo_score,
            "artistic_margin": round(artistic_margin, 4),
            "method": "clip_zero_shot",
        }

    def _classify_heuristic(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Heuristic content classification when CLIP is unavailable.

        Uses visual features that distinguish photos from artwork:
        - Color quantization level (cartoons have few discrete colors)
        - Edge sharpness distribution (cartoons have crisp outlines)
        - Texture complexity (photos have fine textures, cartoons are smooth)
        - Color saturation patterns (anime tends to be highly saturated)
        """
        small = cv2.resize(image, (256, 256))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        scores = {cat: 0.0 for cat in CONTENT_CATEGORIES}

        # ── 1. Flat region analysis (cartoons have large flat-color areas) ──
        block_size = 8
        h, w = gray.shape
        flat_blocks = 0
        total_blocks = 0
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y:y + block_size, x:x + block_size]
                if np.std(block) < 3.0:
                    flat_blocks += 1
                total_blocks += 1
        flat_ratio = flat_blocks / max(total_blocks, 1)

        # ── 2. Edge analysis (cartoons have strong, clean edges) ──
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges > 0)

        # Strong edges with lots of flat areas = cartoon/anime
        # Strong edges with texture = photograph
        # Weak edges with flat areas = CGI/illustration

        # ── 3. Color uniqueness (unique colors after quantization) ──
        # Cartoons use a limited palette; photos use millions of colors
        small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        quantized = (small_rgb // 32) * 32  # Quantize to 8 levels per channel
        unique_colors = len(np.unique(quantized.reshape(-1, 3), axis=0))
        color_diversity = unique_colors / 512  # Normalize (max ~512 with this quantization)

        # ── 4. Saturation analysis ──
        avg_saturation = np.mean(hsv[:, :, 1]) / 255.0
        saturation_std = np.std(hsv[:, :, 1]) / 255.0

        # ── 5. Gradient smoothness (photos have continuous gradients) ──
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        # Bimodal gradient = cartoon (either flat or sharp edge, nothing in between)
        moderate_grad = np.sum((grad_mag > 5) & (grad_mag < 30))
        strong_grad = np.sum(grad_mag > 30)
        total_nonzero = np.sum(grad_mag > 1) + 1
        bimodal_ratio = strong_grad / total_nonzero  # High = cartoon-like edges

        # ── 6. Histogram coverage ──
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        used_bins = np.sum(hist > 0) / 256.0

        # ── Score each category ──

        # Photograph: high color diversity, moderate edges, smooth gradients, natural saturation
        scores["photograph"] = (
            min(1.0, color_diversity) * 0.3
            + (1.0 - flat_ratio) * 0.25
            + used_bins * 0.25
            + (moderate_grad / total_nonzero) * 0.2
        )

        # Cartoon 2D: flat areas, strong edges, limited colors, high saturation
        scores["cartoon_2d"] = (
            flat_ratio * 0.3
            + bimodal_ratio * 0.25
            + (1.0 - color_diversity) * 0.2
            + avg_saturation * 0.15
            + edge_density * 0.1
        )

        # Anime: similar to cartoon but even higher saturation, specific edge patterns
        scores["anime"] = (
            flat_ratio * 0.25
            + avg_saturation * 0.25
            + bimodal_ratio * 0.2
            + (1.0 - color_diversity) * 0.15
            + edge_density * 0.15
        )

        # Stop motion: moderate flat areas (clay has texture), moderate edges
        scores["stop_motion"] = (
            min(flat_ratio * 2, 0.5) * 0.2  # Some flat but not as much as cartoon
            + (1.0 - bimodal_ratio) * 0.2   # Softer edges than cartoon
            + avg_saturation * 0.2
            + (1.0 - used_bins) * 0.2
            + (moderate_grad / total_nonzero) * 0.2
        )

        # CGI 3D: smooth shading, moderate edges, high color diversity but smooth
        scores["cgi_3d"] = (
            (1.0 - bimodal_ratio) * 0.25
            + avg_saturation * 0.2
            + (1.0 - flat_ratio) * 0.2
            + (moderate_grad / total_nonzero) * 0.2
            + saturation_std * 0.15
        )

        # Illustration: moderate flat areas, varied saturation, artistic gradients
        scores["illustration"] = (
            flat_ratio * 0.2
            + avg_saturation * 0.2
            + saturation_std * 0.2
            + (1.0 - used_bins) * 0.2
            + (moderate_grad / total_nonzero) * 0.2
        )

        # Screenshot: very flat, low saturation, very limited colors, grid-like
        scores["screenshot_ui"] = (
            flat_ratio * 0.35
            + (1.0 - avg_saturation) * 0.25
            + (1.0 - edge_density) * 0.2
            + (1.0 - color_diversity) * 0.2
        )

        # Meme: mixed signals (hard to detect heuristically)
        scores["meme_composite"] = 0.1  # Low baseline; CLIP is needed for memes

        # Normalize to probabilities
        total = sum(scores.values())
        if total > 0:
            prob_dict = {k: round(v / total, 4) for k, v in scores.items()}
        else:
            prob_dict = {k: round(1.0 / len(scores), 4) for k in scores}

        top_category = max(prob_dict, key=prob_dict.get)
        top_confidence = prob_dict[top_category]
        photo_score = prob_dict.get("photograph", 0.0)

        artistic_scores = {k: v for k, v in prob_dict.items() if k in ARTISTIC_CATEGORIES}
        top_artistic = max(artistic_scores.values()) if artistic_scores else 0.0
        artistic_margin = top_artistic - photo_score

        # Heuristic needs higher confidence since it's less reliable
        is_artistic = (
            top_category in ARTISTIC_CATEGORIES
            and top_confidence >= 0.25
            and artistic_margin >= 0.05
        )

        is_hybrid = top_category in HYBRID_CATEGORIES and top_confidence >= 0.3

        return {
            "content_type": top_category,
            "content_type_display": CATEGORY_DISPLAY_NAMES.get(top_category, top_category),
            "confidence": round(top_confidence, 4),
            "is_artistic": is_artistic,
            "is_hybrid": is_hybrid,
            "category_scores": prob_dict,
            "photo_score": photo_score,
            "artistic_margin": round(artistic_margin, 4),
            "method": "heuristic",
        }

    def classify_video_frames(
        self,
        frames: list,
        sample_count: int = 5,
    ) -> Dict[str, Any]:
        """
        Classify content type from a set of video frames.
        Uses majority vote across sampled frames for robustness.

        Args:
            frames: List of BGR numpy arrays
            sample_count: Number of frames to sample (evenly spaced)

        Returns:
            Same format as classify() but with additional:
            - frame_votes: per-frame classifications
            - consensus_strength: how much frames agree (0-1)
        """
        if not frames:
            return {
                "content_type": "photograph",
                "content_type_display": "Photograph",
                "confidence": 0.0,
                "is_artistic": False,
                "is_hybrid": False,
                "category_scores": {},
                "photo_score": 1.0,
                "artistic_margin": 0.0,
                "method": "fallback_empty",
            }

        # Sample evenly spaced frames
        n = len(frames)
        step = max(1, n // sample_count)
        sampled = [frames[i] for i in range(0, n, step)][:sample_count]

        # Classify each frame
        frame_results = [self.classify(f) for f in sampled]

        # Aggregate category scores across frames
        all_categories = set()
        for r in frame_results:
            all_categories.update(r["category_scores"].keys())

        avg_scores = {}
        for cat in all_categories:
            cat_scores = [r["category_scores"].get(cat, 0.0) for r in frame_results]
            avg_scores[cat] = round(float(np.mean(cat_scores)), 4)

        # Majority vote on content type
        type_votes = [r["content_type"] for r in frame_results]
        from collections import Counter
        vote_counts = Counter(type_votes)
        top_type, top_count = vote_counts.most_common(1)[0]

        consensus_strength = top_count / len(type_votes)
        top_confidence = avg_scores.get(top_type, 0.0)
        photo_score = avg_scores.get("photograph", 0.0)

        artistic_scores = {k: v for k, v in avg_scores.items() if k in ARTISTIC_CATEGORIES}
        top_artistic = max(artistic_scores.values()) if artistic_scores else 0.0
        artistic_margin = top_artistic - photo_score

        # For video, require stronger consensus since individual frames can mislead
        is_artistic = (
            top_type in ARTISTIC_CATEGORIES
            and consensus_strength >= 0.6
            and (top_confidence >= self.ARTISTIC_CONFIDENCE_THRESHOLD or artistic_margin > 0.15)
        )

        is_hybrid = top_type in HYBRID_CATEGORIES and consensus_strength >= 0.5

        return {
            "content_type": top_type,
            "content_type_display": CATEGORY_DISPLAY_NAMES.get(top_type, top_type),
            "confidence": round(top_confidence, 4),
            "is_artistic": is_artistic,
            "is_hybrid": is_hybrid,
            "category_scores": avg_scores,
            "photo_score": photo_score,
            "artistic_margin": round(artistic_margin, 4),
            "consensus_strength": round(consensus_strength, 4),
            "frame_votes": type_votes,
            "method": frame_results[0]["method"] if frame_results else "unknown",
        }
