"""
Image Forensics -- Main Detector (V2)
=====================================
Professional forensic pipeline combining multiple checks to evaluate
if an image is real or AI-generated.

V2 improvements over V1:
- Compression-aware scoring: detects WhatsApp/Instagram/social media compression
  and dampens unreliable statistical checks instead of generating false positives
- JPEG quality estimation from quantization tables
- Double JPEG compression detection (forensic gold standard)
- Social media platform fingerprinting
- Test-Time Augmentation (TTA) for neural model: 5 augmented views averaged
  for more robust predictions
- Quality-adaptive ELA: uses estimated JPEG quality for better ELA signal
- Better verdict logic with separate thresholds for compressed vs original images

Pipeline:
1. Compression Analysis (JPEG quality, platform ID, double compression)
2. Metadata Provenance (EXIF/C2PA)
3. GAN Fingerprint (Frequency Analysis)
4. Error Level Analysis (ELA) -- quality-adaptive
5. Noise Pattern Analysis
6. Pixel Statistics Analysis
7. Neural Deepfake Detection (ViT-B/16 with TTA)
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Tuple, List

from .frequency_analysis import FrequencyAnalyzer
from .metadata_checker import MetadataChecker
from .ela_analysis import ELAAnalyzer
from .noise_analysis import NoiseAnalyzer
from .pixel_statistics import PixelStatisticsAnalyzer
from .compression_detector import CompressionDetector
from .face_detector import FaceForensicsDetector
from .preprocessing import CompressionNormalizer

# Optional: ML components
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Optional: ViT detector (requires torch + transformers)
try:
    from .vit_detector import ViTDetector
    HAS_VIT = True
except ImportError:
    HAS_VIT = False

# Optional: CLIP detector (requires torch + transformers)
try:
    from .clip_detector import CLIPDetector
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False

# Optional: Content classifier (requires CLIP or falls back to heuristic)
try:
    from .content_classifier import ContentClassifier, ARTISTIC_CATEGORIES, CATEGORY_DISPLAY_NAMES
    HAS_CONTENT_CLASSIFIER = True
except ImportError:
    HAS_CONTENT_CLASSIFIER = False

# Optional: GAN/Diffusion fingerprint identifier
try:
    from .gan_fingerprint import GANFingerprintDetector
    HAS_GAN_FINGERPRINT = True
except ImportError:
    HAS_GAN_FINGERPRINT = False

# Optional: Inpainting & splice detection
try:
    from .inpainting_detector import InpaintingDetector
    HAS_INPAINTING = True
except ImportError:
    HAS_INPAINTING = False

# Phase 8: New forensic analyzers
try:
    from .spectral_analyzer import SpectralDecayAnalyzer
    HAS_SPECTRAL = True
except ImportError:
    HAS_SPECTRAL = False

try:
    from .color_forensics import ColorForensicsAnalyzer
    HAS_COLOR_FORENSICS = True
except ImportError:
    HAS_COLOR_FORENSICS = False

try:
    from .texture_forensics import TextureForensicsAnalyzer
    HAS_TEXTURE_FORENSICS = True
except ImportError:
    HAS_TEXTURE_FORENSICS = False

try:
    from .reconstruction_detector import ReconstructionDetector
    HAS_RECONSTRUCTION = True
except ImportError:
    HAS_RECONSTRUCTION = False

try:
    from .upsampling_detector import UpsamplingDetector
    HAS_UPSAMPLING = True
except ImportError:
    HAS_UPSAMPLING = False

# Infrastructure modules
try:
    from engine.common.uncertainty import UncertaintyQuantifier
    HAS_UNCERTAINTY = True
except ImportError:
    HAS_UNCERTAINTY = False

try:
    from engine.common.generator_attribution import GeneratorAttributor
    HAS_ATTRIBUTION = True
except ImportError:
    HAS_ATTRIBUTION = False

# Default model path
DEFAULT_MODEL_PATH = os.path.join("models", "image_forensics", "deepfake_vit_b16.pt")
DEFAULT_PRETRAINED_DIR = os.path.join("models", "image_forensics", "pretrained_vit")
# Number of frames to sample from video
VIDEO_SAMPLE_FRAMES = 16


def _to_jsonable(obj):
    """
    Recursively convert numpy scalars/arrays to native Python types so the
    forensics result is safely serializable by FastAPI's jsonable_encoder.

    With ~20 forensic modules each producing nested dicts, it is not feasible
    to fix every individual `result["x"] = some_numpy_op(...)` call site --
    new modules will keep introducing the bug. This boundary sanitizer
    catches all numpy types in one place. NaN/Inf are coerced to None so the
    JSON output is valid (FastAPI's encoder rejects non-finite floats).
    """
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):  # NaN/Inf
            return None
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        if f != f or f == float("inf") or f == float("-inf"):
            return None
        return f
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(x) for x in obj]
    # Fallback: anything else (e.g. custom object) -> string repr.
    # Better than crashing the entire request.
    try:
        return str(obj)
    except Exception:
        return None


class ImageForensicsDetector:
    def __init__(self, model_path: str = None, pretrained_dir: str = None, device: str = None):
        """
        Initializes the forensic pipeline.

        If model_path points to a valid ViT-B/16 checkpoint, the neural
        check uses ML inference. Otherwise falls back to frequency + metadata only.
        """
        self.freq_analyzer = FrequencyAnalyzer()
        self.meta_checker = MetadataChecker()
        self.ela_analyzer = ELAAnalyzer()
        self.noise_analyzer = NoiseAnalyzer()
        self.pixel_analyzer = PixelStatisticsAnalyzer()
        self.compression_detector = CompressionDetector()
        self.face_detector = FaceForensicsDetector()
        self.preprocessor = CompressionNormalizer()
        self.neural_detector = None
        self.clip_detector = None

        if HAS_VIT and HAS_TORCH:
            try:
                self.neural_detector = ViTDetector(device=device)
            except Exception as e:
                print(f"[Forensics] Could not initialize ViT: {e}")
                print("[Forensics] Running in statistical-only mode.")
        else:
            print("[Forensics] PyTorch/ViT not available. Using statistical methods only.")

        if HAS_CLIP:
            try:
                clip_probe_path = os.path.join(
                    "models", "image_forensics", "clip_probe.pt"
                )
                self.clip_detector = CLIPDetector(
                    probe_weights_path=clip_probe_path if os.path.exists(clip_probe_path) else None,
                    device=device,
                )
            except Exception as e:
                print(f"[Forensics] CLIP detector not available: {e}")
                print("[Forensics] Zero-shot CLIP will be attempted on first use.")

        # Content classifier: uses CLIP if available, falls back to heuristic
        self.content_classifier = None
        if HAS_CONTENT_CLASSIFIER:
            try:
                self.content_classifier = ContentClassifier(clip_detector=self.clip_detector)
            except Exception as e:
                print(f"[Forensics] Content classifier not available: {e}")

        # GAN/Diffusion fingerprint identifier
        self.gan_fingerprint = None
        if HAS_GAN_FINGERPRINT:
            try:
                self.gan_fingerprint = GANFingerprintDetector()
            except Exception as e:
                print(f"[Forensics] GAN fingerprint detector not available: {e}")

        # Inpainting & splice detector
        self.inpainting_detector = None
        if HAS_INPAINTING:
            try:
                self.inpainting_detector = InpaintingDetector()
            except Exception as e:
                print(f"[Forensics] Inpainting detector not available: {e}")

        # Phase 8: New forensic analyzers
        self.spectral_analyzer = SpectralDecayAnalyzer() if HAS_SPECTRAL else None
        self.color_forensics = ColorForensicsAnalyzer() if HAS_COLOR_FORENSICS else None
        self.texture_forensics = TextureForensicsAnalyzer() if HAS_TEXTURE_FORENSICS else None
        self.reconstruction_detector = ReconstructionDetector() if HAS_RECONSTRUCTION else None
        self.upsampling_detector = UpsamplingDetector() if HAS_UPSAMPLING else None
        self.uncertainty_quantifier = UncertaintyQuantifier() if HAS_UNCERTAINTY else None
        self.generator_attributor = GeneratorAttributor() if HAS_ATTRIBUTION else None

    @property
    def has_neural_model(self) -> bool:
        return self.neural_detector is not None

    @staticmethod
    def _is_photographic(image: np.ndarray, image_path: str) -> Tuple[bool, float]:
        """
        Classify whether the image is a photograph vs a graphic/screenshot/plot.

        Uses multiple heuristics:
        1. File format: JPEG is almost always a photo; PNG could be either
        2. Flat region percentage: graphics have many flat-color blocks
        3. Gradient smoothness: photos have smooth gradients, graphics have sharp edges
        4. Histogram coverage: photos use most of the 256 gray levels

        Returns (is_photo, photo_score) where photo_score 0-1 (higher = more photographic).
        """
        photo_score = 0.0
        small = cv2.resize(image, (256, 256))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # 1. File format heuristic
        ext = os.path.splitext(image_path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            photo_score += 0.35  # JPEG is strongly correlated with photos
        elif ext in (".png", ".webp", ".bmp"):
            photo_score += 0.0   # Could be either

        # 2. Flat region percentage
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
        # Photos: <15% flat blocks; Graphics: >30% flat blocks
        if flat_ratio < 0.15:
            photo_score += 0.25
        elif flat_ratio < 0.25:
            photo_score += 0.15

        # 3. Histogram coverage -- photos use nearly all 256 gray levels
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        used_bins = np.sum(hist > 0)
        bin_coverage = used_bins / 256.0
        if bin_coverage > 0.9:
            photo_score += 0.20
        elif bin_coverage > 0.75:
            photo_score += 0.10

        # 4. Gradient distribution -- photos have many soft gradients
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        # Ratio of moderate gradients (2-30) vs total gradient energy
        moderate = np.sum((grad_mag > 2) & (grad_mag < 30))
        total_grad = np.sum(grad_mag > 0)
        if total_grad > 0:
            soft_ratio = moderate / total_grad
            if soft_ratio > 0.6:
                photo_score += 0.20
            elif soft_ratio > 0.4:
                photo_score += 0.10

        photo_score = min(1.0, photo_score)
        is_photo = photo_score >= 0.45
        return is_photo, photo_score

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Runs the forensic pipeline on the given image.
        Returns a report compatible with the Media Scanner UI.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image at {image_path}")

        print(f"[Forensics] Analyzing: {os.path.basename(image_path)}")

        # ── Step 0: Content Type Classification ──
        # Detect artistic content (cartoons, anime, CGI, illustrations) BEFORE
        # running the deepfake pipeline. These content types trigger false positives
        # on checks designed for photographs.
        content_info = None
        if self.content_classifier:
            try:
                content_info = self.content_classifier.classify(image)
                ctype = content_info["content_type"]
                conf = content_info["confidence"]
                print(f"[Forensics] Content type: {content_info['content_type_display']} "
                      f"(confidence: {conf:.2%}, method: {content_info['method']})")

                if content_info["is_artistic"]:
                    # Artistic content — skip photo-specific deepfake checks entirely
                    return _to_jsonable({
                        "verdict": "artistic-content",
                        "confidence": round(conf * 100, 1),
                        "content_type": ctype,
                        "content_type_display": content_info["content_type_display"],
                        "forensic_checks": [
                            {
                                "id": "content_type",
                                "name": "Content Type Detection",
                                "status": "info",
                                "description": (
                                    f"Detected as {content_info['content_type_display']} "
                                    f"({conf:.0%} confidence). Deepfake analysis is designed "
                                    f"for photographic content and does not apply to artistic media."
                                ),
                            }
                        ],
                        "raw_scores": {
                            "content_classification": content_info["category_scores"],
                            "photo_score": content_info["photo_score"],
                            "artistic_margin": content_info["artistic_margin"],
                        },
                    })
            except Exception as e:
                print(f"[Forensics] Content classification failed: {e}")

        # ── Step 0b: Compression Analysis (V2) ──
        compression_info = self.compression_detector.analyze(image_path)
        estimated_quality = compression_info.get("estimated_quality")
        platform_hint = compression_info.get("platform_hint")
        stat_reliability = compression_info.get("statistical_reliability", 1.0)
        is_compressed = compression_info.get("compression_severity") in ("moderate", "heavy")
        is_social_media = platform_hint is not None and compression_info.get("platform_confidence", 0) >= 0.4

        if platform_hint:
            print(f"[Forensics] Compression: Q={estimated_quality}, platform={platform_hint} "
                  f"(conf={compression_info['platform_confidence']:.2f}), "
                  f"stat_reliability={stat_reliability:.2f}")
        elif estimated_quality:
            print(f"[Forensics] JPEG quality={estimated_quality}, severity={compression_info['compression_severity']}")

        # Detect if image is a photograph vs graphic/screenshot
        is_photo, photo_score = self._is_photographic(image, image_path)
        print(f"[Forensics] Image type: {'photograph' if is_photo else 'graphic/screenshot'} (score: {photo_score:.2f})")

        report: Dict[str, Any] = {
            "verdict": "authentic",
            "confidence": 0.0,
            "forensic_checks": [],
            "raw_scores": {},
        }

        # Include content classification in report (for non-artistic content)
        if content_info:
            report["content_type"] = content_info["content_type"]
            report["content_type_display"] = content_info["content_type_display"]
            report["raw_scores"]["content_classification"] = content_info["category_scores"]
            if content_info["is_hybrid"]:
                report["forensic_checks"].append({
                    "id": "content_type",
                    "name": "Content Type Detection",
                    "status": "info",
                    "description": (
                        f"Detected as {content_info['content_type_display']} — "
                        f"deepfake checks run with adjusted sensitivity."
                    ),
                })

        # Add compression check to report
        if is_social_media:
            report["forensic_checks"].append({
                "id": "compression",
                "name": "Compression Analysis",
                "status": "info",
                "description": (
                    f"Social media compression detected ({platform_hint}, "
                    f"JPEG Q={estimated_quality}). Statistical checks dampened "
                    f"-- metadata stripping and compression artifacts are expected."
                ),
            })
        elif is_compressed:
            report["forensic_checks"].append({
                "id": "compression",
                "name": "Compression Analysis",
                "status": "info",
                "description": (
                    f"Heavy JPEG compression detected (Q={estimated_quality}). "
                    f"Statistical checks have reduced reliability."
                ),
            })
        if compression_info.get("is_double_compressed"):
            report["forensic_checks"].append({
                "id": "double_compression",
                "name": "Double JPEG Compression",
                "status": "info",
                "description": (
                    f"Double JPEG compression detected: {compression_info['double_compression_evidence']}. "
                    f"Image has been saved as JPEG at least twice (common with social media sharing)."
                ),
            })

        report["raw_scores"]["compression"] = {
            "estimated_quality": estimated_quality,
            "platform_hint": platform_hint,
            "platform_confidence": compression_info.get("platform_confidence", 0),
            "is_double_compressed": compression_info.get("is_double_compressed", False),
            "statistical_reliability": stat_reliability,
        }

        # ── Step 1: Metadata Check ──
        meta_score, meta_details = self.meta_checker.analyze(image_path)
        
        # EXIF Triage for Smartphone Computational Photography
        camera_make = str(meta_details.get("exif", {}).get("camera_make", "")).lower()
        has_camera = bool(camera_make)
        has_exif = bool(meta_details.get("exif", {}).get("has_exif"))
        
        is_smartphone = any(m in camera_make for m in ["apple", "samsung", "google", "huawei", "xiaomi", "oppo", "vivo", "oneplus"])
        report["is_smartphone"] = is_smartphone

        # V2: DON'T penalize missing EXIF if social media compression detected
        if not has_exif and not has_camera:
            if is_social_media:
                # Social media strips EXIF -- this is EXPECTED, not suspicious
                meta_score = 0.0
            else:
                meta_score = max(meta_score, 0.3)

        report["raw_scores"]["metadata"] = meta_score

        status = "fail" if meta_score > 0.8 else "warn" if meta_score >= 0.3 else "pass"
        if is_social_media and not has_exif and meta_score < 0.3:
            desc = f"No EXIF (expected for {platform_hint} -- metadata stripped by platform)"
        elif meta_score > 0.8:
            desc = "Suspicious software signatures found"
        elif meta_score >= 0.3:
            desc = "No camera metadata -- possible AI origin"
        else:
            desc = "Metadata consistent with original camera source"
        report["forensic_checks"].append({
            "id": "c2pa",
            "name": "C2PA Provenance & EXIF",
            "status": status,
            "description": desc,
        })

        # ── Step 2: GAN Fingerprint / Frequency Analysis ──
        freq_score, freq_details = self.freq_analyzer.detect_artifacts(image)

        # V2: Dampen frequency score for compressed images
        # Compression creates high-frequency artifacts that mimic GAN signatures
        if is_compressed or is_social_media:
            freq_score *= stat_reliability

        report["raw_scores"]["frequency"] = freq_score

        status = "fail" if freq_score > 0.6 else "warn" if freq_score >= 0.3 else "pass"
        desc = (
            "High-frequency anomalies detected" if freq_score > 0.6
            else "Spectral irregularities found" if freq_score >= 0.3
            else "Natural frequency distribution"
        )
        if is_compressed and stat_reliability < 0.5:
            desc += " [compression-adjusted]"
        report["forensic_checks"].append({
            "id": "fingerprint",
            "name": "Frequency Spectrum Analysis",
            "status": status,
            "description": desc,
        })

        # ── Step 3: Error Level Analysis (ELA) -- quality-adaptive ──
        ela_score, ela_details = self.ela_analyzer.analyze(image, estimated_quality=estimated_quality)
        report["raw_scores"]["ela"] = ela_score

        status = "fail" if ela_score > 0.5 else "warn" if ela_score >= 0.25 else "pass"
        desc = (
            f"Uniform error levels indicate AI generation (score: {ela_score:.2f})"
            if ela_score > 0.5
            else f"Partially uniform compression patterns (score: {ela_score:.2f})"
            if ela_score >= 0.25
            else "Natural compression error distribution"
        )
        if ela_details.get("compression_note"):
            desc += f" [{ela_details['compression_note']}]"
        report["forensic_checks"].append({
            "id": "ela",
            "name": "Error Level Analysis",
            "status": status,
            "description": desc,
        })

        # ── Step 4: Noise Pattern Analysis ──
        noise_score, noise_details = self.noise_analyzer.analyze(image)

        # V2: Dampen noise score for compressed images
        # Compression smooths noise patterns -- creating same pattern as AI
        if is_compressed or is_social_media:
            noise_score *= stat_reliability

        report["raw_scores"]["noise"] = noise_score

        status = "fail" if noise_score > 0.5 else "warn" if noise_score >= 0.25 else "pass"
        desc = (
            f"Synthetic noise patterns detected (score: {noise_score:.2f})"
            if noise_score > 0.5
            else f"Unusual noise characteristics (score: {noise_score:.2f})"
            if noise_score >= 0.25
            else "Natural sensor noise patterns"
        )
        report["forensic_checks"].append({
            "id": "noise",
            "name": "Noise Pattern Analysis",
            "status": status,
            "description": desc,
        })

        # ── Step 5: Pixel Statistics Analysis ──
        pixel_score, pixel_details = self.pixel_analyzer.analyze(image)

        # V2: Dampen pixel score for compressed images (less aggressive than others)
        if is_compressed or is_social_media:
            pixel_score *= max(stat_reliability, 0.4)  # floor at 0.4 -- pixel stats are somewhat robust

        report["raw_scores"]["pixel"] = pixel_score

        status = "fail" if pixel_score > 0.5 else "warn" if pixel_score >= 0.25 else "pass"
        desc = (
            f"AI pixel signatures detected -- histogram/texture anomalies (score: {pixel_score:.2f})"
            if pixel_score > 0.5
            else f"Minor statistical anomalies found (score: {pixel_score:.2f})"
            if pixel_score >= 0.25
            else "Natural pixel statistics"
        )
        report["forensic_checks"].append({
            "id": "pixel",
            "name": "Pixel Statistics Analysis",
            "status": status,
            "description": desc,
        })

        # ── Step 5.5: Face-Specific Micro-Anomaly Detection ──
        face_score, face_details = self.face_detector.analyze_faces(image)
        report["raw_scores"]["face"] = face_score
        
        # If faces are found, it adds a massive flag for AI or Authentic Skin
        if face_details.get("status") == "success" and face_details.get("num_faces", 0) > 0:
            status = "fail" if face_score > 0.6 else "warn" if face_score >= 0.4 else "pass"
            desc = (
                f"Synthetic facial textures/lighting detected (score: {face_score:.2f})"
                if face_score > 0.6
                else f"Minor facial anomalies detected (score: {face_score:.2f})"
                if face_score >= 0.4
                else "Natural facial consistency"
            )
            report["forensic_checks"].append({
                "id": "face",
                "name": "Facial Micro-Consistency Analysis",
                "status": status,
                "description": desc,
            })

        # ── Pre-Step 6: Compression-Aware Preprocessing for Neural Model ──
        # Denoise compression artifacts before the ViT sees the image.
        # This brings compressed images closer to the clean distribution
        # the model was trained on, reducing false positives.
        preprocessed_image = image  # Default: use original
        preprocessing_info = {"preprocessing_applied": False}
        if self.neural_detector and (is_compressed or is_social_media):
            preprocessed_image, preprocessing_info = self.preprocessor.normalize(
                image,
                compression_severity=compression_info.get("compression_severity", "none"),
                is_social_media=is_social_media,
                estimated_quality=estimated_quality,
            )
            if preprocessing_info["preprocessing_applied"]:
                print(f"[Forensics] Preprocessing: {preprocessing_info['method']} (profile: {preprocessing_info.get('profile', 'none')})")

        # ── Step 6: Neural Deepfake Detection (ViT-B/16 with TTA) ──
        if self.neural_detector:
            # V2: Use TTA for more robust predictions (on preprocessed image)
            neural_score, neural_details = self.neural_detector.predict_tta(preprocessed_image)
            report["raw_scores"]["neural"] = neural_score

            if neural_details.get("status") == "success":
                neural_std = neural_details.get("std_fake_probability", 0)
                report["raw_scores"]["neural_tta_std"] = neural_std

                # High TTA variance means the model is uncertain
                # Penalize the effective score when augmented views disagree
                if neural_std > 0.15:
                    effective_neural = neural_score * 0.85
                    report["raw_scores"]["neural_effective"] = effective_neural
                    tta_note = (f" (TTA std={neural_std:.3f} — HIGH variance, "
                                f"effective score reduced to {effective_neural * 100:.1f}%)")
                elif neural_std > 0.08:
                    effective_neural = neural_score * 0.93
                    report["raw_scores"]["neural_effective"] = effective_neural
                    tta_note = (f" (TTA std={neural_std:.3f} — moderate variance, "
                                f"effective score={effective_neural * 100:.1f}%)")
                else:
                    effective_neural = neural_score
                    report["raw_scores"]["neural_effective"] = effective_neural
                    tta_note = f" (TTA: {neural_details.get('num_augmentations', 1)} views, consistent)"

                if preprocessing_info.get("preprocessing_applied"):
                    report["raw_scores"]["preprocessing"] = preprocessing_info

                status = "fail" if effective_neural > 0.65 else "warn" if effective_neural > 0.45 else "pass"
                desc = (
                    f"Neural deepfake detection: {effective_neural * 100:.1f}% AI probability{tta_note}"
                    if effective_neural > 0.55
                    else f"Neural analysis: uncertain ({effective_neural * 100:.1f}%){tta_note}"
                    if effective_neural > 0.35
                    else f"Neural analysis: likely authentic ({(1 - effective_neural) * 100:.1f}%){tta_note}"
                )
                report["forensic_checks"].append({
                    "id": "model",
                    "name": "ViT-B/16 Deepfake Detection",
                    "status": status,
                    "description": desc,
                })

        # ── Step 7: CLIP Semantic Analysis (if available) ──
        clip_score = None
        if self.clip_detector:
            try:
                clip_ai_prob, clip_details = self.clip_detector.predict(image)
                clip_score = clip_ai_prob
                report["raw_scores"]["clip"] = clip_ai_prob
                report["raw_scores"]["clip_details"] = clip_details

                clip_status = "fail" if clip_ai_prob > 0.70 else "warn" if clip_ai_prob > 0.55 else "pass"
                report["forensic_checks"].append({
                    "id": "clip_semantic",
                    "name": "CLIP Semantic Analysis",
                    "status": clip_status,
                    "description": (
                        f"Semantic AI probability: {clip_ai_prob*100:.1f}% "
                        f"({'AI patterns detected' if clip_ai_prob > 0.55 else 'Natural composition'})"
                    ),
                })
            except Exception as e:
                print(f"[Forensics] CLIP analysis failed: {e}")

        # ── Step 8: GAN/Diffusion Model Fingerprint ──
        if self.gan_fingerprint:
            try:
                gan_result = self.gan_fingerprint.analyze(image)
                gan_score = gan_result.get("score", 0.5)
                gen_class = gan_result.get("generator_class", "unknown")
                gen_name = gan_result.get("generator_name", "unknown")
                gen_conf = gan_result.get("generator_confidence", 0)

                report["raw_scores"]["gan_fingerprint"] = gan_score
                report["raw_scores"]["generator_class"] = gen_class
                report["raw_scores"]["generator_name"] = gen_name

                if gen_class != "real" and gen_class != "unknown" and gen_conf > 0.3:
                    gan_status = "fail" if gan_score > 0.6 else "warn" if gan_score > 0.4 else "info"
                    report["forensic_checks"].append({
                        "id": "gan_fingerprint",
                        "name": "AI Generator Fingerprint",
                        "status": gan_status,
                        "description": (
                            f"Generator class: {gen_class} "
                            f"(likely {gen_name}, {gen_conf*100:.0f}% match). "
                            f"Spectral fingerprint score: {gan_score*100:.1f}%."
                        ),
                    })
                else:
                    report["forensic_checks"].append({
                        "id": "gan_fingerprint",
                        "name": "AI Generator Fingerprint",
                        "status": "pass",
                        "description": (
                            f"No AI generator fingerprint detected. "
                            f"Spectral profile consistent with camera capture."
                        ),
                    })
            except Exception as e:
                print(f"[Forensics] GAN fingerprint analysis failed: {e}")

        # ── Step 9: Inpainting & Splice Detection ──
        if self.inpainting_detector:
            try:
                splice_result = self.inpainting_detector.analyze(image)
                splice_score = splice_result.get("score", 0)
                is_manipulated = splice_result.get("is_manipulated", False)
                manip_type = splice_result.get("manipulation_type", "none")
                manip_conf = splice_result.get("manipulation_confidence", 0)
                regions = splice_result.get("manipulation_regions", [])

                report["raw_scores"]["inpainting_score"] = splice_score
                report["raw_scores"]["manipulation_type"] = manip_type

                if is_manipulated and manip_conf > 0.3:
                    splice_status = "fail" if splice_score > 0.6 else "warn"
                    region_text = f" {len(regions)} region(s) identified." if regions else ""
                    report["forensic_checks"].append({
                        "id": "inpainting_detection",
                        "name": "Inpainting & Splice Detection",
                        "status": splice_status,
                        "description": (
                            f"Manipulation detected: {manip_type} ({manip_conf*100:.0f}% confidence). "
                            f"Noise: {splice_result.get('noise_map_anomaly', 0)*100:.0f}%, "
                            f"JPEG grid: {splice_result.get('jpeg_grid_anomaly', 0)*100:.0f}%, "
                            f"Edge: {splice_result.get('edge_anomaly', 0)*100:.0f}%.{region_text}"
                        ),
                    })
                else:
                    report["forensic_checks"].append({
                        "id": "inpainting_detection",
                        "name": "Inpainting & Splice Detection",
                        "status": "pass",
                        "description": (
                            f"No regional manipulation detected. "
                            f"Noise and compression patterns are consistent across image."
                        ),
                    })
            except Exception as e:
                print(f"[Forensics] Inpainting detection failed: {e}")

        # ── Step 10: Spectral Decay Analysis (Phase 8) ──
        if self.spectral_analyzer:
            try:
                spectral_result = self.spectral_analyzer.analyze(image)
                spectral_score = spectral_result["score"]
                if is_compressed or is_social_media:
                    spectral_score *= max(stat_reliability, 0.6)
                report["raw_scores"]["spectral_decay"] = spectral_score
                report["raw_scores"]["spectral_beta"] = spectral_result.get("beta_exponent", 0)

                s_status = "fail" if spectral_score > 0.6 else "warn" if spectral_score > 0.4 else "pass"
                report["forensic_checks"].append({
                    "id": "spectral_decay",
                    "name": "Spectral Decay Analysis",
                    "status": s_status,
                    "description": (
                        f"Power-law decay beta={spectral_result['beta_exponent']:.2f}, "
                        f"AI probability: {spectral_score*100:.1f}%"
                    ),
                })
            except Exception as e:
                print(f"[Forensics] Spectral analysis failed: {e}")

        # ── Step 11: Color Space Forensics (Phase 8) ──
        if self.color_forensics:
            try:
                color_result = self.color_forensics.analyze(image)
                color_score = color_result["score"]
                report["raw_scores"]["color_forensics"] = color_score

                c_status = "fail" if color_score > 0.6 else "warn" if color_score > 0.4 else "pass"
                report["forensic_checks"].append({
                    "id": "color_forensics",
                    "name": "Color Space Forensics",
                    "status": c_status,
                    "description": (
                        f"Color anomaly score: {color_score*100:.1f}% "
                        f"(LAB={color_result['lab_anomaly']:.2f}, "
                        f"HSV={color_result['hsv_anomaly']:.2f})"
                    ),
                })
            except Exception as e:
                print(f"[Forensics] Color forensics failed: {e}")

        # ── Step 12: Texture Synthesis Detection (Phase 8) ──
        if self.texture_forensics:
            try:
                texture_result = self.texture_forensics.analyze(image)
                texture_score = texture_result["score"]
                report["raw_scores"]["texture_forensics"] = texture_score

                t_status = "fail" if texture_score > 0.6 else "warn" if texture_score > 0.4 else "pass"
                report["forensic_checks"].append({
                    "id": "texture_forensics",
                    "name": "Texture Synthesis Detection",
                    "status": t_status,
                    "description": (
                        f"Texture anomaly score: {texture_score*100:.1f}% "
                        f"(LBP={texture_result['lbp_anomaly']:.2f}, "
                        f"GLCM={texture_result['glcm_anomaly']:.2f})"
                    ),
                })
            except Exception as e:
                print(f"[Forensics] Texture forensics failed: {e}")

        # ── Step 13: Reconstruction Consistency (Phase 8) ──
        if self.reconstruction_detector:
            try:
                recon_result = self.reconstruction_detector.analyze(image)
                recon_score = recon_result["score"]
                report["raw_scores"]["reconstruction"] = recon_score

                r_status = "fail" if recon_score > 0.6 else "warn" if recon_score > 0.4 else "pass"
                report["forensic_checks"].append({
                    "id": "reconstruction",
                    "name": "Reconstruction Consistency",
                    "status": r_status,
                    "description": (
                        f"Reconstruction anomaly: {recon_score*100:.1f}% "
                        f"(JPEG={recon_result['jpeg_consistency']:.2f}, "
                        f"blur={recon_result['blur_consistency']:.2f})"
                    ),
                })
            except Exception as e:
                print(f"[Forensics] Reconstruction analysis failed: {e}")

        # ── Step 14: Upsampling Artifact Detection (Phase 8) ──
        if self.upsampling_detector:
            try:
                upsamp_result = self.upsampling_detector.analyze(image)
                upsamp_score = upsamp_result["score"]
                report["raw_scores"]["upsampling"] = upsamp_score
                report["raw_scores"]["detected_patch_size"] = upsamp_result.get("detected_patch_size")

                u_status = "fail" if upsamp_score > 0.6 else "warn" if upsamp_score > 0.4 else "pass"
                patch_info = f", patch={upsamp_result['detected_patch_size']}px" if upsamp_result.get("detected_patch_size") else ""
                report["forensic_checks"].append({
                    "id": "upsampling",
                    "name": "Latent Upsampling Detection",
                    "status": u_status,
                    "description": (
                        f"Upsampling artifact score: {upsamp_score*100:.1f}%{patch_info}"
                    ),
                })
            except Exception as e:
                print(f"[Forensics] Upsampling detection failed: {e}")

        # ── Step 15: Generator Attribution (Phase 8) ──
        if self.generator_attributor:
            try:
                attr_result = self.generator_attributor.attribute_image(image, report["raw_scores"])
                report["raw_scores"]["attribution"] = attr_result
                if attr_result.get("is_ai") and attr_result.get("confidence", 0) > 0.3:
                    report["forensic_checks"].append({
                        "id": "attribution",
                        "name": "AI Generator Attribution",
                        "status": "info",
                        "description": (
                            f"Most likely generator: {attr_result['generator_family']} "
                            f"({attr_result['confidence']*100:.0f}% match)"
                        ),
                    })
                    report["generator_attribution"] = attr_result["generator_family"]
            except Exception as e:
                print(f"[Forensics] Generator attribution failed: {e}")

        # ── Step 16: Uncertainty Quantification (Phase 8) ──
        if self.uncertainty_quantifier:
            try:
                check_scores = {}
                score_keys = ["neural_effective", "neural", "clip", "frequency", "ela",
                              "noise", "pixel", "face", "gan_fingerprint", "inpainting_score",
                              "spectral_decay", "color_forensics", "texture_forensics",
                              "reconstruction", "upsampling"]
                for key in score_keys:
                    if key in report["raw_scores"] and report["raw_scores"][key] is not None:
                        val = report["raw_scores"][key]
                        if isinstance(val, (int, float)):
                            check_scores[key] = float(val)

                if len(check_scores) >= 3:
                    uncertainty_result = self.uncertainty_quantifier.quantify(check_scores)
                    report["raw_scores"]["uncertainty"] = uncertainty_result
                    report["uncertainty"] = uncertainty_result["uncertainty"]
                    report["confidence_level"] = uncertainty_result["confidence_level"]

                    if uncertainty_result["confidence_level"] in ("low", "very_low"):
                        report["forensic_checks"].append({
                            "id": "uncertainty",
                            "name": "Detection Confidence",
                            "status": "warn",
                            "description": uncertainty_result["recommendation"],
                        })
            except Exception as e:
                print(f"[Forensics] Uncertainty quantification failed: {e}")

        # ── Final Verdict (V2: compression-aware ensemble) ──
        scores = report["raw_scores"]

        if not is_photo:
            self._verdict_non_photo(report, scores, is_social_media, is_compressed, stat_reliability)
        else:
            self._verdict_photo(report, scores, is_social_media, is_compressed, stat_reliability, report.get("is_smartphone", False))

        report["confidence"] = round(report["confidence"], 1)

        # Add accuracy disclaimer when running without fine-tuned model
        if not self.neural_detector:
            report["forensic_checks"].append({
                "id": "disclaimer",
                "name": "Analysis Accuracy",
                "status": "warn",
                "description": "Running without neural model -- results based on statistical heuristics only.",
            })

        # Sanitize numpy types before returning -- FastAPI's jsonable_encoder
        # cannot handle np.bool_ / np.integer / np.floating / np.ndarray that
        # leak in from the ~20 forensic submodules.
        return _to_jsonable(report)

    def _verdict_non_photo(
        self,
        report: Dict,
        scores: Dict,
        is_social_media: bool = False,
        is_compressed: bool = False,
        stat_reliability: float = 1.0,
    ):
        """
        Verdict for non-photographic images (screenshots, graphics, diagrams).
        The ViT neural model was trained on photographic content — it is
        out-of-distribution for non-photos, so predictions are heavily discounted.
        V2 model scores screenshots at ~0.80, so thresholds are very high.
        """
        report["forensic_checks"].insert(0, {
            "id": "image_type",
            "name": "Image Type Detection",
            "status": "info",
            "description": (
                "Non-photographic image detected (screenshot/graphic). "
                "Neural deepfake detection has reduced reliability on this content type."
            ),
        })

        neural = scores.get("neural_effective", scores.get("neural", 0.0))
        meta = scores.get("metadata", 0.0)
        face = scores.get("face", 0.0)
        neural_tta_std = scores.get("neural_tta_std", 0.0)

        # EXIF says AI tool — works for any image type
        if meta > 0.8:
            report["verdict"] = "ai-generated"
            report["confidence"] = min(95.0, 65.0 + meta * 30)
            return

        # Face micro-anomaly — only if very strong
        if face > 0.82 and neural > 0.60:
            report["verdict"] = "ai-generated"
            report["confidence"] = min(82.0, 55.0 + (face + neural) / 2 * 30)
            return

        # Force inconclusive on high uncertainty
        if neural_tta_std > 0.15 and not (meta > 0.8 or (face > 0.82 and neural > 0.60)):
            report["verdict"] = "inconclusive"
            report["confidence"] = 50.0
            report["forensic_checks"].append({
                "id": "model_uncertain",
                "name": "High Model Uncertainty",
                "status": "warn",
                "description": (
                    f"High analysis uncertainty (TTA std={neural_tta_std:.3f}). "
                    f"Manual review recommended."
                ),
            })
            return

        # Pretrained model has proper calibration — use reasonable thresholds
        if neural > 0.65 and not is_social_media and not is_compressed:
            report["verdict"] = "ai-generated"
            report["confidence"] = min(80.0, 55.0 + neural * 30)
        elif neural > 0.55:
            report["verdict"] = "inconclusive"
            report["confidence"] = 50.0 + (neural - 0.50) * 40
        else:
            report["verdict"] = "authentic"
            report["confidence"] = min(85.0, 60.0 + (0.55 - neural) * 50)

    def _calibrate_neural_threshold(
        self,
        compression_severity: str,
        is_social_media: bool,
        is_smartphone: bool,
        stat_reliability: float,
        neural_tta_std: float,
    ) -> dict:
        """
        Calibration for the pretrained HuggingFace deepfake detector.

        The pretrained model (prithivMLmods/Deep-Fake-Detector-v2-Model)
        has a properly calibrated output distribution:
          - Real photos: typically 0.0-0.35
          - AI-generated: typically 0.65-1.0
          - Uncertain: 0.35-0.65

        Unlike the previous custom model (where real photos scored 0.80-0.94),
        this model produces meaningful probabilities.
        """
        fake_thresh = 0.60
        real_thresh = 0.40
        conf_cap = 95.0
        force_inconclusive = False

        # HIGH TTA VARIANCE: Model is uncertain, force inconclusive
        if neural_tta_std > 0.20:
            force_inconclusive = True
            conf_cap = 55.0
        elif neural_tta_std > 0.12:
            fake_thresh = 0.70
            real_thresh = 0.30
            conf_cap = 75.0

        # Compression adjustments — social media compression adds artifacts
        # but does NOT erase AI generation patterns. Be moderately conservative:
        # raise threshold enough to filter compression false positives but
        # not so much that we miss actual AI images shared via social media.
        if is_social_media:
            fake_thresh = max(fake_thresh, 0.72)
            real_thresh = min(real_thresh, 0.28)
            conf_cap = min(conf_cap, 82.0)
        elif compression_severity == "heavy":
            fake_thresh = max(fake_thresh, 0.68)
            real_thresh = min(real_thresh, 0.32)
            conf_cap = min(conf_cap, 85.0)

        if is_smartphone:
            conf_cap = min(conf_cap, 92.0)

        return {
            "fake_threshold": min(0.90, fake_thresh),
            "real_threshold": max(0.10, real_thresh),
            "confidence_cap": conf_cap,
            "force_inconclusive": force_inconclusive,
        }

    def _verdict_photo(
        self,
        report: Dict,
        scores: Dict,
        is_social_media: bool,
        is_compressed: bool,
        stat_reliability: float,
        is_smartphone: bool = False,
    ):
        """
        Verdict for photographic images using uncertainty-aware calibration.
        Three zones: confident-real, uncertain (ensemble decides), confident-fake.
        When TTA variance is high, forces inconclusive regardless of score.
        """
        neural = scores.get("neural_effective", scores.get("neural", 0.0))
        meta = scores.get("metadata", 0.0)
        face = scores.get("face", 0.0)
        noise = scores.get("noise", 0.0)
        neural_tta_std = scores.get("neural_tta_std", 0.0)

        compression_severity = "none"
        if is_social_media:
            compression_severity = "heavy"
        elif is_compressed:
            compression_severity = scores.get("compression", {}).get(
                "compression_severity", "moderate"
            ) if isinstance(scores.get("compression"), dict) else "moderate"

        # ── GATE 1: Metadata Override ──
        if meta > 0.8:
            report["verdict"] = "ai-generated"
            report["confidence"] = min(99.0, 70.0 + meta * 25)
            return

        # ── GATE 2: Face Micro-Anomaly ──
        # Strong face evidence (0.82+) combined with neural agreement
        neural = scores.get("neural_effective", scores.get("neural", 0.0))
        if face > 0.82 and neural > 0.60:
            report["verdict"] = "ai-generated"
            report["confidence"] = min(88.0, 60.0 + (face + neural) / 2 * 30)
            return

        # ── ADAPTIVE CALIBRATION ──
        cal = self._calibrate_neural_threshold(
            compression_severity=compression_severity,
            is_social_media=is_social_media,
            is_smartphone=is_smartphone,
            stat_reliability=stat_reliability,
            neural_tta_std=neural_tta_std,
        )

        fake_thresh = cal["fake_threshold"]
        real_thresh = cal["real_threshold"]
        conf_cap = cal["confidence_cap"]

        report["raw_scores"]["calibration"] = {
            "fake_threshold": round(fake_thresh, 3),
            "real_threshold": round(real_thresh, 3),
            "confidence_cap": round(conf_cap, 1),
            "compression_severity": compression_severity,
            "force_inconclusive": cal.get("force_inconclusive", False),
        }

        # ── GATE 3: Noise dead zones ──
        if noise > 0.65:
            report["verdict"] = "ai-generated"
            report["confidence"] = min(conf_cap, 50.0 + noise * 45)
            return

        # ── RULE 0: Forced Inconclusive on High Uncertainty ──
        # When TTA std > 0.15, the model is guessing. Never give a confident
        # verdict unless non-neural signals independently provide strong evidence.
        if cal.get("force_inconclusive") and not (meta > 0.8 or (face > 0.82 and neural > 0.60) or noise > 0.65):
            # Check if CLIP has a confident opinion even when ViT is uncertain
            clip_score = scores.get("clip")
            if clip_score is not None and clip_score > 0.70:
                # CLIP thinks it's AI despite ViT uncertainty — trust CLIP
                report["verdict"] = "ai-generated"
                report["confidence"] = min(75.0, 50.0 + clip_score * 30)
                report["forensic_checks"].append({
                    "id": "clip_override",
                    "name": "CLIP Semantic Override",
                    "status": "fail",
                    "description": (
                        f"While pixel-level analysis is uncertain due to compression, "
                        f"semantic analysis detected AI generation patterns ({clip_score*100:.1f}%)."
                    ),
                })
                return
            elif clip_score is not None and clip_score < 0.35:
                # CLIP thinks it's real despite ViT uncertainty — trust CLIP
                report["verdict"] = "authentic"
                report["confidence"] = min(75.0, 50.0 + (1 - clip_score) * 30)
                return

            # Neither ViT nor CLIP is confident — genuinely inconclusive
            report["verdict"] = "inconclusive"
            report["confidence"] = min(conf_cap, 55.0)
            report["forensic_checks"].append({
                "id": "model_uncertain",
                "name": "High Model Uncertainty",
                "status": "warn",
                "description": (
                    f"The AI detection model shows high uncertainty on this image "
                    f"(TTA std={neural_tta_std:.3f}). Different analysis angles give "
                    f"conflicting results. This often occurs with heavily compressed or "
                    f"multi-platform shared images where AI artifacts have been degraded. "
                    f"Manual review recommended."
                ),
            })
            return

        # ── Build statistical ensemble for the gray zone ──
        stat_checks = [c for c in report["forensic_checks"]
                       if c["id"] not in ("compression", "double_compression", "image_type",
                                          "model", "compression_conflict", "quality_disclaimer",
                                          "video_quality", "motion_blur", "model_uncertain")]

        n_fail = sum(1 for c in stat_checks if c["status"] == "fail")
        n_warn = sum(1 for c in stat_checks if c["status"] == "warn")
        n_pass = sum(1 for c in stat_checks if c["status"] == "pass")
        n_total = max(n_fail + n_warn + n_pass, 1)

        ensemble_score = (n_fail * 1.0 + n_warn * 0.3 + n_pass * (-0.5)) / n_total

        if is_social_media or (is_compressed and stat_reliability < 0.35):
            ensemble_score *= 0.3
        elif is_smartphone:
            ensemble_score *= 0.5

        # Use neural with statistical adjustment
        adjusted_neural = neural + ensemble_score * 0.05
        adjusted_neural = max(0.0, min(1.0, adjusted_neural))

        # ── ZONE 3: Neural confidently says fake (above fake_thresh) ──
        if adjusted_neural > fake_thresh:
            report["verdict"] = "ai-generated"
            raw_conf = 65.0 + (adjusted_neural - fake_thresh) / (1.0 - fake_thresh) * 30.0
            if ensemble_score > 0.2:
                raw_conf += 10
            report["confidence"] = min(conf_cap, raw_conf)
            return

        # ── ZONE 1: Neural confidently says real (below real_thresh) ──
        if adjusted_neural < real_thresh:
            report["verdict"] = "authentic"
            raw_conf = 65.0 + (real_thresh - adjusted_neural) / max(real_thresh, 0.01) * 30.0
            report["confidence"] = min(conf_cap, raw_conf)
            return

        # ── ZONE 2: Neural is uncertain — ensemble decides ──
        neural_lean = (adjusted_neural - 0.5) * 2

        # Include CLIP score if available (it's more reliable than pixel-level neural on compressed images)
        clip_score = scores.get("clip")
        if clip_score is not None:
            clip_lean = (clip_score - 0.5) * 2  # -1 to +1
            combined = neural_lean * 0.25 + clip_lean * 0.35 + ensemble_score * 0.40
        else:
            combined = neural_lean * 0.4 + ensemble_score * 0.6

        if combined > 0.3:
            report["verdict"] = "ai-generated"
            report["confidence"] = min(conf_cap, 55.0 + combined * 25)
        elif combined < -0.3:
            report["verdict"] = "authentic"
            report["confidence"] = min(conf_cap, 55.0 + abs(combined) * 25)
        else:
            report["verdict"] = "inconclusive"
            report["confidence"] = min(conf_cap, 50.0 + abs(combined) * 20)

    def analyze_video(
        self,
        video_path: str,
        num_frames: int = VIDEO_SAMPLE_FRAMES,
        quality_metrics: Dict = None,
    ) -> Dict[str, Any]:
        """
        Analyzes a video by extracting evenly-spaced frames and running
        the FULL forensic pipeline on each — not just ViT and frequency,
        but also CLIP semantic analysis, ELA, noise patterns, and pixel
        statistics.

        V3 improvements over V2:
        - CLIP semantic analysis on sampled frames (strong for AI-gen videos)
        - ELA analysis on frames (catches uniform compression in AI content)
        - Noise pattern analysis across frames (AI noise differs from sensor noise)
        - Pixel statistics aggregation across frames
        - Better scoring: uses all signals instead of just ViT + frequency
        - Compression-aware thresholds for all checks
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found at {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Video has no frames: {video_path}")

        # Pick evenly-spaced frame indices
        sample_count = min(num_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

        print(f"[Forensics] Video analysis V3: sampling {sample_count} frames from {total_frames} total")

        # Compute compression adjustment factor
        compression_adj = 1.0
        if quality_metrics and "compression_score" in quality_metrics:
            comp = quality_metrics["compression_score"]
            compression_adj = 1.0 - (comp * 0.5)

        # Per-frame score accumulators
        frame_scores: List[float] = []   # ViT neural scores
        freq_scores: List[float] = []    # Frequency analysis
        clip_scores: List[float] = []    # CLIP semantic AI probability
        ela_scores: List[float] = []     # ELA anomaly scores
        noise_scores: List[float] = []   # Noise pattern scores
        pixel_scores: List[float] = []   # Pixel statistics scores

        # CLIP is expensive — run on a subset of frames (every 4th)
        clip_indices = set(range(0, sample_count, 4))

        sampled_frames = []
        for frame_idx, idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            sampled_frames.append(frame)

            # ─── ViT Neural Score ───
            if self.neural_detector:
                if quality_metrics and quality_metrics.get("compression_score", 0) > 0.3:
                    frame_processed, _ = self.preprocessor.normalize(
                        frame,
                        compression_severity="moderate" if quality_metrics["compression_score"] < 0.6 else "heavy",
                        is_social_media=False,
                        estimated_quality=None,
                    )
                else:
                    frame_processed = frame
                score, _ = self.neural_detector.predict(frame_processed)
                frame_scores.append(score)

            # ─── Frequency Score ───
            freq_score, _ = self.freq_analyzer.detect_artifacts(frame)
            freq_score *= compression_adj
            freq_scores.append(freq_score)

            # ─── CLIP Semantic Score (subset) ───
            if self.clip_detector and frame_idx in clip_indices:
                try:
                    clip_prob, _ = self.clip_detector.predict(frame)
                    clip_scores.append(clip_prob)
                except Exception:
                    pass

            # ─── ELA Score ───
            try:
                ela_score_val, _ = self.ela_analyzer.analyze(frame)
                ela_scores.append(ela_score_val)
            except Exception:
                pass

            # ─── Noise Pattern Score ───
            try:
                noise_score_val, _ = self.noise_analyzer.analyze(frame)
                noise_scores.append(noise_score_val)
            except Exception:
                pass

            # ─── Pixel Statistics Score ───
            try:
                pixel_score_val, _ = self.pixel_analyzer.analyze(frame)
                pixel_scores.append(pixel_score_val)
            except Exception:
                pass

        cap.release()

        if not freq_scores and not sampled_frames:
            return _to_jsonable({
                "verdict": "inconclusive",
                "confidence": 0.0,
                "forensic_checks": [{
                    "id": "video_error",
                    "name": "Video Analysis",
                    "status": "warn",
                    "description": "Could not extract any valid frames from video.",
                }],
                "raw_scores": {},
            })

        # ── Content type classification on video frames ──
        if self.content_classifier and sampled_frames:
            try:
                video_content = self.content_classifier.classify_video_frames(
                    sampled_frames, sample_count=5
                )
                if video_content["is_artistic"] and video_content["consensus_strength"] >= 0.6:
                    return _to_jsonable({
                        "verdict": "artistic-content",
                        "confidence": round(video_content["confidence"] * 100, 1),
                        "content_type": video_content["content_type"],
                        "content_type_display": video_content["content_type_display"],
                        "forensic_checks": [{
                            "id": "content_type",
                            "name": "Content Type Detection",
                            "status": "info",
                            "description": (
                                f"Video detected as {video_content['content_type_display']} "
                                f"({video_content['consensus_strength']:.0%} frame consensus). "
                                f"Deepfake analysis does not apply to animated/artistic content."
                            ),
                        }],
                        "raw_scores": {
                            "content_classification": video_content["category_scores"],
                            "frame_votes": video_content["frame_votes"],
                            "consensus_strength": video_content["consensus_strength"],
                        },
                    })
            except Exception as e:
                print(f"[Forensics] Video content classification failed: {e}")

        # ─── Aggregate All Scores ───
        report: Dict[str, Any] = {
            "verdict": "authentic",
            "confidence": 0.0,
            "forensic_checks": [],
            "raw_scores": {},
        }

        # Frequency analysis aggregate
        avg_freq = float(np.mean(freq_scores)) if freq_scores else 0
        max_freq = float(np.max(freq_scores)) if freq_scores else 0
        report["raw_scores"]["frequency_avg"] = avg_freq
        report["raw_scores"]["frequency_max"] = max_freq

        status = "fail" if avg_freq > 0.7 else "warn" if avg_freq > 0.4 else "pass"
        desc = (
            f"Frequency anomalies detected across {sample_count} frames (avg: {avg_freq:.2f})"
            if avg_freq > 0.4
            else f"Natural frequency patterns across {sample_count} frames"
        )
        if compression_adj < 1.0:
            desc += f" [compression-adjusted x{compression_adj:.2f}]"
        report["forensic_checks"].append({
            "id": "video_frequency",
            "name": "Video Frequency Analysis",
            "status": status,
            "description": desc,
        })

        # Neural ViT aggregate
        avg_neural = 0.5
        if frame_scores:
            avg_neural = float(np.mean(frame_scores))
            max_neural = float(np.max(frame_scores))
            report["raw_scores"]["neural_avg"] = avg_neural
            report["raw_scores"]["neural_max"] = max_neural

            status = "fail" if avg_neural > 0.97 else "warn" if avg_neural > 0.95 else "pass"
            desc = (
                f"ViT-B/16 detected deepfake patterns in video frames "
                f"(avg: {avg_neural * 100:.1f}%, peak: {max_neural * 100:.1f}%)"
                if avg_neural > 0.4
                else f"Natural patterns confirmed across {sample_count} frames"
            )
            report["forensic_checks"].append({
                "id": "video_neural",
                "name": "ViT-B/16 Video Analysis",
                "status": status,
                "description": desc,
            })

        # CLIP semantic aggregate (strongest signal for AI-gen content)
        avg_clip = 0.5
        if clip_scores:
            avg_clip = float(np.mean(clip_scores))
            max_clip = float(np.max(clip_scores))
            report["raw_scores"]["clip_avg"] = avg_clip
            report["raw_scores"]["clip_max"] = max_clip

            status = "fail" if avg_clip > 0.65 else "warn" if avg_clip > 0.52 else "pass"
            if avg_clip > 0.65:
                desc = f"CLIP semantic analysis: {avg_clip * 100:.1f}% AI probability across {len(clip_scores)} frames (strong AI signal)"
            elif avg_clip > 0.52:
                desc = f"CLIP semantic analysis: {avg_clip * 100:.1f}% AI probability (mild AI patterns)"
            else:
                desc = f"CLIP semantic analysis: natural content ({avg_clip * 100:.1f}% AI probability)"
            report["forensic_checks"].append({
                "id": "video_clip",
                "name": "CLIP Semantic Video Analysis",
                "status": status,
                "description": desc,
            })

        # ELA aggregate
        avg_ela = 0.0
        if ela_scores:
            avg_ela = float(np.mean(ela_scores))
            report["raw_scores"]["ela_avg"] = avg_ela
            status = "fail" if avg_ela > 0.6 else "warn" if avg_ela > 0.3 else "pass"
            desc = f"Frame ELA: avg anomaly score {avg_ela:.2f} across {len(ela_scores)} frames"
            report["forensic_checks"].append({
                "id": "video_ela",
                "name": "Video ELA Analysis",
                "status": status,
                "description": desc,
            })

        # Noise pattern aggregate
        avg_noise = 0.0
        if noise_scores:
            avg_noise = float(np.mean(noise_scores))
            report["raw_scores"]["noise_avg"] = avg_noise
            status = "fail" if avg_noise > 0.6 else "warn" if avg_noise > 0.3 else "pass"
            desc = f"Noise patterns: avg anomaly {avg_noise:.2f} across {len(noise_scores)} frames"
            report["forensic_checks"].append({
                "id": "video_noise",
                "name": "Video Noise Pattern Analysis",
                "status": status,
                "description": desc,
            })

        # Pixel statistics aggregate
        avg_pixel = 0.0
        if pixel_scores:
            avg_pixel = float(np.mean(pixel_scores))
            report["raw_scores"]["pixel_avg"] = avg_pixel
            status = "fail" if avg_pixel > 0.6 else "warn" if avg_pixel > 0.3 else "pass"
            desc = f"Pixel statistics: avg anomaly {avg_pixel:.2f} across {len(pixel_scores)} frames"
            report["forensic_checks"].append({
                "id": "video_pixel",
                "name": "Video Pixel Statistics",
                "status": status,
                "description": desc,
            })

        # Frame consistency check
        if frame_scores and len(frame_scores) >= 2:
            score_std = float(np.std(frame_scores))
            consistency_status = "warn" if score_std > 0.2 else "pass"
            report["forensic_checks"].append({
                "id": "video_consistency",
                "name": "Temporal Consistency",
                "status": consistency_status,
                "description": (
                    f"Frame-to-frame score variance: {score_std:.3f} "
                    f"({'high -- possible splicing' if score_std > 0.2 else 'consistent'})"
                ),
            })

        # ─── V3 Multi-Signal Verdict ───
        # Weight all available signals for the final verdict.
        # ViT is the primary neural signal for video frames.
        # CLIP zero-shot is less reliable on video frames than still images
        # (trained on still images, video frames have motion blur / different
        #  characteristics), so we give it lower weight for video analysis.
        signal_weights = []
        signal_scores = []

        if frame_scores:
            signal_weights.append(2.5)  # ViT is the strongest per-frame signal
            signal_scores.append(avg_neural)

        if clip_scores:
            # CLIP weight reduced for video (less reliable on video frames)
            signal_weights.append(1.2)
            signal_scores.append(avg_clip)

        if freq_scores:
            signal_weights.append(0.6)
            signal_scores.append(avg_freq)

        if ela_scores:
            # ELA is less reliable on video frames (compression artifacts)
            signal_weights.append(0.4)
            signal_scores.append(avg_ela)

        if noise_scores:
            # Noise analysis on video frames is noisy itself — video compression
            # destroys sensor noise patterns in both real AND AI video.
            # Dampen to avoid false positives from compression artifacts.
            signal_weights.append(0.3)
            signal_scores.append(avg_noise)

        if pixel_scores:
            # Pixel stats also affected by video compression — reduce weight
            signal_weights.append(0.3)
            signal_scores.append(avg_pixel)

        if signal_weights:
            total_w = sum(signal_weights)
            combined = sum(s * w for s, w in zip(signal_scores, signal_weights)) / total_w
        else:
            combined = 0.5

        report["raw_scores"]["v3_combined"] = round(combined, 4)

        # Verdict thresholds — adjusted for compression
        fake_thresh = 0.58
        real_thresh = 0.38
        conf_cap = 92.0

        if quality_metrics and "compression_score" in quality_metrics:
            comp = quality_metrics["compression_score"]
            if comp > 0.5:
                fake_thresh += 0.04
                conf_cap -= 8.0
            elif comp > 0.3:
                fake_thresh += 0.02
                conf_cap -= 4.0

        if combined > fake_thresh:
            report["verdict"] = "ai-generated"
            report["confidence"] = round(min(conf_cap, 55.0 + (combined - fake_thresh) * 200), 1)
        elif combined < real_thresh:
            report["verdict"] = "authentic"
            report["confidence"] = round(min(conf_cap, 55.0 + (real_thresh - combined) * 200), 1)
        else:
            report["verdict"] = "inconclusive"
            report["confidence"] = round(50.0 + abs(combined - 0.48) * 100, 1)

        if not self.neural_detector:
            report["forensic_checks"].append({
                "id": "disclaimer",
                "name": "Analysis Accuracy",
                "status": "warn",
                "description": "Running without neural model -- video analysis based on statistical heuristics only (limited accuracy).",
            })

        # Sanitize numpy types before returning -- see analyze() for rationale.
        return _to_jsonable(report)
