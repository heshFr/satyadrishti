"""
Image Forensics -- JPEG Compression Analysis
=============================================
Professional forensic tool for analyzing JPEG compression characteristics.

Capabilities:
1. JPEG quality estimation from quantization tables
2. Double JPEG compression detection (forensic gold standard)
3. Social media platform fingerprinting (WhatsApp, Instagram, Facebook, Telegram, Twitter)
4. Compression-aware scoring adjustments for the forensic pipeline

Why this matters:
- WhatsApp strips ALL EXIF, resizes to max 1600px, compresses at quality ~70-80
- Instagram strips EXIF, resizes to max 1080px, compresses at quality ~70-75
- Facebook strips EXIF, resizes to max 2048px, compresses at quality ~71-85
- Twitter strips EXIF, resizes to max 4096px, compresses at quality ~85
- Telegram (compressed): strips EXIF, compresses at quality ~75-85
- ALL of these destroy metadata, ELA, noise, and frequency patterns that
  the forensic pipeline relies on -- causing massive false positive rates
  on real photos shared through messaging apps.
"""

import io
import os
import struct
import numpy as np
from typing import Dict, Any, Optional, Tuple
from PIL import Image


# Standard JPEG luminance quantization table (quality 50)
# Used as reference for quality estimation
JPEG_STD_LUMINANCE_QT = np.array([
    16, 11, 10, 16,  24,  40,  51,  61,
    12, 12, 14, 19,  26,  58,  60,  55,
    14, 13, 16, 24,  40,  57,  69,  56,
    14, 17, 22, 29,  51,  87,  80,  62,
    18, 22, 37, 56,  68, 109, 103,  77,
    24, 35, 55, 64,  81, 104, 113,  92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103,  99,
], dtype=np.float32)

JPEG_STD_CHROMINANCE_QT = np.array([
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
], dtype=np.float32)


# Known social media platform resolution signatures
PLATFORM_MAX_DIMS = {
    "whatsapp": [(1600, 1600), (1280, 1280)],
    "instagram": [(1080, 1350), (1080, 1080), (1080, 608)],
    "facebook": [(2048, 2048), (960, 960)],
    "twitter": [(4096, 4096), (1200, 675)],
    "telegram": [(2560, 2560), (1280, 1280)],
}


class CompressionDetector:
    """Analyzes JPEG compression characteristics for forensic purposes."""

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Full compression analysis of an image file.

        Returns dict with:
            - is_jpeg: bool
            - estimated_quality: int (1-100) or None
            - is_double_compressed: bool
            - double_compression_evidence: str
            - platform_hint: str or None (e.g. "whatsapp", "instagram")
            - platform_confidence: float (0-1)
            - compression_severity: str ("none", "light", "moderate", "heavy")
            - statistical_reliability: float (0-1) -- how much to trust statistical checks
        """
        result = {
            "is_jpeg": False,
            "estimated_quality": None,
            "quantization_tables": [],
            "is_double_compressed": False,
            "double_compression_evidence": "",
            "platform_hint": None,
            "platform_confidence": 0.0,
            "compression_severity": "none",
            "statistical_reliability": 1.0,
            "image_dimensions": None,
        }

        if not os.path.exists(image_path):
            return result

        # Get image dimensions
        try:
            with Image.open(image_path) as img:
                result["image_dimensions"] = img.size  # (width, height)
        except Exception:
            return result

        # Check if JPEG
        ext = os.path.splitext(image_path)[1].lower()
        try:
            with open(image_path, "rb") as f:
                header = f.read(3)
            is_jpeg = header[:2] == b"\xff\xd8"
        except Exception:
            is_jpeg = False

        if not is_jpeg and ext not in (".jpg", ".jpeg"):
            # Not a JPEG -- compression analysis not applicable
            result["compression_severity"] = "none"
            result["statistical_reliability"] = 1.0
            return result

        result["is_jpeg"] = True

        # Extract quantization tables
        qt_tables = self._extract_quantization_tables(image_path)
        result["quantization_tables"] = [t.tolist() for t in qt_tables]

        # Estimate JPEG quality
        if qt_tables:
            quality = self._estimate_quality(qt_tables[0])
            result["estimated_quality"] = quality
        else:
            quality = None

        # Detect double JPEG compression
        dbl_compressed, dbl_evidence = self._detect_double_compression(image_path, qt_tables)
        result["is_double_compressed"] = dbl_compressed
        result["double_compression_evidence"] = dbl_evidence

        # Platform fingerprinting
        platform, platform_conf = self._identify_platform(
            image_path, result["image_dimensions"], quality, qt_tables
        )
        result["platform_hint"] = platform
        result["platform_confidence"] = platform_conf

        # Determine compression severity and statistical reliability
        if quality is not None:
            if quality <= 50:
                result["compression_severity"] = "heavy"
                result["statistical_reliability"] = 0.15  # almost useless
            elif quality <= 70:
                result["compression_severity"] = "heavy"
                result["statistical_reliability"] = 0.25
            elif quality <= 80:
                result["compression_severity"] = "moderate"
                result["statistical_reliability"] = 0.45
            elif quality <= 90:
                result["compression_severity"] = "light"
                result["statistical_reliability"] = 0.70
            else:
                result["compression_severity"] = "none"
                result["statistical_reliability"] = 0.90

        # Social media compression further reduces reliability
        if platform is not None and platform_conf >= 0.5:
            # Social media always strips metadata and recompresses
            result["statistical_reliability"] = min(
                result["statistical_reliability"], 0.20
            )

        # Double compression always reduces reliability
        if dbl_compressed:
            result["statistical_reliability"] = min(
                result["statistical_reliability"], 0.30
            )

        return result

    def _extract_quantization_tables(self, image_path: str) -> list:
        """
        Extract JPEG quantization tables by parsing the file's DQT markers.
        Returns list of numpy arrays (typically 2: luminance + chrominance).
        """
        tables = []
        try:
            with open(image_path, "rb") as f:
                data = f.read()

            offset = 2  # Skip SOI (0xFFD8)
            while offset < len(data) - 4:
                if data[offset] != 0xFF:
                    break

                marker = struct.unpack(">H", data[offset:offset + 2])[0]
                length = struct.unpack(">H", data[offset + 2:offset + 4])[0]

                if marker == 0xFFDB:  # DQT marker
                    # Parse quantization table(s) within this marker
                    pos = offset + 4
                    end = offset + 2 + length
                    while pos < end:
                        if pos >= len(data):
                            break
                        qt_info = data[pos]
                        precision = (qt_info >> 4) & 0x0F  # 0=8bit, 1=16bit
                        # table_id = qt_info & 0x0F
                        pos += 1

                        if precision == 0:
                            qt_size = 64
                        else:
                            qt_size = 128

                        if pos + qt_size > len(data):
                            break

                        if precision == 0:
                            qt = np.array(list(data[pos:pos + 64]), dtype=np.float32)
                        else:
                            qt = np.array(
                                [struct.unpack(">H", data[pos + i * 2:pos + i * 2 + 2])[0]
                                 for i in range(64)],
                                dtype=np.float32,
                            )
                        tables.append(qt)
                        pos += qt_size

                elif marker == 0xFFDA:  # SOS -- stop parsing headers
                    break
                elif marker in range(0xFFD0, 0xFFD8):  # RST markers
                    offset += 2
                    continue

                offset += 2 + length

        except Exception:
            pass

        return tables

    def _estimate_quality(self, qt_luminance: np.ndarray) -> int:
        """
        Estimate JPEG quality factor from the luminance quantization table.
        Uses the IJG (Independent JPEG Group) formula that maps QT values
        back to quality percentages.
        """
        if len(qt_luminance) != 64:
            return 75  # default guess

        # Compare against standard table to estimate quality scaling factor
        # Quality is inverse of quantization: higher QT values = lower quality
        std_qt = JPEG_STD_LUMINANCE_QT

        # Avoid division by zero
        ratios = qt_luminance / np.maximum(std_qt, 1.0)
        avg_ratio = np.mean(ratios)

        if avg_ratio < 0.01:
            return 100

        # Invert the IJG quality formula
        # For quality >= 50: scale = (200 - 2*quality) / 100
        # For quality < 50: scale = 50 / quality
        if avg_ratio < 1.0:
            # Quality >= 50
            quality = int(round(100 - (avg_ratio * 50)))
        else:
            # Quality < 50
            quality = int(round(50 / avg_ratio))

        return max(1, min(100, quality))

    def _detect_double_compression(
        self, image_path: str, qt_tables: list
    ) -> Tuple[bool, str]:
        """
        Detect double JPEG compression using DCT coefficient histogram analysis.

        When a JPEG is decompressed and recompressed (as social media apps do),
        the DCT coefficients show characteristic periodic artifacts -- specifically,
        the histogram of DCT coefficients develops periodic peaks aligned with
        the quantization step size from the FIRST compression.

        This is a forensic gold standard technique.
        """
        evidence = []

        try:
            # Load image and recompress to analyze DCT artifacts
            img = Image.open(image_path)
            if img.mode != "RGB":
                img = img.convert("RGB")

            img_array = np.array(img, dtype=np.float32)

            # Analyze each 8x8 block's DCT coefficients
            gray = np.mean(img_array, axis=2)
            h, w = gray.shape

            # Pad to multiple of 8
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            if pad_h > 0 or pad_w > 0:
                gray = np.pad(gray, ((0, pad_h), (0, pad_w)), mode="edge")

            h, w = gray.shape

            # Collect DCT coefficients (sample for speed)
            from scipy.fft import dctn
            all_ac_coeffs = []
            max_blocks = 2000  # sample for speed
            block_count = 0

            step_y = max(1, h // (8 * int(np.sqrt(max_blocks))))
            step_x = max(1, w // (8 * int(np.sqrt(max_blocks))))

            for y in range(0, h - 7, 8 * max(1, step_y)):
                for x in range(0, w - 7, 8 * max(1, step_x)):
                    block = gray[y:y + 8, x:x + 8]
                    dct_block = dctn(block, type=2, norm="ortho")
                    # Collect non-DC coefficients (skip [0,0])
                    ac = dct_block.flatten()[1:]
                    all_ac_coeffs.extend(ac.tolist())
                    block_count += 1
                    if block_count >= max_blocks:
                        break
                if block_count >= max_blocks:
                    break

            if len(all_ac_coeffs) < 100:
                return False, "insufficient data"

            all_ac_coeffs = np.array(all_ac_coeffs)

            # Histogram of DCT coefficients
            # Double compression creates periodic peaks at multiples of the
            # first compression's quantization step
            bin_range = 50
            hist, bin_edges = np.histogram(
                all_ac_coeffs,
                bins=np.arange(-bin_range, bin_range + 1, 1),
            )

            # Look for periodicity in histogram using autocorrelation
            hist_centered = hist.astype(np.float64) - np.mean(hist)
            autocorr = np.correlate(hist_centered, hist_centered, mode="full")
            autocorr = autocorr[len(autocorr) // 2:]  # take positive lags only

            if autocorr[0] > 0:
                autocorr = autocorr / autocorr[0]  # normalize

            # Check for periodic peaks (lag 2-15 are typical for QT steps)
            peaks = []
            for lag in range(2, min(16, len(autocorr))):
                if lag < len(autocorr) - 1:
                    if autocorr[lag] > autocorr[lag - 1] and autocorr[lag] > autocorr[lag + 1]:
                        if autocorr[lag] > 0.15:  # significant peak
                            peaks.append((lag, float(autocorr[lag])))

            if peaks:
                best_peak = max(peaks, key=lambda p: p[1])
                if best_peak[1] > 0.25:
                    evidence.append(
                        f"DCT histogram periodicity at step={best_peak[0]} "
                        f"(strength={best_peak[1]:.3f})"
                    )
                    return True, "; ".join(evidence)
                elif best_peak[1] > 0.15:
                    evidence.append(
                        f"Weak DCT periodicity at step={best_peak[0]} "
                        f"(strength={best_peak[1]:.3f})"
                    )

            # Also check for characteristic "comb" pattern in coefficient histogram
            # Real single-compressed images have smooth coefficient distributions
            # Double-compressed have sawtooth/comb patterns
            if len(hist) > 10:
                diffs = np.diff(hist.astype(np.float64))
                sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
                roughness = sign_changes / max(len(diffs) - 1, 1)

                if roughness > 0.7:
                    evidence.append(f"Comb pattern in DCT histogram (roughness={roughness:.3f})")
                    if roughness > 0.8:
                        return True, "; ".join(evidence)

        except ImportError:
            # scipy not available -- skip DCT analysis
            pass
        except Exception:
            pass

        if evidence:
            return False, "; ".join(evidence) + " (below threshold)"
        return False, "no double compression detected"

    def _identify_platform(
        self,
        image_path: str,
        dimensions: Optional[tuple],
        quality: Optional[int],
        qt_tables: list,
    ) -> Tuple[Optional[str], float]:
        """
        Identify if the image was processed by a known social media platform.

        Uses a combination of:
        - Image dimensions (platforms have characteristic max sizes)
        - JPEG quality level (platforms use specific quality ranges)
        - EXIF stripping (all platforms strip EXIF)
        - File size vs dimensions ratio
        """
        if dimensions is None:
            return None, 0.0

        w, h = dimensions
        max_dim = max(w, h)
        scores: Dict[str, float] = {}

        # Check EXIF presence (platforms strip it)
        has_exif = False
        try:
            with Image.open(image_path) as img:
                exif = img.getexif()
                has_exif = bool(exif) and len(exif) > 0
                # Check for camera make -- definitive sign of original photo
                for tag_id, value in exif.items():
                    from PIL import ExifTags
                    tag_name = ExifTags.TAGS.get(tag_id, "")
                    if tag_name == "Make" and value:
                        # Has camera make -- definitely NOT from social media
                        return None, 0.0
        except Exception:
            pass

        no_exif = not has_exif

        # WhatsApp: max 1600px, quality 70-80, no EXIF
        wa_score = 0.0
        if no_exif:
            wa_score += 0.3
        if quality and 60 <= quality <= 82:
            wa_score += 0.3
        if max_dim <= 1600 and max_dim >= 800:
            wa_score += 0.25
            if abs(max_dim - 1600) < 20:  # very close to 1600
                wa_score += 0.15
        scores["whatsapp"] = wa_score

        # Instagram: max 1080px, quality 70-78, no EXIF
        ig_score = 0.0
        if no_exif:
            ig_score += 0.3
        if quality and 65 <= quality <= 80:
            ig_score += 0.3
        if max_dim <= 1080 and max_dim >= 600:
            ig_score += 0.25
            if abs(max_dim - 1080) < 10:
                ig_score += 0.15
        scores["instagram"] = ig_score

        # Facebook: max 2048px, quality 71-85, no EXIF
        fb_score = 0.0
        if no_exif:
            fb_score += 0.3
        if quality and 70 <= quality <= 87:
            fb_score += 0.25
        if max_dim <= 2048 and max_dim >= 800:
            fb_score += 0.2
            if abs(max_dim - 2048) < 20 or abs(max_dim - 960) < 20:
                fb_score += 0.15
        scores["facebook"] = fb_score

        # Twitter/X: max 4096px, quality 80-90, no EXIF
        tw_score = 0.0
        if no_exif:
            tw_score += 0.3
        if quality and 78 <= quality <= 92:
            tw_score += 0.25
        if max_dim <= 4096 and max_dim >= 600:
            tw_score += 0.15
        scores["twitter"] = tw_score

        # Telegram (compressed mode): max 2560px, quality 75-85
        tg_score = 0.0
        if no_exif:
            tg_score += 0.3
        if quality and 72 <= quality <= 87:
            tg_score += 0.25
        if max_dim <= 2560 and max_dim >= 800:
            tg_score += 0.2
        scores["telegram"] = tg_score

        # Find best match
        if not scores:
            return None, 0.0

        best_platform = max(scores, key=scores.get)
        best_score = scores[best_platform]

        # Need at least 0.5 confidence to claim platform identification
        if best_score >= 0.5:
            return best_platform, min(1.0, best_score)

        # If no specific platform but clearly social-media-like
        # (no EXIF + moderate quality + reasonable dimensions)
        if no_exif and quality and quality < 88 and max_dim < 2600:
            return "social_media_generic", 0.4

        return None, 0.0
