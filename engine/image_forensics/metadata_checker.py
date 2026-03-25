"""
Image Forensics — Metadata Checker
==================================
Extracts EXIF data and checks for C2PA (Coalition for Content Provenance
and Authenticity) digital signatures. AI generators often strip EXIF
or leave specific software tags (e.g., "Midjourney", "Stable Diffusion").
"""

import os
import struct
from PIL import Image, ExifTags
from typing import Dict, Any, Tuple

# Known tags inserted by AI generation tools
KNOWN_AI_SOFTWARE_TAGS = [
    "midjourney",
    "stable diffusion",
    "dall-e",
    "dalle",
    "novelai",
    "artbreeder",
    "runwayml",
    "dreamstudio",
    "adobe firefly",
    "civitai",
    "comfyui",
    "automatic1111",
    "invoke ai",
]

# C2PA JUMBF box type used in JPEG/PNG files
C2PA_JUMBF_LABEL = b"c2pa"
C2PA_MANIFEST_LABEL = b"c2pa.manifest"
JUMBF_BOX_TYPE = b"jumb"


class MetadataChecker:
    def __init__(self):
        pass

    def check_exif(self, image_path: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Extracts EXIF data and looks for AI generation flags.
        Returns (is_suspicious, details).
        """
        details = {
            "has_exif": False,
            "software": None,
            "camera_make": None,
            "camera_model": None,
            "datetime": None,
            "suspicious_tags_found": [],
        }

        try:
            with Image.open(image_path) as img:
                exif = img.getexif()

                if not exif:
                    return False, details

                details["has_exif"] = True

                for tag_id, value in exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, tag_id)

                    if tag_name == "Software":
                        details["software"] = str(value)
                    elif tag_name == "Make":
                        details["camera_make"] = str(value)
                    elif tag_name == "Model":
                        details["camera_model"] = str(value)
                    elif tag_name == "DateTime":
                        details["datetime"] = str(value)

                # Check for known AI software tags
                if details["software"]:
                    software_lower = details["software"].lower()
                    for ai_tag in KNOWN_AI_SOFTWARE_TAGS:
                        if ai_tag in software_lower:
                            details["suspicious_tags_found"].append(ai_tag)

                is_suspicious = len(details["suspicious_tags_found"]) > 0
                return is_suspicious, details

        except Exception as e:
            details["error"] = str(e)
            return False, details

    def check_c2pa(self, image_path: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check for C2PA content provenance manifests embedded in the file.

        C2PA manifests are stored as JUMBF (ISO 19566-5) boxes within JPEG APP11
        markers or PNG ancillary chunks. We scan the raw file bytes for the
        c2pa JUMBF signature to detect presence without requiring c2patool.

        Returns (has_c2pa, details).
        """
        details = {
            "has_c2pa_manifest": False,
            "c2pa_detected_in": None,
            "origin": "unknown",
        }

        try:
            file_size = os.path.getsize(image_path)
            # Only scan files under 200MB to avoid memory issues
            if file_size > 200 * 1024 * 1024:
                return False, details

            with open(image_path, "rb") as f:
                data = f.read()

            # Method 1: Search for JUMBF "c2pa" label in raw bytes
            # C2PA manifests contain the bytes "c2pa" as a JUMBF content type label
            if C2PA_JUMBF_LABEL in data:
                details["has_c2pa_manifest"] = True

                # Try to identify the source from manifest content
                if C2PA_MANIFEST_LABEL in data:
                    details["c2pa_detected_in"] = "jumbf_manifest"

                # Check for known C2PA signers
                data_lower = data.lower()
                if b"adobe" in data_lower:
                    details["origin"] = "Adobe (Content Credentials)"
                elif b"microsoft" in data_lower:
                    details["origin"] = "Microsoft"
                elif b"google" in data_lower:
                    details["origin"] = "Google"
                elif b"openai" in data_lower:
                    details["origin"] = "OpenAI"
                elif b"midjourney" in data_lower:
                    details["origin"] = "Midjourney"
                else:
                    details["origin"] = "C2PA signer (unknown)"

                return True, details

            # Method 2: Check JPEG APP11 markers (where JUMBF is stored in JPEG)
            if data[:2] == b"\xff\xd8":  # JPEG SOI marker
                has_jumbf = self._scan_jpeg_app11(data)
                if has_jumbf:
                    details["has_c2pa_manifest"] = True
                    details["c2pa_detected_in"] = "jpeg_app11"
                    return True, details

            return False, details

        except Exception as e:
            details["error"] = str(e)
            return False, details

    def _scan_jpeg_app11(self, data: bytes) -> bool:
        """Scan JPEG APP11 (0xFFEB) markers for JUMBF boxes."""
        offset = 2  # Skip SOI
        while offset < len(data) - 4:
            if data[offset] != 0xFF:
                break
            marker = data[offset:offset + 2]

            if marker == b"\xff\xeb":  # APP11 — JUMBF container
                if offset + 4 < len(data):
                    length = struct.unpack(">H", data[offset + 2:offset + 4])[0]
                    segment = data[offset + 4:offset + 2 + length]
                    if JUMBF_BOX_TYPE in segment or C2PA_JUMBF_LABEL in segment:
                        return True
                    offset += 2 + length
                else:
                    break
            elif marker == b"\xff\xda":  # SOS — start of scan data, stop parsing
                break
            elif marker in (b"\xff\xd0", b"\xff\xd1", b"\xff\xd2", b"\xff\xd3",
                            b"\xff\xd4", b"\xff\xd5", b"\xff\xd6", b"\xff\xd7"):
                offset += 2  # RST markers have no length
            else:
                if offset + 4 <= len(data):
                    length = struct.unpack(">H", data[offset + 2:offset + 4])[0]
                    offset += 2 + length
                else:
                    break

        return False

    def analyze(self, image_path: str) -> Tuple[float, Dict[str, Any]]:
        """
        Performs full metadata analysis.
        Returns an anomaly score (0.0=authentic, 1.0=AI) and details.
        """
        score = 0.0

        exif_suspicious, exif_details = self.check_exif(image_path)
        c2pa_present, c2pa_details = self.check_c2pa(image_path)

        if exif_suspicious:
            score += 0.9  # Very strong indicator if EXIF says "Midjourney"

        if not exif_details["has_exif"]:
            score += 0.1  # Weak indicator, common on web

        # C2PA presence is informational — it means the image has provenance
        # metadata which could indicate either authentic origin or AI-generated
        # with proper disclosure. We don't add to score since it's not suspicious.

        details = {
            "exif": exif_details,
            "c2pa": c2pa_details,
            "anomaly_score": min(1.0, score),
        }

        return score, details
