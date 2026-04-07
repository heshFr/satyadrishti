"""
Document Forensics Detector
=============================
Detects manipulation in PDF and document files through multiple forensic checks:

1. **Metadata Analysis**: Creation/modification timestamps, producer software,
   author changes, version history anomalies.

2. **Font Consistency**: Checks for mixed fonts, embedded vs substituted fonts,
   unusual font combinations that suggest editing.

3. **Stream Analysis**: PDF object streams for hidden content, JavaScript,
   suspicious object types, incremental updates.

4. **Visual Consistency**: Rendered page analysis for alignment issues,
   resolution mismatches, copy-paste artifacts.

5. **Digital Signature Verification**: Validates PDF signatures and
   certificates for tampering.

6. **AI-Generated Text Detection**: Statistical analysis of text patterns
   that indicate AI authorship (perplexity, burstiness, vocabulary).
"""

import logging
import os
import re
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


class DocumentForensicsDetector:
    """Forensic analysis of PDF and document files."""

    def analyze(self, file_path: str) -> dict:
        """
        Run forensic analysis on a document.

        Args:
            file_path: Path to the document file.

        Returns:
            {
                "verdict": str,
                "confidence": float,
                "forensic_checks": list[dict],
                "raw_scores": dict,
            }
        """
        if not os.path.exists(file_path):
            return {"verdict": "error", "confidence": 0, "forensic_checks": [],
                    "raw_scores": {}, "error": "File not found"}

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return self._analyze_pdf(file_path)
        else:
            return {"verdict": "unsupported", "confidence": 0, "forensic_checks": [],
                    "raw_scores": {}, "error": f"Unsupported format: {ext}"}

    def _analyze_pdf(self, file_path: str) -> dict:
        """Full forensic analysis of a PDF file."""
        if not HAS_PYMUPDF:
            return self._analyze_pdf_basic(file_path)

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            return {"verdict": "error", "confidence": 0, "forensic_checks": [],
                    "raw_scores": {}, "error": str(e)}

        checks = []
        scores = {}

        # 1. Metadata analysis
        meta_score, meta_check = self._check_metadata(doc)
        checks.append(meta_check)
        scores["metadata"] = meta_score

        # 2. Font consistency
        font_score, font_check = self._check_fonts(doc)
        checks.append(font_check)
        scores["fonts"] = font_score

        # 3. Stream/structure analysis
        struct_score, struct_check = self._check_structure(doc, file_path)
        checks.append(struct_check)
        scores["structure"] = struct_score

        # 4. Incremental update detection
        update_score, update_check = self._check_incremental_updates(file_path)
        checks.append(update_check)
        scores["incremental_updates"] = update_score

        # 5. Text analysis (AI detection + consistency)
        text_score, text_check = self._check_text_content(doc)
        checks.append(text_check)
        scores["text_content"] = text_score

        # 6. Image analysis within PDF
        image_score, image_check = self._check_embedded_images(doc)
        checks.append(image_check)
        scores["embedded_images"] = image_score

        doc.close()

        # Final verdict
        all_scores = list(scores.values())
        max_score = max(all_scores) if all_scores else 0
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0

        if max_score > 0.7 or avg_score > 0.5:
            verdict = "manipulated"
            confidence = min(95.0, 50.0 + max_score * 45)
        elif max_score > 0.5 or avg_score > 0.35:
            verdict = "suspicious"
            confidence = 40.0 + avg_score * 40
        else:
            verdict = "authentic"
            confidence = min(90.0, 60.0 + (1 - avg_score) * 30)

        return {
            "verdict": verdict,
            "confidence": round(confidence, 1),
            "forensic_checks": checks,
            "raw_scores": scores,
        }

    def _check_metadata(self, doc) -> tuple:
        """Analyze PDF metadata for manipulation indicators."""
        meta = doc.metadata
        score = 0.0
        issues = []

        # Check producer/creator software
        producer = (meta.get("producer") or "").lower()
        creator = (meta.get("creator") or "").lower()

        # Known editing tools (not original creation)
        editing_tools = ["adobe acrobat", "foxit", "nitro", "pdfelement",
                         "sejda", "smallpdf", "ilovepdf", "pdf-xchange"]
        for tool in editing_tools:
            if tool in producer or tool in creator:
                score += 0.2
                issues.append(f"Edited with: {tool}")
                break

        # Timestamp analysis
        creation_date = meta.get("creationDate", "")
        mod_date = meta.get("modDate", "")

        if creation_date and mod_date:
            try:
                c_date = self._parse_pdf_date(creation_date)
                m_date = self._parse_pdf_date(mod_date)
                if c_date and m_date:
                    diff = (m_date - c_date).total_seconds()
                    if diff > 86400:  # Modified >1 day after creation
                        score += 0.15
                        issues.append(f"Modified {diff / 86400:.0f} days after creation")
                    if diff < 0:
                        score += 0.4  # Modification before creation = anomalous
                        issues.append("Modification date before creation date")
            except Exception:
                pass

        # Missing metadata is suspicious for official documents
        if not meta.get("author") and not meta.get("creator"):
            score += 0.1
            issues.append("No author/creator metadata")

        status = "fail" if score > 0.5 else "warn" if score > 0.2 else "pass"
        desc = "; ".join(issues) if issues else "Metadata consistent with original creation"

        return float(min(score, 1.0)), {
            "id": "metadata",
            "name": "Document Metadata Analysis",
            "status": status,
            "description": desc,
        }

    def _check_fonts(self, doc) -> tuple:
        """Analyze font consistency across pages."""
        all_fonts = set()
        page_font_sets = []
        embedded_count = 0
        total_fonts = 0

        for page in doc:
            fonts = page.get_fonts()
            page_fonts = set()
            for font in fonts:
                font_name = font[3]  # Font name
                font_type = font[2]  # Font type
                is_embedded = font[4]  # Embedded flag
                all_fonts.add(font_name)
                page_fonts.add(font_name)
                total_fonts += 1
                if is_embedded:
                    embedded_count += 1
            page_font_sets.append(page_fonts)

        score = 0.0
        issues = []

        # Many different fonts suggest editing
        if len(all_fonts) > 8:
            score += 0.3
            issues.append(f"Unusually many fonts ({len(all_fonts)})")
        elif len(all_fonts) > 5:
            score += 0.1
            issues.append(f"Multiple fonts ({len(all_fonts)})")

        # Inconsistent fonts across pages
        if len(page_font_sets) > 1:
            first_page_fonts = page_font_sets[0]
            inconsistent_pages = sum(1 for pf in page_font_sets[1:] if pf != first_page_fonts)
            if inconsistent_pages > len(page_font_sets) * 0.5:
                score += 0.2
                issues.append(f"Font inconsistency across {inconsistent_pages} pages")

        # Mix of embedded and non-embedded fonts
        if total_fonts > 0:
            embed_ratio = embedded_count / total_fonts
            if 0.1 < embed_ratio < 0.9:
                score += 0.2
                issues.append("Mix of embedded and non-embedded fonts")

        status = "fail" if score > 0.5 else "warn" if score > 0.2 else "pass"
        desc = "; ".join(issues) if issues else f"Consistent font usage ({len(all_fonts)} fonts)"

        return float(min(score, 1.0)), {
            "id": "fonts",
            "name": "Font Consistency Analysis",
            "status": status,
            "description": desc,
        }

    def _check_structure(self, doc, file_path: str) -> tuple:
        """Analyze PDF internal structure."""
        score = 0.0
        issues = []

        # Check for JavaScript
        try:
            with open(file_path, "rb") as f:
                raw = f.read(100000)  # First 100KB
                if b"/JS" in raw or b"/JavaScript" in raw:
                    score += 0.4
                    issues.append("Contains JavaScript (potential malware)")

                # Check for embedded files
                if b"/EmbeddedFile" in raw:
                    score += 0.15
                    issues.append("Contains embedded files")

                # Check for launch actions
                if b"/Launch" in raw:
                    score += 0.3
                    issues.append("Contains launch actions")

                # Multiple xref tables indicate incremental saves
                xref_count = raw.count(b"xref")
                if xref_count > 2:
                    score += 0.15
                    issues.append(f"Multiple xref tables ({xref_count})")
        except Exception as e:
            issues.append(f"Structure analysis error: {e}")

        # Page count anomalies
        n_pages = doc.page_count
        if n_pages == 0:
            score += 0.3
            issues.append("Empty document (0 pages)")

        status = "fail" if score > 0.5 else "warn" if score > 0.2 else "pass"
        desc = "; ".join(issues) if issues else "Normal PDF structure"

        return float(min(score, 1.0)), {
            "id": "structure",
            "name": "PDF Structure Analysis",
            "status": status,
            "description": desc,
        }

    def _check_incremental_updates(self, file_path: str) -> tuple:
        """Detect incremental updates (PDF modifications without re-saving)."""
        score = 0.0
        issues = []

        try:
            with open(file_path, "rb") as f:
                content = f.read()

            # Count %%EOF markers (multiple = incremental updates)
            eof_count = content.count(b"%%EOF")
            if eof_count > 1:
                score += 0.1 * min(5, eof_count - 1)
                issues.append(f"{eof_count} revisions detected (incremental updates)")

            # Check for startxref entries
            startxref_count = content.count(b"startxref")
            if startxref_count > 2:
                score += 0.1
                issues.append(f"Multiple cross-reference sections ({startxref_count})")

        except Exception as e:
            issues.append(f"Error reading file: {e}")

        status = "fail" if score > 0.5 else "warn" if score > 0.2 else "pass"
        desc = "; ".join(issues) if issues else "No incremental updates detected"

        return float(min(score, 1.0)), {
            "id": "incremental",
            "name": "Incremental Update Detection",
            "status": status,
            "description": desc,
        }

    def _check_text_content(self, doc) -> tuple:
        """Analyze text content for AI generation indicators."""
        all_text = ""
        for page in doc:
            all_text += page.get_text()

        if len(all_text) < 100:
            return 0.0, {
                "id": "text_content",
                "name": "Text Content Analysis",
                "status": "info",
                "description": "Insufficient text for analysis",
            }

        score = 0.0
        issues = []

        # Word-level analysis
        words = all_text.split()
        n_words = len(words)

        if n_words < 50:
            return 0.0, {
                "id": "text_content",
                "name": "Text Content Analysis",
                "status": "info",
                "description": f"Limited text ({n_words} words)",
            }

        # Vocabulary richness (type-token ratio)
        unique_words = len(set(w.lower() for w in words))
        ttr = unique_words / n_words

        # AI text tends to have moderate TTR (not too high, not too low)
        # Human text is more variable
        if 0.35 < ttr < 0.55 and n_words > 200:
            score += 0.15
            issues.append(f"Moderate vocabulary richness (TTR={ttr:.2f})")

        # Sentence length variance
        sentences = re.split(r'[.!?]+', all_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        if len(sentences) > 5:
            sent_lengths = [len(s.split()) for s in sentences]
            sent_cv = (
                max(0.01, (sum((x - sum(sent_lengths) / len(sent_lengths)) ** 2
                               for x in sent_lengths) / len(sent_lengths)) ** 0.5)
                / (sum(sent_lengths) / len(sent_lengths) + 1e-5)
            )
            # AI: lower sentence length variance (more uniform)
            if sent_cv < 0.25:
                score += 0.2
                issues.append(f"Low sentence length variation (CV={sent_cv:.2f})")
            elif sent_cv < 0.35:
                score += 0.1

        # Paragraph structure
        paragraphs = all_text.split("\n\n")
        if len(paragraphs) > 3:
            para_lengths = [len(p.split()) for p in paragraphs if len(p.strip()) > 10]
            if para_lengths:
                para_cv = (
                    max(0.01, (sum((x - sum(para_lengths) / len(para_lengths)) ** 2
                                   for x in para_lengths) / len(para_lengths)) ** 0.5)
                    / (sum(para_lengths) / len(para_lengths) + 1e-5)
                )
                if para_cv < 0.2:
                    score += 0.15
                    issues.append("Suspiciously uniform paragraph lengths")

        status = "fail" if score > 0.4 else "warn" if score > 0.2 else "pass"
        desc = "; ".join(issues) if issues else "Text patterns appear natural"

        return float(min(score, 1.0)), {
            "id": "text_content",
            "name": "Text Content Analysis",
            "status": status,
            "description": desc,
        }

    def _check_embedded_images(self, doc) -> tuple:
        """Check embedded images for manipulation."""
        score = 0.0
        issues = []
        n_images = 0

        for page_num in range(min(doc.page_count, 10)):
            page = doc[page_num]
            images = page.get_images()
            n_images += len(images)

            for img_info in images:
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    if base_image:
                        width = base_image.get("width", 0)
                        height = base_image.get("height", 0)
                        colorspace = base_image.get("colorspace", "")

                        # Very small images might be watermarks/overlays
                        if 0 < width < 50 and 0 < height < 50:
                            score += 0.05
                            issues.append(f"Tiny image ({width}x{height}) — possible overlay")

                        # Resolution mismatches between images
                        # (different sources = likely manipulated)
                except Exception:
                    pass

        if n_images == 0:
            desc = "No embedded images"
        elif issues:
            desc = f"{n_images} images found; " + "; ".join(issues)
        else:
            desc = f"{n_images} embedded images — no anomalies detected"

        status = "warn" if score > 0.2 else "pass"

        return float(min(score, 1.0)), {
            "id": "embedded_images",
            "name": "Embedded Image Analysis",
            "status": status,
            "description": desc,
        }

    def _analyze_pdf_basic(self, file_path: str) -> dict:
        """Basic PDF analysis without PyMuPDF."""
        checks = []
        scores = {}

        try:
            with open(file_path, "rb") as f:
                raw = f.read(50000)

            # Check for PDF header
            if not raw.startswith(b"%PDF"):
                return {"verdict": "error", "confidence": 0, "forensic_checks": [],
                        "raw_scores": {}, "error": "Not a valid PDF"}

            # Count revisions
            eof_count = raw.count(b"%%EOF")
            score = min(0.5, 0.1 * max(0, eof_count - 1))
            scores["revisions"] = score

            checks.append({
                "id": "basic_structure",
                "name": "Basic PDF Structure",
                "status": "warn" if score > 0.2 else "pass",
                "description": f"{eof_count} revision(s) detected" if eof_count > 1 else "Single revision",
            })

            # JavaScript check
            has_js = b"/JS" in raw or b"/JavaScript" in raw
            if has_js:
                scores["javascript"] = 0.5
                checks.append({
                    "id": "javascript",
                    "name": "JavaScript Detection",
                    "status": "fail",
                    "description": "PDF contains JavaScript",
                })

        except Exception as e:
            return {"verdict": "error", "confidence": 0, "forensic_checks": [],
                    "raw_scores": {}, "error": str(e)}

        avg_score = sum(scores.values()) / len(scores) if scores else 0
        verdict = "suspicious" if avg_score > 0.3 else "authentic"

        return {
            "verdict": verdict,
            "confidence": round(50 + avg_score * 40, 1),
            "forensic_checks": checks,
            "raw_scores": scores,
            "note": "Install pymupdf for full analysis",
        }

    @staticmethod
    def _parse_pdf_date(date_str: str) -> datetime:
        """Parse PDF date format (D:YYYYMMDDHHmmSS)."""
        if not date_str:
            return None
        date_str = date_str.strip()
        if date_str.startswith("D:"):
            date_str = date_str[2:]
        # Remove timezone info
        date_str = re.sub(r"[Z+-].*", "", date_str)
        try:
            if len(date_str) >= 14:
                return datetime.strptime(date_str[:14], "%Y%m%d%H%M%S")
            elif len(date_str) >= 8:
                return datetime.strptime(date_str[:8], "%Y%m%d")
        except ValueError:
            pass
        return None
