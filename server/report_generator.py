"""
Satya Drishti -- PDF Evidence Report Generator
==============================================
Generates a branded, professional PDF evidence report for a scan.
Uses reportlab when available; falls back to raw PDF generation otherwise.
"""

from io import BytesIO
from typing import List


def generate_report(
    scan_id: str,
    file_name: str,
    verdict: str,
    confidence: float,
    forensic_data: List[dict],
    created_at: str,
) -> bytes:
    """Generate a PDF evidence report and return the bytes."""
    try:
        return _generate_reportlab_pdf(scan_id, file_name, verdict, confidence, forensic_data, created_at)
    except ImportError:
        return _generate_fallback_pdf(scan_id, file_name, verdict, confidence, forensic_data, created_at)


# ═══════════════════════════════════════════════════════════════
# ReportLab-based PDF (full-featured, professional layout)
# ═══════════════════════════════════════════════════════════════

def _generate_reportlab_pdf(
    scan_id: str,
    file_name: str,
    verdict: str,
    confidence: float,
    forensic_data: List[dict],
    created_at: str,
) -> bytes:
    """Generate a professional PDF using reportlab."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        HRFlowable,
        KeepTogether,
    )

    # --- Color Palette ---
    DARK_BG = colors.HexColor("#0D1320")
    DARK_CARD = colors.HexColor("#141E32")
    PRIMARY = colors.HexColor("#4FC3F7")
    PRIMARY_DARK = colors.HexColor("#0288D1")
    INDIGO = colors.HexColor("#6366F1")
    TEXT_PRIMARY = colors.HexColor("#1A1F2E")
    TEXT_SECONDARY = colors.HexColor("#555555")
    TEXT_MUTED = colors.HexColor("#888888")
    BORDER_LIGHT = colors.HexColor("#E0E0E0")
    BORDER_SUBTLE = colors.HexColor("#F0F0F0")
    WHITE = colors.white
    SAFE = colors.HexColor("#66BB6A")
    DANGER = colors.HexColor("#EF5350")
    WARNING = colors.HexColor("#FFA726")

    VERDICT_COLORS = {
        "ai-generated": DANGER,
        "authentic": SAFE,
        "uncertain": WARNING,
        "inconclusive": WARNING,
    }

    STATUS_COLORS = {
        "pass": SAFE,
        "fail": DANGER,
        "warn": WARNING,
    }

    STATUS_BG = {
        "pass": colors.HexColor("#E8F5E9"),
        "fail": colors.HexColor("#FFEBEE"),
        "warn": colors.HexColor("#FFF3E0"),
    }

    buffer = BytesIO()
    page_w, page_h = A4
    margin = 18 * mm

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=margin,
        bottomMargin=margin,
        leftMargin=margin,
        rightMargin=margin,
    )

    usable_width = page_w - 2 * margin

    # --- Styles ---
    styles = getSampleStyleSheet()

    s_title = ParagraphStyle(
        "S_Title",
        parent=styles["Title"],
        fontSize=26,
        leading=30,
        textColor=TEXT_PRIMARY,
        fontName="Helvetica-Bold",
        alignment=TA_CENTER,
        spaceAfter=2 * mm,
    )
    s_subtitle = ParagraphStyle(
        "S_Subtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=TEXT_SECONDARY,
        alignment=TA_CENTER,
        spaceAfter=0,
    )
    s_section = ParagraphStyle(
        "S_Section",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=PRIMARY_DARK,
        fontName="Helvetica-Bold",
        spaceBefore=4 * mm,
        spaceAfter=3 * mm,
    )
    s_body = ParagraphStyle(
        "S_Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=TEXT_PRIMARY,
    )
    s_body_small = ParagraphStyle(
        "S_BodySmall",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=TEXT_SECONDARY,
    )
    s_label = ParagraphStyle(
        "S_Label",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        fontName="Helvetica-Bold",
        textColor=TEXT_PRIMARY,
    )
    s_value = ParagraphStyle(
        "S_Value",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=TEXT_SECONDARY,
    )
    s_table_header = ParagraphStyle(
        "S_TableHeader",
        parent=styles["Normal"],
        fontSize=9,
        fontName="Helvetica-Bold",
        textColor=WHITE,
        leading=12,
    )
    s_table_cell = ParagraphStyle(
        "S_TableCell",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=TEXT_PRIMARY,
    )
    s_table_detail = ParagraphStyle(
        "S_TableDetail",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=12,
        textColor=TEXT_SECONDARY,
    )
    s_disclaimer = ParagraphStyle(
        "S_Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        leading=12,
        textColor=TEXT_MUTED,
    )
    s_footer_bold = ParagraphStyle(
        "S_FooterBold",
        parent=styles["Normal"],
        fontSize=8,
        leading=12,
        fontName="Helvetica-Bold",
        textColor=TEXT_SECONDARY,
    )

    elements = []

    # === HEADER ===
    elements.append(HRFlowable(width="100%", thickness=3, color=PRIMARY, spaceAfter=4 * mm))
    elements.append(Paragraph("SATYA DRISHTI", s_title))
    elements.append(Paragraph("Forensic Analysis Report", s_subtitle))
    elements.append(Spacer(1, 2 * mm))
    elements.append(HRFlowable(width="40%", thickness=1.5, color=PRIMARY, spaceAfter=2 * mm))
    elements.append(Spacer(1, 4 * mm))

    # === SCAN DETAILS ===
    elements.append(Paragraph("Scan Details", s_section))

    verdict_display = verdict.upper().replace("-", " ")
    verdict_color = VERDICT_COLORS.get(verdict, TEXT_PRIMARY)

    detail_rows = [
        [Paragraph("Report ID", s_label), Paragraph(scan_id, s_value)],
        [Paragraph("File Analyzed", s_label), Paragraph(file_name, s_value)],
        [Paragraph("Analysis Date", s_label), Paragraph(created_at, s_value)],
        [
            Paragraph("Verdict", s_label),
            Paragraph(
                f'<font color="#{verdict_color.hexval()[2:]}">{verdict_display}</font>',
                ParagraphStyle("verdict_val", parent=s_value, fontName="Helvetica-Bold", fontSize=11),
            ),
        ],
        [
            Paragraph("Confidence", s_label),
            Paragraph(
                f'<font color="#{verdict_color.hexval()[2:]}"><b>{confidence}%</b></font>',
                s_value,
            ),
        ],
    ]

    col1_w = 110
    col2_w = usable_width - col1_w

    detail_table = Table(detail_rows, colWidths=[col1_w, col2_w])
    detail_table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            *[
                ("BACKGROUND", (0, i), (-1, i), BORDER_SUBTLE if i % 2 == 0 else WHITE)
                for i in range(len(detail_rows))
            ],
            ("LINEABOVE", (0, 0), (-1, 0), 1, BORDER_LIGHT),
            ("LINEBELOW", (0, -1), (-1, -1), 1, BORDER_LIGHT),
        ])
    )
    elements.append(detail_table)
    elements.append(Spacer(1, 8 * mm))

    # === FORENSIC ANALYSIS TABLE ===
    if forensic_data:
        elements.append(Paragraph("Forensic Analysis Results", s_section))

        header_row = [
            Paragraph("Check", s_table_header),
            Paragraph("Status", s_table_header),
            Paragraph("Details", s_table_header),
        ]

        data_rows = [header_row]
        status_row_map = {}

        for i, item in enumerate(forensic_data):
            check_name = item.get("name", item.get("label", "Unknown"))
            status = item.get("status", "N/A")
            detail = item.get("description", item.get("detail", ""))

            status_upper = status.upper() if isinstance(status, str) else str(status)
            status_color = STATUS_COLORS.get(status, TEXT_PRIMARY)

            data_rows.append([
                Paragraph(check_name, s_table_cell),
                Paragraph(
                    f'<font color="#{status_color.hexval()[2:]}"><b>{status_upper}</b></font>',
                    ParagraphStyle("status_cell", parent=s_table_cell, alignment=TA_CENTER),
                ),
                Paragraph(detail, s_table_detail),
            ])
            status_row_map[i + 1] = status

        check_w = 120
        status_w = 50
        detail_w = usable_width - check_w - status_w

        forensic_table = Table(data_rows, colWidths=[check_w, status_w, detail_w], repeatRows=1)

        style_commands = [
            ("BACKGROUND", (0, 0), (-1, 0), DARK_BG),
            ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("LINEBELOW", (0, 0), (-1, 0), 1.5, PRIMARY),
            ("LINEBELOW", (0, 1), (-1, -1), 0.5, BORDER_LIGHT),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ]

        for row_i, status in status_row_map.items():
            if row_i % 2 == 0:
                style_commands.append(("BACKGROUND", (0, row_i), (-1, row_i), BORDER_SUBTLE))
            if status in STATUS_BG:
                style_commands.append(("BACKGROUND", (1, row_i), (1, row_i), STATUS_BG[status]))

        forensic_table.setStyle(TableStyle(style_commands))
        elements.append(forensic_table)

    elements.append(Spacer(1, 8 * mm))

    # === VERDICT SUMMARY BOX + FOOTER ===
    verdict_bg = {
        "ai-generated": colors.HexColor("#FFEBEE"),
        "authentic": colors.HexColor("#E8F5E9"),
        "uncertain": colors.HexColor("#FFF3E0"),
        "inconclusive": colors.HexColor("#FFF3E0"),
    }

    verdict_messages = {
        "ai-generated": "This media has been identified as AI-generated or manipulated. Exercise caution and verify through additional sources.",
        "authentic": "This media appears to be authentic based on our forensic analysis pipeline. No signs of AI manipulation detected.",
        "uncertain": "Analysis was inconclusive. The confidence level is not high enough to make a definitive determination.",
        "inconclusive": "Analysis was inconclusive. The confidence level is not high enough to make a definitive determination.",
    }

    box_bg = verdict_bg.get(verdict, colors.HexColor("#F5F5F5"))
    box_msg = verdict_messages.get(verdict, "Analysis complete.")

    s_verdict_title = ParagraphStyle(
        "S_VerdictTitle",
        parent=s_body,
        fontSize=12,
        fontName="Helvetica-Bold",
        textColor=verdict_color,
    )
    s_verdict_body = ParagraphStyle(
        "S_VerdictBody",
        parent=s_body,
        fontSize=10,
        textColor=TEXT_SECONDARY,
    )

    verdict_box_data = [[
        [
            Paragraph(f"Verdict: {verdict_display}", s_verdict_title),
            Spacer(1, 2 * mm),
            Paragraph(box_msg, s_verdict_body),
        ]
    ]]

    verdict_box = Table(verdict_box_data, colWidths=[usable_width - 16])
    verdict_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), box_bg),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("LINEABOVE", (0, 0), (-1, 0), 2, verdict_color),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))

    footer_elements = [
        verdict_box,
        Spacer(1, 8 * mm),
        HRFlowable(width="100%", thickness=0.5, color=BORDER_LIGHT, spaceAfter=3 * mm),
        Paragraph(
            "<b>Disclaimer:</b> This report is generated by Satya Drishti's AI forensics engine. "
            "Results are probabilistic and should be used as supporting evidence, not as definitive proof. "
            "For legal proceedings, consult with a certified digital forensics expert.",
            s_disclaimer,
        ),
        Spacer(1, 2 * mm),
        Paragraph(
            "If you are a victim of deepfake abuse, contact the <b>National Cyber Crime Helpline: 1930</b> "
            "or visit <b>cybercrime.gov.in</b>",
            s_disclaimer,
        ),
        Spacer(1, 5 * mm),
        HRFlowable(width="100%", thickness=2, color=PRIMARY),
        Spacer(1, 2 * mm),
        Paragraph(
            "Generated by Satya Drishti Deepfake Detection System",
            ParagraphStyle("tagline", parent=s_disclaimer, alignment=TA_CENTER, textColor=PRIMARY_DARK, fontName="Helvetica-BoldOblique"),
        ),
    ]
    elements.append(KeepTogether(footer_elements))

    doc.build(elements)
    return buffer.getvalue()


# ═══════════════════════════════════════════════════════════════
# Fallback: raw PDF generation (no external dependencies)
# ═══════════════════════════════════════════════════════════════

def _generate_fallback_pdf(
    scan_id: str,
    file_name: str,
    verdict: str,
    confidence: float,
    forensic_data: List[dict],
    created_at: str,
) -> bytes:
    """
    Generate a valid PDF without any external libraries.

    Builds the PDF byte stream manually using the PDF 1.4 specification.
    The result is a simple, text-based single-page report with Helvetica font.
    """

    verdict_display = verdict.upper().replace("-", " ")

    # Build the text content lines
    lines = []
    lines.append("Satya Drishti -- Forensic Analysis Report")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Scan ID:     {scan_id}")
    lines.append(f"File Name:   {file_name}")
    lines.append(f"Date:        {created_at}")
    lines.append("")
    lines.append(f"Verdict:     {verdict_display}")
    lines.append(f"Confidence:  {confidence}%")
    lines.append("")

    if forensic_data:
        lines.append("-" * 50)
        lines.append("FORENSIC ANALYSIS RESULTS")
        lines.append("-" * 50)
        lines.append("")

        # Calculate column widths for alignment
        max_name_len = max(
            (len(item.get("name", item.get("label", "Unknown"))) for item in forensic_data),
            default=20,
        )
        max_name_len = max(max_name_len, 5)  # minimum width for "Check"
        max_status_len = 6  # "STATUS"

        header = f"{'Check':<{max_name_len}}  {'Status':<{max_status_len}}  Description"
        lines.append(header)
        lines.append("-" * len(header))

        for item in forensic_data:
            check_name = item.get("name", item.get("label", "Unknown"))
            status = item.get("status", "N/A")
            detail = item.get("description", item.get("detail", ""))
            status_str = status.upper() if isinstance(status, str) else str(status)

            # Truncate description to fit reasonably
            if len(detail) > 60:
                detail = detail[:57] + "..."

            lines.append(f"{check_name:<{max_name_len}}  {status_str:<{max_status_len}}  {detail}")

        lines.append("")

    lines.append("=" * 50)
    lines.append("Generated by Satya Drishti Deepfake Detection System")

    return _build_raw_pdf(lines)


def _build_raw_pdf(lines: list) -> bytes:
    """
    Build a minimal valid PDF 1.4 file from a list of text lines.

    Structure:
      - Object 1: Catalog
      - Object 2: Pages
      - Object 3: Page
      - Object 4: Font (Helvetica)
      - Object 5: Content stream (the text)
    """

    # PDF text rendering: place each line using Tj operator
    # Start at top of page (A4: 595 x 842 points), with margin
    page_width = 595
    page_height = 842
    margin_left = 50
    margin_top = 50
    font_size = 10
    title_font_size = 16
    line_height = 14
    title_line_height = 22

    # Build the content stream
    stream_parts = []

    # Start text object
    stream_parts.append("BT")

    y = page_height - margin_top

    for i, line in enumerate(lines):
        # Use larger font for the title (first line)
        if i == 0:
            stream_parts.append(f"/F1 {title_font_size} Tf")
            stream_parts.append(f"{margin_left} {y} Td")
            stream_parts.append(f"({_pdf_escape(line)}) Tj")
            y -= title_line_height
        elif i == 1:
            # Separator after title -- stay with title font briefly
            stream_parts.append(f"/F1 {font_size} Tf")
            stream_parts.append(f"{margin_left} {y} Td")
            stream_parts.append(f"({_pdf_escape(line)}) Tj")
            y -= line_height
        else:
            # Reset position for each line (absolute positioning)
            stream_parts.append(f"{margin_left} {y} Td")
            stream_parts.append(f"({_pdf_escape(line)}) Tj")
            y -= line_height

        # Prevent going off-page (simple single-page limit)
        if y < margin_top:
            break

    stream_parts.append("ET")
    content_stream = "\n".join(stream_parts)

    # Now build the PDF objects
    objects = []
    offsets = []

    # We'll collect everything into a bytearray
    pdf = bytearray()

    # Header
    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    pdf.extend(header)

    # Object 1: Catalog
    offsets.append(len(pdf))
    obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    pdf.extend(obj1)

    # Object 2: Pages
    offsets.append(len(pdf))
    obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    pdf.extend(obj2)

    # Object 3: Page
    offsets.append(len(pdf))
    obj3 = (
        f"3 0 obj\n"
        f"<< /Type /Page /Parent 2 0 R "
        f"/MediaBox [0 0 {page_width} {page_height}] "
        f"/Contents 5 0 R "
        f"/Resources << /Font << /F1 4 0 R >> >> "
        f">>\n"
        f"endobj\n"
    ).encode()
    pdf.extend(obj3)

    # Object 4: Font (Helvetica -- built-in, no embedding needed)
    offsets.append(len(pdf))
    obj4 = b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    pdf.extend(obj4)

    # Object 5: Content stream
    stream_bytes = content_stream.encode("latin-1", errors="replace")
    offsets.append(len(pdf))
    obj5_header = f"5 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode()
    obj5_footer = b"\nendstream\nendobj\n"
    pdf.extend(obj5_header)
    pdf.extend(stream_bytes)
    pdf.extend(obj5_footer)

    # Cross-reference table
    xref_offset = len(pdf)
    num_objects = len(offsets) + 1  # +1 for the free object entry

    xref = f"xref\n0 {num_objects}\n"
    xref += "0000000000 65535 f \n"
    for offset in offsets:
        xref += f"{offset:010d} 00000 n \n"

    pdf.extend(xref.encode())

    # Trailer
    trailer = (
        f"trailer\n"
        f"<< /Size {num_objects} /Root 1 0 R >>\n"
        f"startxref\n"
        f"{xref_offset}\n"
        f"%%EOF\n"
    )
    pdf.extend(trailer.encode())

    return bytes(pdf)


def _pdf_escape(text: str) -> str:
    """Escape special characters for PDF string literals."""
    text = text.replace("\\", "\\\\")
    text = text.replace("(", "\\(")
    text = text.replace(")", "\\)")
    text = text.replace("\r", "\\r")
    text = text.replace("\n", "\\n")
    return text
