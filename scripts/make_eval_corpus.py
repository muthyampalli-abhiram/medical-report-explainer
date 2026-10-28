"""Deterministic synthetic evaluation corpus generator for medical report explainer.

Generates:
  - 30 digital PDFs in 3 layouts (bordered_table, plain_lines, reordered_table)
  - 30 corresponding .labels.json files with ground-truth labels and planted PII
  - 10 image sets (clean.png, skewed.jpg, poor.jpg) for OCR evaluation

Dependencies: Standard library, ReportLab, and Pillow only.
"""

from datetime import date
import hashlib
import json
from pathlib import Path
import random
import re

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# ---------------------------------------------------------------------------
# INDEPENDENT CONVERSION FACTORS TABLE
#
# NOTE: This table and the flag logic below are intentionally implemented
# independently within this generator without importing or calling
# app/services/flagger.py, units.py, or pipeline.py.
#
# This architectural independence is essential: evaluation must measure production
# behavior against an independently derived ground truth, rather than having
# production code evaluate itself.
# ---------------------------------------------------------------------------
INDEPENDENT_CONVERSION_TABLE: dict[tuple[str, str], tuple[float, str]] = {
    ("Fasting Glucose", "mmol/L"): (18.016, "mg/dL"),
    ("Total Cholesterol", "mmol/L"): (38.67, "mg/dL"),
    ("LDL Cholesterol", "mmol/L"): (38.67, "mg/dL"),
    ("HDL Cholesterol", "mmol/L"): (38.67, "mg/dL"),
    ("Triglycerides", "mmol/L"): (88.57, "mg/dL"),
    ("Hemoglobin", "g/L"): (0.1, "g/dL"),
    ("Potassium", "mEq/L"): (1.0, "mmol/L"),
}

CATALOG_SPECS = {
    "HbA1c": {
        "aliases": [
            "HbA1c",
            "Glycosylated Hb",
            "Glycated Hemoglobin",
            "Glycosylated Hemoglobin",
            "A1C",
            "HbA1C",
            "Hb A1c",
        ],
        "canonical_unit": "%",
        "default_range": (4.0, 5.6),
        "si_range": (20.0, 42.0),
        "si_unit": "mmol/mol",
        "critical_low": None,
        "critical_high": None,
    },
    "Fasting Glucose": {
        "aliases": [
            "Fasting Glucose",
            "FBS",
            "Fasting Blood Sugar",
            "Glucose Fasting",
            "Glucose, Fasting",
            "Fasting Blood Glucose",
        ],
        "canonical_unit": "mg/dL",
        "default_range": (70.0, 99.0),
        "si_range": (3.9, 5.5),
        "si_unit": "mmol/L",
        "critical_low": 40.0,
        "critical_high": 400.0,
    },
    "Total Cholesterol": {
        "aliases": [
            "Total Cholesterol",
            "Cholesterol Total",
            "Cholesterol",
            "Serum Cholesterol",
        ],
        "canonical_unit": "mg/dL",
        "default_range": (None, 200.0),
        "si_range": (None, 5.2),
        "si_unit": "mmol/L",
        "critical_low": None,
        "critical_high": None,
    },
    "LDL Cholesterol": {
        "aliases": [
            "LDL Cholesterol",
            "LDL-C",
            "LDL",
            "LDL Cholesterol Direct",
        ],
        "canonical_unit": "mg/dL",
        "default_range": (None, 100.0),
        "si_range": (None, 2.6),
        "si_unit": "mmol/L",
        "critical_low": None,
        "critical_high": None,
    },
    "HDL Cholesterol": {
        "aliases": [
            "HDL Cholesterol",
            "HDL-C",
            "HDL",
        ],
        "canonical_unit": "mg/dL",
        "default_range": (40.0, None),
        "si_range": (1.0, None),
        "si_unit": "mmol/L",
        "critical_low": None,
        "critical_high": None,
    },
    "Triglycerides": {
        "aliases": [
            "Triglycerides",
            "TG",
            "TGL",
            "Triglyceride",
        ],
        "canonical_unit": "mg/dL",
        "default_range": (None, 150.0),
        "si_range": (None, 1.7),
        "si_unit": "mmol/L",
        "critical_low": None,
        "critical_high": None,
    },
    "Hemoglobin": {
        "aliases": [
            "Hemoglobin",
            "Haemoglobin",
            "Hb",
            "Hgb",
        ],
        "canonical_unit": "g/dL",
        "default_range": (13.0, 17.0),
        "si_range": (130.0, 170.0),
        "si_unit": "g/L",
        "critical_low": 7.0,
        "critical_high": 20.0,
    },
    "Potassium": {
        "aliases": [
            "Potassium",
            "Serum Potassium",
            "K+",
            "K",
        ],
        "canonical_unit": "mmol/L",
        "default_range": (3.5, 5.0),
        "si_range": (3.5, 5.0),
        "si_unit": "mEq/L",
        "critical_low": 2.5,
        "critical_high": 6.5,
    },
    "Urine Protein": {
        "aliases": [
            "Urine Protein",
            "Protein, Urine",
            "Urine Albumin",
        ],
        "canonical_unit": None,
        "default_range": (None, None),
        "si_range": (None, None),
        "si_unit": None,
        "critical_low": None,
        "critical_high": None,
    },
}

UNRECOGNIZED_TEST_POOL = [
    {
        "raw_name": "Serum Amylase",
        "unit": "U/L",
        "ref_text": "28 - 100",
        "ref_low": 28.0,
        "ref_high": 100.0,
        "normal_val": 65.0,
        "abnormal_val": 160.0,
    },
    {
        "raw_name": "ESR",
        "unit": "mm/hr",
        "ref_text": "< 20",
        "ref_low": None,
        "ref_high": 20.0,
        "normal_val": 10.0,
        "abnormal_val": 42.0,
    },
    {
        "raw_name": "Serum Ferritin",
        "unit": "ng/mL",
        "ref_text": "30 - 400",
        "ref_low": 30.0,
        "ref_high": 400.0,
        "normal_val": 180.0,
        "abnormal_val": 15.0,
    },
]

FAKE_PATIENT_POOL = [
    {
        "name": "Ramesh Kumar",
        "id": "8492019",
        "phone": "+91 98765 43210",
        "email": "ramesh.kumar98@example.org",
        "aadhaar": "4829 1048 5729",
        "address": "42 Palm Grove Road, Bengaluru 560001",
    },
    {
        "name": "Priya Sharma",
        "id": "7391042",
        "phone": "+91 91234 56789",
        "email": "priya.sharma99@example.com",
        "aadhaar": "7492 1048 3920",
        "address": "14 MG Road, Indiranagar, Bengaluru 560038",
    },
    {
        "name": "Anil Verma",
        "id": "6291048",
        "phone": "+91 98450 11223",
        "email": "anil.verma21@example.net",
        "aadhaar": "5820 9182 3019",
        "address": "88 Jubilee Hills, Road No 36, Hyderabad 500033",
    },
    {
        "name": "Sunita Patel",
        "id": "5192038",
        "phone": "+91 94480 33445",
        "email": "sunita.patel84@example.org",
        "aadhaar": "3910 4820 1928",
        "address": "23 Ashram Road, Navrangpura, Ahmedabad 380009",
    },
    {
        "name": "Vikram Malhotra",
        "id": "4091827",
        "phone": "+91 98100 55667",
        "email": "v.malhotra77@example.com",
        "aadhaar": "9182 3019 4820",
        "address": "102 Connaught Place, Block B, New Delhi 110001",
    },
]

LAB_NAMES = [
    "Apex Diagnostics Laboratory",
    "Metro Healthcare Lab",
    "Sunrise Pathology Centre",
    "CarePoint Clinical Diagnostics",
    "Nova Medical Laboratory",
]


def _independent_convert(
    canonical_name: str | None, value: float | None, unit: str | None
) -> tuple[float | None, str | None]:
    """Independent unit conversion without calling production code."""
    if value is None or unit is None or canonical_name is None:
        return value, unit

    # HbA1c formula
    if canonical_name == "HbA1c" and unit.lower() == "mmol/mol":
        val_conv = round((value / 10.929) + 2.15, 2)
        return val_conv, "%"

    key = (canonical_name, unit)
    if key in INDEPENDENT_CONVERSION_TABLE:
        factor, target_unit = INDEPENDENT_CONVERSION_TABLE[key]
        return round(value * factor, 2), target_unit

    spec = CATALOG_SPECS.get(canonical_name)
    canon_unit = spec["canonical_unit"] if spec else unit
    return round(value, 2), canon_unit


def _independent_compute_flag(
    canonical_name: str | None,
    val_canon: float | None,
    val_text: str | None,
    ref_low_orig: float | None,
    ref_high_orig: float | None,
    ref_text: str | None,
    unit_orig: str | None,
) -> str:
    """Independent flag calculation without calling production code."""
    # Critical overrides (checked on canonical numeric value for catalog tests)
    if canonical_name and val_canon is not None:
        spec = CATALOG_SPECS.get(canonical_name)
        if spec:
            c_low = spec["critical_low"]
            c_high = spec["critical_high"]
            if c_low is not None and val_canon < c_low:
                return "CRITICAL_LOW"
            if c_high is not None and val_canon > c_high:
                return "CRITICAL_HIGH"

    # Text results
    if val_text is not None:
        if not ref_text:
            return "UNKNOWN"
        if val_text.strip().lower() == ref_text.strip().lower():
            return "NORMAL"
        return "UNKNOWN"

    # Numeric range checks
    if val_canon is not None:
        if ref_low_orig is None and ref_high_orig is None:
            return "UNKNOWN"

        # Convert ref_low and ref_high to canonical for proper comparison
        ref_low_canon = ref_low_orig
        ref_high_canon = ref_high_orig
        if canonical_name and unit_orig:
            if ref_low_orig is not None:
                c_low, _ = _independent_convert(canonical_name, ref_low_orig, unit_orig)
                ref_low_canon = c_low
            if ref_high_orig is not None:
                c_high, _ = _independent_convert(canonical_name, ref_high_orig, unit_orig)
                ref_high_canon = c_high

        rt = (ref_text or "").strip()
        # Strict less than '< x'
        if ref_high_canon is not None:
            if rt.startswith("<") and not rt.startswith("<="):
                if val_canon >= ref_high_canon:
                    return "HIGH"
            else:
                if val_canon > ref_high_canon:
                    return "HIGH"

        # Strict greater than '> x'
        if ref_low_canon is not None:
            if rt.startswith(">") and not rt.startswith(">="):
                if val_canon <= ref_low_canon:
                    return "LOW"
            else:
                if val_canon < ref_low_canon:
                    return "LOW"

        return "NORMAL"

    return "UNKNOWN"


def _generate_safe_value(
    rng: random.Random,
    low: float | None,
    high: float | None,
    is_normal: bool,
    critical_kind: str | None = None,
) -> float:
    """Generate a value respecting the 2% boundary exclusion zone."""
    if critical_kind == "glucose_low":
        return round(rng.uniform(25.0, 38.0), 1)
    if critical_kind == "glucose_high":
        return round(rng.uniform(415.0, 480.0), 1)
    if critical_kind == "potassium_high":
        return round(rng.uniform(6.8, 8.0), 1)
    if critical_kind == "hemoglobin_low":
        return round(rng.uniform(4.5, 6.4), 1)

    # Normal range: strictly inside [low * 1.03, high * 0.97]
    if is_normal:
        if low is not None and high is not None:
            val_min = low * 1.03
            val_max = high * 0.97
            return round(rng.uniform(val_min, val_max), 1)
        if high is not None:
            return round(rng.uniform(high * 0.4, high * 0.97), 1)
        if low is not None:
            return round(rng.uniform(low * 1.03, low * 1.8), 1)
        return 50.0

    # Abnormal: outside boundary by at least 3%
    pick_high = rng.choice([True, False]) if (low is not None and high is not None) else (low is None)
    if pick_high and high is not None:
        return round(rng.uniform(high * 1.04, high * 1.5), 1)
    if low is not None:
        return round(rng.uniform(low * 0.5, low * 0.96), 1)
    return 120.0


def _build_report_data(report_idx: int, rng: random.Random) -> dict:
    """Construct deterministic data and labels for one report."""
    report_id = f"eval_{report_idx:02d}"

    # Layout mapping: 1-10 bordered_table, 11-20 plain_lines, 21-30 reordered_table
    if 1 <= report_idx <= 10:
        layout = "bordered_table"
    elif 11 <= report_idx <= 20:
        layout = "plain_lines"
    else:
        layout = "reordered_table"

    # Dates in ISO or DD-Mon-YYYY
    day = (report_idx * 3) % 27 + 1
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]
    month_name = months[report_idx % len(months)]
    month_num = (report_idx % 10) + 1
    if report_idx % 2 == 0:
        date_str = f"2026-{month_num:02d}-{day:02d}"
    else:
        date_str = f"{day:02d}-{month_name}-2026"

    lab_name = LAB_NAMES[report_idx % len(LAB_NAMES)]
    patient = FAKE_PATIENT_POOL[report_idx % len(FAKE_PATIENT_POOL)]
    planted_pii = [
        patient["name"],
        f"ID: {patient['id']}",
        patient["phone"],
        patient["email"],
        patient["aadhaar"],
        patient["address"],
    ]

    # Special conditions assignments
    use_si_units = report_idx in (2, 7, 14, 21, 28)
    has_critical = report_idx in (4, 11, 18, 25)
    has_text_result = report_idx in (1, 9, 16, 23)
    has_unrecognized = report_idx in (3, 13, 26)

    # Number of tests: 4-9
    num_tests = rng.randint(4, 9)
    available_catalog = list(CATALOG_SPECS.keys())
    # Reserve text test if required
    selected_tests = []
    if has_text_result:
        selected_tests.append("Urine Protein")
        available_catalog.remove("Urine Protein")

    # Add critical target if needed
    critical_target = None
    if has_critical:
        crit_options = {
            4: ("Fasting Glucose", "glucose_low"),
            11: ("Fasting Glucose", "glucose_high"),
            18: ("Potassium", "potassium_high"),
            25: ("Hemoglobin", "hemoglobin_low"),
        }
        test_name, crit_kind = crit_options[report_idx]
        critical_target = (test_name, crit_kind)
        if test_name not in selected_tests:
            selected_tests.append(test_name)
            if test_name in available_catalog:
                available_catalog.remove(test_name)

    remaining_needed = num_tests - len(selected_tests)
    if has_unrecognized:
        remaining_needed = max(0, remaining_needed - 1)

    rng.shuffle(available_catalog)
    selected_tests.extend(available_catalog[:remaining_needed])
    rng.shuffle(selected_tests)

    results = []

    # Target roughly 40% normal, 60% abnormal
    for test_name in selected_tests:
        spec = CATALOG_SPECS[test_name]
        raw_name = rng.choice(spec["aliases"])

        if test_name == "Urine Protein":
            val_text = rng.choice(["Negative", "Positive", "Trace"])
            ref_text = "Negative"
            flag = "NORMAL" if val_text == "Negative" else "UNKNOWN"
            results.append(
                {
                    "raw_name": raw_name,
                    "canonical_name": test_name,
                    "value": None,
                    "value_text": val_text,
                    "unit_original": None,
                    "ref_text": ref_text,
                    "ref_low": None,
                    "ref_high": None,
                    "expected_value_canonical": None,
                    "expected_unit_canonical": None,
                    "expected_flag": flag,
                }
            )
            continue

        unit = spec["si_unit"] if (use_si_units and spec["si_unit"]) else spec["canonical_unit"]
        low, high = spec["si_range"] if (use_si_units and spec["si_range"]) else spec["default_range"]

        # Check if critical applies
        crit_kind = None
        if critical_target and critical_target[0] == test_name:
            crit_kind = critical_target[1]

        # Normal/Abnormal probability
        is_normal = (rng.random() < 0.4) if not crit_kind else False

        val = _generate_safe_value(rng, low, high, is_normal, crit_kind)

        # Reference range style
        if low is not None and high is not None:
            style = rng.choice(["a - b", "a-b"])
            ref_text = f"{low:g} - {high:g}" if style == "a - b" else f"{low:g}-{high:g}"
            ref_l, ref_h = float(low), float(high)
        elif high is not None:
            style = rng.choice(["< x", "<= x"])
            ref_text = f"< {high:g}" if style == "< x" else f"<= {high:g}"
            ref_l, ref_h = None, float(high)
        elif low is not None:
            style = rng.choice(["> x", ">= x"])
            ref_text = f"> {low:g}" if style == "> x" else f">= {low:g}"
            ref_l, ref_h = float(low), None
        else:
            ref_text = None
            ref_l, ref_h = None, None

        # Compute independent canonical value and flag
        val_canon, unit_canon = _independent_convert(test_name, val, unit)
        flag = _independent_compute_flag(test_name, val_canon, None, ref_l, ref_h, ref_text, unit)

        results.append(
            {
                "raw_name": raw_name,
                "canonical_name": test_name,
                "value": val,
                "value_text": None,
                "unit_original": unit,
                "ref_text": ref_text,
                "ref_low": ref_l,
                "ref_high": ref_h,
                "expected_value_canonical": val_canon,
                "expected_unit_canonical": unit_canon,
                "expected_flag": flag,
            }
        )

    # Insert unrecognized test if needed
    if has_unrecognized:
        unrec = UNRECOGNIZED_TEST_POOL[report_idx % len(UNRECOGNIZED_TEST_POOL)]
        is_norm = rng.random() < 0.4
        val = unrec["normal_val"] if is_norm else unrec["abnormal_val"]
        flag = _independent_compute_flag(
            None, val, None, unrec["ref_low"], unrec["ref_high"], unrec["ref_text"], unrec["unit"]
        )
        results.append(
            {
                "raw_name": unrec["raw_name"],
                "canonical_name": None,
                "value": val,
                "value_text": None,
                "unit_original": unrec["unit"],
                "ref_text": unrec["ref_text"],
                "ref_low": unrec["ref_low"],
                "ref_high": unrec["ref_high"],
                "expected_value_canonical": val,
                "expected_unit_canonical": unrec["unit"],
                "expected_flag": flag,
            }
        )

    return {
        "report_id": report_id,
        "layout": layout,
        "report_date": date_str,
        "lab_name": lab_name,
        "results": results,
        "planted_pii": planted_pii,
    }


def _render_pdf(data: dict, output_pdf: Path) -> None:
    """Render digital PDF for a given layout."""
    layout = data["layout"]

    if layout == "bordered_table":
        doc = SimpleDocTemplate(
            str(output_pdf),
            pagesize=letter,
            leftMargin=54,
            rightMargin=54,
            topMargin=54,
            bottomMargin=54,
        )
        styles = getSampleStyleSheet()
        elements = [
            Paragraph(f"<b>{data['lab_name']}</b>", styles["Heading1"]),
            Paragraph(f"Date: {data['report_date']}", styles["Normal"]),
            Paragraph(f"Patient: {data['planted_pii'][0]}, {data['planted_pii'][1]}", styles["Normal"]),
            Paragraph(f"{data['planted_pii'][2]} | {data['planted_pii'][3]}", styles["Normal"]),
            Paragraph(f"Aadhaar: {data['planted_pii'][4]}", styles["Normal"]),
            Paragraph(f"Address: {data['planted_pii'][5]}", styles["Normal"]),
            Spacer(1, 15),
        ]
        table_rows = [["Test Name", "Result", "Unit", "Reference Range"]]
        for r in data["results"]:
            val_str = str(r["value"]) if r["value"] is not None else str(r["value_text"])
            unit_str = r["unit_original"] or ""
            ref_str = r["ref_text"] or ""
            table_rows.append([r["raw_name"], val_str, unit_str, ref_str])

        table = Table(table_rows, colWidths=[170, 70, 70, 140])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        elements.append(table)
        doc.build(elements)

    elif layout == "plain_lines":
        c = canvas.Canvas(str(output_pdf), pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(54, 730, data["lab_name"])
        c.setFont("Helvetica", 9)
        c.drawString(54, 712, f"Date: {data['report_date']}")
        c.drawString(54, 698, f"Patient: {data['planted_pii'][0]}, {data['planted_pii'][1]}")
        c.drawString(54, 684, f"Phone: {data['planted_pii'][2]}  Email: {data['planted_pii'][3]}")
        c.drawString(54, 670, f"Aadhaar: {data['planted_pii'][4]}")
        c.drawString(54, 656, f"Address: {data['planted_pii'][5]}")

        y = 615
        c.setFont("Courier-Bold", 10)
        c.drawString(54, y, "Test Name                 Result    Unit     Reference Range")

        c.setFont("Courier", 10)
        for r in data["results"]:
            y -= 22
            val_str = str(r["value"]) if r["value"] is not None else str(r["value_text"])
            unit_str = r["unit_original"] or ""
            ref_str = r["ref_text"] or ""
            line = f"{r['raw_name']:<26}{val_str:<10}{unit_str:<9}{ref_str}"
            c.drawString(54, y, line)
        c.save()

    elif layout == "reordered_table":
        doc = SimpleDocTemplate(
            str(output_pdf),
            pagesize=letter,
            leftMargin=54,
            rightMargin=54,
            topMargin=54,
            bottomMargin=54,
        )
        styles = getSampleStyleSheet()
        elements = [
            Paragraph(f"<b>{data['lab_name']}</b>", styles["Heading1"]),
            Paragraph(f"Date: {data['report_date']}", styles["Normal"]),
            Paragraph(f"Patient: {data['planted_pii'][0]}, {data['planted_pii'][1]}", styles["Normal"]),
            Paragraph(f"{data['planted_pii'][2]} | {data['planted_pii'][3]}", styles["Normal"]),
            Paragraph(f"Aadhaar: {data['planted_pii'][4]}", styles["Normal"]),
            Paragraph(f"Address: {data['planted_pii'][5]}", styles["Normal"]),
            Spacer(1, 15),
        ]
        # Columns reordered: Test Name, Units, Result, Ref. Interval
        table_rows = [["Test Name", "Units", "Result", "Ref. Interval"]]
        for r in data["results"]:
            val_str = str(r["value"]) if r["value"] is not None else str(r["value_text"])
            unit_str = r["unit_original"] or ""
            ref_str = r["ref_text"] or ""
            table_rows.append([r["raw_name"], unit_str, val_str, ref_str])

        table = Table(table_rows, colWidths=[170, 70, 70, 140])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        elements.append(table)
        doc.build(elements)


def _get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load standard Windows font with fallback."""
    candidates = (
        ["arialbd.ttf", "trebucbd.ttf", "calibrib.ttf"]
        if bold
        else ["arial.ttf", "trebuc.ttf", "calibri.ttf"]
    )
    for fn in candidates:
        try:
            return ImageFont.truetype(fn, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _render_clean_image(data: dict) -> Image.Image:
    """Render clean A4 image of report data (1240 x 1754 px) with Pillow only."""
    img = Image.new("RGB", (1240, 1754), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    f_title = _get_font(42, bold=True)
    f_sub = _get_font(24, bold=False)
    f_cell_bold = _get_font(26, bold=True)
    f_cell = _get_font(26, bold=False)

    draw.text((100, 90), data["lab_name"], fill=(0, 0, 0), font=f_title)
    draw.text((100, 160), f"Date: {data['report_date']}", fill=(0, 0, 0), font=f_sub)
    draw.text((100, 205), f"Patient: {data['planted_pii'][0]}, {data['planted_pii'][1]}", fill=(0, 0, 0), font=f_sub)
    draw.text((100, 250), f"{data['planted_pii'][2]} | {data['planted_pii'][3]}", fill=(0, 0, 0), font=f_sub)
    draw.text((100, 295), f"Address: {data['planted_pii'][5]}", fill=(0, 0, 0), font=f_sub)

    # Table header
    y = 390
    draw.text((100, y), "Test Name", fill=(0, 0, 0), font=f_cell_bold)
    draw.text((540, y), "Result", fill=(0, 0, 0), font=f_cell_bold)
    draw.text((740, y), "Unit", fill=(0, 0, 0), font=f_cell_bold)
    draw.text((920, y), "Reference Range", fill=(0, 0, 0), font=f_cell_bold)

    draw.line([(100, y + 40), (1140, y + 40)], fill=(0, 0, 0), width=2)
    y += 50

    for r in data["results"]:
        val_str = str(r["value"]) if r["value"] is not None else str(r["value_text"])
        unit_str = r["unit_original"] or ""
        ref_str = r["ref_text"] or ""

        draw.text((100, y), r["raw_name"], fill=(0, 0, 0), font=f_cell)
        draw.text((540, y), val_str, fill=(0, 0, 0), font=f_cell)
        draw.text((740, y), unit_str, fill=(0, 0, 0), font=f_cell)
        draw.text((920, y), ref_str, fill=(0, 0, 0), font=f_cell)
        y += 65

    return img


def _render_image_set(data: dict, out_clean: Path, out_skewed: Path, out_poor: Path, rng: random.Random) -> None:
    """Render clean, skewed, and poor images using Pillow only (no numpy)."""
    clean_img = _render_clean_image(data)
    clean_img.save(out_clean)

    # 1. Skewed: 3 degrees rotation, blur, noise, quality 70
    skewed = clean_img.rotate(3.0, resample=Image.Resampling.BICUBIC, fillcolor=(240, 240, 240))
    skewed = skewed.filter(ImageFilter.GaussianBlur(radius=0.5))

    # Add noise using Pillow
    tile_w, tile_h = 310, 439
    noise_bytes = bytearray([rng.randint(124, 132) for _ in range(tile_w * tile_h)])
    tile = Image.frombytes("L", (tile_w, tile_h), bytes(noise_bytes))
    noise_img = Image.new("L", clean_img.size)
    for x in range(0, clean_img.width, tile_w):
        for y in range(0, clean_img.height, tile_h):
            noise_img.paste(tile, (x, y))
    noise_rgb = Image.merge("RGB", (noise_img, noise_img, noise_img))
    skewed_noisy = Image.blend(skewed, noise_rgb, 0.05)
    skewed_noisy.save(out_skewed, quality=70)

    # 2. Poor: downscale 40% then up, strong blur, uneven brightness, quality 40
    w, h = clean_img.size
    small = clean_img.resize((int(w * 0.4), int(h * 0.4)), Image.Resampling.BILINEAR)
    restored = small.resize((w, h), Image.Resampling.BILINEAR)
    blurred = restored.filter(ImageFilter.GaussianBlur(radius=2.0))

    # Horizontal brightness gradient: darker on left (160), brighter on right (255)
    grad = Image.linear_gradient("L").rotate(-90).resize((w, h))
    grad = grad.point(lambda p: int(160 + p * (255 - 160) / 255))
    grad_rgb = Image.merge("RGB", (grad, grad, grad))
    poor_img = ImageChops.multiply(blurred, grad_rgb)
    poor_img.save(out_poor, quality=40)


def generate_corpus(output_dir: Path | str, seed: int = 42, target_dir: Path | str | None = None) -> None:
    """Generate deterministic 30 PDFs, labels.json, and 10 OCR image sets."""
    rng = random.Random(seed)
    if target_dir is not None:
        corpus_dir = Path(target_dir)
    else:
        corpus_dir = Path(output_dir) / "eval" / "corpus"
    images_dir = corpus_dir / "images"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic evaluation corpus at {corpus_dir} (seed={seed})...")

    for i in range(1, 31):
        report_data = _build_report_data(i, rng)
        report_id = report_data["report_id"]

        pdf_path = corpus_dir / f"{report_id}.pdf"
        labels_path = corpus_dir / f"{report_id}.labels.json"

        # Write PDF
        _render_pdf(report_data, pdf_path)

        # Write labels.json
        labels_dict = {
            "report_date": report_data["report_date"],
            "lab_name": report_data["lab_name"],
            "layout": report_data["layout"],
            "results": report_data["results"],
            "planted_pii": report_data["planted_pii"],
        }
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels_dict, f, indent=2)

        # Render images for reports 1..10
        if 1 <= i <= 10:
            c_png = images_dir / f"{report_id}_clean.png"
            s_jpg = images_dir / f"{report_id}_skewed.jpg"
            p_jpg = images_dir / f"{report_id}_poor.jpg"
            _render_image_set(report_data, c_png, s_jpg, p_jpg, rng)

    print("Corpus generation complete: 30 PDFs, 30 labels.json, 30 images.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic evaluation corpus.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--out", type=str, default="eval/corpus", help="Output directory (default: eval/corpus)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    target_path = Path(args.out)
    if not target_path.is_absolute():
        target_path = base_dir / target_path
    generate_corpus(base_dir, seed=args.seed, target_dir=target_path)

