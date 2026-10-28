"""Script to generate synthetic medical report PDFs and expected JSON answer keys."""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def generate_report_table_pdf(output_path: Path) -> None:
    """Generate report_table.pdf with a bordered table."""
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    styles = getSampleStyleSheet()
    elements = []

    # Header section with fake lab and fake patient info
    elements.append(Paragraph("<b>Sample Diagnostics Lab</b>", styles["Heading1"]))
    elements.append(Paragraph("Date: 2026-01-15", styles["Normal"]))
    elements.append(Paragraph("Patient: TEST PATIENT, ID 000000", styles["Normal"]))
    elements.append(Spacer(1, 15))

    # Bordered table with columns: Test | Result | Unit | Reference Range
    data = [
        ["Test", "Result", "Unit", "Reference Range"],
        ["HbA1c", "7.4", "%", "4.0 - 5.6"],
        ["Fasting Glucose", "132", "mg/dL", "70 - 99"],
        ["Total Cholesterol", "214", "mg/dL", "< 200"],
        ["LDL Cholesterol", "138", "mg/dL", "< 100"],
        ["HDL Cholesterol", "42", "mg/dL", "> 40"],
        ["Triglycerides", "168", "mg/dL", "< 150"],
        ["Hemoglobin", "13.1", "g/dL", "13.0 - 17.0"],
        ["Urine Protein", "Negative", "", "Negative"],
    ]

    table = Table(data, colWidths=[160, 70, 70, 130])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    elements.append(table)
    doc.build(elements)


def generate_report_lines_pdf(output_path: Path) -> None:
    """Generate report_lines.pdf with plain text lines and no table borders."""
    c = canvas.Canvas(str(output_path), pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(54, 730, "Sample Diagnostics Lab")
    c.setFont("Helvetica", 10)
    c.drawString(54, 710, "Date: 2026-04-20")
    c.drawString(54, 695, "Patient: TEST PATIENT, ID 000000")

    y = 650
    c.setFont("Courier-Bold", 10)
    c.drawString(54, y, "Test Name                 Result    Unit     Reference Range")

    rows = [
        ("Glycosylated Hb", "6.9", "%", "4.0 - 5.6"),
        ("FBS", "110", "mg/dL", "70 - 99"),
        ("Cholesterol Total", "195", "mg/dL", "< 200"),
        ("LDL-C", "115", "mg/dL", "< 100"),
        ("HDL-C", "48", "mg/dL", "> 40"),
        ("TG", "140", "mg/dL", "< 150"),
        ("Haemoglobin", "14.2", "g/dL", "13.0 - 17.0"),
    ]

    c.setFont("Courier", 10)
    for test, result, unit, ref in rows:
        y -= 22
        # Format as space-separated columns
        line = f"{test:<26}{result:<10}{unit:<9}{ref}"
        c.drawString(54, y, line)

    c.save()


def generate_report_scanned_pdf(output_path: Path) -> None:
    """Generate an image-only PDF with no text layer using Pillow."""
    img = Image.new("RGB", (1240, 1754), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        f_title = ImageFont.truetype("trebucbd.ttf", 44)
        f_sub = ImageFont.truetype("trebuc.ttf", 28)
        f_cell = ImageFont.truetype("trebuc.ttf", 30)
    except Exception:
        try:
            f_title = ImageFont.truetype("arialbd.ttf", 44)
            f_sub = ImageFont.truetype("arial.ttf", 28)
            f_cell = ImageFont.truetype("arial.ttf", 30)
        except Exception:
            f_title = f_sub = f_cell = ImageFont.load_default()

    draw.text((100, 100), "Sample Diagnostics Lab", fill=(0, 0, 0), font=f_title)
    draw.text((100, 180), "Date: 2026-01-15", fill=(0, 0, 0), font=f_sub)
    draw.text((100, 240), "Patient: TEST PATIENT, ID 000000", fill=(0, 0, 0), font=f_sub)

    y = 350
    draw.text((100, y), "Test Name", fill=(0, 0, 0), font=f_sub)
    draw.text((560, y), "Result", fill=(0, 0, 0), font=f_sub)
    draw.text((760, y), "Unit", fill=(0, 0, 0), font=f_sub)
    draw.text((940, y), "Reference Range", fill=(0, 0, 0), font=f_sub)

    rows = [
        ("HbA1c", "7.4", "%", "4.0 - 5.6"),
        ("Fasting Glucose", "132", "mg/dL", "70 - 99"),
        ("Total Cholesterol", "214", "mg/dL", "< 200"),
        ("LDL Cholesterol", "138", "mg/dL", "< 100"),
        ("HDL Cholesterol", "42", "mg/dL", "> 40"),
        ("Triglycerides", "168", "mg/dL", "< 150"),
        ("Hemoglobin", "13.1", "g/dL", "13.0 - 17.0"),
        ("Urine Protein", "Negative", "", "Negative"),
    ]

    for row in rows:
        y += 75
        draw.text((100, y), row[0], fill=(0, 0, 0), font=f_cell)
        draw.text((560, y), row[1], fill=(0, 0, 0), font=f_cell)
        draw.text((760, y), row[2], fill=(0, 0, 0), font=f_cell)
        draw.text((940, y), row[3], fill=(0, 0, 0), font=f_cell)

    img.save(str(output_path), "PDF", resolution=150.0)


def generate_expected_json_files() -> None:
    """Write expected JSON answer keys for report_table and report_lines."""
    table_expected = {
        "report_date": "2026-01-15",
        "lab_name": "Sample Diagnostics Lab",
        "results": [
            {
                "test_name_raw": "HbA1c",
                "value": 7.4,
                "unit": "%",
                "ref_low": 4.0,
                "ref_high": 5.6,
            },
            {
                "test_name_raw": "Fasting Glucose",
                "value": 132.0,
                "unit": "mg/dL",
                "ref_low": 70.0,
                "ref_high": 99.0,
            },
            {
                "test_name_raw": "Total Cholesterol",
                "value": 214.0,
                "unit": "mg/dL",
                "ref_low": None,
                "ref_high": 200.0,
            },
            {
                "test_name_raw": "LDL Cholesterol",
                "value": 138.0,
                "unit": "mg/dL",
                "ref_low": None,
                "ref_high": 100.0,
            },
            {
                "test_name_raw": "HDL Cholesterol",
                "value": 42.0,
                "unit": "mg/dL",
                "ref_low": 40.0,
                "ref_high": None,
            },
            {
                "test_name_raw": "Triglycerides",
                "value": 168.0,
                "unit": "mg/dL",
                "ref_low": None,
                "ref_high": 150.0,
            },
            {
                "test_name_raw": "Hemoglobin",
                "value": 13.1,
                "unit": "g/dL",
                "ref_low": 13.0,
                "ref_high": 17.0,
            },
            {
                "test_name_raw": "Urine Protein",
                "value_text": "Negative",
                "unit": None,
                "ref_low": None,
                "ref_high": None,
            },
        ],
    }

    lines_expected = {
        "report_date": "2026-04-20",
        "lab_name": "Sample Diagnostics Lab",
        "results": [
            {
                "test_name_raw": "Glycosylated Hb",
                "value": 6.9,
                "unit": "%",
                "ref_low": 4.0,
                "ref_high": 5.6,
            },
            {
                "test_name_raw": "FBS",
                "value": 110.0,
                "unit": "mg/dL",
                "ref_low": 70.0,
                "ref_high": 99.0,
            },
            {
                "test_name_raw": "Cholesterol Total",
                "value": 195.0,
                "unit": "mg/dL",
                "ref_low": None,
                "ref_high": 200.0,
            },
            {
                "test_name_raw": "LDL-C",
                "value": 115.0,
                "unit": "mg/dL",
                "ref_low": None,
                "ref_high": 100.0,
            },
            {
                "test_name_raw": "HDL-C",
                "value": 48.0,
                "unit": "mg/dL",
                "ref_low": 40.0,
                "ref_high": None,
            },
            {
                "test_name_raw": "TG",
                "value": 140.0,
                "unit": "mg/dL",
                "ref_low": None,
                "ref_high": 150.0,
            },
            {
                "test_name_raw": "Haemoglobin",
                "value": 14.2,
                "unit": "g/dL",
                "ref_low": 13.0,
                "ref_high": 17.0,
            },
        ],
    }

    with open(SAMPLES_DIR / "report_table.expected.json", "w", encoding="utf-8") as f:
        json.dump(table_expected, f, indent=2)

    with open(SAMPLES_DIR / "report_lines.expected.json", "w", encoding="utf-8") as f:
        json.dump(lines_expected, f, indent=2)


def main() -> None:
    """Generate all sample reports and answer keys."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    table_pdf = SAMPLES_DIR / "report_table.pdf"
    lines_pdf = SAMPLES_DIR / "report_lines.pdf"
    scanned_pdf = SAMPLES_DIR / "report_scanned.pdf"

    print(f"Generating {table_pdf}...")
    generate_report_table_pdf(table_pdf)

    print(f"Generating {lines_pdf}...")
    generate_report_lines_pdf(lines_pdf)

    print(f"Generating {scanned_pdf}...")
    generate_report_scanned_pdf(scanned_pdf)

    print("Generating expected JSON answer keys...")
    generate_expected_json_files()

    print("All sample files generated successfully.")


if __name__ == "__main__":
    main()
