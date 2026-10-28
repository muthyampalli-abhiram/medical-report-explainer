"""Script to generate synthetic prescription PDFs (digital text and scanned image-only)."""

from pathlib import Path
from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def generate_prescription_text_pdf(output_path: Path) -> None:
    """Generate prescription_text.pdf with digital text layer."""
    c = canvas.Canvas(str(output_path), pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(54, 730, "Sample Clinic")
    c.setFont("Helvetica", 10)
    c.drawString(54, 710, "Patient: TEST PATIENT")
    c.drawString(54, 695, "Dr. Sample Doctor")
    c.drawString(54, 680, "Ph: 9876543210")

    y = 640
    lines = [
        "Rx",
        "1. Tab Metformin 500 mg  1-0-1  after food  x 30 days",
        "2. Tab Atorvastatin 10 mg  0-0-1  at bedtime  x 30 days",
        "3. Cap Omeprazole 20 mg  OD  before food  x 14 days",
        "4. Syp Paracetamol 650 mg  SOS  for fever",
        "Avoid sugary drinks",
        "Review after 2 weeks with HbA1c report",
    ]
    for line in lines:
        if line == "Rx":
            c.setFont("Helvetica-Bold", 12)
        else:
            c.setFont("Helvetica", 10)
        c.drawString(54, y, line)
        y -= 25

    c.save()


def generate_prescription_scanned_pdf(output_path: Path) -> None:
    """Generate prescription_scanned.pdf as image-only PDF with no text layer."""
    img = Image.new("RGB", (612, 792), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.text((54, 54), "Sample Clinic", fill=(0, 0, 0))
    draw.text((54, 76), "Patient: TEST PATIENT", fill=(0, 0, 0))
    draw.text((54, 94), "Dr. Sample Doctor", fill=(0, 0, 0))
    draw.text((54, 112), "Ph: 9876543210", fill=(0, 0, 0))
    draw.text((54, 140), "Rx", fill=(0, 0, 0))
    draw.text((54, 165), "1. Tab Metformin 500 mg 1-0-1 after food x 30 days", fill=(0, 0, 0))
    draw.text((54, 190), "2. Tab Atorvastatin 10 mg 0-0-1 at bedtime x 30 days", fill=(0, 0, 0))
    draw.text((54, 215), "Avoid sugary drinks", fill=(0, 0, 0))

    img.save(str(output_path), "PDF", resolution=72.0)


def main() -> None:
    """Generate both sample prescription PDFs."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    text_pdf = SAMPLES_DIR / "prescription_text.pdf"
    scanned_pdf = SAMPLES_DIR / "prescription_scanned.pdf"

    generate_prescription_text_pdf(text_pdf)
    print(f"Generated: {text_pdf}")

    generate_prescription_scanned_pdf(scanned_pdf)
    print(f"Generated: {scanned_pdf}")


if __name__ == "__main__":
    main()
