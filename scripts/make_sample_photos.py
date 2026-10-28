"""Generate synthetic medical report photo samples (clean, skewed, poor)."""

import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load standard Windows font with fallbacks."""
    candidates = (
        ["trebucbd.ttf", "arialbd.ttf", "calibrib.ttf"]
        if bold
        else ["trebuc.ttf", "arial.ttf", "calibri.ttf"]
    )
    for fn in candidates:
        try:
            return ImageFont.truetype(fn, size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_report_table_image() -> Image.Image:
    """Render a clean A4 table report image (1240 x 1754 px)."""
    img = Image.new("RGB", (1240, 1754), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    f_title = get_font(44, bold=True)
    f_sub = get_font(28, bold=False)
    f_cell = get_font(30, bold=False)

    # Header section
    draw.text((100, 100), "Sample Diagnostics Lab", fill=(0, 0, 0), font=f_title)
    draw.text((100, 180), "Date: 2026-01-15", fill=(0, 0, 0), font=f_sub)
    draw.text((100, 240), "Patient: TEST PATIENT, ID 000000", fill=(0, 0, 0), font=f_sub)

    # Table header
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

    return img


def generate_photos() -> None:
    """Generate clean.png, skewed.jpg, and poor.jpg sample photos."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    clean_img = render_report_table_image()
    clean_path = SAMPLES_DIR / "photo_clean.png"
    clean_img.save(clean_path)
    print(f"Generated {clean_path}")

    # Skewed: rotated 3 degrees on light grey background with slight blur and mild noise, quality 70
    skewed = clean_img.rotate(3.0, resample=Image.Resampling.BICUBIC, fillcolor=(240, 240, 240))
    skewed = skewed.filter(ImageFilter.GaussianBlur(radius=0.5))
    np.random.seed(42)
    arr = np.array(skewed, dtype=np.int16)
    noise = np.random.normal(0, 4, arr.shape).astype(np.int16)
    skewed_noisy = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    skewed_path = SAMPLES_DIR / "photo_skewed.jpg"
    skewed_noisy.save(skewed_path, quality=70)
    print(f"Generated {skewed_path}")

    # Poor: heavily degraded (downscaled to 40% then back up, strong blur, uneven brightness gradient, quality 40)
    w, h = clean_img.size
    small = clean_img.resize((int(w * 0.4), int(h * 0.4)), Image.Resampling.BILINEAR)
    restored = small.resize((w, h), Image.Resampling.BILINEAR)
    blurred = restored.filter(ImageFilter.GaussianBlur(radius=2.0))
    grad = np.linspace(0.6, 1.2, w, dtype=np.float32)
    arr_poor = np.clip(np.array(blurred, dtype=np.float32) * grad[np.newaxis, :, np.newaxis], 0, 255).astype(np.uint8)
    poor_img = Image.fromarray(arr_poor)
    poor_path = SAMPLES_DIR / "photo_poor.jpg"
    poor_img.save(poor_path, quality=40)
    print(f"Generated {poor_path}")


if __name__ == "__main__":
    generate_photos()
