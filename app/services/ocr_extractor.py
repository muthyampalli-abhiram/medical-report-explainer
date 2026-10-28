"""OCR engine for medical lab reports from photographed/scanned images and PDFs."""

from collections import defaultdict
from datetime import date
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import pdfplumber
from PIL import Image, ImageFilter, ImageOps
import pytesseract

from app.config import settings
from app.models.lab import ExtractionResponse, LabReport, LabResult
from app.services.ocr_safeguards import apply_ocr_safeguards
from app.services.pdf_extractor import (
    contains_patient_info,
    detect_header,
    extract_report_from_pdf,
    parse_text_lines,
)


class OCRUnavailableError(Exception):
    """Raised when Tesseract OCR is not installed, not found, or fails to execute."""


def _ensure_tesseract() -> None:
    """Ensure Tesseract executable is found and configured without exposing paths."""
    if settings.TESSERACT_CMD:
        cmd_path = Path(settings.TESSERACT_CMD)
        if not cmd_path.exists():
            raise OCRUnavailableError("OCR engine is not available on this server.")
        pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
    else:
        cmd = pytesseract.pytesseract.tesseract_cmd
        if not shutil.which(cmd):
            raise OCRUnavailableError("OCR engine is not available on this server.")


def deskew_image(img: Image.Image) -> Image.Image:
    """Small-angle deskew between -5 and +5 degrees using projection-profile method."""
    w, h = img.size
    scale = min(1.0, 1000.0 / max(w, h))
    if scale < 1.0:
        thumb = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
    else:
        thumb = img

    arr = np.array(thumb)
    th, tw = arr.shape
    crop = arr[int(th * 0.1) : int(th * 0.9), int(tw * 0.1) : int(tw * 0.9)]
    fill_val = int(np.median(crop))

    crop_img = Image.fromarray(crop)
    angles = np.arange(-5.0, 5.25, 0.25)
    best_angle = 0.0
    best_var = -1.0

    for a in angles:
        if a == 0.0:
            rot_crop = crop
        else:
            rot = crop_img.rotate(a, resample=Image.Resampling.BILINEAR, fillcolor=fill_val)
            rot_crop = np.array(rot)
        v = np.var(np.sum(255 - rot_crop, axis=1, dtype=np.float64))
        if v > best_var:
            best_var = v
            best_angle = a

    if best_angle != 0.0:
        return img.rotate(best_angle, resample=Image.Resampling.BILINEAR, fillcolor=fill_val)
    return img


def preprocess_image(img: Image.Image) -> Image.Image:
    """Preprocess image for OCR using Pillow and numpy only.

    Steps:
      - check pixel limit vs OCR_MAX_IMAGE_PIXELS
      - EXIF orientation
      - grayscale
      - upscale so shorter side is at least 1600 px (never downscale below that)
      - auto-contrast
      - mild sharpen
      - deskew (-5 to +5 degrees)
    """
    Image.MAX_IMAGE_PIXELS = settings.OCR_MAX_IMAGE_PIXELS
    if img.width * img.height > settings.OCR_MAX_IMAGE_PIXELS:
        raise ValueError("Image is too large.")

    # 1. EXIF orientation
    img = ImageOps.exif_transpose(img)

    # 2. Convert to grayscale
    img = img.convert("L")

    # 3. Upscale shorter side to at least 1600 px (never downscale below that)
    w, h = img.size
    shorter = min(w, h)
    if shorter < 1600:
        scale = 1600.0 / shorter
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        if new_w * new_h > settings.OCR_MAX_IMAGE_PIXELS:
            raise ValueError("Image is too large.")
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 4. Auto-contrast
    img = ImageOps.autocontrast(img)

    # 5. Mild sharpen
    img = img.filter(ImageFilter.SHARPEN)

    # 6. Deskew
    img = deskew_image(img)

    return img


def _is_numeric_token(tok: str) -> bool:
    """Check if token is a numeric value, range, or measurement number."""
    cleaned = re.sub(r"^[<>=~±+\-]+|[%,]+$", "", tok.strip())
    if not cleaned:
        return False
    try:
        float(cleaned)
        return True
    except ValueError:
        pass
    if re.match(r"^\d+(?:\.\d+)?\s*[-–—]\s*\d+(?:\.\d+)?$", cleaned):
        return True
    return any(c.isdigit() for c in tok) and not any(c.isalpha() for c in tok)


def _ocr_single_image(
    prep: Image.Image, page_num: int = 1
) -> tuple[list[LabResult], date | None, str | None, list[str]]:
    """Run Tesseract OCR on a preprocessed image and parse results."""
    _ensure_tesseract()

    try:
        data = pytesseract.image_to_data(
            prep,
            lang="eng",
            config="--oem 1 --psm 6",
            timeout=30,
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractNotFoundError:
        raise OCRUnavailableError("OCR engine is not available on this server.") from None
    except pytesseract.TesseractError:
        raise OCRUnavailableError("OCR engine failed to execute on this server.") from None
    except (FileNotFoundError, OSError):
        raise OCRUnavailableError("OCR engine is not available on this server.") from None

    # Group words into lines by (block_num, par_num, line_num)
    lines_dict = defaultdict(list)
    for i in range(len(data["text"])):
        w = str(data["text"][i]).strip()
        c = float(data["conf"][i])
        if c == -1 or not w:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines_dict[key].append((w, c))

    ocr_lines: list[tuple[str, float]] = []
    for key, words in sorted(lines_dict.items()):
        line_text = " ".join(w[0] for w in words)
        confs = [w[1] for w in words]
        num_confs = [w[1] for w in words if _is_numeric_token(w[0])]
        mean_conf = sum(confs) / len(confs)
        if num_confs:
            line_conf = min(min(num_confs), mean_conf) / 100.0
        else:
            line_conf = mean_conf / 100.0
        line_conf = round(max(0.0, min(1.0, line_conf)), 4)
        ocr_lines.append((line_text, line_conf))

    # Guard: drop lines containing patient info before anything is stored
    safe_lines = [(text, conf) for text, conf in ocr_lines if not contains_patient_info(text)]
    line_texts = [text for text, _ in safe_lines]

    report_date, lab_name = detect_header(line_texts)
    raw_results = parse_text_lines(line_texts, page=page_num)

    # Map each parsed result back to its OCR line confidence and snippet
    results: list[LabResult] = []
    safe_ptr = 0
    for r in raw_results:
        while safe_ptr < len(safe_lines):
            text, conf = safe_lines[safe_ptr]
            safe_ptr += 1
            if text.strip() == r.source_snippet.strip():
                r.confidence = conf
                r.source_snippet = text
                break
        results.append(r)

    return results, report_date, lab_name, line_texts


def extract_report_from_image(path: str | Path) -> ExtractionResponse:
    """Extract medical report data and lab results from a photographed/scanned image."""
    _ensure_tesseract()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    Image.MAX_IMAGE_PIXELS = settings.OCR_MAX_IMAGE_PIXELS
    try:
        with Image.open(p) as img:
            prep = preprocess_image(img)
    except Image.DecompressionBombError:
        raise ValueError("Image is too large.")

    results, report_date, lab_name, _ = _ocr_single_image(prep, page_num=1)

    report = LabReport(
        report_date=report_date,
        lab_name=lab_name,
        source_filename=p.name,
        results=results,
    )
    return ExtractionResponse(report=report, warnings=[])


def extract_report_with_ocr_fallback(pdf_path: str | Path) -> tuple[ExtractionResponse, str]:
    """Extract report from PDF, falling back to OCR only if no text layer is found."""
    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    text_response = extract_report_from_pdf(str(p))

    needs_ocr = (
        len(text_response.report.results) == 0
        and any("No text layer found" in w for w in text_response.warnings)
    )

    if not needs_ocr:
        return text_response, "text"

    # OCR fallback
    _ensure_tesseract()
    warnings: list[str] = []
    all_results: list[LabResult] = []
    all_lines: list[str] = []

    with pdfplumber.open(str(p)) as pdf:
        total_pages = len(pdf.pages)
        pages_to_process = pdf.pages[: settings.OCR_MAX_PAGES]
        if total_pages > settings.OCR_MAX_PAGES:
            warnings.append(f"Only the first {settings.OCR_MAX_PAGES} pages were read.")

        for page_idx, page in enumerate(pages_to_process, start=1):
            page_img = page.to_image(resolution=300).original
            prep = preprocess_image(page_img)
            page_results, _, _, p_lines = _ocr_single_image(prep, page_num=page_idx)
            all_results.extend(page_results)
            all_lines.extend(p_lines)

    report_date, lab_name = detect_header(all_lines)

    report = LabReport(
        report_date=report_date,
        lab_name=lab_name,
        source_filename=p.name,
        results=all_results,
    )
    return ExtractionResponse(report=report, warnings=warnings), "ocr"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
        if not input_path.exists():
            print(f"File not found: {input_path}")
            sys.exit(1)

        try:
            if input_path.suffix.lower() == ".pdf":
                response, method = extract_report_with_ocr_fallback(input_path)
            else:
                response = extract_report_from_image(input_path)
                method = "ocr"

            from app.services.ocr_safeguards import apply_ocr_safeguards, filter_ocr_junk
            from app.services.pipeline import process_report

            processed = process_report(response)
            if method == "ocr":
                processed = filter_ocr_junk(processed)
                processed = apply_ocr_safeguards(processed)

            headers = f"{'Test':<25} {'Value':<12} {'Unit':<10} {'Reference':<18} {'Flag':<15} {'Confidence'}"
            print(headers)
            print("-" * len(headers))
            for r in processed.report.results:
                test = r.test_name_raw
                val = str(r.value) if r.value is not None else (r.value_text or "")
                unit = r.unit or ""
                ref = r.ref_text or (
                    f"{r.ref_low} - {r.ref_high}" if r.ref_low and r.ref_high else ""
                )
                flag = r.flag.value if hasattr(r.flag, "value") else str(r.flag)
                conf = f"{r.confidence:.2f}"
                print(f"{test:<25} {val:<12} {unit:<10} {ref:<18} {flag:<15} {conf}")

            if processed.warnings:
                print("\nWarnings:")
                for w in processed.warnings:
                    print(f"  - {w}")

        except OCRUnavailableError as e:
            print(f"OCR Error: {e}")
            sys.exit(1)
    else:
        print("Usage: python -m app.services.ocr_extractor <path to image or pdf>")
