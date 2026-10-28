"""Rule-based PDF extraction for medical lab reports using pdfplumber."""

from datetime import date, datetime
from pathlib import Path
import re
import sys
import pdfplumber

from app.models.lab import ExtractionResponse, LabReport, LabResult, ResultFlag

# Privacy guard patterns to ensure patient identifiers are never captured
PATIENT_GUARD_PATTERNS = [
    re.compile(r"\bpatient[:\s]", re.IGNORECASE),
    re.compile(r"\btest\s+patient\b", re.IGNORECASE),
    re.compile(r"\bid[:\s]*\d+", re.IGNORECASE),
    re.compile(r"\b000000\b"),
    re.compile(r"\bphone[:\s]", re.IGNORECASE),
    re.compile(r"\baddress[:\s]", re.IGNORECASE),
    re.compile(r"\bdob[:\s]", re.IGNORECASE),
    re.compile(r"\bage[:\s]", re.IGNORECASE),
    re.compile(r"\bgender[:\s]", re.IGNORECASE),
    re.compile(r"\bsex[:\s]", re.IGNORECASE),
]

LINE_RESULT_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9\s\-_/().,\+]*?)\s+"
    r"(?P<val>[<>]?\s*\d+(?:\.\d+)?|Negative|Positive|Trace|Reactive|Non-Reactive)\s+"
    r"(?![\s]*$)"
    r"(?:(?P<unit>%|mg/dL|g/dL|mmol/L|uIU/mL|cells/mcL|fl|pg|g/L|IU/L|U/L|mIU/L|[A-Za-z0-9/%μµ^.-]+)\s+)?\s*"
    r"(?P<ref>(?:<=?|>=?)\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?|Negative|Positive|Normal|Trace)?$",
    re.IGNORECASE,
)


def contains_patient_info(text: str) -> bool:
    """Check if a string contains patient identifying keywords or patterns."""
    return any(p.search(text) for p in PATIENT_GUARD_PATTERNS)


def parse_reference_range(ref_text: str | None) -> tuple[float | None, float | None]:
    """Parse reference range string into numeric (ref_low, ref_high).

    Supports formats:
    - "a - b", "a-b", "a – b"
    - "< x", "<= x"
    - "> x", ">= x"
    - Returns (None, None) for non-numeric/text ranges or garbage inputs.
    """
    if not ref_text or not isinstance(ref_text, str):
        return None, None

    cleaned = ref_text.strip().replace("–", "-").replace("—", "-")
    if not cleaned:
        return None, None

    # Check inequality formats: <=, >=, <, >
    m_lte = re.fullmatch(r"<=\s*(\d+(?:\.\d+)?)", cleaned)
    if m_lte:
        return None, float(m_lte.group(1))

    m_gte = re.fullmatch(r">=\s*(\d+(?:\.\d+)?)", cleaned)
    if m_gte:
        return float(m_gte.group(1)), None

    m_lt = re.fullmatch(r"<\s*(\d+(?:\.\d+)?)", cleaned)
    if m_lt:
        return None, float(m_lt.group(1))

    m_gt = re.fullmatch(r">\s*(\d+(?:\.\d+)?)", cleaned)
    if m_gt:
        return float(m_gt.group(1)), None

    # Check range format: a - b or a-b
    m_range = re.fullmatch(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", cleaned)
    if m_range:
        low = float(m_range.group(1))
        high = float(m_range.group(2))
        return low, high

    return None, None


def parse_result_value(raw_val: str | None) -> tuple[float | None, str | None]:
    """Parse test result value into numeric (value) or text (value_text)."""
    if not raw_val or not isinstance(raw_val, str):
        return None, None

    cleaned = raw_val.strip()
    if not cleaned:
        return None, None

    try:
        val_float = float(cleaned.replace(",", ""))
        return val_float, None
    except ValueError:
        return None, cleaned


def parse_report_date(text: str) -> date | None:
    """Extract report date from text supporting multiple formats."""
    # Format: YYYY-MM-DD
    m_iso = re.search(r"\b(20\d\d-[01]\d-[0-3]\d)\b", text)
    if m_iso:
        try:
            return datetime.strptime(m_iso.group(1), "%Y-%m-%d").date()
        except ValueError:
            pass

    # Format: DD-Mon-YYYY (e.g. 15-Jan-2026)
    m_mon = re.search(r"\b([0-3]?\d-[A-Za-z]{3}-20\d\d)\b", text)
    if m_mon:
        try:
            return datetime.strptime(m_mon.group(1), "%d-%b-%Y").date()
        except ValueError:
            pass

    # Format: DD/MM/YYYY
    m_slash = re.search(r"\b([0-3]?\d/[01]?\d/20\d\d)\b", text)
    if m_slash:
        try:
            return datetime.strptime(m_slash.group(1), "%d/%m/%Y").date()
        except ValueError:
            pass

    return None


def parse_lab_name(text: str) -> str | None:
    """Extract laboratory name from header lines containing 'Lab' or 'Diagnostics'."""
    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        cleaned_lower = cleaned.lower()
        if contains_patient_info(cleaned):
            continue
        if any(bad in cleaned_lower for bad in ["doctor", "dr.", "test", "result", "date", "page"]):
            continue
        if "lab" in cleaned_lower or "diagnostics" in cleaned_lower or "pathology" in cleaned_lower:
            return cleaned
    return None


def detect_header(lines: list[str]) -> tuple[date | None, str | None]:
    """Detect report date and lab name from header text lines."""
    full_text = "\n".join(lines)
    return parse_report_date(full_text), parse_lab_name(full_text)


def extract_results_from_tables(page, page_idx: int) -> list[LabResult]:
    """Extract lab results using pdfplumber table extraction."""
    tables = page.extract_tables()
    if not tables:
        return []

    results: list[LabResult] = []
    for table in tables:
        if not table or len(table) < 2:
            continue

        # Detect column indices from header row
        header_idx = -1
        col_test = col_result = col_unit = col_ref = None

        for r_idx, row in enumerate(table):
            if not row:
                continue
            row_lower = [str(cell).lower().strip() if cell else "" for cell in row]
            t_col = r_col = u_col = ref_col = None
            for c_idx, cell_str in enumerate(row_lower):
                if any(k in cell_str for k in ["test", "parameter", "investigation", "analyte"]):
                    t_col = c_idx
                elif any(k in cell_str for k in ["result", "value", "observed", "finding"]):
                    r_col = c_idx
                elif "unit" in cell_str:
                    u_col = c_idx
                elif any(k in cell_str for k in ["reference", "ref", "range", "interval", "normal"]):
                    ref_col = c_idx

            if t_col is not None and r_col is not None:
                header_idx = r_idx
                col_test = t_col
                col_result = r_col
                col_unit = u_col
                col_ref = ref_col
                break

        if header_idx == -1:
            continue

        # Parse subsequent rows
        for row in table[header_idx + 1:]:
            if not row or len(row) <= max(col_test, col_result):
                continue

            test_name = str(row[col_test]).strip() if row[col_test] is not None else ""
            result_str = str(row[col_result]).strip() if row[col_result] is not None else ""

            if not test_name or not result_str:
                continue

            if contains_patient_info(test_name) or contains_patient_info(result_str):
                continue

            unit = (
                str(row[col_unit]).strip()
                if (col_unit is not None and col_unit < len(row) and row[col_unit] is not None)
                else None
            )
            if unit == "":
                unit = None

            ref_str = (
                str(row[col_ref]).strip()
                if (col_ref is not None and col_ref < len(row) and row[col_ref] is not None)
                else None
            )
            if ref_str == "":
                ref_str = None

            val, val_text = parse_result_value(result_str)
            ref_low, ref_high = parse_reference_range(ref_str)

            row_cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
            snippet = " | ".join(row_cells)

            try:
                lab_result = LabResult(
                    test_name_raw=test_name,
                    test_name_canonical=None,
                    value=val,
                    value_text=val_text,
                    unit=unit,
                    ref_low=ref_low,
                    ref_high=ref_high,
                    ref_text=ref_str,
                    flag=ResultFlag.UNKNOWN,
                    source_page=page_idx,
                    source_snippet=snippet,
                    confidence=1.0,
                )
                results.append(lab_result)
            except Exception:
                continue

    return results


def parse_text_lines(lines: list[str], page: int = 1) -> list[LabResult]:
    """Parse a list of text lines into LabResults using regex matching."""
    results: list[LabResult] = []

    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue

        if contains_patient_info(cleaned):
            continue

        # Skip column header lines or lab header
        cleaned_lower = cleaned.lower()
        if "test name" in cleaned_lower or "reference range" in cleaned_lower:
            continue
        if "diagnostics" in cleaned_lower or "sample" in cleaned_lower:
            continue

        m = LINE_RESULT_RE.match(cleaned)
        if not m:
            continue

        test_name = m.group("name").strip()
        raw_val = m.group("val").strip()
        unit = m.group("unit").strip() if m.group("unit") else None
        ref_str = m.group("ref").strip() if m.group("ref") else None

        if contains_patient_info(test_name):
            continue

        val, val_text = parse_result_value(raw_val)
        ref_low, ref_high = parse_reference_range(ref_str)

        try:
            lab_result = LabResult(
                test_name_raw=test_name,
                test_name_canonical=None,
                value=val,
                value_text=val_text,
                unit=unit,
                ref_low=ref_low,
                ref_high=ref_high,
                ref_text=ref_str,
                flag=ResultFlag.UNKNOWN,
                source_page=page,
                source_snippet=cleaned,
                confidence=1.0,
            )
            results.append(lab_result)
        except Exception:
            continue

    return results


def extract_results_from_lines(page_text: str, page_idx: int) -> list[LabResult]:
    """Fall back to regex parsing of text lines when tables are not available."""
    return parse_text_lines(page_text.splitlines(), page=page_idx)


def extract_report_from_pdf(path: str) -> ExtractionResponse:
    """Extract medical report data and lab results from a digital PDF file."""
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    filename = pdf_path.name
    warnings: list[str] = []

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            pages_text: list[str] = []
            for p in pdf.pages:
                t = p.extract_text() or ""
                pages_text.append(t)

            all_text = "\n".join(pages_text)
            if not all_text.strip():
                return ExtractionResponse(
                    report=LabReport(
                        source_filename=filename,
                        results=[],
                    ),
                    warnings=["No text layer found. OCR is required for scanned documents."],
                )

            report_date, lab_name = detect_header(pages_text)

            # 1. Try table extraction first
            results: list[LabResult] = []
            for page_idx, page in enumerate(pdf.pages, start=1):
                table_results = extract_results_from_tables(page, page_idx)
                results.extend(table_results)

            # 2. If no usable table is found, fall back to line-by-line regex parsing
            if not results:
                for page_idx, page_text in enumerate(pages_text, start=1):
                    line_results = extract_results_from_lines(page_text, page_idx)
                    results.extend(line_results)

            report = LabReport(
                report_date=report_date,
                lab_name=lab_name,
                source_filename=filename,
                results=results,
            )

            return ExtractionResponse(report=report, warnings=warnings)
    except Exception:
        return ExtractionResponse(
            report=LabReport(
                source_filename=filename,
                results=[],
            ),
            warnings=["Could not read PDF file: file may be corrupt or invalid."],
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        response = extract_report_from_pdf(pdf_file)
        print(response.model_dump_json(indent=2))
    else:
        print("Usage: python -m app.services.pdf_extractor <path_to_pdf>")
