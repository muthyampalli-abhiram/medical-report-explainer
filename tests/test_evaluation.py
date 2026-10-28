"""Automated evaluation test suite verifying hard invariants on quick subset.

Runs quickly without network access or live LLM calls.
"""

import hashlib
import json
from pathlib import Path
import pytest
import shutil

from app.config import settings
from app.models.explanation import ReportExplanation
from app.models.lab import ExtractionResponse, LabReport, LabResult, ResultFlag
from app.services.explainer import explain_report
from app.services.flagger import compute_flag
from app.services.llm_client import FakeLLMClient
from app.services.normalizer import canonicalize_name
from app.services.ocr_extractor import extract_report_from_image
from app.services.ocr_safeguards import apply_ocr_safeguards, filter_ocr_junk
from app.services.pdf_extractor import (
    LINE_RESULT_RE,
    extract_report_from_pdf,
    parse_lab_name,
    parse_reference_range,
    parse_report_date,
    parse_text_lines,
)
from app.services.pipeline import process_report
from app.services.safety import check_text
from app.services.units import clean_unit, convert_to_canonical
from scripts.evaluate_all import (
    CASES_FILE,
    CORPUS_DIR,
    UNSAFE_LLM_OUTPUTS,
    evaluate_case_lists,
    evaluate_digital_extraction,
    evaluate_end_to_end,
    evaluate_privacy,
    evaluate_safety,
    evaluate_translation_integrity,
)
from scripts.make_eval_corpus import generate_corpus

ROOT_DIR = Path(__file__).resolve().parent.parent


def is_tesseract_available() -> bool:
    """Check if Tesseract binary is available."""
    if settings.TESSERACT_CMD:
        return Path(settings.TESSERACT_CMD).exists()
    return shutil.which("tesseract") is not None


requires_tesseract = pytest.mark.skipif(
    not is_tesseract_available(),
    reason="Tesseract OCR executable not found",
)


def test_generator_determinism(tmp_path):
    """Corpus generator must be deterministic: generating twice gives byte-identical labels.json files."""
    run1_dir = tmp_path / "run1"
    run2_dir = tmp_path / "run2"

    generate_corpus(run1_dir, seed=42)
    generate_corpus(run2_dir, seed=42)

    def hash_labels(d: Path) -> str:
        h = hashlib.sha256()
        for p in sorted((d / "eval" / "corpus").glob("*.labels.json")):
            h.update(p.read_bytes())
        return h.hexdigest()

    assert hash_labels(run1_dir) == hash_labels(run2_dir)


def test_flag_cases_100_percent():
    """Verify that all 40 hand-derived flag cases pass 100%."""
    m_cases, failures = evaluate_case_lists(CASES_FILE)
    assert failures == []
    assert m_cases["flag_cases"]["accuracy"] == 1.0
    assert m_cases["flag_cases"]["passed"] == 40


def test_unit_cases_100_percent():
    """Verify that all 20 hand-derived unit cases pass 100%."""
    m_cases, failures = evaluate_case_lists(CASES_FILE)
    assert failures == []
    assert m_cases["unit_cases"]["accuracy"] == 1.0
    assert m_cases["unit_cases"]["passed"] == 20


def test_digital_quick_subset_field_accuracy_and_flags():
    """Assert digital field accuracy is at least 0.95 and flag accuracy is 100% on quick subset."""
    all_pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    quick_pdfs = [p for p in all_pdfs if int(p.stem.split("_")[1]) <= 10]
    assert len(quick_pdfs) == 10

    m_dig, f_dig = evaluate_digital_extraction(quick_pdfs)
    assert m_dig["field_accuracy"] >= 0.95
    assert m_dig["row_recall"] >= 0.95

    m_e2e, f_e2e = evaluate_end_to_end(quick_pdfs)
    assert m_e2e["flag_accuracy"] == 1.0


def test_privacy_leaks_zero():
    """Assert zero PII leaks on quick corpus subset through full API scan."""
    all_pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    quick_pdfs = [p for p in all_pdfs if int(p.stem.split("_")[1]) <= 5]

    m_priv, f_priv = evaluate_privacy(quick_pdfs)
    assert f_priv == []
    assert m_priv["leaks_found"] == 0
    assert m_priv["strings_scanned"] > 0


def test_translation_mismatches_zero():
    """Assert zero translation mismatches across en, hi, te for numbers, flags, and test names."""
    all_pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
    quick_pdfs = [p for p in all_pdfs if int(p.stem.split("_")[1]) <= 5]

    m_trans, f_trans = evaluate_translation_integrity(quick_pdfs)
    assert f_trans == []
    assert m_trans["mismatches"] == 0


def test_unsafe_outputs_neutralized_end_to_end():
    """Assert every unsafe output is neutralized end-to-end and used_fallback is true."""
    m_safe, f_safe = evaluate_safety(live=False)
    assert m_safe["unsafe_detection_rate"] == 1.0
    assert m_safe["false_positive_rate"] == 0.0
    assert m_safe["end_to_end_neutralized_count"] == len(UNSAFE_LLM_OUTPUTS)


def test_prompt_injection_never_reaches_llm_prompt():
    """Assert prompt injection attempts never appear in LLM prompts, explanations, or questions."""
    m_safe, f_safe = evaluate_safety(live=False)
    assert m_safe["prompt_injection"]["passed"] == m_safe["prompt_injection"]["total"]


@requires_tesseract
def test_ocr_quick_subset_safety_count_zero():
    """Assert OCR safety count is 0 on quick subset images."""
    quick_images = [
        CORPUS_DIR / "images" / "eval_01_clean.png",
        CORPUS_DIR / "images" / "eval_01_skewed.jpg",
        CORPUS_DIR / "images" / "eval_01_poor.jpg",
    ]
    quick_images = [p for p in quick_images if p.exists()]
    from scripts.evaluate_all import evaluate_ocr
    m_ocr, f_ocr = evaluate_ocr(quick_images, CORPUS_DIR)
    assert m_ocr["overall_safety_count"] == 0


# ---------------------------------------------------------------------------
# REGRESSION TESTS FOR PRODUCTION CODE FIXES (PART F)
# ---------------------------------------------------------------------------
def test_regression_single_letter_test_name():
    """Regression test Part F(b): Potassium alias 'K' matches in LINE_RESULT_RE."""
    line = "K 1.9 mmol/L 3.5-5"
    m = LINE_RESULT_RE.match(line)
    assert m is not None
    assert m.group("name") == "K"
    assert m.group("val") == "1.9"


def test_regression_comma_in_test_name():
    """Regression test Part F(b): Comma-containing names match in LINE_RESULT_RE."""
    line_gluc = "Glucose, Fasting          95      mg/dL    70-99"
    m_gluc = LINE_RESULT_RE.match(line_gluc)
    assert m_gluc is not None
    assert m_gluc.group("name") == "Glucose, Fasting"

    line_prot = "Protein, Urine           Negative            Negative"
    m_prot = LINE_RESULT_RE.match(line_prot)
    assert m_prot is not None
    assert m_prot.group("name") == "Protein, Urine"


def test_regression_pathology_lab_name():
    """Regression test Part F(b): parse_lab_name recognizes 'Pathology' in headers."""
    text = "Sunrise Pathology Centre\nDate: 2026-03-01\nPatient: John Doe"
    lab = parse_lab_name(text)
    assert lab == "Sunrise Pathology Centre"


def test_regression_extended_safety_rules():
    """Regression test Part F(a): New safety patterns correctly catch prohibited statements."""
    assert len(check_text("This suggests diabetes.")) > 0
    assert len(check_text("You are prediabetic.")) > 0
    assert len(check_text("You might have anemia.")) > 0
    assert len(check_text("Consider starting metformin.")) > 0
    assert len(check_text("Your doctor will probably prescribe a statin.")) > 0
    assert len(check_text("Reduce your insulin.")) > 0
    assert len(check_text("Begin taking 10 mg of amlodipine.")) > 0
    assert len(check_text("This is perfectly fine.")) > 0
    assert len(check_text("Nothing serious here.")) > 0
    assert len(check_text("This could be fatal.")) > 0
    assert len(check_text("Seek emergency care now.")) > 0


def test_regression_non_result_lines_not_parsed():
    """Regression test Part B: Non-result text lines must not be parsed into results."""
    lines = [
        "Page 1 of 2",
        "Reg No 12345 / 2026",
        "Sample collected at 10:30 on 12/04/2026",
        "Dr. John Doe, MD - Pathologist",
        "Total Cost: $ 150.00",
        "Phone: +1 (555) 019-2834",
    ]
    results = parse_text_lines(lines)
    # They must either produce no result or result.value is None
    assert len(results) == 0 or all(r.value is None for r in results)


