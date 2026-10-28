"""Integration tests for the complete extraction and processing pipeline."""

from pathlib import Path

from app.models.lab import ResultFlag
from app.services.pdf_extractor import extract_report_from_pdf
from app.services.pipeline import process_report

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def test_pipeline_on_report_table():
    """Verify processing pipeline on report_table.pdf with hard-coded expected flags."""
    pdf_path = SAMPLES_DIR / "report_table.pdf"
    raw_extraction = extract_report_from_pdf(str(pdf_path))
    processed = process_report(raw_extraction)
    results = processed.report.results

    # Hard-coded expected flags as mandated by specification
    expected_flags = {
        "HbA1c": ResultFlag.HIGH,
        "Fasting Glucose": ResultFlag.HIGH,
        "Total Cholesterol": ResultFlag.HIGH,
        "LDL Cholesterol": ResultFlag.HIGH,
        "HDL Cholesterol": ResultFlag.NORMAL,
        "Triglycerides": ResultFlag.HIGH,
        "Hemoglobin": ResultFlag.NORMAL,
        "Urine Protein": ResultFlag.NORMAL,
    }

    assert len(results) == len(expected_flags)

    for r in results:
        # Every result must receive a non-None canonical name
        assert r.test_name_canonical is not None
        assert r.test_name_canonical in expected_flags
        expected_flag = expected_flags[r.test_name_canonical]
        assert r.flag == expected_flag, f"Test {r.test_name_canonical} got flag {r.flag}, expected {expected_flag}"

    # Verify HbA1c mapping in report_table
    hba1c_res = next(r for r in results if r.test_name_raw == "HbA1c")
    assert hba1c_res.test_name_canonical == "HbA1c"


def test_pipeline_on_report_lines():
    """Verify processing pipeline on report_lines.pdf with manually verified hard-coded flags."""
    pdf_path = SAMPLES_DIR / "report_lines.pdf"
    raw_extraction = extract_report_from_pdf(str(pdf_path))
    processed = process_report(raw_extraction)
    results = processed.report.results

    # Hard-coded expected flags manually evaluated from samples/report_lines.expected.json
    expected_flags = {
        "Glycosylated Hb": ResultFlag.HIGH,       # 6.9 % vs 4.0 - 5.6
        "FBS": ResultFlag.HIGH,                   # 110.0 mg/dL vs 70 - 99
        "Cholesterol Total": ResultFlag.NORMAL,   # 195.0 mg/dL vs < 200 (strict inequality)
        "LDL-C": ResultFlag.HIGH,                 # 115.0 mg/dL vs < 100
        "HDL-C": ResultFlag.NORMAL,               # 48.0 mg/dL vs > 40
        "TG": ResultFlag.NORMAL,                  # 140.0 mg/dL vs < 150
        "Haemoglobin": ResultFlag.NORMAL,         # 14.2 g/dL vs 13.0 - 17.0
    }

    assert len(results) == len(expected_flags)

    for r in results:
        # Every result must receive a non-None canonical name
        assert r.test_name_canonical is not None
        assert r.test_name_raw in expected_flags
        expected_flag = expected_flags[r.test_name_raw]
        assert r.flag == expected_flag, f"Test {r.test_name_raw} got flag {r.flag}, expected {expected_flag}"

    # Verify Glycosylated Hb in report_lines maps to HbA1c
    glyc_res = next(r for r in results if r.test_name_raw == "Glycosylated Hb")
    assert glyc_res.test_name_canonical == "HbA1c"
