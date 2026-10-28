"""Tests for local SQLite repository and trend query layer."""

from pathlib import Path
import sqlite3
import pytest

from app.db.database import get_connection, init_db
from app.models.lab import ExtractionResponse, LabReport, LabResult, ResultFlag
from app.services.pdf_extractor import extract_report_from_pdf
from app.services.pipeline import process_report
from app.services.repository import (
    DuplicateReportError,
    create_profile,
    delete_profile_data,
    delete_report,
    get_latest_results,
    get_report,
    get_trend,
    list_profiles,
    list_reports,
    save_report,
)

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def test_init_db_creates_tables_and_default_me_profile(tmp_path):
    """Verify init_db creates all required tables and default 'Me' profile."""
    db_file = tmp_path / "test.db"
    init_db(db_file)

    conn = get_connection(db_file)
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        )
        tables = {row["name"] for row in cursor.fetchall()}
        assert "profiles" in tables
        assert "reports" in tables
        assert "results" in tables

        # Default profile 'Me' must exist
        profiles = list_profiles(db_file)
        assert len(profiles) == 1
        assert profiles[0]["label"] == "Me"

        # Duplicate profile label raises error
        with pytest.raises(ValueError, match="already exists"):
            create_profile("Me", db_file)

        # Creating new profile succeeds
        new_id = create_profile("Mom", db_file)
        assert isinstance(new_id, int)
        assert len(list_profiles(db_file)) == 2
    finally:
        conn.close()


def test_save_and_get_report_equality(tmp_path):
    """Verify saving processed reports and reconstructing equal LabReport objects."""
    db_file = tmp_path / "test.db"
    init_db(db_file)

    rep1 = process_report(extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf")))
    rep2 = process_report(extract_report_from_pdf(str(SAMPLES_DIR / "report_lines.pdf")))

    id1 = save_report(1, rep1, db_file)
    id2 = save_report(1, rep2, db_file)

    assert id1 == rep1.report.report_id
    assert id2 == rep2.report.report_id

    reconstructed1 = get_report(id1, db_file)
    reconstructed2 = get_report(id2, db_file)

    assert reconstructed1 == rep1.report
    assert reconstructed2 == rep2.report

    # Non-existent report returns None
    assert get_report("non-existent-uuid", db_file) is None


def test_save_duplicate_report_raises_error(tmp_path):
    """Verify saving the same report twice raises DuplicateReportError with existing id."""
    db_file = tmp_path / "test.db"
    init_db(db_file)

    rep = process_report(extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf")))
    original_id = save_report(1, rep, db_file)

    # Re-saving same content under same profile must raise DuplicateReportError
    with pytest.raises(DuplicateReportError) as exc_info:
        save_report(1, rep, db_file)

    assert exc_info.value.existing_report_id == original_id


def test_transaction_rollback_leaves_no_partial_data(tmp_path):
    """Verify that a failure halfway through save_report rolls back completely."""
    db_file = tmp_path / "test.db"
    init_db(db_file)

    rep = process_report(extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf")))
    target_report_id = rep.report.report_id

    # Introduce a NOT NULL constraint violation on the second result
    bad_result = rep.report.results[1]
    object.__setattr__(bad_result, "confidence", None)

    with pytest.raises(sqlite3.IntegrityError):
        save_report(1, rep, db_file)

    # Verify atomic rollback: no report row and no result rows
    conn = get_connection(db_file)
    try:
        rep_count = conn.execute(
            "SELECT COUNT(*) AS cnt FROM reports WHERE id = ?;", (target_report_id,)
        ).fetchone()["cnt"]
        res_count = conn.execute(
            "SELECT COUNT(*) AS cnt FROM results WHERE report_id = ?;", (target_report_id,)
        ).fetchone()["cnt"]
        assert rep_count == 0
        assert res_count == 0
    finally:
        conn.close()


def test_get_trend_hardcoded_values(tmp_path):
    """Verify get_trend returns exact hard-coded historical measurements in ascending date order."""
    db_file = tmp_path / "test.db"
    init_db(db_file)

    rep1 = process_report(extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf")))
    rep2 = process_report(extract_report_from_pdf(str(SAMPLES_DIR / "report_lines.pdf")))

    save_report(1, rep1, db_file)
    save_report(1, rep2, db_file)

    # Hard-coded HbA1c trend
    hba1c_pts = get_trend(1, "HbA1c", db_file)
    hba1c_values = [(pt.report_date.isoformat(), pt.value) for pt in hba1c_pts]
    assert hba1c_values == [("2026-01-15", 7.4), ("2026-04-20", 6.9)]

    # Hard-coded Total Cholesterol trend
    tc_pts = get_trend(1, "Total Cholesterol", db_file)
    tc_values = [(pt.report_date.isoformat(), pt.value) for pt in tc_pts]
    assert tc_values == [("2026-01-15", 214.0), ("2026-04-20", 195.0)]


def test_get_trend_ignores_text_only_results(tmp_path):
    """Verify get_trend ignores non-numeric results like Urine Protein."""
    db_file = tmp_path / "test.db"
    init_db(db_file)

    rep1 = process_report(extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf")))
    save_report(1, rep1, db_file)

    urine_pts = get_trend(1, "Urine Protein", db_file)
    assert urine_pts == []


def test_report_with_null_date_saved_and_omitted_from_trends(tmp_path):
    """Verify null-date report is saved, appears last in list_reports, and ignored by get_trend."""
    db_file = tmp_path / "test.db"
    init_db(db_file)

    rep_dated = process_report(extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf")))
    save_report(1, rep_dated, db_file)

    # Create report with null date
    undated_report = LabReport(
        report_date=None,
        lab_name="Undated Lab",
        results=[
            LabResult(
                test_name_raw="HbA1c",
                test_name_canonical="HbA1c",
                value=8.0,
                unit="%",
                flag=ResultFlag.HIGH,
            )
        ],
    )
    save_report(1, ExtractionResponse(report=undated_report), db_file)

    # list_reports must have null dates last
    summaries = list_reports(1, db_file)
    assert len(summaries) == 2
    assert summaries[0]["report_date"] == "2026-01-15"
    assert summaries[1]["report_date"] is None

    # Trend must only contain dated report
    hba1c_trend = get_trend(1, "HbA1c", db_file)
    assert len(hba1c_trend) == 1
    assert hba1c_trend[0].value == 7.4


def test_get_latest_results(tmp_path):
    """Verify get_latest_results returns the most recent (2026-04-20) values for common tests."""
    db_file = tmp_path / "test.db"
    init_db(db_file)

    rep1 = process_report(extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf")))
    rep2 = process_report(extract_report_from_pdf(str(SAMPLES_DIR / "report_lines.pdf")))

    save_report(1, rep1, db_file)
    save_report(1, rep2, db_file)

    latest = get_latest_results(1, db_file)

    # All tests in common must reflect 2026-04-20 values
    assert latest["HbA1c"].value == 6.9
    assert latest["Fasting Glucose"].value == 110.0
    assert latest["Total Cholesterol"].value == 195.0
    assert latest["LDL Cholesterol"].value == 115.0
    assert latest["HDL Cholesterol"].value == 48.0
    assert latest["Triglycerides"].value == 140.0
    assert latest["Hemoglobin"].value == 14.2


def test_profile_isolation(tmp_path):
    """Verify complete isolation between separate profiles."""
    db_file = tmp_path / "test.db"
    init_db(db_file)

    mom_id = create_profile("Mom", db_file)
    me_id = 1

    rep1 = process_report(extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf")))
    save_report(me_id, rep1, db_file)

    # Mom's profile should have no reports, no trends, no latest results
    assert list_reports(mom_id, db_file) == []
    assert get_trend(mom_id, "HbA1c", db_file) == []
    assert get_latest_results(mom_id, db_file) == {}


def test_delete_report_and_cascade(tmp_path):
    """Verify deleting reports cascade-deletes results rows and returns counts."""
    db_file = tmp_path / "test.db"
    init_db(db_file)

    rep1 = process_report(extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf")))
    rep2 = process_report(extract_report_from_pdf(str(SAMPLES_DIR / "report_lines.pdf")))

    id1 = save_report(1, rep1, db_file)
    id2 = save_report(1, rep2, db_file)

    # Delete first report
    deleted = delete_report(id1, db_file)
    assert deleted is True

    conn = get_connection(db_file)
    try:
        # Check direct SQL count for report and cascaded results
        r_count = conn.execute("SELECT COUNT(*) AS cnt FROM reports WHERE id = ?;", (id1,)).fetchone()["cnt"]
        res_count = conn.execute("SELECT COUNT(*) AS cnt FROM results WHERE report_id = ?;", (id1,)).fetchone()["cnt"]
        assert r_count == 0
        assert res_count == 0

        # Delete remaining profile data
        deleted_count = delete_profile_data(1, db_file)
        assert deleted_count == 1

        total_reports = conn.execute("SELECT COUNT(*) AS cnt FROM reports WHERE profile_id = 1;").fetchone()["cnt"]
        total_results = conn.execute("SELECT COUNT(*) AS cnt FROM results;").fetchone()["cnt"]
        assert total_reports == 0
        assert total_results == 0
    finally:
        conn.close()


def test_sql_injection_resistance(tmp_path):
    """Verify parameterized queries defend against SQL injection strings."""
    db_file = tmp_path / "test.db"
    init_db(db_file)

    rep1 = process_report(extract_report_from_pdf(str(SAMPLES_DIR / "report_table.pdf")))
    save_report(1, rep1, db_file)

    injection_string = "HbA1c'; DROP TABLE results; --"
    pts = get_trend(1, injection_string, db_file)
    assert pts == []

    # Verify results table still exists and data is intact
    conn = get_connection(db_file)
    try:
        cnt = conn.execute("SELECT COUNT(*) AS cnt FROM results;").fetchone()["cnt"]
        assert cnt > 0
    finally:
        conn.close()
