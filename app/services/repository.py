"""Repository layer for local SQLite storage of processed medical reports and biomarker trends."""

from datetime import date, datetime, timezone
import hashlib
from pathlib import Path
import sqlite3
from typing import Any

from app.db.database import get_connection, init_db
from app.models.lab import ExtractionResponse, LabReport, LabResult, ResultFlag
from app.models.trend import TrendPoint


class DuplicateReportError(Exception):
    """Raised when a report with identical content hash already exists for the profile."""

    def __init__(self, existing_report_id: str, message: str | None = None):
        self.existing_report_id = existing_report_id
        super().__init__(
            message or f"Report with duplicate content already exists (ID: {existing_report_id})."
        )


def compute_report_hash(report: LabReport) -> str:
    """Compute SHA-256 hash of canonical string representation of report content."""
    date_part = report.report_date.isoformat() if report.report_date else ""
    lab_part = report.lab_name or ""
    results_parts = []
    for r in report.results:
        name = r.test_name_canonical or r.test_name_raw
        val_str = str(r.value) if r.value is not None else ""
        val_text_str = r.value_text or ""
        unit_str = r.unit or ""
        results_parts.append(f"{name}|{val_str}|{val_text_str}|{unit_str}")

    canonical_str = f"{date_part}|{lab_part}|" + ";".join(results_parts)
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


def create_profile(label: str, db_path: Path | str | None = None) -> int:
    """Create a new profile with a unique nickname label."""
    if not label or not label.strip():
        raise ValueError("Profile label cannot be empty.")

    label_clean = label.strip()
    now_iso = datetime.now(timezone.utc).isoformat()
    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                "INSERT INTO profiles (label, created_at) VALUES (?, ?);",
                (label_clean, now_iso),
            )
            return cursor.lastrowid
    except sqlite3.IntegrityError as e:
        raise ValueError(f"Profile with label '{label_clean}' already exists.") from e
    finally:
        conn.close()


def list_profiles(db_path: Path | str | None = None) -> list[dict[str, Any]]:
    """List all user profiles."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute("SELECT id, label, created_at FROM profiles ORDER BY id ASC;")
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def save_report(
    profile_id: int, response: ExtractionResponse, db_path: Path | str | None = None
) -> str:
    """Save an ExtractionResponse and all its results in a single atomic transaction.

    Raises DuplicateReportError if the same content hash already exists for the profile.
    """
    report = response.report
    content_hash = compute_report_hash(report)
    now_iso = datetime.now(timezone.utc).isoformat()
    date_str = report.report_date.isoformat() if report.report_date else None

    conn = get_connection(db_path)
    try:
        # Check duplicate hash for this profile
        dup_cursor = conn.execute(
            "SELECT id FROM reports WHERE profile_id = ? AND content_hash = ?;",
            (profile_id, content_hash),
        )
        dup_row = dup_cursor.fetchone()
        if dup_row:
            raise DuplicateReportError(dup_row["id"])

        # Execute insertion in one transaction
        with conn:
            conn.execute(
                """
                INSERT INTO reports (
                    id, profile_id, report_date, lab_name, source_filename, content_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    report.report_id,
                    profile_id,
                    date_str,
                    report.lab_name,
                    report.source_filename,
                    content_hash,
                    now_iso,
                ),
            )

            for r in report.results:
                flag_str = r.flag.value if isinstance(r.flag, ResultFlag) else str(r.flag)
                conn.execute(
                    """
                    INSERT INTO results (
                        report_id, test_name_raw, test_name_canonical, value, value_text, unit,
                        ref_low, ref_high, ref_text, flag, value_original, unit_original,
                        source_page, source_snippet, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        report.report_id,
                        r.test_name_raw,
                        r.test_name_canonical,
                        r.value,
                        r.value_text,
                        r.unit,
                        r.ref_low,
                        r.ref_high,
                        r.ref_text,
                        flag_str,
                        r.value_original,
                        r.unit_original,
                        r.source_page,
                        r.source_snippet,
                        r.confidence,
                    ),
                )

        return report.report_id
    finally:
        conn.close()


def get_report(report_id: str, db_path: Path | str | None = None) -> LabReport | None:
    """Retrieve and reconstruct a full LabReport object by report_id."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT id, report_date, lab_name, source_filename FROM reports WHERE id = ?;",
            (report_id,),
        )
        rep_row = cursor.fetchone()
        if not rep_row:
            return None

        res_cursor = conn.execute(
            """
            SELECT test_name_raw, test_name_canonical, value, value_text, unit,
                   ref_low, ref_high, ref_text, flag, value_original, unit_original,
                   source_page, source_snippet, confidence
            FROM results
            WHERE report_id = ?
            ORDER BY id ASC;
            """,
            (report_id,),
        )
        results: list[LabResult] = []
        for row in res_cursor.fetchall():
            results.append(
                LabResult(
                    test_name_raw=row["test_name_raw"],
                    test_name_canonical=row["test_name_canonical"],
                    value=row["value"],
                    value_text=row["value_text"],
                    unit=row["unit"],
                    ref_low=row["ref_low"],
                    ref_high=row["ref_high"],
                    ref_text=row["ref_text"],
                    flag=ResultFlag(row["flag"]),
                    value_original=row["value_original"],
                    unit_original=row["unit_original"],
                    source_page=row["source_page"],
                    source_snippet=row["source_snippet"],
                    confidence=row["confidence"],
                )
            )

        report_date_val = (
            date.fromisoformat(rep_row["report_date"]) if rep_row["report_date"] else None
        )
        return LabReport(
            report_id=rep_row["id"],
            report_date=report_date_val,
            lab_name=rep_row["lab_name"],
            source_filename=rep_row["source_filename"],
            results=results,
        )
    finally:
        conn.close()


def list_reports(profile_id: int, db_path: Path | str | None = None) -> list[dict[str, Any]]:
    """List report summaries for a profile, ordered newest first with null dates last."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT r.id AS report_id, r.report_date, r.lab_name, COUNT(res.id) AS result_count
            FROM reports r
            LEFT JOIN results res ON r.id = res.report_id
            WHERE r.profile_id = ?
            GROUP BY r.id
            ORDER BY (r.report_date IS NULL) ASC, r.report_date DESC, r.created_at DESC;
            """,
            (profile_id,),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_trend(
    profile_id: int, canonical_name: str, db_path: Path | str | None = None
) -> list[TrendPoint]:
    """Retrieve historical chronological trend points for a given canonical biomarker."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT r.id AS report_id, r.report_date, res.value, res.unit, res.flag,
                   res.ref_low, res.ref_high
            FROM results res
            JOIN reports r ON res.report_id = r.id
            WHERE r.profile_id = ?
              AND res.test_name_canonical = ?
              AND res.value IS NOT NULL
              AND r.report_date IS NOT NULL
            ORDER BY r.report_date ASC, r.created_at ASC, res.id ASC;
            """,
            (profile_id, canonical_name),
        )
        points: list[TrendPoint] = []
        for row in cursor.fetchall():
            points.append(
                TrendPoint(
                    report_id=row["report_id"],
                    report_date=date.fromisoformat(row["report_date"]),
                    value=row["value"],
                    unit=row["unit"],
                    flag=ResultFlag(row["flag"]),
                    ref_low=row["ref_low"],
                    ref_high=row["ref_high"],
                )
            )
        return points
    finally:
        conn.close()


def get_latest_results(
    profile_id: int, db_path: Path | str | None = None
) -> dict[str, LabResult]:
    """Return the most recent numeric result (by report_date) for each canonical test."""
    conn = get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT res.test_name_raw, res.test_name_canonical, res.value, res.value_text, res.unit,
                   res.ref_low, res.ref_high, res.ref_text, res.flag, res.value_original, res.unit_original,
                   res.source_page, res.source_snippet, res.confidence
            FROM results res
            JOIN reports r ON res.report_id = r.id
            WHERE r.profile_id = ?
              AND res.test_name_canonical IS NOT NULL
              AND res.value IS NOT NULL
              AND r.report_date IS NOT NULL
            ORDER BY r.report_date DESC, r.created_at DESC, res.id DESC;
            """,
            (profile_id,),
        )
        latest_map: dict[str, LabResult] = {}
        for row in cursor.fetchall():
            canon = row["test_name_canonical"]
            if canon not in latest_map:
                latest_map[canon] = LabResult(
                    test_name_raw=row["test_name_raw"],
                    test_name_canonical=canon,
                    value=row["value"],
                    value_text=row["value_text"],
                    unit=row["unit"],
                    ref_low=row["ref_low"],
                    ref_high=row["ref_high"],
                    ref_text=row["ref_text"],
                    flag=ResultFlag(row["flag"]),
                    value_original=row["value_original"],
                    unit_original=row["unit_original"],
                    source_page=row["source_page"],
                    source_snippet=row["source_snippet"],
                    confidence=row["confidence"],
                )
        return latest_map
    finally:
        conn.close()


def delete_report(report_id: str, db_path: Path | str | None = None) -> bool:
    """Delete a report and cascade-delete its results."""
    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.execute("DELETE FROM reports WHERE id = ?;", (report_id,))
            return cursor.rowcount > 0
    finally:
        conn.close()


def delete_profile_data(profile_id: int, db_path: Path | str | None = None) -> int:
    """Delete all reports and associated results for a profile. Returns number of reports deleted."""
    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.execute("DELETE FROM reports WHERE profile_id = ?;", (profile_id,))
            return cursor.rowcount
    finally:
        conn.close()


def run_demo() -> None:
    """CLI demo saving sample reports and rendering biomarker trend tables in a temporary database."""
    import tempfile
    from app.services.pdf_extractor import extract_report_from_pdf
    from app.services.pipeline import process_report

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        temp_db = Path(tmpdir) / "demo.db"
        init_db(temp_db)
        profiles = list_profiles(temp_db)
        profile_id = profiles[0]["id"]

        samples_dir = Path(__file__).resolve().parent.parent.parent / "samples"
        table_pdf = samples_dir / "report_table.pdf"
        lines_pdf = samples_dir / "report_lines.pdf"

        rep1 = process_report(extract_report_from_pdf(str(table_pdf)))
        rep2 = process_report(extract_report_from_pdf(str(lines_pdf)))

        save_report(profile_id, rep1, temp_db)
        save_report(profile_id, rep2, temp_db)

        # 1. HbA1c Trend Table
        hba1c_trend = get_trend(profile_id, "HbA1c", temp_db)
        print("HbA1c Trend:")
        print(f"{'Date':<12} {'Value':<10} {'Unit':<10} {'Flag'}")
        print("-" * 42)
        for pt in hba1c_trend:
            date_str = pt.report_date.isoformat()
            unit_str = pt.unit or ""
            flag_str = pt.flag.value
            print(f"{date_str:<12} {pt.value:<10} {unit_str:<10} {flag_str}")

        print()

        # 2. Total Cholesterol Trend Table
        tc_trend = get_trend(profile_id, "Total Cholesterol", temp_db)
        print("Total Cholesterol Trend:")
        print(f"{'Date':<12} {'Value':<10} {'Unit':<10} {'Flag'}")
        print("-" * 42)
        for pt in tc_trend:
            date_str = pt.report_date.isoformat()
            unit_str = pt.unit or ""
            flag_str = pt.flag.value
            print(f"{date_str:<12} {pt.value:<10} {unit_str:<10} {flag_str}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_demo()
    else:
        print("Usage: python -m app.services.repository demo")
