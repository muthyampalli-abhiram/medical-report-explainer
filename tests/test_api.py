"""Comprehensive integration test suite for FastAPI routes."""

import io
import json
from pathlib import Path
import shutil
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_db_path, get_llm
from app.api.uploads import UPLOAD_TEMP_DIR
from app.config import settings
from app.db.database import get_connection, init_db
from app.main import app
from app.services.explainer import FIXED_DISCLAIMER
from app.services.llm_client import FakeLLMClient, UnavailableLLMClient
from app.services.repository import list_reports

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
REPORT_TABLE_PDF = SAMPLES_DIR / "report_table.pdf"
REPORT_LINES_PDF = SAMPLES_DIR / "report_lines.pdf"
REPORT_SCANNED_PDF = SAMPLES_DIR / "report_scanned.pdf"
PRESCRIPTION_TEXT_PDF = SAMPLES_DIR / "prescription_text.pdf"
PHOTO_CLEAN = SAMPLES_DIR / "photo_clean.png"
PHOTO_POOR = SAMPLES_DIR / "photo_poor.jpg"
EXPECTED_JSON = SAMPLES_DIR / "report_table.expected.json"


def is_tesseract_available() -> bool:
    """Check if Tesseract binary is available."""
    if settings.TESSERACT_CMD:
        return Path(settings.TESSERACT_CMD).exists()
    return shutil.which("tesseract") is not None


@pytest.fixture
def client(tmp_path: Path):
    """TestClient fixture with isolated SQLite database and mock LLM client."""
    db_file = tmp_path / "test_api_health.db"
    init_db(db_file)

    fake_llm = FakeLLMClient(
        {
            "tests": [
                {"name": "HbA1c", "what_it_measures": "Measures average blood sugar."},
                {"name": "Fasting Glucose", "what_it_measures": "Measures blood glucose after fasting."},
                {"name": "Total Cholesterol", "what_it_measures": "Measures overall cholesterol levels."},
                {"name": "LDL Cholesterol", "what_it_measures": "Measures LDL cholesterol particles."},
                {"name": "HDL Cholesterol", "what_it_measures": "Measures HDL protective cholesterol."},
                {"name": "Triglycerides", "what_it_measures": "Measures fat lipids in the bloodstream."},
                {"name": "Hemoglobin", "what_it_measures": "Measures oxygen carrying protein in red cells."},
                {"name": "Urine Protein", "what_it_measures": "Measures kidney filtration of proteins."},
            ]
        }
    )

    app.dependency_overrides[get_db_path] = lambda: db_file
    app.dependency_overrides[get_llm] = lambda: fake_llm

    with TestClient(app, raise_server_exceptions=False) as test_cli:
        yield test_cli, db_file, fake_llm

    app.dependency_overrides.clear()


def test_upload_report_table_success_and_duplicate(client) -> None:
    """Upload report_table.pdf returns 200 with exact counts, scrubbed metadata, and 409 on duplicate."""
    cli, db_file, _ = client

    with open(REPORT_TABLE_PDF, "rb") as f:
        resp = cli.post(
            "/api/reports/upload",
            files={"file": ("report_table.pdf", f, "application/pdf")},
            data={"profile_id": 1},
        )

    assert resp.status_code == 200
    data = resp.json()
    first_id = data["report_id"]

    assert data["result_count"] == 8
    assert data["abnormal_count"] == 5
    assert data["critical_count"] == 0
    assert data["disclaimer"] == FIXED_DISCLAIMER

    hba1c = next(r for r in data["results"] if r["test_name_canonical"] == "HbA1c")
    assert hba1c["flag"] == "HIGH"

    # Verify no source_filename value and no PII anywhere
    assert data.get("source_filename") is None
    resp_text = resp.text
    assert "TEST PATIENT" not in resp_text
    assert "000000" not in resp_text
    assert "report_table" not in resp_text

    # Upload same file again -> 409 with existing_report_id
    with open(REPORT_TABLE_PDF, "rb") as f:
        dup_resp = cli.post(
            "/api/reports/upload",
            files={"file": ("report_table.pdf", f, "application/pdf")},
            data={"profile_id": 1},
        )

    assert dup_resp.status_code == 409
    dup_data = dup_resp.json()
    assert dup_data["existing_report_id"] == first_id
    assert dup_data["detail"] == "This report has already been uploaded."

    # Temp directory clean
    assert list(UPLOAD_TEMP_DIR.glob("*")) == []


def test_trends_queries_and_aliases(client) -> None:
    """Upload report_table and report_lines, then test trend queries, aliases, and 404s."""
    cli, db_file, _ = client

    # Upload first report (2026-01-15)
    with open(REPORT_TABLE_PDF, "rb") as f:
        cli.post(
            "/api/reports/upload",
            files={"file": ("report_table.pdf", f, "application/pdf")},
            data={"profile_id": 1},
        )

    # Upload second report (2026-04-20)
    with open(REPORT_LINES_PDF, "rb") as f:
        cli.post(
            "/api/reports/upload",
            files={"file": ("report_lines.pdf", f, "application/pdf")},
            data={"profile_id": 1},
        )

    # Test exact hard-coded trend points
    resp = cli.get("/api/profiles/1/trends/HbA1c")
    assert resp.status_code == 200
    points = [(pt["report_date"], pt["value"]) for pt in resp.json()]
    assert points == [("2026-01-15", 7.4), ("2026-04-20", 6.9)]

    # Test alias lowercase
    resp_lower = cli.get("/api/profiles/1/trends/hba1c")
    assert resp_lower.status_code == 200
    assert [(pt["report_date"], pt["value"]) for pt in resp_lower.json()] == [
        ("2026-01-15", 7.4),
        ("2026-04-20", 6.9),
    ]

    # Test alias original raw name
    resp_alias = cli.get("/api/profiles/1/trends/Glycosylated%20Hb")
    assert resp_alias.status_code == 200
    assert [(pt["report_date"], pt["value"]) for pt in resp_alias.json()] == [
        ("2026-01-15", 7.4),
        ("2026-04-20", 6.9),
    ]

    # Unknown test name returns 404
    resp_unknown = cli.get("/api/profiles/1/trends/NotATest")
    assert resp_unknown.status_code == 404
    assert resp_unknown.json()["detail"] == "Unknown test name."

    # Latest results check
    resp_latest = cli.get("/api/profiles/1/latest")
    assert resp_latest.status_code == 200
    latest_data = resp_latest.json()
    assert "HbA1c" in latest_data
    assert latest_data["HbA1c"]["value"] == 6.9


def test_questions_and_explanation_endpoints(client) -> None:
    """Test questions with trends and explanation generation with fallback."""
    cli, db_file, _ = client

    # Upload first report
    with open(REPORT_TABLE_PDF, "rb") as f:
        rep1_resp = cli.post(
            "/api/reports/upload",
            files={"file": ("report_table.pdf", f, "application/pdf")},
            data={"profile_id": 1},
        )
    first_id = rep1_resp.json()["report_id"]

    # Upload second report
    with open(REPORT_LINES_PDF, "rb") as f:
        cli.post(
            "/api/reports/upload",
            files={"file": ("report_lines.pdf", f, "application/pdf")},
            data={"profile_id": 1},
        )

    # Questions endpoint
    q_resp = cli.get(f"/api/reports/{first_id}/questions")
    assert q_resp.status_code == 200
    q_data = q_resp.json()
    questions = q_data["questions"]
    assert any("7.4" in q["text"] and "HbA1c" in q["text"] for q in questions)
    assert any("changed from" in q["text"] for q in questions)

    # Explanation endpoint
    exp_resp = cli.get(f"/api/reports/{first_id}/explanation")
    assert exp_resp.status_code == 200
    exp_data = exp_resp.json()
    assert len(exp_data["explanations"]) == 8
    assert exp_data["disclaimer"] == FIXED_DISCLAIMER

    # Explanation endpoint with UnavailableLLMClient returns 200 with used_fallback True
    app.dependency_overrides[get_llm] = lambda: UnavailableLLMClient()
    fb_resp = cli.get(f"/api/reports/{first_id}/explanation")
    assert fb_resp.status_code == 200
    fb_data = fb_resp.json()
    assert len(fb_data["explanations"]) == 8
    assert all(item["used_fallback"] is True for item in fb_data["explanations"])


def test_explanation_endpoint_i18n(client) -> None:
    """GET /api/reports/{id}/explanation with lang query parameter supports hi, te and rejects invalid languages with 422."""
    cli, db_file, _ = client

    with open(REPORT_TABLE_PDF, "rb") as f:
        upload_resp = cli.post(
            "/api/reports/upload",
            files={"file": ("report_table.pdf", f, "application/pdf")},
            data={"profile_id": 1},
        )
    assert upload_resp.status_code == 200
    report_id = upload_resp.json()["report_id"]

    # 1. No lang parameter: defaults to en, language="en", translation_reviewed=True, draft_notice=None, 8 explanations
    default_resp = cli.get(f"/api/reports/{report_id}/explanation")
    assert default_resp.status_code == 200
    default_data = default_resp.json()
    assert default_data["language"] == "en"
    assert default_data["translation_reviewed"] is True
    assert default_data["draft_notice"] is None
    assert len(default_data["explanations"]) == 8
    assert default_data["disclaimer"] == FIXED_DISCLAIMER

    # Explicit ?lang=en is identical
    en_resp = cli.get(f"/api/reports/{report_id}/explanation?lang=en")
    assert en_resp.status_code == 200
    assert en_resp.json() == default_data

    # 2. ?lang=hi: returns 200 with language="hi", translation_reviewed=False, non-empty draft_notice, 8 explanations
    hi_resp = cli.get(f"/api/reports/{report_id}/explanation?lang=hi")
    assert hi_resp.status_code == 200
    hi_data = hi_resp.json()
    assert hi_data["language"] == "hi"
    assert hi_data["translation_reviewed"] is False
    assert isinstance(hi_data["draft_notice"], str) and len(hi_data["draft_notice"].strip()) > 0
    assert len(hi_data["explanations"]) == 8

    # 3. ?lang=te: returns 200 with language="te", translation_reviewed=False, non-empty draft_notice, 8 explanations
    te_resp = cli.get(f"/api/reports/{report_id}/explanation?lang=te")
    assert te_resp.status_code == 200
    te_data = te_resp.json()
    assert te_data["language"] == "te"
    assert te_data["translation_reviewed"] is False
    assert isinstance(te_data["draft_notice"], str) and len(te_data["draft_notice"].strip()) > 0
    assert len(te_data["explanations"]) == 8

    # 4. ?lang=xx: returns 422 with exact message "Unsupported language. Supported: en, hi, te."
    invalid_resp = cli.get(f"/api/reports/{report_id}/explanation?lang=xx")
    assert invalid_resp.status_code == 422
    assert invalid_resp.json()["detail"] == "Unsupported language. Supported: en, hi, te."


def test_scanned_pdf_upload_returns_422_and_stores_nothing(client) -> None:
    """Uploading scanned PDF succeeds with 'ocr' if Tesseract available; blank image returns 422 and stores nothing."""
    cli, db_file, _ = client

    # 1. Scanned PDF upload: succeeds with extraction_method "ocr" when Tesseract is available
    if is_tesseract_available():
        with open(REPORT_SCANNED_PDF, "rb") as f:
            resp = cli.post(
                "/api/reports/upload",
                files={"file": ("report_scanned.pdf", f, "application/pdf")},
                data={"profile_id": 1},
            )
        assert resp.status_code == 200
        assert resp.json()["extraction_method"] == "ocr"

    # 2. To keep a 422 test, upload a blank white PNG and assert 422 with the new message and that nothing was stored
    blank_img = Image.new("RGB", (200, 200), color="white")
    buf = io.BytesIO()
    blank_img.save(buf, format="PNG")

    cli.post("/api/profiles", json={"label": "Blank Test Profile"})
    resp_blank = cli.post(
        "/api/reports/upload",
        files={"file": ("blank.png", buf.getvalue(), "image/png")},
        data={"profile_id": 2},
    )
    assert resp_blank.status_code == 422
    assert (
        resp_blank.json()["detail"]
        == "We could not read any results from this file. Try a clearer, flat, well-lit photo of the full report."
    )
    assert list_reports(2, db_file) == []
    assert list(UPLOAD_TEMP_DIR.glob("*")) == []


def test_upload_validations_and_temp_file_cleanup(client, monkeypatch) -> None:
    """Validate magic bytes, empty file, size limit, image extensions, and malicious paths."""
    cli, db_file, _ = client

    # 1. Plain text with .pdf extension -> 415
    resp_text = cli.post(
        "/api/reports/upload",
        files={"file": ("fake.pdf", b"This is not a PDF file.", "application/pdf")},
    )
    assert resp_text.status_code == 415
    assert resp_text.json()["detail"] == "Supported formats: PDF, JPG, PNG."

    # 2. Empty file -> 400
    resp_empty = cli.post(
        "/api/reports/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert resp_empty.status_code == 400
    assert resp_empty.json()["detail"] == "The uploaded file is empty."

    # 3. File size exceeds MAX_UPLOAD_MB -> 413
    monkeypatch.setattr(settings, "MAX_UPLOAD_MB", 1)
    large_payload = b"%PDF-" + b"0" * int(1.5 * 1024 * 1024)
    resp_large = cli.post(
        "/api/reports/upload",
        files={"file": ("large.pdf", large_payload, "application/pdf")},
    )
    assert resp_large.status_code == 413
    assert resp_large.json()["detail"] == "File is too large."

    # 4. Photo/image report with corrupt header only -> 415
    resp_img = cli.post(
        "/api/reports/upload",
        files={"file": ("photo.png", b"\x89PNG\r\n\x1a\n", "image/png")},
    )
    assert resp_img.status_code == 415
    assert resp_img.json()["detail"] == "The image could not be read."

    # 5. Malicious path traversal filename -> does not escape, stores source_filename None
    with open(REPORT_TABLE_PDF, "rb") as f:
        valid_pdf = f.read()

    resp_traversal = cli.post(
        "/api/reports/upload",
        files={"file": ("../../evil.pdf", valid_pdf, "application/pdf")},
        data={"profile_id": 1},
    )
    assert resp_traversal.status_code == 200
    assert resp_traversal.json().get("source_filename") is None
    assert not Path("../../evil.pdf").exists()

    # 6. Non-existent profile -> 404
    resp_noprofile = cli.post(
        "/api/reports/upload",
        files={"file": ("report.pdf", valid_pdf, "application/pdf")},
        data={"profile_id": 9999},
    )
    assert resp_noprofile.status_code == 404

    # Temp directory clean after all success/error cases
    assert list(UPLOAD_TEMP_DIR.glob("*")) == []


def test_profiles_crud(client) -> None:
    """Test profiles creation, validation, duplicates, and data deletion."""
    cli, db_file, _ = client

    # List default profile
    resp_list = cli.get("/api/profiles")
    assert resp_list.status_code == 200
    assert any(p["label"] == "Me" for p in resp_list.json())

    # Create new profile
    resp_create = cli.post("/api/profiles", json={"label": "Mom"})
    assert resp_create.status_code == 201
    mom_id = resp_create.json()["id"]

    # Duplicate profile -> 409
    resp_dup = cli.post("/api/profiles", json={"label": "Mom"})
    assert resp_dup.status_code == 409

    # Empty label -> 422
    resp_empty = cli.post("/api/profiles", json={"label": ""})
    assert resp_empty.status_code == 422

    # 41-char label -> 422
    resp_long = cli.post("/api/profiles", json={"label": "A" * 41})
    assert resp_long.status_code == 422

    # Delete profile data
    resp_del_data = cli.delete(f"/api/profiles/{mom_id}/data")
    assert resp_del_data.status_code == 200
    assert resp_del_data.json() == {"deleted_reports": 0}


def test_reports_crud(client) -> None:
    """Test single report GET and DELETE endpoints."""
    cli, db_file, _ = client

    with open(REPORT_TABLE_PDF, "rb") as f:
        up_resp = cli.post(
            "/api/reports/upload",
            files={"file": ("report.pdf", f, "application/pdf")},
            data={"profile_id": 1},
        )
    rep_id = up_resp.json()["report_id"]

    # GET report
    get_resp = cli.get(f"/api/reports/{rep_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["report_id"] == rep_id

    # DELETE report
    del_resp = cli.delete(f"/api/reports/{rep_id}")
    assert del_resp.status_code == 200
    assert del_resp.json() == {"deleted": True, "report_id": rep_id}

    # Subsequent GET returns 404
    assert cli.get(f"/api/reports/{rep_id}").status_code == 404


def test_prescriptions_parse_endpoint(client) -> None:
    """POST /api/prescriptions/parse returns medications and saves nothing to database."""
    cli, db_file, _ = client

    conn = get_connection(db_file)
    counts_before = {
        "profiles": conn.execute("SELECT COUNT(*) FROM profiles;").fetchone()[0],
        "reports": conn.execute("SELECT COUNT(*) FROM reports;").fetchone()[0],
        "results": conn.execute("SELECT COUNT(*) FROM results;").fetchone()[0],
    }
    conn.close()

    with open(PRESCRIPTION_TEXT_PDF, "rb") as f:
        resp = cli.post(
            "/api/prescriptions/parse",
            files={"file": ("prescription_text.pdf", f, "application/pdf")},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["medications"]) == 4
    assert "This is a summary of what is written on your prescription." in data["disclaimer"]

    # Verify zero database changes
    conn = get_connection(db_file)
    counts_after = {
        "profiles": conn.execute("SELECT COUNT(*) FROM profiles;").fetchone()[0],
        "reports": conn.execute("SELECT COUNT(*) FROM reports;").fetchone()[0],
        "results": conn.execute("SELECT COUNT(*) FROM results;").fetchone()[0],
    }
    conn.close()
    assert counts_after == counts_before

    assert list(UPLOAD_TEMP_DIR.glob("*")) == []


def test_unexpected_exception_returns_500_without_leaks(client, monkeypatch) -> None:
    """Unexpected exception in route returns generic 500 without stack trace leaks."""
    cli, db_file, _ = client

    def _crash(*args, **kwargs):
        raise RuntimeError("SecretDatabaseCrash: /internal/db/path.sqlite failed SQL query")

    monkeypatch.setattr("app.api.routes_profiles.list_profiles", _crash)

    resp = cli.get("/api/profiles")
    assert resp.status_code == 500
    assert resp.json() == {"detail": "Something went wrong while processing this request."}
    assert "traceback" not in resp.text.lower()
    assert "secretdatabasecrash" not in resp.text.lower()
    assert "internal/db/path" not in resp.text.lower()


def test_health_check_still_works(client) -> None:
    """Health check endpoint returns ok status."""
    cli, _, _ = client
    resp = cli.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "app": "medical-report-explainer"}


def test_upload_photo_clean_api(client) -> None:
    """Upload samples/photo_clean.png returns 200 with extraction_method 'ocr', 8 results, and low_confidence_count 0."""
    if not is_tesseract_available():
        pytest.skip("Tesseract OCR executable not found")

    cli, db_file, _ = client
    with open(PHOTO_CLEAN, "rb") as f:
        resp = cli.post(
            "/api/reports/upload",
            files={"file": ("photo_clean.png", f, "image/png")},
            data={"profile_id": 1},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["extraction_method"] == "ocr"
    assert data["result_count"] == 8
    assert data["low_confidence_count"] == 0
    assert list(UPLOAD_TEMP_DIR.glob("*")) == []


def test_upload_photo_poor_api_safety_property(client) -> None:
    """Upload photo_poor.jpg returns 422 with nothing stored or 200 where every result has a canonical name, with no junk strings."""
    if not is_tesseract_available():
        pytest.skip("Tesseract OCR executable not found")

    cli, db_file, _ = client
    with open(EXPECTED_JSON, encoding="utf-8") as f:
        expected_data = json.load(f)

    conn = get_connection(db_file)
    reports_before = conn.execute("SELECT COUNT(*) FROM reports;").fetchone()[0]
    conn.close()

    with open(PHOTO_POOR, "rb") as f:
        resp = cli.post(
            "/api/reports/upload",
            files={"file": ("photo_poor.jpg", f, "image/jpeg")},
            data={"profile_id": 1},
        )
    assert resp.status_code in (200, 422)

    # In both cases the response must not contain the strings "im mg/dl", "214 mg/d", or "13.1 e/a"
    assert "im mg/dl" not in resp.text
    assert "214 mg/d" not in resp.text
    assert "13.1 e/a" not in resp.text

    if resp.status_code == 422:
        conn = get_connection(db_file)
        reports_after = conn.execute("SELECT COUNT(*) FROM reports;").fetchone()[0]
        conn.close()
        assert reports_after == reports_before
    elif resp.status_code == 200:
        data = resp.json()
        for act in data["results"]:
            assert act.get("test_name_canonical") is not None, (
                f"Result without canonical name found: {act.get('test_name_raw')}"
            )
            is_exact_match = False
            for exp in expected_data["results"]:
                if act.get("test_name_raw") == exp["test_name_raw"]:
                    val = act.get("value")
                    val_ok = (
                        (val is not None and abs(val - exp["value"]) < 1e-4)
                        if exp.get("value") is not None
                        else (act.get("value_text") == exp.get("value_text"))
                    )
                    unit_ok = act.get("unit") == exp["unit"]
                    low_ok = act.get("ref_low") == exp["ref_low"]
                    high_ok = act.get("ref_high") == exp["ref_high"]
                    if val_ok and unit_ok and low_ok and high_ok:
                        is_exact_match = True
                        break

            if not is_exact_match:
                conf = act.get("confidence", 0.0)
                flag = act.get("flag")
                is_safe = (conf < settings.OCR_VERIFY_BELOW) or (flag == "UNKNOWN")
                assert is_safe, (
                    f"Safety violation on photo_poor.jpg for {act.get('test_name_raw')}: "
                    f"value={act.get('value')}, conf={conf}, flag={flag}"
                )
    assert list(UPLOAD_TEMP_DIR.glob("*")) == []


def test_upload_heic_returns_415(client) -> None:
    """A HEIC-looking file returns 415 with the specific HEIC message."""
    cli, _, _ = client
    heic_bytes = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00fakeheicdata"
    resp = cli.post(
        "/api/reports/upload",
        files={"file": ("photo.heic", heic_bytes, "image/heic")},
        data={"profile_id": 1},
    )
    assert resp.status_code == 415
    assert (
        resp.json()["detail"]
        == "HEIC photos are not supported. Please save the photo as JPG or PNG."
    )
    assert list(UPLOAD_TEMP_DIR.glob("*")) == []


def test_upload_gif_returns_415(client) -> None:
    """A GIF file returns 415 'Supported formats: PDF, JPG, PNG.'."""
    cli, _, _ = client
    gif_bytes = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
    resp = cli.post(
        "/api/reports/upload",
        files={"file": ("test.gif", gif_bytes, "image/gif")},
        data={"profile_id": 1},
    )
    assert resp.status_code == 415
    assert resp.json()["detail"] == "Supported formats: PDF, JPG, PNG."
    assert list(UPLOAD_TEMP_DIR.glob("*")) == []


def test_upload_oversized_pixel_image_returns_413(client, monkeypatch) -> None:
    """An oversized-pixel image returns 413 'Image is too large.'."""
    cli, _, _ = client
    monkeypatch.setattr(settings, "OCR_MAX_IMAGE_PIXELS", 500)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 500)
    img = Image.new("RGB", (50, 50), color="white")  # 2500 pixels > 500
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    resp = cli.post(
        "/api/reports/upload",
        files={"file": ("oversized.png", buf.getvalue(), "image/png")},
        data={"profile_id": 1},
    )
    assert resp.status_code == 413
    assert resp.json()["detail"] == "Image is too large."
    assert list(UPLOAD_TEMP_DIR.glob("*")) == []


def test_prescriptions_parse_with_png_returns_415(client) -> None:
    """POST /api/prescriptions/parse only accepts PDF; uploading a PNG returns 415."""
    cli, _, _ = client
    img = Image.new("RGB", (20, 20), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    resp = cli.post(
        "/api/prescriptions/parse",
        files={"file": ("prescription.png", buf.getvalue(), "image/png")},
    )
    assert resp.status_code == 415
    assert resp.json()["detail"] == "Only PDF files are supported right now."
    assert list(UPLOAD_TEMP_DIR.glob("*")) == []


def test_stricter_filename_check(client) -> None:
    """Upload report_table.pdf with client filename 'John_Doe_Report.pdf', verify source_filename is NULL via GET and SQL, and 'John_Doe' appears nowhere."""
    cli, db_file, _ = client

    with open(REPORT_TABLE_PDF, "rb") as f:
        resp = cli.post(
            "/api/reports/upload",
            files={"file": ("John_Doe_Report.pdf", f, "application/pdf")},
            data={"profile_id": 1},
        )
    assert resp.status_code == 200
    upload_data = resp.json()
    report_id = upload_data["report_id"]
    assert upload_data.get("source_filename") is None
    assert "John_Doe" not in resp.text

    # Read stored report through GET /api/reports/{id}
    get_resp = cli.get(f"/api/reports/{report_id}")
    assert get_resp.status_code == 200
    get_data = get_resp.json()
    assert get_data.get("source_filename") is None
    assert "John_Doe" not in get_resp.text

    # Direct SQL query on the test database: source_filename column must be NULL
    conn = get_connection(db_file)
    row = conn.execute("SELECT source_filename FROM reports WHERE id = ?;", (report_id,)).fetchone()
    assert row is not None
    assert row["source_filename"] is None

    # Full dump of the database (all tables, all columns)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
    dump_text = ""
    for table in tables:
        cursor = conn.execute(f"SELECT * FROM {table};")
        for r in cursor.fetchall():
            dump_text += str(dict(r)) + "\n"
    conn.close()

    assert "John_Doe" not in dump_text
    assert list(UPLOAD_TEMP_DIR.glob("*")) == []
