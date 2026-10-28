"""Test suite for web interface, static assets, security headers, and safety properties."""

from pathlib import Path
import re
import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_db_path, get_llm
from app.db.database import init_db
from app.main import app
from app.services.llm_client import FakeLLMClient

STATIC_DIR = Path(__file__).resolve().parent.parent / "app" / "static"
INDEX_HTML = STATIC_DIR / "index.html"
STYLES_CSS = STATIC_DIR / "styles.css"
APP_JS = STATIC_DIR / "app.js"

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
REPORT_TABLE_PDF = SAMPLES_DIR / "report_table.pdf"

EXACT_DISCLAIMER = (
    "This tool explains lab reports in general terms and does not provide a diagnosis or "
    "medical advice. Always discuss your results with your doctor."
)

EXACT_PRIVACY_NOTE = (
    "Your reports are processed and stored on this computer. Uploaded files are deleted "
    "right after reading. If an AI model is enabled, only test names (never your values "
    "or personal details) are sent to it to write general descriptions."
)

REQUIRED_STATUS_LABELS = [
    "▲ Above range",
    "▼ Below range",
    "● Within range",
    "? Not compared",
    "‼ Critical high",
    "‼ Critical low",
]

REQUIRED_VERIFY_TEXT = "Please verify against your original report"

EXPECTED_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; "
    "frame-ancestors 'none'; form-action 'self'"
)


@pytest.fixture
def client(tmp_path: Path):
    """Isolated TestClient fixture with temporary SQLite database and mock LLM."""
    db_file = tmp_path / "test_ui_health.db"
    init_db(db_file)

    fake_llm = FakeLLMClient(
        {
            "tests": [
                {"name": "HbA1c", "what_it_measures": "Measures average blood sugar."},
                {"name": "Fasting Glucose", "what_it_measures": "Measures blood glucose."},
                {"name": "Total Cholesterol", "what_it_measures": "Measures overall cholesterol."},
                {"name": "LDL Cholesterol", "what_it_measures": "Measures LDL cholesterol."},
                {"name": "HDL Cholesterol", "what_it_measures": "Measures HDL cholesterol."},
                {"name": "Triglycerides", "what_it_measures": "Measures triglycerides."},
            ]
        }
    )

    app.dependency_overrides[get_db_path] = lambda: db_file
    app.dependency_overrides[get_llm] = lambda: fake_llm

    with TestClient(app, raise_server_exceptions=False) as test_cli:
        yield test_cli, db_file

    app.dependency_overrides.clear()


def test_get_root_returns_html_with_disclaimer_and_privacy(client):
    """GET / returns 200 with text/html and exact disclaimer and privacy text."""
    cli, _ = client
    response = cli.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")

    html_text = response.text
    assert EXACT_DISCLAIMER in html_text
    assert EXACT_PRIVACY_NOTE in html_text


def test_static_assets_and_vendor_plotly(client):
    """GET /static/app.js, styles.css return 200; /static/vendor/plotly.min.js returns js > 1MB."""
    cli, _ = client

    resp_js = cli.get("/static/app.js")
    assert resp_js.status_code == 200
    assert len(resp_js.content) > 100

    resp_css = cli.get("/static/styles.css")
    assert resp_css.status_code == 200
    assert len(resp_css.content) > 100

    resp_plotly = cli.get("/static/vendor/plotly.min.js")
    assert resp_plotly.status_code == 200
    assert "javascript" in resp_plotly.headers.get("content-type", "").lower()
    assert len(resp_plotly.content) > 1_000_000  # Above 1 MB


def test_security_headers_and_swagger_exemption(client):
    """Security headers are present on /, /static/app.js, and /api/*; /docs and /openapi.json are exempted."""
    cli, _ = client

    for path in ["/", "/static/app.js", "/api/profiles"]:
        resp = cli.get(path)
        assert resp.status_code == 200
        assert resp.headers.get("Content-Security-Policy") == EXPECTED_CSP
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("Referrer-Policy") == "no-referrer"
        assert resp.headers.get("X-Frame-Options") == "DENY"

    # /docs and /openapi.json do NOT get strict CSP and still return 200
    docs_resp = cli.get("/docs")
    assert docs_resp.status_code == 200
    assert "Content-Security-Policy" not in docs_resp.headers

    openapi_resp = cli.get("/openapi.json")
    assert openapi_resp.status_code == 200
    assert "Content-Security-Policy" not in openapi_resp.headers


def test_cache_control_and_health_check(client):
    """Cache-Control: no-store on /api/profiles; GET /health works normally."""
    cli, _ = client

    profiles_resp = cli.get("/api/profiles")
    assert profiles_resp.status_code == 200
    assert profiles_resp.headers.get("Cache-Control") == "no-store"

    health_resp = cli.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json() == {"status": "ok", "app": "medical-report-explainer"}


def test_static_files_security_scans():
    """Static scans of UI files: no external URLs, no forbidden DOM APIs, no inline handlers."""
    # 1. No http:// or https:// anywhere across the three files
    for filepath in [INDEX_HTML, STYLES_CSS, APP_JS]:
        content = filepath.read_text(encoding="utf-8")
        assert "http://" not in content, f"Found http:// in {filepath.name}"
        assert "https://" not in content, f"Found https:// in {filepath.name}"

    # 2. app.js contains none of forbidden APIs
    js_content = APP_JS.read_text(encoding="utf-8")
    forbidden_js = ["innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"]
    for forbidden in forbidden_js:
        assert forbidden not in js_content, f"Found forbidden {forbidden} in app.js"

    # 3. index.html contains no inline handlers on...= and no <script> without src
    html_content = INDEX_HTML.read_text(encoding="utf-8")
    inline_handlers = re.findall(r"\bon[a-zA-Z]+\s*=", html_content)
    assert not inline_handlers, f"Found inline event handlers in index.html: {inline_handlers}"

    # Verify every <script> has a src attribute
    script_tags = re.findall(r"<script\b([^>]*)>", html_content, re.IGNORECASE)
    for tag in script_tags:
        assert "src=" in tag, f"Found <script> tag without src in index.html: {tag}"

    # Verify index.html references only /static/... resources
    hrefs_and_srcs = re.findall(r'(?:href|src)=["\']([^"\']+)["\']', html_content)
    for resource in hrefs_and_srcs:
        # Ignore intra-page anchor links or meta/svg attributes
        if resource.startswith("#") or resource == "en":
            continue
        assert resource.startswith("/static/"), f"Resource in index.html does not start with /static/: {resource}"


def test_app_js_contains_required_status_labels_and_verify_text():
    """app.js contains each exact fixed status label and the verification text."""
    js_content = APP_JS.read_text(encoding="utf-8")

    for label in REQUIRED_STATUS_LABELS:
        assert label in js_content, f"Missing required status label in app.js: {label}"

    assert REQUIRED_VERIFY_TEXT in js_content, f"Missing required verify text in app.js: {REQUIRED_VERIFY_TEXT}"


def test_upload_and_latest_returns_trend_dropdown_shape(client):
    """Uploading a report returns numeric results accessible via /api/profiles/{id}/latest with expected keys."""
    cli, _ = client

    with open(REPORT_TABLE_PDF, "rb") as f:
        upload_resp = cli.post(
            "/api/reports/upload",
            files={"file": ("report_table.pdf", f, "application/pdf")},
            data={"profile_id": 1},
        )
    assert upload_resp.status_code == 200

    latest_resp = cli.get("/api/profiles/1/latest")
    assert latest_resp.status_code == 200
    latest_data = latest_resp.json()

    # Must be a dictionary where keys are canonical test names
    assert isinstance(latest_data, dict)
    assert len(latest_data) > 0

    # Assert expected shape that the UI code relies on for populating trend dropdown and plotting
    for test_name, item in latest_data.items():
        assert isinstance(test_name, str)
        assert item["value"] is not None
        assert "unit" in item
        assert "ref_low" in item
        assert "ref_high" in item
        assert "flag" in item
        assert "test_name_canonical" in item


def test_app_js_single_point_trend_handling():
    """app.js contains single-point trend handling with consistent date formatting and prompt."""
    js_content = APP_JS.read_text(encoding="utf-8")
    assert "%b %d, %Y" in js_content
    assert "Upload another report to see a trend." in js_content


def test_ui_i18n_elements_and_font_stack():
    """app.js and index.html contain the three option labels, the 'Explanation language' label

    and the sentence about doctor questions; styles.css contains 'Nirmala UI'.
    """
    html_content = INDEX_HTML.read_text(encoding="utf-8")
    js_content = APP_JS.read_text(encoding="utf-8")
    css_content = STYLES_CSS.read_text(encoding="utf-8")

    # Options labels
    for opt in ["English", "हिन्दी", "తెలుగు"]:
        assert opt in html_content, f"Option label '{opt}' not found in index.html"
        assert opt in js_content, f"Option label '{opt}' not found in app.js"

    # Explanation language label
    assert "Explanation language" in html_content
    assert "Explanation language" in js_content

    # Doctor questions hint sentence
    hint = "Doctor questions are shown in English so you can share them with your doctor."
    assert hint in html_content
    assert hint in js_content

    # styles.css contains "Nirmala UI"
    assert "Nirmala UI" in css_content

