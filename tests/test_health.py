from fastapi.testclient import TestClient

from app.main import app


def test_health_check():
    """Verify that GET /health returns 200 and expected payload."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "app": "medical-report-explainer",
        }
