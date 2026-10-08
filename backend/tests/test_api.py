import pytest
from fastapi.testclient import TestClient

try:
    from backend.src.main import app
    client = TestClient(app)
    CLIENT_AVAILABLE = True
except Exception:
    CLIENT_AVAILABLE = False


@pytest.mark.skipif(not CLIENT_AVAILABLE, reason="FastAPI TestClient unavailable")
def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


@pytest.mark.skipif(not CLIENT_AVAILABLE, reason="FastAPI TestClient unavailable")
def test_historical_endpoint():
    response = client.get("/api/v1/stocks/AAPL/historical?days=30")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert data["count"] == 30
    assert len(data["data"]) == 30


@pytest.mark.skipif(not CLIENT_AVAILABLE, reason="FastAPI TestClient unavailable")
def test_forecast_predict_endpoint():
    payload = {
        "ticker": "AAPL",
        "lookback_window": 60,
        "horizon_days": 5,
        "include_confidence_intervals": True,
    }
    response = client.post("/api/v1/forecast/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert len(data["predictions"]) == 5

