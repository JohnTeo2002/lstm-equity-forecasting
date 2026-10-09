import pytest
from fastapi.testclient import TestClient
from backend.src.main import app
from backend.src.config import settings
from backend.src.api.routes import data_service, forecast_service

# Ensure tests can run deterministically without external network calls
data_service.allow_synthetic = True

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data


def test_models_endpoint():
    response = client.get("/api/v1/forecast/models")
    assert response.status_code == 200
    data = response.json()
    assert data["active_model"] in ("lstm-v2.1", "non-lstm-heuristic-v1")
    assert len(data["models"]) >= 1
    assert data["models"][0]["name"] == "Stacked LSTM"
    assert "sma" in data["models"][0]["features"]


def test_historical_endpoint():
    response = client.get("/api/v1/stocks/AAPL/historical?days=30")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "AAPL"
    assert data["count"] == 30
    assert len(data["data"]) == 30
    assert data["data"][0]["is_synthetic"] is True


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
    assert data["predictions"][0]["step"] == 1
    assert data["predictions"][0]["predicted_price"] > 0
