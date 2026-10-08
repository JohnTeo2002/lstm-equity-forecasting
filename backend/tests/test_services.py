import pytest
from backend.src.services.data_service import DataService
from backend.src.services.feature_service import FeatureService
from backend.src.services.forecast_service import ForecastService


def test_data_service_synthetic_generation():
    service = DataService()
    bars = service.fetch_historical_bars("AAPL", days=40)
    assert len(bars) == 40
    assert "date" in bars[0]
    assert "close" in bars[0]
    assert bars[0]["close"] > 0
    assert bars[0]["high"] >= bars[0]["low"]


def test_feature_service_indicators():
    service = FeatureService()
    prices = [100.0, 102.0, 101.0, 105.0, 107.0, 106.0, 108.0, 110.0, 112.0, 115.0, 114.0]
    sma = service.compute_sma(prices, window=5)
    assert len(sma) == len(prices)
    assert sma[-1] == sum(prices[-5:]) / 5.0

    rsi = service.compute_rsi(prices, period=5)
    assert len(rsi) == len(prices)
    assert all(0.0 <= r <= 100.0 for r in rsi)


def test_forecast_service_generation():
    service = ForecastService()
    result = service.generate_forecast("MSFT", lookback_window=50, horizon_days=5)

    assert result["ticker"] == "MSFT"
    assert len(result["predictions"]) == 5
    assert result["last_historical_close"] > 0

    first_pred = result["predictions"][0]
    assert "step" in first_pred
    assert first_pred["step"] == 1
    assert "predicted_price" in first_pred
    assert "lower_bound" in first_pred
    assert "upper_bound" in first_pred
    assert first_pred["lower_bound"] <= first_pred["predicted_price"] <= first_pred["upper_bound"]

