import datetime
import os
import tempfile
import pytest
from backend.src.services.data_service import DataService
from backend.src.services.db_service import DatabaseService
from backend.src.services.feature_service import FeatureService
from backend.src.services.forecast_service import ForecastService


def test_data_service_synthetic_generation():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db = DatabaseService(db_path=tmp.name)
        service = DataService(allow_synthetic=True, db_service=db)
        bars = service.fetch_historical_bars("AAPL", days=40)
        assert len(bars) == 40
        assert "date" in bars[0]
        assert "close" in bars[0]
        assert bars[0]["close"] > 0
        assert bars[0]["high"] >= bars[0]["low"]
        assert bars[0]["is_synthetic"] is True

        # Verify that all synthetic dates are strictly weekdays (eligible trading days)
        for bar in bars:
            d = datetime.datetime.strptime(bar["date"], "%Y-%m-%d").date()
            assert d.weekday() < 5, f"Date {bar['date']} falls on a weekend!"


def test_data_service_propagates_error_without_synthetic():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db = DatabaseService(db_path=tmp.name)
        service = DataService(provider="invalid_provider", allow_synthetic=False, db_service=db)
        with pytest.raises(ValueError, match="not found or market data unavailable"):
            service.fetch_historical_bars("NONEXISTENT_XYZ", days=20)


def test_impute_missing_data():
    raw_bars = [
        {"date": "2026-01-01", "open": 100.0, "high": 105.0, "low": 98.0, "close": 102.0, "volume": 1000},
        # Missing close, invalid low > high, volume None
        {"date": "2026-01-02", "open": None, "high": 90.0, "low": 120.0, "close": None, "volume": None},
    ]
    imputed = DataService.impute_missing_data(raw_bars)
    assert len(imputed) == 2
    assert imputed[0]["is_imputed"] is False

    second = imputed[1]
    assert second["is_imputed"] is True
    # Close was forward-filled from previous bar (102.0)
    assert second["close"] == 102.0
    assert second["open"] == 102.0
    assert second["high"] >= max(second["open"], second["close"])
    assert second["low"] <= min(second["open"], second["close"])
    assert second["volume"] > 0


def test_database_service_caching():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db = DatabaseService(db_path=tmp.name)
        bars = [
            {"date": "2026-01-01", "open": 150.0, "high": 155.0, "low": 149.0, "close": 152.0, "volume": 50000, "is_imputed": False, "is_synthetic": True},
            {"date": "2026-01-02", "open": 152.0, "high": 158.0, "low": 151.0, "close": 157.0, "volume": 60000, "is_imputed": True, "is_synthetic": True},
        ]
        db.save_bars("AAPL", bars)
        cached = db.get_cached_bars("AAPL", days=5)

        assert len(cached) == 2
        assert cached[0]["date"] == "2026-01-01"
        assert cached[1]["date"] == "2026-01-02"
        assert cached[1]["is_imputed"] is True

        # Test forecast saving
        db.save_forecast("AAPL-123", "AAPL", "lstm-v2.1", 5, 157.0, [{"step": 1, "predicted_price": 160.0}])


def test_data_service_uses_cache():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db = DatabaseService(db_path=tmp.name)
        service = DataService(provider="invalid_provider", allow_synthetic=False, db_service=db)

        # Prepopulate cache with fresh trading days ending on most recent trading day
        most_recent = DataService._get_most_recent_trading_day()
        trading_dates = []
        curr = most_recent
        while len(trading_dates) < 15:
            if curr.weekday() < 5:
                trading_dates.append(curr)
            curr -= datetime.timedelta(days=1)
        trading_dates.reverse()

        bars = [
            {
                "date": d.strftime("%Y-%m-%d"),
                "open": 100.0 + i,
                "high": 105.0 + i,
                "low": 95.0 + i,
                "close": 102.0 + i,
                "volume": 10000,
                "is_imputed": False,
                "is_synthetic": False,
            }
            for i, d in enumerate(trading_dates)
        ]
        db.save_bars("TEST_TICKER", bars)

        # Fetch using cache — should succeed even though provider is invalid
        result = service.fetch_historical_bars("TEST_TICKER", days=10, use_cache=True)
        assert len(result) == 10
        assert result[-1]["close"] == 102.0 + 14


def test_data_service_rejects_stale_or_synthetic_cache_when_disallowed():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db = DatabaseService(db_path=tmp.name)
        service = DataService(provider="invalid_provider", allow_synthetic=False, db_service=db)

        # Save stale bars (from year 2020)
        stale_bars = [
            {
                "date": f"2020-01-{i+1:02d}",
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 100.0,
                "volume": 1000,
                "is_imputed": False,
                "is_synthetic": False,
            }
            for i in range(10)
        ]
        db.save_bars("STALE_TICKER", stale_bars)

        # Should reject stale cache and attempt provider (which fails because provider is invalid)
        with pytest.raises(ValueError):
            service.fetch_historical_bars("STALE_TICKER", days=5, use_cache=True)


def test_feature_service_indicators():
    service = FeatureService()
    prices = [100.0, 102.0, 101.0, 105.0, 107.0, 106.0, 108.0, 110.0, 112.0, 115.0, 114.0]
    sma = service.compute_sma(prices, window=5)
    assert len(sma) == len(prices)
    assert sma[-1] == sum(prices[-5:]) / 5.0

    rsi = service.compute_rsi(prices, period=5)
    assert len(rsi) == len(prices)
    assert all(0.0 <= r <= 100.0 for r in rsi)


def test_forecast_service_generation_and_dates():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db = DatabaseService(db_path=tmp.name)
        data_svc = DataService(allow_synthetic=True, db_service=db)
        service = ForecastService(data_service=data_svc)
        result = service.generate_forecast("MSFT", lookback_window=50, horizon_days=7)

        assert result["ticker"] == "MSFT"
        assert len(result["predictions"]) == 7
        assert result["last_historical_close"] > 0
        assert result["model_version"] in ("lstm-v2.1", "non-lstm-heuristic-v1")

        # Verify consecutive forecast dates advance properly and skip weekends
        prev_date = None
        for pred in result["predictions"]:
            curr_d = datetime.datetime.strptime(pred["target_date"], "%Y-%m-%d").date()
            assert curr_d.weekday() < 5, f"Target date {pred['target_date']} is on weekend"
            if prev_date is not None:
                assert curr_d > prev_date, f"Consecutive forecast dates must strictly increase: {curr_d} <= {prev_date}"
            prev_date = curr_d

        first_pred = result["predictions"][0]
        assert "step" in first_pred
        assert first_pred["step"] == 1
        assert "predicted_price" in first_pred
        assert "lower_bound" in first_pred
        assert "upper_bound" in first_pred
        assert first_pred["lower_bound"] <= first_pred["predicted_price"] <= first_pred["upper_bound"]
