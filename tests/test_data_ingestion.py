import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from lstm_forecasting import data_ingestion


class TestOhlcvFetch:
    """Test OHLCV download and caching logic."""

    @patch("lstm_forecasting.data_ingestion.yf.download")
    def test_fetch_ohlcv_success(self, mock_download, tmp_path):
        """Test successful OHLCV fetch without cache."""
        mock_df = pd.DataFrame({
            "Open": [150.0, 151.0],
            "High": [152.0, 153.0],
            "Low": [149.0, 150.0],
            "Close": [151.5, 152.5],
            "Volume": [1000000, 1100000],
        }, index=pd.DatetimeIndex(["2023-01-01", "2023-01-02"], name="Date"))

        mock_download.return_value = mock_df

        result = data_ingestion.fetch_ohlcv(
            "TEST", start="2023-01-01", cache_dir=tmp_path, use_cache=False
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]

    @patch("lstm_forecasting.data_ingestion.yf.download")
    def test_fetch_ohlcv_caching(self, mock_download, tmp_path):
        """Test that cached OHLCV is reused on second fetch."""
        mock_df = pd.DataFrame({
            "Open": [149.0, 150.0],
            "High": [152.0, 153.0],
            "Low": [148.0, 149.0],
            "Close": [150.0, 151.0],
            "Volume": [1e6, 1.1e6],
        }, index=pd.DatetimeIndex(["2023-01-01", "2023-01-02"], name="Date"))

        mock_download.return_value = mock_df

        # First fetch writes to cache
        result1 = data_ingestion.fetch_ohlcv(
            "TEST", cache_dir=tmp_path, use_cache=True
        )
        assert mock_download.call_count == 1

        # Second fetch reads from cache (no yfinance call)
        result2 = data_ingestion.fetch_ohlcv(
            "TEST", cache_dir=tmp_path, use_cache=True
        )
        assert mock_download.call_count == 1  # unchanged
        assert result1.equals(result2)

    @patch("lstm_forecasting.data_ingestion.yf.download")
    def test_fetch_ohlcv_retry_logic(self, mock_download, tmp_path):
        """Test exponential backoff retry on transient failure."""
        success_df = pd.DataFrame({
            "Open": [99.0],
            "High": [101.0],
            "Low": [98.0],
            "Close": [100.0],
            "Volume": [1e6],
        }, index=pd.DatetimeIndex(["2023-01-01"], name="Date"))
        
        mock_download.side_effect = [
            Exception("Timeout"),
            success_df
        ]

        # Should succeed on second attempt
        result = data_ingestion.fetch_ohlcv(
            "TEST", cache_dir=tmp_path, use_cache=False, retries=2
        )
        assert mock_download.call_count == 2

    def test_scrape_sp500_tickers_fallback(self):
        """Test that SP500 scrape falls back to static list on failure."""
        with patch("lstm_forecasting.data_ingestion.requests.get") as mock_get:
            mock_get.side_effect = Exception("Network error")
            tickers = data_ingestion.scrape_sp500_tickers()

        # Should return fallback list
        assert tickers == data_ingestion.FALLBACK_SP500_SAMPLE
        assert len(tickers) > 0


class TestFetchMany:
    """Test batch fetching across multiple tickers."""

    @patch("lstm_forecasting.data_ingestion.fetch_ohlcv")
    def test_fetch_many_partial_failure(self, mock_fetch, tmp_path):
        """Test that fetch_many skips failed tickers and returns partial results."""
        mock_fetch.side_effect = [
            pd.DataFrame({"Close": [100.0]}, index=pd.DatetimeIndex(["2023-01-01"])),
            Exception("Failed for MSFT"),
            pd.DataFrame({"Close": [50.0]}, index=pd.DatetimeIndex(["2023-01-01"])),
        ]

        tickers = ["AAPL", "MSFT", "GOOGL"]
        results = data_ingestion.fetch_many(
            tickers, cache_dir=tmp_path, use_cache=False
        )

        assert len(results) == 2
        assert "AAPL" in results
        assert "MSFT" not in results
        assert "GOOGL" in results