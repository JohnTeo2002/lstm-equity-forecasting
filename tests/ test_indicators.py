import numpy as np
import pandas as pd
import pytest

from lstm_forecasting import indicators


class TestSimpleIndicators:
    """Test basic indicator computations."""

    def test_exponential_moving_average(self):
        """Test EMA calculation."""
        prices = pd.Series([100, 101, 102, 103, 104])
        ema = indicators.exponential_moving_average(prices, span=2)

        assert len(ema) == len(prices)
        assert not ema.iloc[-1] == prices.iloc[-1]  # EMA should smooth
        assert ema.iloc[-1] < prices.iloc[-1]  # EMA lags behind (for rising series)

    def test_relative_strength_index(self):
        """Test RSI oscillates in [0, 100]."""
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        rsi = indicators.relative_strength_index(prices, window=14)

        assert rsi.notna().sum() > 0
        assert (rsi.dropna() >= 0).all() and (rsi.dropna() <= 100).all()

    def test_macd(self):
        """Test MACD returns line, signal, and histogram."""
        prices = pd.Series(np.linspace(100, 110, 100))
        macd_df = indicators.macd(prices, fast=12, slow=26, signal=9)

        assert isinstance(macd_df, pd.DataFrame)
        assert set(macd_df.columns) == {"macd_line", "macd_signal", "macd_hist"}
        assert len(macd_df) == len(prices)

    def test_bollinger_bands(self):
        """Test Bollinger Bands computation."""
        prices = pd.Series(np.random.randn(50) + 100)
        bb = indicators.bollinger_bands(prices, window=20, num_std=2.0)

        assert set(bb.columns) == {"bb_mid", "bb_upper", "bb_lower"}
        # Upper band should always be >= mid >= lower
        assert (bb["bb_upper"] >= bb["bb_mid"]).all()
        assert (bb["bb_mid"] >= bb["bb_lower"]).all()

    def test_log_returns(self):
        """Test log return computation."""
        prices = pd.Series([100, 105, 110])
        log_ret = indicators.log_returns(prices)

        assert len(log_ret) == len(prices)
        assert pd.isna(log_ret.iloc[0])  # First return is NaN
        expected_ret_1 = np.log(105 / 100)
        assert np.isclose(log_ret.iloc[1], expected_ret_1)

    def test_rolling_volatility(self):
        """Test rolling volatility from log returns."""
        prices = pd.Series(np.random.randn(50) + 100)
        vol = indicators.rolling_volatility(prices, window=20)

        assert len(vol) == len(prices)
        assert vol.notna().sum() > 0
        assert (vol.dropna() >= 0).all()  # Volatility is non-negative


class TestAddTechnicalIndicators:
    """Test the full feature engineering pipeline."""

    @pytest.fixture
    def sample_ohlcv_df(self):
        """Create a small OHLCV DataFrame for testing."""
        n = 200
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        prices = pd.Series(np.linspace(100, 110, n) + np.random.randn(n) * 0.5)
        return pd.DataFrame(
            {
                "Date": dates,
                "Open": prices * 0.98,
                "High": prices * 1.02,
                "Low": prices * 0.96,
                "Close": prices,
                "Volume": np.random.randint(1e6, 2e6, n),
            }
        )

    def test_add_technical_indicators_complete(self, sample_ohlcv_df):
        """Test that all indicators are added and no NaNs remain."""
        result = indicators.add_technical_indicators(
            sample_ohlcv_df,
            price_col="Close",
            ema_spans=(10, 20, 50),
            rsi_window=14,
            bb_window=20,
            vol_window=20,
        )

        assert isinstance(result, pd.DataFrame)
        # Should have original OHLCV + engineered features
        assert "ema_10" in result.columns
        assert "ema_20" in result.columns
        assert "ema_50" in result.columns
        assert "rsi" in result.columns
        assert "macd_line" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns
        assert "bb_mid" in result.columns
        assert "volatility" in result.columns
        assert "log_return" in result.columns

        # No NaNs after dropna
        assert result.isna().sum().sum() == 0

    def test_add_technical_indicators_with_date_column(self, sample_ohlcv_df):
        """Test that Date column is preserved through feature engineering."""
        result = indicators.add_technical_indicators(
            sample_ohlcv_df,
            price_col="Close",
        )

        assert "Date" in result.columns
        assert len(result) < len(sample_ohlcv_df)  # Some rows dropped for NaN

    def test_add_technical_indicators_custom_params(self, sample_ohlcv_df):
        """Test with custom indicator parameters."""
        result = indicators.add_technical_indicators(
            sample_ohlcv_df,
            price_col="Close",
            ema_spans=(5, 15),
            rsi_window=7,
            macd_params=(5, 10, 3),
            bb_window=10,
            vol_window=10,
        )

        assert "ema_5" in result.columns
        assert "ema_15" in result.columns
        assert "ema_20" not in result.columns  # Not added with custom spans
        assert result.isna().sum().sum() == 0
