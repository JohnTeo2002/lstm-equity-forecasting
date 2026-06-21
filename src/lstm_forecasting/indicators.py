"""
indicators.py
=============

Technical indicator feature engineering on top of raw OHLCV data.

Implements the core momentum/trend indicators — EMA, RSI, MACD — and complementary
indicators (Bollinger Bands, rolling volatility, log returns) that
are standard additions to an LSTM feature set for short-term price
forecasting. All functions are pure (take a DataFrame, return a
DataFrame/Series) so they can be unit-tested in isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def exponential_moving_average(series: pd.Series, span: int = 20) -> pd.Series:
    """Exponential Moving Average (EMA) with the given span (in periods)."""
    return series.ewm(span=span, adjust=False).mean()


def relative_strength_index(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (RSI), Wilder's smoothing method.

    RSI oscillates between 0 and 100; values above 70 are
    conventionally read as overbought, below 30 as oversold.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Moving Average Convergence Divergence (MACD).

    Returns a DataFrame with columns ``macd_line``, ``macd_signal``,
    and ``macd_hist`` (the difference between the two, commonly
    plotted as a histogram and used to flag momentum/trend reversals).
    """
    ema_fast = exponential_moving_average(series, span=fast)
    ema_slow = exponential_moving_average(series, span=slow)
    macd_line = ema_fast - ema_slow
    macd_signal = exponential_moving_average(macd_line, span=signal)
    macd_hist = macd_line - macd_signal

    return pd.DataFrame(
        {"macd_line": macd_line, "macd_signal": macd_signal, "macd_hist": macd_hist}
    )


def bollinger_bands(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands: rolling mean +/- ``num_std`` rolling standard deviations."""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper = rolling_mean + num_std * rolling_std
    lower = rolling_mean - num_std * rolling_std
    return pd.DataFrame({"bb_mid": rolling_mean, "bb_upper": upper, "bb_lower": lower})


def log_returns(series: pd.Series) -> pd.Series:
    """1-period log return: ln(P_t / P_{t-1})."""
    return np.log(series / series.shift(1))


def rolling_volatility(series: pd.Series, window: int = 20) -> pd.Series:
    """Rolling standard deviation of log returns, a simple realized-vol proxy."""
    return log_returns(series).rolling(window=window).std()


def add_technical_indicators(
    df: pd.DataFrame,
    price_col: str = "Close",
    ema_spans: tuple[int, ...] = (10, 20, 50),
    rsi_window: int = 14,
    macd_params: tuple[int, int, int] = (12, 26, 9),
    bb_window: int = 20,
    vol_window: int = 20,
) -> pd.DataFrame:
    """Append the full engineered feature set to a raw OHLCV DataFrame.

    Adds: multiple EMAs, RSI, MACD (line/signal/hist), Bollinger Bands,
    rolling volatility, and 1-day log return. Rows containing NaNs
    introduced by the indicator warm-up windows are dropped at the end
    so every remaining row has a complete feature vector.
    """
    out = df.copy()
    price = out[price_col]

    for span in ema_spans:
        out[f"ema_{span}"] = exponential_moving_average(price, span=span)

    out["rsi"] = relative_strength_index(price, window=rsi_window)

    fast, slow, signal = macd_params
    macd_df = macd(price, fast=fast, slow=slow, signal=signal)
    out = out.join(macd_df)

    bb_df = bollinger_bands(price, window=bb_window)
    out = out.join(bb_df)

    out["volatility"] = rolling_volatility(price, window=vol_window)
    out["log_return"] = log_returns(price)

    out = out.dropna().reset_index(drop=False)
    return out
