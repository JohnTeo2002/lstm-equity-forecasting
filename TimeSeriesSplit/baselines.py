"""
Traditional statistical forecasting benchmarks — ARIMA for the price
level and GARCH for conditional volatility — used as a point of
comparison against the LSTM model, per the project brief's claim of
"reduction in forecasting lag compared to traditional ARIMA/GARCH
benchmarks".

Both baselines are fit using a rolling/walk-forward one-step-ahead
scheme so they are evaluated on the same out-of-sample philosophy as
the LSTM's TimeSeriesSplit folds (train on the past, predict the next
unseen point, never the reverse).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from .logging_utils import get_logger

logger = get_logger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")


@dataclass
class BaselineResult:
    """Predictions and ground truth from a baseline model, aligned by index."""

    y_true: np.ndarray
    y_pred: np.ndarray
    name: str


def rolling_arima_forecast(
    series: pd.Series,
    order: tuple[int, int, int] = (5, 1, 0),
    test_size: int = 60,
    refit_every: int = 1,
) -> BaselineResult:
    """Walk-forward one-step-ahead ARIMA forecast over the last ``test_size`` points.

    At each step, fits ARIMA(``order``) on all data up to (not
    including) that point and forecasts exactly one step ahead. This
    mirrors how the LSTM is evaluated: no future information is ever
    used to predict the past. Refitting every step is the most
    faithful (and most expensive) approach; set ``refit_every`` > 1 to
    only refit periodically and reuse the model parameters in between,
    trading a little accuracy for speed on long test windows.
    """
    series = series.dropna()
    n = len(series)
    if test_size >= n:
        raise ValueError("test_size must be smaller than the full series length")

    history = series.iloc[: n - test_size].tolist()
    actuals = series.iloc[n - test_size:].tolist()
    predictions: list[float] = []

    fitted_model = None
    for step, actual in enumerate(actuals):
        if fitted_model is None or step % refit_every == 0:
            try:
                fitted_model = ARIMA(history, order=order).fit()
            except Exception as exc:  # noqa: BLE001
                logger.warning("ARIMA fit failed at step %d (%s); reusing prior fit", step, exc)

        forecast = fitted_model.forecast(steps=1)
        pred = float(np.asarray(forecast)[0])
        predictions.append(pred)
        history.append(actual)

    logger.info("Rolling ARIMA%s forecast complete over %d test points", order, test_size)
    return BaselineResult(
        y_true=np.array(actuals), y_pred=np.array(predictions), name=f"ARIMA{order}"
    )


def rolling_garch_volatility_forecast(
    returns: pd.Series,
    test_size: int = 60,
    p: int = 1,
    q: int = 1,
    refit_every: int = 5,
) -> BaselineResult:
    """Walk-forward one-step-ahead GARCH(p, q) conditional-volatility forecast.

    ``returns`` should be percentage (or log) returns, NOT raw price
    levels — GARCH models the conditional variance of returns, which
    is the standard volatility benchmark in financial forecasting
    research (as opposed to ARIMA, which models the price/return
    level itself). Requires the ``arch`` package.
    """
    from arch import arch_model  # local import: optional heavy dependency

    returns = returns.dropna() * 100  # arch_model expects returns scaled in percent
    n = len(returns)
    if test_size >= n:
        raise ValueError("test_size must be smaller than the full series length")

    history = returns.iloc[: n - test_size].tolist()
    actual_abs_returns = returns.iloc[n - test_size:].abs().tolist()
    predicted_vol: list[float] = []

    fitted_model = None
    for step in range(test_size):
        if fitted_model is None or step % refit_every == 0:
            am = arch_model(history, vol="GARCH", p=p, q=q, rescale=False)
            fitted_model = am.fit(disp="off")

        forecast = fitted_model.forecast(horizon=1, reindex=False)
        cond_vol = float(np.sqrt(forecast.variance.values[-1, 0]))
        predicted_vol.append(cond_vol)
        history.append(returns.iloc[n - test_size + step])

    logger.info("Rolling GARCH(%d,%d) volatility forecast complete over %d test points", p, q, test_size)
    return BaselineResult(
        y_true=np.array(actual_abs_returns), y_pred=np.array(predicted_vol),
        name=f"GARCH({p},{q})",
    )