"""
Evaluation metrics (RMSE, MAE) and diagnostic plots for comparing the
LSTM model against the ARIMA/GARCH baselines, including a simple
"forecasting lag" diagnostic — the cross-correlation lag at which a
prediction series best aligns with the actual series. A naive model
that just echoes yesterday's price typically shows lag = 1; a model
that has genuinely learned leading signal should show lag close to 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Metrics:
    rmse: float
    mae: float
    mape: float
    lag: int
    name: str = "model"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.name}: RMSE={self.rmse:.5f}  MAE={self.mae:.5f}  "
            f"MAPE={self.mape:.2f}%  lag={self.lag}"
        )


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error, expressed in percent."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def estimate_forecasting_lag(
    y_true: np.ndarray, y_pred: np.ndarray, max_lag: int = 10
) -> int:
    """Estimate the lag (in steps) that maximizes correlation between pred and true.

    Computes corr(y_true[t], y_pred[t - k]) for k in [0, max_lag] and
    returns the k with the highest correlation. A model that mostly
    just reproduces a delayed version of the actual series (a common
    failure mode for naive sequence models on near-random-walk price
    data) will show a lag > 0; lower is better.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    best_lag, best_corr = 0, -np.inf

    for lag in range(0, max_lag + 1):
        if lag == 0:
            a, b = y_true, y_pred
        else:
            a, b = y_true[lag:], y_pred[:-lag]
        if len(a) < 2:
            continue
        corr = np.corrcoef(a, b)[0, 1]
        if np.isnan(corr):
            continue
        if corr > best_corr:
            best_corr, best_lag = corr, lag

    return best_lag


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, name: str = "model", max_lag: int = 10
) -> Metrics:
    """Compute RMSE, MAE, MAPE, and lag metrics for one model's predictions."""
    metrics = Metrics(
        rmse=compute_rmse(y_true, y_pred),
        mae=compute_mae(y_true, y_pred),
        mape=compute_mape(y_true, y_pred),
        lag=estimate_forecasting_lag(y_true, y_pred, max_lag=max_lag),
        name=name,
    )
    logger.info(str(metrics))
    return metrics


def ensemble_predictions(
    predictions: dict[str, np.ndarray],
    weights: dict[str, float] | None = None,
    method: str = "weighted_average",
) -> np.ndarray:
    """Combine multiple models' predictions via ensemble.

    Args:
        predictions: Mapping of model name to prediction array (all same length).
        weights: Optional mapping of model name to weight. If None, uses equal weights.
        method: "weighted_average" or "median".

    Returns:
        Ensemble predictions (1-D array, same length as input predictions).
    """
    if not predictions:
        raise ValueError("predictions dict cannot be empty")

    pred_arrays = list(predictions.values())
    if not all(len(p) == len(pred_arrays[0]) for p in pred_arrays):
        raise ValueError("all prediction arrays must have the same length")

    if weights is None:
        weights = {name: 1.0 / len(predictions) for name in predictions.keys()}

    if method == "weighted_average":
        ensemble = np.zeros_like(pred_arrays[0], dtype=float)
        for name, pred in predictions.items():
            weight = weights.get(name, 1.0 / len(predictions))
            ensemble += weight * np.asarray(pred)
        logger.info("Ensemble created: %s with weights %s", method, weights)
        return ensemble

    elif method == "median":
        stacked = np.column_stack(pred_arrays)
        ensemble = np.median(stacked, axis=1)
        logger.info("Ensemble created: %s", method)
        return ensemble

    else:
        raise ValueError(f"Unknown ensemble method: {method}")



def plot_predictions(
    dates,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    title: str = "Predicted vs. Actual Price",
    save_path: str | None = None,
):
    """Plot actual price against one or more models' predictions over time.

    ``predictions`` maps a model name (e.g. "LSTM", "ARIMA(5,1,0)") to
    its prediction array, all aligned to the same ``dates``/``y_true``.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, y_true, label="Actual", color="black", linewidth=1.5)

    for name, preds in predictions.items():
        ax.plot(dates, preds, label=name, linewidth=1.2, alpha=0.85)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("Saved prediction plot to %s", save_path)

    return fig


def plot_training_history(history, save_path: str | None = None):
    """Plot training/validation loss curves from a Keras ``History`` object."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.history["loss"], label="train_loss")
    if "val_loss" in history.history:
        ax.plot(history.history["val_loss"], label="val_loss")
    ax.set_title("Training History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("Saved training history plot to %s", save_path)

    return fig


def plot_metric_comparison(metrics_list: list[Metrics], save_path: str | None = None):
    """Bar chart comparing RMSE and MAE across multiple models (e.g. LSTM vs ARIMA)."""
    names = [m.name for m in metrics_list]
    rmses = [m.rmse for m in metrics_list]
    maes = [m.mae for m in metrics_list]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, rmses, width, label="RMSE")
    ax.bar(x + width / 2, maes, width, label="MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Error (scaled units)")
    ax.set_title("Model Comparison: RMSE & MAE")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("Saved metric comparison plot to %s", save_path)

    return fig
