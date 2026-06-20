"""
Turns an engineered feature DataFrame into supervised-learning
sequences suitable for an LSTM and provides walk-forward
cross-validation splits via sklearn.model_selection.TimeSeriesSplit.

Design notes:
- Scaling is fit ONLY on the training portion of each split and then
  applied to the corresponding validation/test portion, to avoid
  look-ahead bias (a common and serious leakage bug in financial ML).
- Sequences are built as sliding windows of lookback past time
  steps used to predict the target horizon steps ahead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

from .logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "ema_10", "ema_20", "ema_50",
    "rsi",
    "macd_line", "macd_signal", "macd_hist",
    "bb_mid", "bb_upper", "bb_lower",
    "volatility", "log_return",
]


@dataclass
class WindowedDataset:
    """Container for a windowed train/test split, plus the fitted scalers.

    ``X`` has shape (n_samples, lookback, n_features); ``y`` has shape
    (n_samples,) and is the scaled target value ``horizon`` steps
    ahead of the end of each window. The scalers are retained so
    predictions can be inverse-transformed back to price units.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler
    feature_columns: list[str] = field(default_factory=list)
    test_dates: pd.Series | None = None


def make_sequences(
    feature_array: np.ndarray,
    target_array: np.ndarray,
    lookback: int,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a fixed-length window across the arrays to build (X, y) pairs.

    For each index ``i`` from ``lookback`` to ``len(array) - horizon``,
    builds one sample: the previous ``lookback`` rows of features as
    ``X``, and the target value ``horizon`` steps after the window end
    as ``y``.
    """
    n = len(feature_array)
    X, y = [], []
    for end in range(lookback, n - horizon + 1):
        X.append(feature_array[end - lookback:end])
        y.append(target_array[end + horizon - 1])
    return np.array(X), np.array(y)


def scale_and_window_split(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_column: str = "Close",
    lookback: int = 60,
    horizon: int = 1,
    train_frac: float = 0.8,
    date_column: str = "Date",
) -> WindowedDataset:
    """Single chronological train/test split with leakage-safe scaling.

    The DataFrame must already be sorted ascending by date. Scalers
    are fit on the training slice only, then applied (not re-fit) to
    the test slice, mirroring how the model would be deployed on
    genuinely unseen future data.
    """
    feature_columns = feature_columns or [c for c in DEFAULT_FEATURE_COLUMNS if c in df.columns]
    df = df.sort_values(date_column).reset_index(drop=True)

    split_idx = int(len(df) * train_frac)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    train_features = feature_scaler.fit_transform(train_df[feature_columns].values)
    train_target = target_scaler.fit_transform(train_df[[target_column]].values).ravel()

    test_features = feature_scaler.transform(test_df[feature_columns].values)
    test_target = target_scaler.transform(test_df[[target_column]].values).ravel()

    X_train, y_train = make_sequences(train_features, train_target, lookback, horizon)
    X_test, y_test = make_sequences(test_features, test_target, lookback, horizon)

    test_dates = test_df[date_column].iloc[lookback + horizon - 1:].reset_index(drop=True)

    logger.info(
        "Windowed dataset built: X_train=%s, X_test=%s, lookback=%d, horizon=%d",
        X_train.shape, X_test.shape, lookback, horizon,
    )

    return WindowedDataset(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        feature_scaler=feature_scaler, target_scaler=target_scaler,
        feature_columns=feature_columns, test_dates=test_dates,
    )


def time_series_cv_splits(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_column: str = "Close",
    lookback: int = 60,
    horizon: int = 1,
    n_splits: int = 5,
    date_column: str = "Date",
):
    """Generate walk-forward (expanding-window) CV folds via TimeSeriesSplit.

    Yields ``WindowedDataset`` objects, one per fold, each scaled
    independently on that fold's training portion only. This is the
    "rigorous cross-validation" piece called out in the project brief:
    unlike a random k-fold, TimeSeriesSplit always trains on the past
    and validates on a contiguous future block, respecting the
    temporal ordering of financial data.
    """
    feature_columns = feature_columns or [c for c in DEFAULT_FEATURE_COLUMNS if c in df.columns]
    df = df.sort_values(date_column).reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    full_features = df[feature_columns].values
    full_target = df[[target_column]].values

    for fold, (train_idx, test_idx) in enumerate(tscv.split(full_features), start=1):
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        train_features = feature_scaler.fit_transform(full_features[train_idx])
        train_target = target_scaler.fit_transform(full_target[train_idx]).ravel()

        test_features = feature_scaler.transform(full_features[test_idx])
        test_target = target_scaler.transform(full_target[test_idx]).ravel()

        X_train, y_train = make_sequences(train_features, train_target, lookback, horizon)
        X_test, y_test = make_sequences(test_features, test_target, lookback, horizon)

        if len(X_train) == 0 or len(X_test) == 0:
            logger.warning(
                "Fold %d skipped: insufficient rows for lookback=%d (train_idx=%d, test_idx=%d)",
                fold, lookback, len(train_idx), len(test_idx),
            )
            continue

        test_dates = df[date_column].iloc[test_idx].iloc[lookback + horizon - 1:].reset_index(drop=True)

        logger.info(
            "Fold %d/%d: X_train=%s, X_test=%s", fold, n_splits, X_train.shape, X_test.shape
        )

        yield WindowedDataset(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            feature_scaler=feature_scaler, target_scaler=target_scaler,
            feature_columns=feature_columns, test_dates=test_dates,
        )