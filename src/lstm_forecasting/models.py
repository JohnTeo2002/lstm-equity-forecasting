"""
LSTM model architecture and training utilities (TensorFlow / Keras).

NOTE: This module requires TensorFlow. Install with:
  pip install tensorflow  # For most platforms
  pip install tensorflow-macos  # For Apple Silicon (M1/M2/M3)

For Python 3.14, TensorFlow support is limited. Use Python 3.12 or 3.13 instead.

The model takes a sliding window of ``lookback`` past time steps,
each with the engineered OHLCV + technical-indicator feature vector,
and predicts the scaled closing price ``horizon`` steps ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class LSTMConfig:
    """Hyperparameters for the LSTM forecasting model."""

    lstm_units: tuple[int, ...] = (64, 32)
    dropout: float = 0.2
    dense_units: int = 16
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 100
    patience: int = 10  # early stopping patience


def build_lstm_model(
    input_shape: tuple[int, int], config: LSTMConfig | None = None
) -> tf.keras.Model:
    """Construct a stacked LSTM regressor.

    Architecture: N stacked LSTM layers (return_sequences=True on all
    but the last) each followed by Dropout for regularization, then a
    Dense bottleneck layer, then a single linear output unit
    predicting the (scaled) next price.

    ``input_shape`` is ``(lookback, n_features)``.
    """
    config = config or LSTMConfig()
    model = models.Sequential(name="lstm_equity_forecaster")
    model.add(layers.Input(shape=input_shape))

    for i, units in enumerate(config.lstm_units):
        return_sequences = i < len(config.lstm_units) - 1
        model.add(layers.LSTM(units, return_sequences=return_sequences))
        model.add(layers.Dropout(config.dropout))

    model.add(layers.Dense(config.dense_units, activation="relu"))
    model.add(layers.Dense(1, activation="linear"))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    logger.info("Built LSTM model: %s", model.summary())
    return model


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    config: LSTMConfig | None = None,
    verbose: int = 1,
) -> tf.keras.callbacks.History:
    """Train with early stopping (and LR reduction) on a validation set.

    If no explicit validation arrays are given, Keras carves off the
    last 10% of the training data chronologically (``validation_split``
    does NOT shuffle, so this remains temporally valid).
    """
    config = config or LSTMConfig()

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=config.patience, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(1, config.patience // 2),
            min_lr=1e-6,
        ),
    ]

    fit_kwargs = dict(
        x=X_train,
        y=y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        callbacks=cb_list,
        verbose=verbose,
        shuffle=False,  # preserve temporal order within mini-batches
    )

    if X_val is not None and y_val is not None:
        fit_kwargs["validation_data"] = (X_val, y_val)
    else:
        fit_kwargs["validation_split"] = 0.1

    logger.info(
        "Starting training for up to %d epochs (patience=%d)",
        config.epochs,
        config.patience,
    )
    history = model.fit(**fit_kwargs)
    logger.info("Training finished after %d epochs", len(history.history["loss"]))
    return history


def predict(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    """Run inference, returning a flat 1-D array of (scaled) predictions."""
    preds = model.predict(X, verbose=0)
    return preds.ravel()
