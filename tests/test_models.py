import numpy as np
import pytest
import tensorflow as tf

from lstm_forecasting import models


class TestLSTMConfig:
    """Test LSTM hyperparameter configuration."""

    def test_lstm_config_defaults(self):
        """Test default LSTM configuration values."""
        cfg = models.LSTMConfig()

        assert cfg.lstm_units == (64, 32)
        assert cfg.dropout == 0.2
        assert cfg.dense_units == 16
        assert cfg.epochs == 100
        assert cfg.patience == 10

    def test_lstm_config_custom(self):
        """Test custom LSTM configuration."""
        cfg = models.LSTMConfig(
            lstm_units=(128, 64, 32),
            dropout=0.3,
            epochs=50,
        )

        assert cfg.lstm_units == (128, 64, 32)
        assert cfg.dropout == 0.3
        assert cfg.epochs == 50


class TestBuildLSTMModel:
    """Test LSTM model architecture construction."""

    def test_build_lstm_basic(self):
        """Test basic LSTM model build."""
        input_shape = (60, 15)  # (lookback, n_features)
        model = models.build_lstm_model(input_shape)

        assert isinstance(model, tf.keras.Model)
        assert model.input_shape == (None, 60, 15)
        assert model.output_shape == (None, 1)

    def test_build_lstm_stacked(self):
        """Test stacked LSTM with multiple layers."""
        input_shape = (60, 15)
        cfg = models.LSTMConfig(lstm_units=(64, 32))
        model = models.build_lstm_model(input_shape, config=cfg)

        # Count LSTM layers (should be 2)
        lstm_count = sum(1 for layer in model.layers if isinstance(layer, tf.keras.layers.LSTM))
        assert lstm_count == 2

    def test_build_lstm_with_dropout(self):
        """Test that dropout layers are added."""
        input_shape = (60, 15)
        cfg = models.LSTMConfig(dropout=0.2)
        model = models.build_lstm_model(input_shape, config=cfg)

        dropout_count = sum(
            1 for layer in model.layers if isinstance(layer, tf.keras.layers.Dropout)
        )
        assert dropout_count > 0

    def test_build_lstm_output_layer(self):
        """Test that output layer is linear regression (1 unit, no activation)."""
        input_shape = (60, 15)
        model = models.build_lstm_model(input_shape)

        last_layer = model.layers[-1]
        assert isinstance(last_layer, tf.keras.layers.Dense)
        assert last_layer.units == 1
        assert last_layer.activation is not None  # Linear is the default


class TestTrainModel:
    """Test model training and early stopping."""

    @pytest.fixture
    def toy_data(self):
        """Create small toy dataset for quick training."""
        X_train = np.random.randn(100, 60, 15).astype(np.float32)
        y_train = np.random.randn(100).astype(np.float32)
        X_val = np.random.randn(20, 60, 15).astype(np.float32)
        y_val = np.random.randn(20).astype(np.float32)
        return X_train, y_train, X_val, y_val

    def test_train_model_with_validation(self, toy_data):
        """Test training with explicit validation data."""
        X_train, y_train, X_val, y_val = toy_data
        model = models.build_lstm_model((60, 15))
        cfg = models.LSTMConfig(epochs=5, patience=2)

        history = models.train_model(
            model, X_train, y_train, X_val=X_val, y_val=y_val, config=cfg, verbose=0
        )

        assert hasattr(history, "history")
        assert "loss" in history.history
        assert len(history.history["loss"]) > 0
        assert len(history.history["loss"]) <= 5

    def test_train_model_with_validation_split(self, toy_data):
        """Test training with validation_split instead of explicit val data."""
        X_train, y_train, _, _ = toy_data
        model = models.build_lstm_model((60, 15))
        cfg = models.LSTMConfig(epochs=5, patience=2)

        history = models.train_model(
            model, X_train, y_train, config=cfg, verbose=0
        )

        assert "val_loss" in history.history

    def test_train_model_early_stopping(self, toy_data):
        """Test that early stopping terminates training early."""
        X_train, y_train, X_val, y_val = toy_data
        model = models.build_lstm_model((60, 15))
        cfg = models.LSTMConfig(epochs=100, patience=1)  # Very low patience

        history = models.train_model(
            model, X_train, y_train, X_val=X_val, y_val=y_val, config=cfg, verbose=0
        )

        # Training should stop well before 100 epochs
        assert len(history.history["loss"]) < 50


class TestPredict:
    """Test inference."""

    @pytest.fixture
    def trained_model_and_data(self):
        """Create a trained model for inference testing."""
        X_train = np.random.randn(50, 60, 15).astype(np.float32)
        y_train = np.random.randn(50).astype(np.float32)
        model = models.build_lstm_model((60, 15))
        cfg = models.LSTMConfig(epochs=2)
        models.train_model(model, X_train, y_train, config=cfg, verbose=0)

        X_test = np.random.randn(10, 60, 15).astype(np.float32)
        return model, X_test

    def test_predict_shape(self, trained_model_and_data):
        """Test that predictions have correct shape."""
        model, X_test = trained_model_and_data
        preds = models.predict(model, X_test)

        assert preds.shape == (10,)  # Flattened 1-D array

    def test_predict_is_numeric(self, trained_model_and_data):
        """Test that predictions are valid numeric values."""
        model, X_test = trained_model_and_data
        preds = models.predict(model, X_test)

        assert np.isfinite(preds).all()
        assert preds.dtype in [np.float32, np.float64]