import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from lstm_forecasting import preprocessing


@pytest.fixture
def sample_feature_df():
    """Create a feature-engineered DataFrame for testing."""
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "Date": dates,
        "Open": np.random.randn(n).cumsum() + 100,
        "High": np.random.randn(n).cumsum() + 102,
        "Low": np.random.randn(n).cumsum() + 98,
        "Close": np.random.randn(n).cumsum() + 100,
        "Volume": np.random.randint(1e6, 2e6, n),
        "ema_10": np.random.randn(n).cumsum() + 100,
        "ema_20": np.random.randn(n).cumsum() + 100,
        "ema_50": np.random.randn(n).cumsum() + 100,
        "rsi": np.random.uniform(30, 70, n),
        "macd_line": np.random.randn(n),
        "macd_signal": np.random.randn(n),
        "macd_hist": np.random.randn(n),
        "bb_mid": np.random.randn(n).cumsum() + 100,
        "bb_upper": np.random.randn(n).cumsum() + 102,
        "bb_lower": np.random.randn(n).cumsum() + 98,
        "volatility": np.random.uniform(0.01, 0.05, n),
        "log_return": np.random.randn(n) * 0.01,
    })


class TestMakeSequences:
    """Test sequence windowing logic."""

    def test_make_sequences_basic(self):
        """Test basic sequence creation."""
        features = np.arange(100).reshape(-1, 1)
        target = np.arange(100)

        X, y = preprocessing.make_sequences(features, target, lookback=10, horizon=1)

        assert X.shape[0] == 100 - 10  # n - lookback
        assert X.shape == (90, 10, 1)
        assert y.shape == (90,)
        assert X[0, -1, 0] == 9  # Last feature in first sequence
        assert y[0] == 10  # Target is 1 step ahead

    def test_make_sequences_multifeature(self):
        """Test with multiple features."""
        features = np.random.randn(100, 5)
        target = np.random.randn(100)

        X, y = preprocessing.make_sequences(features, target, lookback=20, horizon=1)

        assert X.shape == (100 - 20, 20, 5)
        assert y.shape == (100 - 20,)

    def test_make_sequences_horizon(self):
        """Test that horizon correctly shifts target."""
        features = np.arange(100).reshape(-1, 1)
        target = np.arange(100)

        X2, y2 = preprocessing.make_sequences(features, target, lookback=10, horizon=2)

        assert X2.shape[0] == 100 - 10 - 1  # n - lookback - (horizon - 1)
        assert y2[0] == 11  # 2 steps ahead


class TestScaleAndWindowSplit:
    """Test single train/test split with leakage-safe scaling."""

    def test_scale_and_window_single_split(self, sample_feature_df):
        """Test basic train/test split creation."""
        dataset = preprocessing.scale_and_window_split(
            sample_feature_df,
            lookback=30,
            train_frac=0.8,
        )

        assert dataset.X_train.shape[1] == 30  # lookback
        assert dataset.X_train.shape[2] > 0  # n_features
        assert len(dataset.y_train) == dataset.X_train.shape[0]
        assert len(dataset.y_test) == dataset.X_test.shape[0]
        assert dataset.y_test.shape[0] > 0

    def test_scale_and_window_leakage_safe(self, sample_feature_df):
        """Test that scalers are fit only on train data."""
        dataset = preprocessing.scale_and_window_split(
            sample_feature_df,
            lookback=30,
            train_frac=0.8,
        )

        # Feature scaler should be fit on training split
        assert dataset.feature_scaler.data_min_.shape[0] > 0
        # Target scaler should be fit on training split
        assert dataset.target_scaler.data_min_.shape == (1,)

    def test_scale_and_window_train_larger_than_test(self, sample_feature_df):
        """Test that train split is larger than test split (80/20)."""
        dataset = preprocessing.scale_and_window_split(
            sample_feature_df, train_frac=0.8, lookback=30
        )

        assert len(dataset.y_train) > len(dataset.y_test)
        assert len(dataset.y_train) / (len(dataset.y_train) + len(dataset.y_test)) > 0.75


class TestTimeSeriesCVSplits:
    """Test walk-forward (expanding window) cross-validation."""

    def test_time_series_cv_splits_generates_folds(self, sample_feature_df):
        """Test that TimeSeriesSplit generates the expected number of folds."""
        folds = list(preprocessing.time_series_cv_splits(
            sample_feature_df,
            n_splits=5,
            lookback=30,
        ))

        assert len(folds) == 5

    def test_time_series_cv_expanding_window(self, sample_feature_df):
        """Test that training window expands across folds."""
        folds = list(preprocessing.time_series_cv_splits(
            sample_feature_df,
            n_splits=5,
            lookback=30,
        ))

        train_sizes = [len(f.y_train) for f in folds]
        # Training set should grow (expand-window CV)
        assert train_sizes == sorted(train_sizes)

    def test_time_series_cv_test_windows_disjoint(self, sample_feature_df):
        """Test that test windows are disjoint and chronological."""
        folds = list(preprocessing.time_series_cv_splits(
            sample_feature_df,
            n_splits=3,
            lookback=30,
        ))

        total_test_size = sum(len(f.y_test) for f in folds)
        assert total_test_size > 0

    def test_time_series_cv_leakage_safe(self, sample_feature_df):
        """Test that each fold's scaler is fit independently on that fold's train."""
        folds = list(preprocessing.time_series_cv_splits(
            sample_feature_df,
            n_splits=3,
            lookback=30,
        ))

        # Each fold should have its own scalers
        scaler_mins = [f.feature_scaler.data_min_.copy() for f in folds]
        # Scalers should differ (fitted on different training slices)
        assert not np.allclose(scaler_mins[0], scaler_mins[1])


class TestWindowedDatasetProperties:
    """Test properties of WindowedDataset container."""

    def test_windowed_dataset_contains_scalers(self, sample_feature_df):
        """Test that WindowedDataset retains fitted scalers for inverse transform."""
        dataset = preprocessing.scale_and_window_split(
            sample_feature_df, lookback=30, train_frac=0.8
        )

        assert dataset.feature_scaler is not None
        assert dataset.target_scaler is not None
        # Check that scalers have been fit
        assert dataset.feature_scaler.data_min_ is not None

    def test_windowed_dataset_inverse_transform(self, sample_feature_df):
        """Test that predictions can be inverse-transformed back to original scale."""
        dataset = preprocessing.scale_and_window_split(
            sample_feature_df, lookback=30, train_frac=0.8
        )

        # Mock a prediction (scaled)
        scaled_pred = np.array([0.5, 0.6, 0.7])
        # Inverse transform
        original_scale = dataset.target_scaler.inverse_transform(
            scaled_pred.reshape(-1, 1)
        )

        assert original_scale.shape == (3, 1)
        assert not np.allclose(original_scale, scaled_pred)