import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from lstm_forecasting import evaluate


class TestRegressionMetrics:
    """Test standard regression metrics."""

    def test_compute_rmse(self):
        """Test RMSE computation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.9])

        rmse = evaluate.compute_rmse(y_true, y_pred)
        expected = np.sqrt(np.mean([0.01, 0.01, 0.01, 0.01]))
        assert np.isclose(rmse, expected)

    def test_compute_mae(self):
        """Test MAE computation."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])

        mae = evaluate.compute_mae(y_true, y_pred)
        assert mae == 0.5

    def test_compute_mape(self):
        """Test MAPE (percentage error) computation."""
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 220.0])  # 10% error on both

        mape = evaluate.compute_mape(y_true, y_pred)
        assert np.isclose(mape, 10.0, atol=0.1)

    def test_compute_mape_with_small_values(self):
        """Test MAPE with epsilon for numerical stability."""
        y_true = np.array([0.0, 1e-10])
        y_pred = np.array([0.1, 1e-9])

        mape = evaluate.compute_mape(y_true, y_pred, eps=1e-8)
        assert np.isfinite(mape)


class TestForecastingLag:
    """Test lag estimation diagnostic."""

    def test_estimate_lag_perfect_fit_lag_zero(self):
        """Test that perfect predictions have lag=0."""
        y_true = np.sin(np.linspace(0, 4 * np.pi, 100))
        y_pred = y_true.copy()

        lag = evaluate.estimate_forecasting_lag(y_true, y_pred, max_lag=10)
        assert lag == 0

    def test_estimate_lag_one_step_delay(self):
        """Test that one-step-delayed predictions have lag=1."""
        y_true = np.sin(np.linspace(0, 4 * np.pi, 100))
        y_pred = np.roll(y_true, -1)  # Shift left by 1 (predictions are 1-step ahead)

        lag = evaluate.estimate_forecasting_lag(y_true, y_pred, max_lag=5)
        assert lag == 1

    def test_estimate_lag_respects_max_lag(self):
        """Test that lag estimate is bounded by max_lag."""
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)  # Independent random

        lag = evaluate.estimate_forecasting_lag(y_true, y_pred, max_lag=5)
        assert lag <= 5


class TestEvaluatePredictions:
    """Test full metrics bundle computation."""

    def test_evaluate_predictions_complete(self):
        """Test that evaluate_predictions returns all metrics."""
        y_true = np.array([100.0, 101.0, 102.0, 103.0])
        y_pred = np.array([100.5, 101.5, 102.5, 103.5])

        metrics = evaluate.evaluate_predictions(y_true, y_pred, name="TestModel")

        assert metrics.rmse > 0
        assert metrics.mae > 0
        assert metrics.mape > 0
        assert metrics.lag >= 0
        assert metrics.name == "TestModel"

    def test_evaluate_predictions_string_repr(self):
        """Test that metrics can be converted to string."""
        y_true = np.array([100.0, 101.0, 102.0])
        y_pred = np.array([100.5, 101.5, 102.5])

        metrics = evaluate.evaluate_predictions(y_true, y_pred, name="Test")
        metrics_str = str(metrics)

        assert "Test" in metrics_str
        assert "RMSE" in metrics_str
        assert "MAE" in metrics_str


class TestPlotPredictions:
    """Test prediction visualization."""

    @pytest.fixture
    def plot_data(self):
        """Create sample data for plotting."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        y_true = np.sin(np.linspace(0, 4 * np.pi, 100)) + 100
        y_pred = y_true + np.random.randn(100) * 0.5
        return dates, y_true, y_pred

    def test_plot_predictions_creates_figure(self, plot_data):
        """Test that plot_predictions creates a figure."""
        dates, y_true, y_pred = plot_data
        fig = evaluate.plot_predictions(
            dates, y_true, {"LSTM": y_pred}, title="Test"
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_predictions_saves_to_file(self, plot_data, tmp_path):
        """Test that plot can be saved to disk."""
        dates, y_true, y_pred = plot_data
        save_path = tmp_path / "test_plot.png"

        fig = evaluate.plot_predictions(
            dates, y_true, {"LSTM": y_pred},
            title="Test", save_path=str(save_path)
        )

        assert save_path.exists()
        plt.close(fig)

    def test_plot_predictions_multiple_models(self, plot_data):
        """Test plotting multiple models together."""
        dates, y_true, y_pred1 = plot_data
        y_pred2 = y_true + np.random.randn(100) * 0.3

        fig = evaluate.plot_predictions(
            dates, y_true,
            {"LSTM": y_pred1, "ARIMA": y_pred2},
            title="Comparison"
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotMetricComparison:
    """Test metrics comparison bar chart."""

    def test_plot_metric_comparison_creates_figure(self):
        """Test that metric comparison plot is created."""
        metrics = [
            evaluate.Metrics(rmse=1.5, mae=1.2, mape=0.5, lag=0, name="LSTM"),
            evaluate.Metrics(rmse=2.0, mae=1.6, mape=0.7, lag=1, name="ARIMA"),
        ]

        fig = evaluate.plot_metric_comparison(metrics)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_metric_comparison_saves(self, tmp_path):
        """Test that comparison plot can be saved."""
        metrics = [
            evaluate.Metrics(rmse=1.5, mae=1.2, mape=0.5, lag=0, name="LSTM"),
            evaluate.Metrics(rmse=2.0, mae=1.6, mape=0.7, lag=1, name="ARIMA"),
        ]
        save_path = tmp_path / "comparison.png"

        fig = evaluate.plot_metric_comparison(metrics, save_path=str(save_path))
        assert save_path.exists()
        plt.close(fig)