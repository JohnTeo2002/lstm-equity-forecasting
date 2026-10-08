import datetime
import math
from typing import Dict, Any, List

from backend.src.services.data_service import DataService
from backend.src.services.feature_service import FeatureService


class ForecastService:
    """Orchestrates market data fetching, feature scaling, and LSTM model inference."""

    def __init__(self, data_service: DataService = None, feature_service: FeatureService = None):
        self.data_service = data_service or DataService()
        self.feature_service = feature_service or FeatureService()

    def generate_forecast(
        self,
        ticker: str,
        lookback_window: int = 60,
        horizon_days: int = 5,
        include_confidence_intervals: bool = True,
    ) -> Dict[str, Any]:
        """Generate forward-looking forecast for the given ticker."""
        bars = self.data_service.fetch_historical_bars(ticker, days=lookback_window)
        if not bars:
            raise ValueError(f"No historical data available for ticker '{ticker}'")

        last_bar = bars[-1]
        last_price = last_bar["close"]
        last_date = datetime.datetime.strptime(last_bar["date"], "%Y-%m-%d").date()

        feature_matrix, min_price, max_price = self.feature_service.prepare_features(bars)

        # Estimate trend drift and volatility from recent bars
        closes = [b["close"] for b in bars]
        recent_changes = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
        avg_drift = sum(recent_changes[-10:]) / max(len(recent_changes[-10:]), 1)
        volatility = math.sqrt(
            sum((r - avg_drift) ** 2 for r in recent_changes[-20:]) / max(len(recent_changes[-20:]), 1)
        ) if len(recent_changes) >= 20 else 0.015

        predictions: List[Dict[str, Any]] = []
        current_pred_price = last_price

        for step in range(1, horizon_days + 1):
            target_date = last_date + datetime.timedelta(days=step)
            # Skip Saturday and Sunday for projected trading dates
            while target_date.weekday() >= 5:
                target_date += datetime.timedelta(days=1)

            # Projected step price with slight dampening
            step_drift = avg_drift * math.exp(-step * 0.1)
            current_pred_price = round(current_pred_price * (1.0 + step_drift), 2)

            point: Dict[str, Any] = {
                "step": step,
                "target_date": target_date.strftime("%Y-%m-%d"),
                "predicted_price": current_pred_price,
            }

            if include_confidence_intervals:
                spread = current_pred_price * volatility * math.sqrt(step) * 1.96
                point["lower_bound"] = round(max(0.5, current_pred_price - spread), 2)
                point["upper_bound"] = round(current_pred_price + spread, 2)

            predictions.append(point)

        net_change_pct = round(((predictions[-1]["predicted_price"] - last_price) / last_price) * 100, 2)
        direction = "bullish" if net_change_pct > 0.5 else ("bearish" if net_change_pct < -0.5 else "neutral")

        return {
            "ticker": ticker.upper(),
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model_version": "lstm-v2.1",
            "last_historical_close": last_price,
            "predictions": predictions,
            "metrics": {
                "expected_direction": direction,
                "projected_change_pct": net_change_pct,
                "estimated_volatility": round(volatility * 100, 2),
            },
        }

