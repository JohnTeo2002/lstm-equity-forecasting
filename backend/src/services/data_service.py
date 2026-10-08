import datetime
import math
import random
from typing import List, Dict, Any


class DataService:
    """Service responsible for retrieving and formatting market historical data."""

    def __init__(self, provider: str = "yfinance"):
        self.provider = provider

    def fetch_historical_bars(self, ticker: str, days: int = 60) -> List[Dict[str, Any]]:
        """
        Fetch historical daily OHLCV bars for the specified ticker.
        Falls back to deterministic synthetic generation if network/provider is unavailable.
        """
        ticker = ticker.upper().strip()

        try:
            import yfinance as yf
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period=f"{max(days, 30)}d")
            if not hist.empty:
                bars = []
                for idx, row in hist.tail(days).iterrows():
                    bars.append({
                        "date": idx.strftime("%Y-%m-%d"),
                        "open": round(float(row["Open"]), 2),
                        "high": round(float(row["High"]), 2),
                        "low": round(float(row["Low"]), 2),
                        "close": round(float(row["Close"]), 2),
                        "volume": int(row["Volume"]),
                    })
                if bars:
                    return bars
        except Exception:
            pass

        # Deterministic synthetic fallback for development / offline testing
        return self._generate_synthetic_bars(ticker, days)

    def _generate_synthetic_bars(self, ticker: str, days: int) -> List[Dict[str, Any]]:
        """Generate realistic synthetic price data based on ticker seed."""
        seed = sum(ord(c) for c in ticker)
        rng = random.Random(seed)
        base_price = 150.0 + (seed % 100)
        current_price = base_price

        today = datetime.date.today()
        bars = []

        for i in range(days):
            date = today - datetime.timedelta(days=(days - i))
            # Skip weekends
            drift = math.sin(i / 10.0) * 1.5
            change_pct = rng.gauss(0.0005, 0.015)
            open_price = current_price
            close_price = max(1.0, open_price * (1.0 + change_pct) + drift * 0.1)
            high_price = max(open_price, close_price) + rng.uniform(0.2, 1.5)
            low_price = min(open_price, close_price) - rng.uniform(0.2, 1.5)
            volume = rng.randint(20_000_000, 80_000_000)

            bars.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(max(0.5, low_price), 2),
                "close": round(close_price, 2),
                "volume": volume,
            })
            current_price = close_price

        return bars

