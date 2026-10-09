import datetime
import math
import random
from typing import List, Dict, Any, Optional

from backend.src.services.db_service import DatabaseService


class DataService:
    """Service responsible for retrieving, imputing, and caching market historical data."""

    def __init__(
        self,
        provider: str = "yfinance",
        allow_synthetic: bool = False,
        db_service: Optional[DatabaseService] = None,
    ):
        self.provider = provider
        self.allow_synthetic = allow_synthetic
        self.db = db_service or DatabaseService()

    @staticmethod
    def impute_missing_data(bars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Impute missing, NaN, or non-positive values using forward-fill and high/low bounding.
        Marks any modified bar with is_imputed = True.
        """
        if not bars:
            return []

        cleaned_bars = []
        prev_close = 150.0

        for bar in bars:
            b = dict(bar)
            was_imputed = b.get("is_imputed", False)

            close_val = b.get("close")
            if close_val is None or (isinstance(close_val, float) and math.isnan(close_val)) or close_val <= 0:
                b["close"] = round(prev_close, 2)
                was_imputed = True
            else:
                prev_close = float(close_val)

            for price_key in ("open", "high", "low"):
                val = b.get(price_key)
                if val is None or (isinstance(val, float) and math.isnan(val)) or val <= 0:
                    b[price_key] = b["close"]
                    was_imputed = True

            # Enforce physical constraints: High >= max(Open, Close), Low <= min(Open, Close)
            max_body = max(b["open"], b["close"])
            min_body = min(b["open"], b["close"])
            if b["high"] < max_body:
                b["high"] = max_body
                was_imputed = True
            if b["low"] > min_body:
                b["low"] = min_body
                was_imputed = True

            vol_val = b.get("volume")
            if vol_val is None or vol_val < 0:
                b["volume"] = 10_000_000
                was_imputed = True

            b["is_imputed"] = was_imputed
            cleaned_bars.append(b)

        return cleaned_bars

    @staticmethod
    def _get_most_recent_trading_day() -> datetime.date:
        """Return the most recent completed trading day.
        If the current local time is before typical market close (16:00),
        the most recent *completed* session is the previous weekday.
        Weekends are always skipped.
        """
        now = datetime.datetime.now()
        market_close = datetime.time(16, 0)
        if now.weekday() < 5 and now.time() < market_close:
            d = (now - datetime.timedelta(days=1)).date()
        else:
            d = now.date()
        while d.weekday() >= 5:
            d -= datetime.timedelta(days=1)
        return d

    def fetch_historical_bars(
        self,
        ticker: str,
        days: int = 60,
        allow_synthetic: Optional[bool] = None,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical daily OHLCV bars for ticker:
        1. Checks local SQLite cache for fresh bars (verifying trading-day freshness & synthetic rules)
        2. If cache miss or stale, queries provider (e.g. yfinance)
        3. Imputes missing data points
        4. Caches result to SQLite
        5. Falls back to deterministic synthetic generation only if allow_synthetic is True.
        """
        clean_ticker = ticker.upper().strip()
        use_synthetic = self.allow_synthetic if allow_synthetic is None else allow_synthetic

        # Check local cache first if enabled
        if use_cache:
            try:
                cached_bars = self.db.get_cached_bars(clean_ticker, days)
                if len(cached_bars) >= days:
                    candidate_bars = cached_bars[-days:]
                    # Exclude bars marked is_synthetic when synthetic is disallowed
                    has_synthetic = any(b.get("is_synthetic") for b in candidate_bars)
                    if not (not use_synthetic and has_synthetic):
                        last_cached_date = datetime.datetime.strptime(candidate_bars[-1]["date"], "%Y-%m-%d").date()
                        most_recent_trading_day = self._get_most_recent_trading_day()
                        if last_cached_date >= most_recent_trading_day:
                            return candidate_bars
            except Exception:
                pass

        if self.provider == "yfinance":
            try:
                import yfinance as yf
                ticker_obj = yf.Ticker(clean_ticker)

                initial_calendar_days = max(int(days * 1.7) + 15, 30)
                hist = ticker_obj.history(period=f"{initial_calendar_days}d")

                if len(hist) < days:
                    if days <= 120:
                        hist = ticker_obj.history(period="1y")
                    elif days <= 250:
                        hist = ticker_obj.history(period="2y")
                    else:
                        hist = ticker_obj.history(period="5y")

                if not hist.empty:
                    bars = []
                    target_rows = hist.tail(days)
                    for idx, row in target_rows.iterrows():
                        bars.append({
                            "date": idx.strftime("%Y-%m-%d"),
                            "open": round(float(row["Open"]), 2),
                            "high": round(float(row["High"]), 2),
                            "low": round(float(row["Low"]), 2),
                            "close": round(float(row["Close"]), 2),
                            "volume": int(row["Volume"]),
                            "is_imputed": False,
                            "is_synthetic": False,
                        })
                    if bars:
                        imputed_bars = self.impute_missing_data(bars)
                        try:
                            self.db.save_bars(clean_ticker, imputed_bars)
                        except Exception:
                            pass
                        return imputed_bars
            except Exception as e:
                if not use_synthetic:
                    raise RuntimeError(f"Market data provider error for '{clean_ticker}': {str(e)}") from e

        if not use_synthetic:
            raise ValueError(f"Historical data for ticker '{clean_ticker}' not found or market data unavailable.")

        # Synthetic fallback in explicit demo/synthetic mode
        synthetic_bars = self._generate_synthetic_bars(clean_ticker, days)
        try:
            self.db.save_bars(clean_ticker, synthetic_bars)
        except Exception:
            pass
        return synthetic_bars

    def _generate_synthetic_bars(self, ticker: str, days: int) -> List[Dict[str, Any]]:
        """Generate realistic synthetic price data strictly on eligible trading days."""
        seed = sum(ord(c) for c in ticker)
        rng = random.Random(seed)
        base_price = 150.0 + (seed % 100)
        current_price = base_price

        today = datetime.date.today()
        trading_dates = []
        curr = today
        while len(trading_dates) < days:
            if curr.weekday() < 5:  # Monday to Friday
                trading_dates.append(curr)
            curr -= datetime.timedelta(days=1)
        trading_dates.reverse()

        bars = []
        for i, date in enumerate(trading_dates):
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
                "is_imputed": False,
                "is_synthetic": True,
            })
            current_price = close_price

        return bars
