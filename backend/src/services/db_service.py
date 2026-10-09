import json
import os
import sqlite3
import datetime
from typing import List, Dict, Any, Optional

DEFAULT_DB_PATH = os.getenv("SQLITE_DB_PATH", "equity_cache.db")


class DatabaseService:
    """Embedded SQLite database service for local caching and data persistence."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_db(self):
        """Initialize database schema with tables and indexes."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tickers (
                    ticker TEXT PRIMARY KEY,
                    name TEXT,
                    currency TEXT DEFAULT 'USD',
                    last_updated TIMESTAMP
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS historical_bars (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    is_imputed INTEGER DEFAULT 0,
                    is_synthetic INTEGER DEFAULT 0,
                    UNIQUE(ticker, date)
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_historical_ticker_date 
                ON historical_bars(ticker, date);
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS forecast_runs (
                    run_id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    generated_at TIMESTAMP NOT NULL,
                    model_version TEXT NOT NULL,
                    horizon_days INTEGER NOT NULL,
                    last_close REAL NOT NULL,
                    predictions_json TEXT NOT NULL
                );
            """)

    def get_cached_bars(self, ticker: str, days: int) -> List[Dict[str, Any]]:
        """Retrieve cached historical bars for ticker ordered chronologically."""
        clean_ticker = ticker.upper().strip()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT date, open, high, low, close, volume, is_imputed, is_synthetic
                FROM historical_bars
                WHERE ticker = ?
                ORDER BY date DESC
                LIMIT ?
                """,
                (clean_ticker, days),
            )
            rows = cursor.fetchall()
            if not rows:
                return []
            
            # Return in chronological order
            results = [
                {
                    "date": row["date"],
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                    "is_imputed": bool(row["is_imputed"]),
                    "is_synthetic": bool(row["is_synthetic"]),
                }
                for row in reversed(rows)
            ]
            return results

    def save_bars(self, ticker: str, bars: List[Dict[str, Any]]) -> None:
        """Upsert historical bars into the database cache."""
        clean_ticker = ticker.upper().strip()
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO tickers (ticker, name, currency, last_updated)
                VALUES (?, ?, 'USD', ?)
                ON CONFLICT(ticker) DO UPDATE SET last_updated = excluded.last_updated
                """,
                (clean_ticker, clean_ticker, now),
            )

            records = [
                (
                    clean_ticker,
                    bar["date"],
                    bar["open"],
                    bar["high"],
                    bar["low"],
                    bar["close"],
                    bar["volume"],
                    1 if bar.get("is_imputed") else 0,
                    1 if bar.get("is_synthetic") else 0,
                )
                for bar in bars
            ]

            conn.executemany(
                """
                INSERT INTO historical_bars (ticker, date, open, high, low, close, volume, is_imputed, is_synthetic)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, date) DO UPDATE SET
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    volume = excluded.volume,
                    is_imputed = excluded.is_imputed,
                    is_synthetic = excluded.is_synthetic
                """,
                records,
            )

    def save_forecast(
        self,
        run_id: str,
        ticker: str,
        model_version: str,
        horizon_days: int,
        last_close: float,
        predictions: List[Dict[str, Any]],
    ) -> None:
        """Persist forecast run output to database."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO forecast_runs 
                (run_id, ticker, generated_at, model_version, horizon_days, last_close, predictions_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    ticker.upper().strip(),
                    now,
                    model_version,
                    horizon_days,
                    last_close,
                    json.dumps(predictions),
                ),
            )

