"""
Automated data pipeline for historical OHLCV time-series data.

Two sources are used:

1. "requests" + "BeautifulSoup" to scrape constituent ticker lists
2. The "yfinance" wrapper around Yahoo Finance's API to download the
   actual decades-long OHLCV time series for each ticker

All downloaded frames are cached to disk as Parquet/CSV so repeated
pipeline runs do not re-hit the network.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

from .logging_utils import get_logger

logger = get_logger(__name__)

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
STI_WIKI_URL = "https://en.wikipedia.org/wiki/Straits_Times_Index"

# A small, stable fallback list used if scraping fails (e.g. offline,
# Wikipedia layout change, or network policy blocking the request).
FALLBACK_SP500_SAMPLE = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM"]
FALLBACK_STI_SAMPLE = ["D05.SI", "O39.SI", "U11.SI", "Z74.SI", "C38U.SI"]

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; lstm-equity-forecasting/0.1; "
        "+https://github.com/your-username/lstm-equity-forecasting)"
    )
}


@dataclass
class IngestionConfig:
    """Configuration for a single ticker's data pull."""

    ticker: str
    start: str = "2000-01-01"
    end: str | None = None  # None => up to today
    interval: str = "1d"
    cache_dir: Path = Path("data/cache")


def scrape_sp500_tickers(timeout: int = 10) -> list[str]:
    """Scrape the current S&P 500 constituent tickers from Wikipedia.

    Uses ``requests`` to fetch the page and ``BeautifulSoup`` to parse
    the constituents table, satisfying the "Requests and BeautifulSoup"
    ingestion requirement for ticker-universe discovery.

    Falls back to a small static sample list on any failure so the
    rest of the pipeline can still run end-to-end (e.g. in CI, or
    offline development).
    """
    try:
        resp = requests.get(SP500_WIKI_URL, headers=DEFAULT_HEADERS, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"id": "constituents"})
        if table is None:
            raise ValueError("Could not locate constituents table on page")

        tickers: list[str] = []
        for row in table.find_all("tr")[1:]:
            cells = row.find_all("td")
            if not cells:
                continue
            symbol = cells[0].get_text(strip=True).replace(".", "-")
            tickers.append(symbol)

        if not tickers:
            raise ValueError("Parsed zero tickers from constituents table")

        logger.info("Scraped %d S&P 500 tickers from Wikipedia", len(tickers))
        return tickers

    except Exception as exc:  # noqa: BLE001 - log and degrade gracefully
        logger.warning(
            "S&P 500 scrape failed (%s); falling back to static sample list", exc
        )
        return FALLBACK_SP500_SAMPLE


def scrape_sti_tickers(timeout: int = 10) -> list[str]:
    """Scrape Straits Times Index (STI) constituent tickers from Wikipedia.

    SGX tickers are mapped to the ``.SI`` suffix expected by Yahoo
    Finance. As with :func:`scrape_sp500_tickers`, this degrades to a
    static fallback list if the page structure changes or the request
    fails, rather than crashing the whole pipeline.
    """
    try:
        resp = requests.get(STI_WIKI_URL, headers=DEFAULT_HEADERS, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        tickers: list[str] = []
        for table in soup.find_all("table", {"class": "wikitable"}):
            headers_text = [
                th.get_text(strip=True).lower() for th in table.find_all("th")
            ]
            if not any(
                "code" in h or "ticker" in h or "sgx" in h for h in headers_text
            ):
                continue
            for row in table.find_all("tr")[1:]:
                cells = row.find_all("td")
                if len(cells) < 2:
                    continue
                code = cells[-1].get_text(strip=True)
                if code:
                    tickers.append(f"{code}.SI" if not code.endswith(".SI") else code)
            if tickers:
                break

        if not tickers:
            raise ValueError("Parsed zero tickers from STI page")

        logger.info("Scraped %d STI tickers from Wikipedia", len(tickers))
        return tickers

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "STI scrape failed (%s); falling back to static sample list", exc
        )
        return FALLBACK_STI_SAMPLE


def _cache_path(cache_dir: Path, ticker: str, interval: str) -> Path:
    safe_ticker = ticker.replace("/", "-")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{safe_ticker}_{interval}.parquet"


def fetch_ohlcv(
    ticker: str,
    start: str = "2000-01-01",
    end: str | None = None,
    interval: str = "1d",
    cache_dir: str | Path = "data/cache",
    use_cache: bool = True,
    retries: int = 3,
    retry_backoff: float = 2.0,
) -> pd.DataFrame:
    """Download historical OHLCV data for a single ticker via Yahoo Finance.

    Results are cached to a local Parquet file under ``cache_dir`` so
    subsequent pipeline runs avoid redundant network calls. Network
    calls are retried with exponential backoff to absorb transient
    rate-limit or connectivity errors.

    Returns a DataFrame indexed by date with columns
    ``[Open, High, Low, Close, Volume]``.
    """
    cache_dir = Path(cache_dir)
    cache_file = _cache_path(cache_dir, ticker, interval)

    if use_cache and cache_file.exists():
        logger.info("Loading cached OHLCV for %s from %s", ticker, cache_file)
        df = pd.read_parquet(cache_file)
        df.index = pd.to_datetime(df.index)
        return df

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            logger.info(
                "Downloading %s OHLCV (%s -> %s, interval=%s) [attempt %d/%d]",
                ticker,
                start,
                end or "today",
                interval,
                attempt,
                retries,
            )
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
            if raw is None or raw.empty:
                raise ValueError(f"yfinance returned no data for {ticker}")

            # yfinance can return a MultiIndex column structure for some
            # interval/auto_adjust combinations; flatten defensively.
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df = df.dropna(how="all")
            df.index.name = "Date"

            if use_cache:
                df.to_parquet(cache_file)
                logger.info("Cached %s OHLCV to %s", ticker, cache_file)

            return df

        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "Download attempt %d for %s failed: %s", attempt, ticker, exc
            )
            if attempt < retries:
                time.sleep(retry_backoff**attempt)

    raise RuntimeError(
        f"Failed to download OHLCV for {ticker} after {retries} attempts"
    ) from last_exc


def fetch_many(
    tickers: Iterable[str],
    start: str = "2000-01-01",
    end: str | None = None,
    interval: str = "1d",
    cache_dir: str | Path = "data/cache",
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV data for multiple tickers, skipping ones that fail.

    Returns a dict mapping ticker -> DataFrame for every ticker that
    downloaded successfully; failures are logged and excluded rather
    than aborting the whole batch.
    """
    results: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            results[ticker] = fetch_ohlcv(
                ticker,
                start=start,
                end=end,
                interval=interval,
                cache_dir=cache_dir,
                use_cache=use_cache,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Skipping %s: %s", ticker, exc)
    logger.info("Successfully fetched %d/%d tickers", len(results), len(list(tickers)))
    return results
