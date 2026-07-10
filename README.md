# LSTM-Based Equity Forecasting Pipeline

Production-grade deep learning system for short-term price prediction combining stacked LSTMs, ensemble methods, and rigorous statistical benchmarking with full test coverage.

## Core Innovation

Solves **forecasting lag** (the gap between prediction and true signal) by:
- **Stacked LSTM architecture** capturing non-linear temporal dependencies
- **Ensemble weighting** (60% LSTM + 40% ARIMA) balancing deep learning with mean-reversion
- **Walk-forward CV** preventing look-ahead bias on time-series data
- **Production safeguards**: error handling, graceful degradation, comprehensive testing

---

## Key Features

| Category | Capability |
|----------|-----------|
| **Data Pipeline** | Automated OHLCV ingestion (yfinance) + exponential backoff + Parquet caching |
| **Feature Engineering** | EMA, RSI, MACD, Bollinger Bands, rolling volatility, log returns (15 indicators) |
| **ML Architecture** | Stacked LSTM (64→32 units) + ARIMA(5,1,0) + GARCH(1,1) baseline + weighted ensemble |
| **Validation** | TimeSeriesSplit walk-forward CV, expanding train/test windows, leakage-safe scaling |
| **Evaluation** | RMSE, MAE, MAPE, lag diagnostics (identifies overfitting to random walk) |
| **Production** | YAML config-driven, CLI + Python API, 56 unit tests, PEP 621 packaging, GitHub Actions CI |

---

## Quick Start

### Installation

```bash
git clone https://github.com/your-username/lstm-equity-forecasting
cd lstm-equity-forecasting
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Run Pipeline

```bash
# Single ticker
lstm-forecast --ticker AAPL

# Multiple tickers with custom hyperparameters
lstm-forecast --ticker AAPL --ticker MSFT --epochs 150 --lookback 90
```

### Programmatic Usage

```python
from lstm_forecasting.pipeline import run_for_ticker, load_config

cfg = load_config("configs/default.yaml")
summary = run_for_ticker("AAPL", cfg, Path("artifacts"))

print(f"LSTM RMSE: {summary['lstm_cv_avg_rmse']:.5f}")
print(f"ARIMA RMSE: {summary['arima']['rmse']:.5f}")
```

---

## Architecture

### Data Ingestion
- **yfinance** downloads OHLCV with 3-attempt exponential backoff
- **BeautifulSoup** scrapes S&P 500 & STI constituent lists
- **Parquet caching** avoids redundant downloads

### Feature Engineering
Computes 15 technical indicators on closing prices, fit only on training splits (prevents look-ahead bias):
- Momentum: EMA (10/20/50), RSI, MACD
- Volatility: Bollinger Bands, rolling std
- Returns: Log returns, daily changes

### Model Ensemble

```
Ensemble = 0.6 * LSTM + 0.4 * ARIMA
```

**Why ensemble?**
- LSTM captures non-linear patterns, excels on 3+ day horizons
- ARIMA dominates mean-reverting 1-day regimes
- Weighted combination reduces individual model risk (standard quant practice)
- Configurable weights enable A/B testing across market regimes

### Validation Strategy
- **TimeSeriesSplit**: Expanding training window, rolling test window
- **Walk-forward CV**: Respects temporal ordering (no future leakage)
- **Lag diagnostics**: Identifies if model merely echoes history (lag ≥ 1)

---

## Results (Illustrative)

**AAPL, 10-year lookback, 60-day window, 1-day horizon:**

| Model | RMSE | MAE | MAPE | Lag |
|-------|------|-----|------|-----|
| LSTM (CV avg) | 2.35 | 1.89 | 0.45% | **0** ✓ |
| ARIMA(5,1,0) | 3.12 | 2.41 | 0.61% | 1 |
| GARCH(1,1) | 4.27 | 3.56 | 0.89% | 2 |

**Key insight**: LSTM lag = 0 indicates genuine predictive signal; ARIMA lag = 1 shows it's overfitting to random walk.

---

## Repository Structure

```
lstm-equity-forecasting/
├── src/lstm_forecasting/              # Main package
│   ├── data_ingestion.py              # yfinance, ticker scraping, caching
│   ├── indicators.py                  # EMA, RSI, MACD, Bollinger Bands, etc.
│   ├── preprocessing.py               # Scaling, windowing, TimeSeriesSplit
│   ├── models.py                      # LSTM architecture & training
│   ├── baselines.py                   # ARIMA/GARCH statistical benchmarks
│   ├── evaluate.py                    # Metrics, lag diagnostics, plots
│   └── pipeline.py                    # CLI orchestration
├── tests/                             # 56 unit tests (51 core + 5 ensemble)
├── configs/default.yaml               # Hyperparameters & ensemble weights
├── data/                              # Cache, raw, processed (git-ignored)
└── artifacts/                         # Results & plots (git-ignored)
```

---

## Testing & Quality

```bash
pytest tests/ -v --cov=src/lstm_forecasting
```

**Coverage**: Data ingestion, indicators, preprocessing, LSTM training, ARIMA/GARCH, evaluation, ensemble logic.

**CI/CD**: GitHub Actions validates on every push (pytest, linting).

---

## Design Decisions

### Why TimeSeriesSplit over RandomSplit?
Random shuffling violates temporal order, introducing look-ahead bias. TimeSeriesSplit ensures training always precedes testing.

### Why Ensemble?
Quant firms require redundancy; single-model systems fail in regime shifts. Ensemble (60/40 LSTM/ARIMA) balances momentum-capture with mean-reversion.

### Why Lag Diagnostics?
1-day horizon predictions easily overfit to random-walk structure. Lag = 0 confirms the model captures genuine signal.

### Why YAML Config?
Enables rapid A/B testing without code changes; hyperparameter sweeps stay reproducible and auditable.

---

## Limitations

- **Single-step horizon**: Predicts 1-day close only (multi-step requires seq2seq)
- **Price-only features**: Excludes sentiment, macro data, earnings surprises
- **No transaction costs**: Real trading incurs slippage, fees, spreads
- **Stationarity assumptions**: ARIMA/GARCH may fail in crypto/extreme volatility regimes
- **1-day edge fragile**: Close to random-walk limit; out-of-sample alpha requires robust backtesting

---

## Dependencies

- Python 3.10+
- TensorFlow/Keras (deep learning)
- scikit-learn (preprocessing, metrics)
- statsmodels (ARIMA, GARCH)
- yfinance (market data)
- pandas, NumPy, Matplotlib (data/viz)

---

## License & Disclaimer

**This software is provided for educational and research purposes only.** Financial forecasting is inherently uncertain; past performance does not guarantee future results. Do not use this code for actual trading without thorough backtesting, risk management, and professional financial advice. The authors assume no liability for trading losses or incorrect predictions.