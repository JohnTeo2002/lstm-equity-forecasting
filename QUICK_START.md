# LSTM Equity Forecasting Pipeline — Quick Reference

## How to Run

### **Option 1: Python Module (Recommended - Always Works)**
```bash
python -m src.lstm_forecasting.pipeline --ticker AAPL --epochs 5
```

### **Option 2: Wrapper Script**
```bash
# macOS / Linux
./run.sh --ticker AAPL

# Windows
run.bat --ticker AAPL
```

### **Option 3: Console Script (After pip install)**
```bash
pip install -e .
lstm-forecast --ticker AAPL
```

---

## Common Commands

### Single Ticker (Quick Test - 5 Epochs)
```bash
python -m src.lstm_forecasting.pipeline --ticker AAPL --epochs 5
```

### Full Training (100 Epochs)
```bash
python -m src.lstm_forecasting.pipeline --ticker AAPL --epochs 100
```

### Multiple Tickers
```bash
python -m src.lstm_forecasting.pipeline --ticker AAPL --ticker MSFT --ticker D05.SI
```

### Custom Settings
```bash
python -m src.lstm_forecasting.pipeline \
  --ticker AAPL \
  --start 2015-01-01 \
  --lookback 90 \
  --epochs 150
```

---

## Complete Troubleshooting Guide

### Installation Order (CRITICAL!)

**Always install TensorFlow FIRST** before pandas/scipy to avoid NumPy conflicts:

```bash
# CORRECT ORDER:
pip install --upgrade pip setuptools wheel
pip install tensorflow-macos==2.16.2    # Install FIRST - locks numpy<2.0
pip install pyarrow                      # Parquet support
pip install pandas scikit-learn matplotlib yfinance requests beautifulsoup4 'scipy<1.13' statsmodels PyYAML tqdm arch

# WRONG ORDER (will cause NumPy conflicts):
pip install pandas scipy
pip install tensorflow-macos             # Too late - scipy already installed numpy 2.4.6
```

---

### Common Errors & Solutions

#### TOML parsing error: `tomllib.TOMLDecodeError: Cannot overwrite a value`
**Cause:** Corrupted or duplicate `pyproject.toml`

**Solution:** Re-download the repository from GitHub:
```bash
git clone https://github.com/your-username/lstm-equity-forecasting.git
cd lstm-equity-forecasting
```

---

#### NumPy version conflict: `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.4.6`
**Cause:** pandas/scipy installed NumPy 2.4.6, but TensorFlow needs NumPy <2.0

**Solution:**
```bash
# Option 1: Install TensorFlow FIRST (prevents this)
pip install tensorflow-macos==2.16.2
pip install pandas scipy                 # Now respects TensorFlow's numpy<2.0

# Option 2: If already installed wrong, fix it:
pip install 'numpy<2'
```

---

#### scipy error: `scipy 1.18.0 requires numpy<2.8,>=2.0.0, but you have numpy 1.26.4`
**Cause:** scipy 1.18 needs NumPy 2.0+, but TensorFlow locked numpy 1.26.4

**Solution:** Downgrade scipy:
```bash
pip install 'scipy<1.13'
```

---

#### `AttributeError: module 'numpy' has no attribute 'long'`
**Cause:** scipy 1.18 trying to use NumPy 2 syntax with NumPy 1.26

**Solution:**
```bash
pip install 'scipy<1.13'
```

---

#### `tensorflow-macos==2.13.0` not found
**Cause:** TensorFlow 2.13 is no longer available on PyPI

**Solution:** Use 2.16.2:
```bash
pip install tensorflow-macos==2.16.2
```

---

#### `ImportError: Unable to find a usable engine; tried using 'pyarrow', 'fastparquet'`
**Cause:** Missing Parquet support library

**Solution:** Install pyarrow:
```bash
pip install pyarrow
```

Then re-run:
```bash
python -m src.lstm_forecasting.pipeline --ticker AAPL --epochs 5
```

---

#### `FileNotFoundError: configs/default.yaml`
**Cause:** Missing configuration directory

**Solution:** Ensure you have the complete repository with all folders. The `configs/` folder is included in the repo.

If missing, create it:
```bash
mkdir -p configs
cat > configs/default.yaml << 'EOF'
data:
  tickers: ["AAPL"]
  start: "2015-01-01"
  end: null
  interval: "1d"
  cache_dir: "data/cache"

features:
  price_col: "Close"
  ema_spans: [10, 20, 50]
  rsi_window: 14
  macd: [12, 26, 9]
  bb_window: 20
  vol_window: 20

windowing:
  lookback: 60
  horizon: 1
  train_frac: 0.8
  n_cv_splits: 5

model:
  lstm_units: [64, 32]
  dropout: 0.2
  dense_units: 16
  learning_rate: 0.001
  batch_size: 32
  epochs: 100
  patience: 10

baselines:
  arima_order: [5, 1, 0]
  garch_p: 1
  garch_q: 1
  test_size: 60

output:
  artifacts_dir: "artifacts"
EOF
```

---

#### `ImportError: attempted relative import beyond top-level package`
**Cause:** Using wrong command to run the pipeline

**Solution:** Use the correct Python module syntax:
```bash
# Correct
python -m src.lstm_forecasting.pipeline --ticker AAPL

# Wrong
python src/lstm_forecasting/pipeline.py --ticker AAPL
```

---

#### `command not found: lstm-forecast`
**Cause:** Console script not installed (only works after `pip install -e .`)

**Solution:** Use Python module syntax instead:
```bash
python -m src.lstm_forecasting.pipeline --ticker AAPL
```

Or install the package:
```bash
pip install -e .
lstm-forecast --ticker AAPL
```

---

### Python Version Issues

#### Using Python 3.14
**Cause:** TensorFlow doesn't support Python 3.14 yet

**Solution:** Use Python 3.12 or 3.13:
```bash
# Create venv with Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate

# Then install as normal
pip install tensorflow-macos==2.16.2
pip install pyarrow
pip install -e .
```

---

### Complete Fresh Install (Guaranteed to Work)

If hitting multiple issues, start completely fresh:

```bash
# 1. Remove old venv
deactivate
rm -rf .venv

# 2. Create fresh venv with Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip setuptools wheel

# 4. Install TensorFlow FIRST (critical!)
pip install tensorflow-macos==2.16.2

# 5. Install all other dependencies
pip install pyarrow
pip install pandas scikit-learn matplotlib yfinance requests beautifulsoup4 'scipy<1.13' statsmodels PyYAML tqdm arch

# 6. Verify installation
python -c "import tensorflow as tf; print(f'✓ TensorFlow {tf.__version__}')"
python -c "import lstm_forecasting; print('✓ Package OK')"
python -c "import pyarrow; print('✓ PyArrow OK')"

# 7. Run!
python -m src.lstm_forecasting.pipeline --ticker AAPL --epochs 5
```

---

## Expected Output

When it's working correctly, you should see:
```
2026-06-21 18:16:47 | INFO     | __main__ | === Running pipeline for AAPL ===
2026-06-21 18:16:48 | INFO     | src.lstm_forecasting.data_ingestion | Downloading AAPL OHLCV (2015-01-01 -> today, interval=1d)
2026-06-21 18:16:50 | INFO     | src.lstm_forecasting.indicators | Feature-engineered dataset: 3650 rows, 22 columns
...
```

Results appear in `artifacts/AAPL/`:
- `summary.json` — Metrics
- `lstm_predictions.png` — Prediction plot
- `model_comparison.png` — Model comparison

---

## Configuration

Edit `configs/default.yaml`:
```yaml
data:
  tickers: ["AAPL"]
  start: "2015-01-01"

model:
  epochs: 100
  lstm_units: [64, 32]

windowing:
  lookback: 60
```

Override with CLI:
```bash
python -m src.lstm_forecasting.pipeline --ticker AAPL --epochs 50 --lookback 90
```

---

## Run Tests

```bash
pytest tests/ -v --cov=src/lstm_forecasting
```

---

## Documentation

- **README.md** – Full documentation
- **notebooks/demo.ipynb** – Jupyter walkthrough

---

**Happy forecasting!** 