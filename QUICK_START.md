# LSTM Equity Forecasting Pipeline — Quick Reference

## 🚀 How to Run

### **Option 1: Python Module (Recommended - Always Works)**
```bash
python -m lstm_forecasting.pipeline --ticker AAPL
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

## 📋 Common Commands

### Single Ticker
```bash
python -m lstm_forecasting.pipeline --ticker AAPL
```

### Multiple Tickers
```bash
python -m lstm_forecasting.pipeline --ticker AAPL --ticker MSFT --ticker D05.SI
```

### Custom Settings
```bash
python -m lstm_forecasting.pipeline \
  --ticker AAPL \
  --start 2015-01-01 \
  --lookback 90 \
  --epochs 150
```

### Custom Configuration File
```bash
python -m lstm_forecasting.pipeline --config my_config.yaml
```

### Save Results to Different Location
```bash
python -m lstm_forecasting.pipeline --ticker AAPL --artifacts-dir ./results/
```

### See All Options
```bash
python -m lstm_forecasting.pipeline --help
```

---

## 📊 Output Location

Results will be saved in `artifacts/`:
```
artifacts/
├── AAPL/
│   ├── summary.json                 # Metrics
│   ├── lstm_predictions.png         # Prediction plot
│   └── model_comparison.png         # Model comparison chart
├── MSFT/
│   ├── summary.json
│   ├── lstm_predictions.png
│   └── model_comparison.png
└── all_tickers_summary.json         # Combined results
```

---

## ⚙️ Configuration

Edit `configs/default.yaml` for default hyperparameters:
```yaml
data:
  tickers: ["AAPL", "MSFT", "D05.SI"]
  start: "2010-01-01"

model:
  epochs: 100
  lstm_units: [64, 32]
  dropout: 0.2

windowing:
  lookback: 60
  n_cv_splits: 5
```

Override with CLI flags:
```bash
python -m lstm_forecasting.pipeline --ticker AAPL --epochs 50 --lookback 90
```

---

## 🧪 Run Tests

```bash
pytest tests/ -v --cov=src/lstm_forecasting
```

---

## 🐍 Python API

```python
from lstm_forecasting import data_ingestion, indicators, preprocessing, models, evaluate

# 1. Fetch data
raw_df = data_ingestion.fetch_ohlcv("AAPL", start="2015-01-01")

# 2. Add features
raw_df = raw_df.reset_index().rename(columns={"index": "Date"})
feature_df = indicators.add_technical_indicators(raw_df.set_index("Date"))

# 3. Split & scale
dataset = preprocessing.scale_and_window_split(feature_df, lookback=60)

# 4. Train LSTM
lstm = models.build_lstm_model((60, feature_df.shape[1]))
history = models.train_model(lstm, dataset.X_train, dataset.y_train, verbose=1)

# 5. Predict & evaluate
y_pred = models.predict(lstm, dataset.X_test)
y_pred_unscaled = dataset.target_scaler.inverse_transform(y_pred.reshape(-1, 1))
metrics = evaluate.evaluate_predictions(dataset.y_test, y_pred_unscaled)
print(metrics)
```

---

## 🆘 Troubleshooting

**Q: `command not found: lstm-forecast`**  
A: Use `python -m lstm_forecasting.pipeline` instead, or run `pip install -e .` first

**Q: `ModuleNotFoundError: No module named 'lstm_forecasting'`**  
A: Run `pip install -e ".[dev]"` from the repo root to install the package

**Q: `ImportError: No module named 'tensorflow'`**  
A: Run `pip install tensorflow` (or `tensorflow-macos` on Apple Silicon)

**Q: Tests fail**  
A: Ensure all dev dependencies installed: `pip install -e ".[dev]"`

---

## 📚 Documentation

- **README.md** – Full documentation
- **notebooks/demo.ipynb** – Jupyter walkthrough
- **configs/default.yaml** – Configuration reference

---

**Happy forecasting!** 📈