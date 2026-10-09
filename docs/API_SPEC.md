# API Specification

## Base URL
```
http://localhost:8000/api/v1
```

All requests and responses use `application/json` content type unless specified otherwise.

---

## Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Service health status and uptime probe |
| `GET`  | `/stocks/{ticker}/historical` | Retrieve historical OHLCV candles |
| `POST` | `/forecast/predict` | Generate forward-looking price trajectory predictions |
| `GET`  | `/forecast/models` | List available models, parameters, and version metadata |

---

## Detailed Endpoint Definitions

### 1. Health Check
Checks backend operational status and model readiness.

- **URL**: `/health`
- **Method**: `GET`
- **Response `200 OK`**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "device": "cpu",
  "timestamp": "2026-10-08T09:00:00Z"
}
```

---

### 2. Historical Stock Data
Fetches historical market prices for a given ticker symbol.

- **URL**: `/stocks/{ticker}/historical`
- **Method**: `GET`
- **Path Parameters**:
  - `ticker` (string, required): Standard equity ticker symbol (e.g., `AAPL`, `MSFT`, `SPY`).
- **Query Parameters**:
  - `days` (integer, optional, default: `60`): Number of daily trading bars to retrieve (range: 10 to 365). Daily OHLCV bars are returned.
- **Response `200 OK`**:
```json
{
  "ticker": "AAPL",
  "count": 60,
  "data": [
    {
      "date": "2026-07-15",
      "open": 228.40,
      "high": 230.15,
      "low": 227.80,
      "close": 229.50,
      "volume": 48200300
    }
  ]
}
```
- **Error Response `404 Not Found`**:
```json
{
  "detail": "Ticker symbol 'INVALID' not found or market data unavailable."
}
```

---

### 3. Forecast Prediction
Runs the LSTM neural network model on historical sequence data to predict future prices.

- **URL**: `/forecast/predict`
- **Method**: `POST`
- **Request Body**:
```json
{
  "ticker": "AAPL",
  "lookback_window": 60,
  "horizon_days": 5,
  "include_confidence_intervals": true
}
```

- **Field Validations**:
  - `ticker`: string, 1 to 10 characters (e.g., `AAPL`, `GOOGL`).
  - `lookback_window`: integer, between 30 and 180 (default: 60).
  - `horizon_days`: integer, between 1 and 30 (default: 5).
  - `include_confidence_intervals`: boolean (default: true).

- **Response `200 OK`**:
```json
{
  "ticker": "AAPL",
  "generated_at": "2026-10-08T09:05:00Z",
  "model_version": "lstm-v2.1",
  "last_historical_close": 235.20,
  "predictions": [
    {
      "step": 1,
      "target_date": "2026-10-09",
      "predicted_price": 236.15,
      "lower_bound": 233.80,
      "upper_bound": 238.50
    },
    {
      "step": 2,
      "target_date": "2026-10-12",
      "predicted_price": 237.05,
      "lower_bound": 234.10,
      "upper_bound": 240.00
    }
  ],
  "metrics": {
    "expected_direction": "bullish",
    "projected_change_pct": 1.25
  }
}
```

---

### 4. Available Models Metadata
Lists registered model weights and checkpoint details.

- **URL**: `/forecast/models`
- **Method**: `GET`
- **Response `200 OK`**:
```json
{
  "active_model": "lstm-v2.1",
  "models": [
    {
      "id": "lstm-v2.1",
      "name": "Stacked LSTM",
      "sequence_length": 60,
      "features": ["close", "volume", "rsi", "sma"],
      "hidden_dim": 128,
      "num_layers": 2
    }
  ]
}
```

---

## Data Models (Schemas)

### `HistoricalBar`
| Field | Type | Description |
|-------|------|-------------|
| `date` | `string` (date) | Trading date (ISO-8601 `YYYY-MM-DD`) |
| `open` | `float` | Opening price in USD |
| `high` | `float` | Session high price in USD |
| `low` | `float` | Session low price in USD |
| `close` | `float` | Closing / Adjusted closing price in USD |
| `volume` | `integer` | Trading volume |

### `ForecastPoint`
| Field | Type | Description |
|-------|------|-------------|
| `step` | `integer` | Horizon step index (1-based) |
| `target_date` | `string` (date) | Projected trading date |
| `predicted_price` | `float` | Point estimate forecast |
| `lower_bound` | `float` (optional) | 95% lower confidence bound |
| `upper_bound` | `float` (optional) | 95% upper confidence bound |

