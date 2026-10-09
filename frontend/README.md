# LSTM Equity Forecasting - Frontend

Interactive React & TypeScript web application for equity trend visualization and multi-step LSTM forecast trajectory analysis.

## Features
- Real-time ticker query and interactive parameter controls (lookback sequence window, prediction horizon).
- Metrics summary cards (latest close, predicted target, expected net return, directional outlook).
- Responsive time-series chart showing historical price data alongside LSTM predicted projections.

## Development Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The application runs on `http://localhost:5173` and proxies API requests to `http://localhost:8000`.

