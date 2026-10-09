# LSTM Equity Forecasting (v2)

An end-to-end quantitative analytics platform and deep learning engine designed to model and forecast equity price trajectories using Long Short-Term Memory (LSTM) recurrent neural networks.

---

## 📌 Problem Statement

Predicting financial asset prices is notoriously challenging due to market volatility, non-stationary price dynamics, and multi-scale temporal dependencies. Classical linear models (ARIMA, GARCH) often struggle with non-linear patterns across multi-feature temporal inputs.

This platform leverages stacked **Long Short-Term Memory (LSTM)** neural network architectures to:
1. Capture sequential dependencies across historical Open, High, Low, Close, Volume (OHLCV) records.
2. Ingest engineered momentum indicators (RSI, Moving Averages, Volatility).
3. Project multi-step forward-looking price targets with dynamic confidence intervals.
4. Provide a low-latency REST API and an intuitive, interactive charting frontend for real-time inference and analysis.

---

## 🛠️ Tech Stack

### Backend
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python 3.12+)
- **Deep Learning**: [PyTorch](https://pytorch.org/) (Stacked LSTM neural network)
- **Data & Numerical Processing**: [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)
- **Data Ingestion**: Yahoo Finance (`yfinance`) / Alpha Vantage / Polygon
- **Data Validation & Settings**: [Pydantic v2](https://docs.pydantic.dev/) & `pydantic-settings`
- **Testing**: [Pytest](https://docs.pytest.org/), `httpx`

### Frontend
- **Framework**: [React 18](https://react.dev/) + [TypeScript](https://www.typescriptlang.org/)
- **Build Tool**: [Vite](https://vitejs.dev/)
- **Icons**: [Lucide React](https://lucide.dev/)
- **Visualization**: Responsive SVG time-series price & trajectory charts

### Quality & Governance
- **Code Review**: Automated AI code review via [CodeRabbit](https://coderabbit.ai/) (`.coderabbit.yaml`)
- **Version Control & CI**: GitHub pull request template (`.github/PULL_REQUEST_TEMPLATE.md`)

---

## 📂 Repository Structure

```
lstm-equity-forecasting/
├── .github/
│   └── PULL_REQUEST_TEMPLATE.md   # Standard PR submission guidelines
├── docs/
│   ├── ARCHITECTURE.md            # High-level system design & diagrams
│   └── API_SPEC.md                # Route definitions & data models
├── backend/                       # Python FastAPI & PyTorch service
│   ├── src/
│   │   ├── api/                   # REST API routes & controllers
│   │   ├── services/              # Ingestion, feature engineering, forecasting
│   │   ├── models/                # PyTorch LSTM network & Pydantic schemas
│   │   ├── config.py              # Environment configuration loader
│   │   └── main.py                # FastAPI application entrypoint
│   ├── tests/                     # Unit and integration test suite
│   └── requirements.txt           # Python backend dependencies
├── frontend/                      # React / TypeScript web application
│   ├── src/                       # Components, charting & API clients
│   ├── package.json               # NPM scripts and dependencies
│   └── vite.config.ts             # Vite proxy and build configuration
├── .env.example                   # Template for environment variables
├── .coderabbit.yaml               # CodeRabbit configuration
└── README.md                      # Problem statement, setup instructions, tech stack
```

---

## 🚀 Quickstart & Setup Instructions

### 1. Prerequisites
- **Python**: `>= 3.10` (Python 3.12 recommended)
- **Node.js**: `>= 18.0` (v22+ supported)
- **Git**

### 2. Environment Configuration
Clone the repository and copy the environment configuration template:
```bash
cp .env.example .env
```
Customize any environment variables in `.env` as required (e.g., API keys, port settings).

---

### 3. Backend Setup

1. Create and activate your virtual environment:
   ```bash
   # Create virtual environment
   python3 -m venv .venv

   # On macOS / Linux
   source .venv/bin/activate
   ```
2. Install Python dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
3. Run the backend server:
   ```bash
   uvicorn backend.src.main:app --reload --host 0.0.0.0 --port 8000
   ```
4. Access interactive API documentation at:
   - **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

### 4. Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install frontend dependencies:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```
4. Open [http://localhost:5173](http://localhost:5173) in your browser.

---

### 5. Running Tests

To run the backend test suite:
```bash
pytest backend/tests
```

---

## 📖 Documentation
- [System Architecture](docs/ARCHITECTURE.md)
- [API Specification](docs/API_SPEC.md)
