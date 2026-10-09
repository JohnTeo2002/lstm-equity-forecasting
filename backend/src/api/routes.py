import datetime
from fastapi import APIRouter, HTTPException, Query, status
from backend.src.models.schemas import (
    HealthResponse,
    HistoricalDataResponse,
    HistoricalBar,
    ForecastRequest,
    ForecastResponse,
    ModelInfoResponse,
    ModelMeta,
)
from backend.src.services.data_service import DataService
from backend.src.services.forecast_service import ForecastService
from backend.src.config import settings

router = APIRouter()
data_service = DataService(
    provider=settings.MARKET_DATA_PROVIDER,
    allow_synthetic=settings.DEMO_MODE,
)
forecast_service = ForecastService(
    data_service=data_service,
    weights_path=settings.MODEL_WEIGHTS_PATH,
)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check probe",
    tags=["System"],
)
async def get_health() -> HealthResponse:
    """Return system operational status and timestamp."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        device="cpu",
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


@router.get(
    "/stocks/{ticker}/historical",
    response_model=HistoricalDataResponse,
    summary="Fetch historical market data",
    tags=["Market Data"],
)
def get_historical_stock_data(
    ticker: str,
    days: int = Query(default=60, ge=10, le=365, description="Number of trading days"),
) -> HistoricalDataResponse:
    """Retrieve historical daily OHLCV bars for the specified ticker. Runs synchronously in thread pool."""
    clean_ticker = ticker.upper().strip()
    try:
        bars = data_service.fetch_historical_bars(clean_ticker, days=days)
        if not bars:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Historical data for ticker '{clean_ticker}' not found.",
            )
        return HistoricalDataResponse(
            ticker=clean_ticker,
            count=len(bars),
            data=[HistoricalBar(**b) for b in bars],
        )
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(ve),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching data: {str(e)}",
        )


@router.post(
    "/forecast/predict",
    response_model=ForecastResponse,
    summary="Generate price forecast",
    tags=["Forecasting"],
)
def generate_prediction(request: ForecastRequest) -> ForecastResponse:
    """Run model inference to generate forward price predictions. Runs synchronously in thread pool."""
    try:
        result = forecast_service.generate_forecast(
            ticker=request.ticker,
            lookback_window=request.lookback_window,
            horizon_days=request.horizon_days,
            include_confidence_intervals=request.include_confidence_intervals,
        )
        return ForecastResponse(**result)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecast failure: {str(e)}",
        )


@router.get(
    "/forecast/models",
    response_model=ModelInfoResponse,
    summary="List registered models",
    tags=["Forecasting"],
)
def list_models() -> ModelInfoResponse:
    """Retrieve metadata of available forecasting models."""
    is_lstm_ready = forecast_service.is_lstm_available()
    active_model = "lstm-v2.1" if is_lstm_ready else "non-lstm-heuristic-v1"

    models = [
        ModelMeta(
            id="lstm-v2.1",
            name="Stacked LSTM",
            sequence_length=60,
            features=["close", "volume", "rsi", "sma"],
            hidden_dim=128,
            num_layers=2,
        )
    ]
    if not is_lstm_ready:
        models.append(
            ModelMeta(
                id="non-lstm-heuristic-v1",
                name="Heuristic Extrapolation Baseline",
                sequence_length=60,
                features=["close", "volume", "rsi", "sma"],
                hidden_dim=0,
                num_layers=0,
            )
        )

    return ModelInfoResponse(
        active_model=active_model,
        models=models,
    )
