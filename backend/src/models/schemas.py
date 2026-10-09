from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    device: str = "cpu"
    timestamp: str


class HistoricalBar(BaseModel):
    date: str = Field(..., description="Trading date in ISO format YYYY-MM-DD")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Closing price")
    volume: int = Field(..., description="Trading volume")
    is_imputed: Optional[bool] = Field(default=False, description="Flag indicating if missing bar values were imputed")
    is_synthetic: Optional[bool] = Field(default=False, description="Flag indicating if price data was synthetically generated")


class HistoricalDataResponse(BaseModel):
    ticker: str
    count: int
    data: List[HistoricalBar]


class ForecastRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10, description="Equity ticker symbol (e.g., AAPL)")
    lookback_window: int = Field(default=60, ge=30, le=180, description="Number of historical days for sequence window")
    horizon_days: int = Field(default=5, ge=1, le=30, description="Number of future days to forecast")
    include_confidence_intervals: bool = Field(default=True, description="Compute lower and upper confidence bands")


class ForecastPoint(BaseModel):
    step: int = Field(..., description="Forecast step index (1-based)")
    target_date: str = Field(..., description="Forecasted trading date")
    predicted_price: float = Field(..., description="Predicted closing price")
    lower_bound: Optional[float] = Field(default=None, description="Lower confidence interval")
    upper_bound: Optional[float] = Field(default=None, description="Upper confidence interval")


class ForecastResponse(BaseModel):
    ticker: str
    generated_at: str
    model_version: str
    last_historical_close: float
    predictions: List[ForecastPoint]
    metrics: Dict[str, Any] = Field(default_factory=dict)


class ModelMeta(BaseModel):
    id: str
    name: str
    sequence_length: int
    features: List[str]
    hidden_dim: int
    num_layers: int


class ModelInfoResponse(BaseModel):
    active_model: str
    models: List[ModelMeta]

