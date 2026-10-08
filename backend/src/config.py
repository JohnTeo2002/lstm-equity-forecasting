from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    MARKET_DATA_PROVIDER: str = "yfinance"
    ALPHA_VANTAGE_API_KEY: str = ""
    POLYGON_API_KEY: str = ""

    MODEL_WEIGHTS_PATH: str = "backend/weights/lstm_equity_v2.pt"
    SEQUENCE_LENGTH: int = 60
    FORECAST_DEFAULT_HORIZON: int = 5
    TORCH_DEVICE: str = "cpu"

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

