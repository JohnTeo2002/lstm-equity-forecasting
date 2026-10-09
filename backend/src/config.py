import os
from typing import List

try:
    from pydantic_settings import BaseSettings

    class Settings(BaseSettings):
        """Application settings loaded from environment variables."""

        ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
        DEBUG: bool = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")
        HOST: str = os.getenv("HOST", "0.0.0.0")
        PORT: int = int(os.getenv("PORT", "8000"))
        CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")

        MARKET_DATA_PROVIDER: str = os.getenv("MARKET_DATA_PROVIDER", "yfinance")
        DEMO_MODE: bool = os.getenv("DEMO_MODE", "false").lower() in ("true", "1", "yes")
        ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        POLYGON_API_KEY: str = os.getenv("POLYGON_API_KEY", "")

        MODEL_WEIGHTS_PATH: str = os.getenv("MODEL_WEIGHTS_PATH", "backend/weights/lstm_equity_v2.pt")
        SEQUENCE_LENGTH: int = int(os.getenv("SEQUENCE_LENGTH", "60"))
        FORECAST_DEFAULT_HORIZON: int = int(os.getenv("FORECAST_DEFAULT_HORIZON", "5"))
        TORCH_DEVICE: str = os.getenv("TORCH_DEVICE", "cpu")

        @property
        def cors_origins_list(self) -> List[str]:
            return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

        class Config:
            env_file = ".env"
            extra = "ignore"

except ImportError:
    from pydantic import BaseModel

    class Settings(BaseModel):
        """Fallback settings model when pydantic-settings is not installed."""

        ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
        DEBUG: bool = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")
        HOST: str = os.getenv("HOST", "0.0.0.0")
        PORT: int = int(os.getenv("PORT", "8000"))
        CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")

        MARKET_DATA_PROVIDER: str = os.getenv("MARKET_DATA_PROVIDER", "yfinance")
        DEMO_MODE: bool = os.getenv("DEMO_MODE", "false").lower() in ("true", "1", "yes")
        ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        POLYGON_API_KEY: str = os.getenv("POLYGON_API_KEY", "")

        MODEL_WEIGHTS_PATH: str = os.getenv("MODEL_WEIGHTS_PATH", "backend/weights/lstm_equity_v2.pt")
        SEQUENCE_LENGTH: int = int(os.getenv("SEQUENCE_LENGTH", "60"))
        FORECAST_DEFAULT_HORIZON: int = int(os.getenv("FORECAST_DEFAULT_HORIZON", "5"))
        TORCH_DEVICE: str = os.getenv("TORCH_DEVICE", "cpu")

        @property
        def cors_origins_list(self) -> List[str]:
            return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


settings = Settings()
