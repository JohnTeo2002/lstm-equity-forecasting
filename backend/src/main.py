from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.src.api.routes import router as api_router
from backend.src.config import settings

app = FastAPI(
    title="LSTM Equity Forecasting API",
    description="REST API service for time-series equity price forecasting with Long Short-Term Memory models.",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes with prefix
app.include_router(api_router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "LSTM Equity Forecasting API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.src.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)

