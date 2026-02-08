# Import der benötigten Module
from fastapi import FastAPI
from src.config import settings
from src.router import router
from src.schemas import HealthResponse

# ------------------------------------------------------------
# FastAPI-Anwendung
app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="A simple ML API that classifies Iris flowers. "
    "Built as a learning exercise for ML deployment patterns.",
)

# ------------------------------------------------------------
# Router für die API
app.include_router(router, prefix="/model", tags=["Model"])

# ------------------------------------------------------------
# Health-Check-Endpoint
@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    # Hier wird der Health-Check durchgeführt
    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
    )