# Import der benötigten Module
from fastapi import APIRouter, HTTPException, Security
from fastapi.security import APIKeyHeader
from src.config import settings
from src.ml_model import IrisModel
from src.schemas import (
    IrisFeatures,
    ModelStatusResponse,
    PredictionResponse,
    TrainingRequest,
    TrainingResponse,
)

# ------------------------------------------------------------
# Router für die API
router = APIRouter()

# ------------------------------------------------------------
# Initialisierung des Modells
iris_model = IrisModel(model_dir=settings.model_dir)

# ------------------------------------------------------------
# API Key Authentifizierung via Header
api_key_header = APIKeyHeader(name="X-API-Key")

# ------------------------------------------------------------
# API Key Authentifizierung via Header
def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    # Hier wird die API Key Authentifizierung via Header durchgeführt
    if api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="API KEY ist ungültig")
    return api_key

# ------------------------------------------------------------
# Endpoint für das Trainieren des Modells
@router.post("/train", response_model=TrainingResponse)
def train_model(
    request: TrainingRequest,
    _api_key: str = Security(verify_api_key),
) -> TrainingResponse:
    # Hier wird das Trainieren des Modells durchgeführt
    return iris_model.train(
        test_size=request.test_size,
        random_state=request.random_state,
    )

# ------------------------------------------------------------
# Endpoint für die Vorhersage der Iris-Spezies
@router.post("/predict", response_model=PredictionResponse)
def predict_species(features: IrisFeatures) -> PredictionResponse:
   # Hier wird die Vorhersage der Iris-Spezies durchgeführt
    if not iris_model.is_trained:
        raise HTTPException(
            status_code=400,
            detail="Modell wurde noch nicht trainiert. Senden Sie einen POST-Request an /train.",
        )

    species, confidence = iris_model.predict([
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ])

    return PredictionResponse(
        predicted_species=species,
        confidence=confidence,
        model_version=iris_model.model_version,
    )

# ------------------------------------------------------------
# Endpoint für den Status des Modells
@router.get("/status", response_model=ModelStatusResponse)
def model_status() -> ModelStatusResponse:
    # Hier wird der Status des Modells durchgeführt
    return ModelStatusResponse(
        is_trained=iris_model.is_trained,
        model_version=iris_model.model_version,
        trained_at=iris_model.trained_at,
        accuracy=iris_model.accuracy,
    )
