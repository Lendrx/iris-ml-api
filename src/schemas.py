# src/schemas.py
from datetime import datetime
from pydantic import BaseModel, Field

# ------------------------------------------------------------
# Anfrage-Schema
class IrisFeatures(BaseModel):
    # Hier werden die Eingabeparameter für die Iris-Klassifizierung definiert
    sepal_length: float = Field(gt=0, lt=10, description="Sepal length in cm")
    sepal_width: float = Field(gt=0, lt=10, description="Sepal width in cm")
    petal_length: float = Field(gt=0, lt=10, description="Petal length in cm")
    petal_width: float = Field(gt=0, lt=10, description="Petal width in cm")

# ------------------------------------------------------------
# Antwort-Schema
class TrainingRequest(BaseModel):
    # Hier werden die Parameter für das Trainieren des Modells definiert\
    test_size: float = Field(default=0.2, gt=0, lt=1)
    random_state: int = Field(default=42)


# ------------------------------------------------------------
# Antwort-Schema
class PredictionResponse(BaseModel):
    # Hier werden die Antwortparameter für die Iris-Klassifizierung definiert
    predicted_species: str
    confidence: float = Field(ge=0, le=1)
    model_version: str

# ------------------------------------------------------------
# Antwort-Schema für das Trainieren des Modells
class TrainingResponse(BaseModel):
    # Hier werden die Antwortparameter für das Trainieren des Modells definiert
    message: str
    accuracy: float
    samples_trained: int
    samples_tested: int
    trained_at: datetime

# ------------------------------------------------------------
# Antwort-Schema für den Status des Modells
class ModelStatusResponse(BaseModel):
    # Hier werden die Antwortparameter für den Status des Modells definiert
    is_trained: bool
    model_version: str | None = None
    trained_at: datetime | None = None
    accuracy: float | None = None

# ------------------------------------------------------------
# Antwort-Schema für den Health-Check
class HealthResponse(BaseModel):
    # Hier werden die Antwortparameter für den Health-Check definiert
    status: str
    app_name: str
