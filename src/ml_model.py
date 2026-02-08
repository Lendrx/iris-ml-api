from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.schemas import TrainingResponse

# ------------------------------------------------------------
# IrisModel-Klasse
class IrisModel:
    # Hier wird die Iris-Klassifizierung des Modells verwaltet
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.pipeline: Pipeline | None = None
        self.target_names: list[str] = []
        self.model_version: str | None = None
        self.trained_at: datetime | None = None
        self.accuracy: float | None = None

        self._model_path = self.model_dir / "iris_model.joblib"
        self._metadata_path = self.model_dir / "iris_metadata.joblib"

        self._try_load_existing()

    @property
    def is_trained(self) -> bool:
        return self.pipeline is not None

    def train(self, test_size: float = 0.2, random_state: int = 42) -> TrainingResponse:
        """Trainiert das Modell auf der Iris-Datenmenge.

        Diese Methode führt die vollständige Trainingspipeline aus:
        1. Lädt die Daten
        2. Teilt die Daten in Trainings- und Testsets auf
        3. Baut und passt eine scikit-learn Pipeline an
        4. Evaluati on der Testset
        5. Persistiert das trainierten Modell auf die Festplatte
        """
        iris = load_iris()
        x_train, x_test, y_train, y_test = train_test_split(
            iris.data,
            iris.target,
            test_size=test_size,
            random_state=random_state,
            stratify=iris.target,
        )

        # A Pipeline chains preprocessing and model into one unit.
        # This ensures the same preprocessing is applied during training
        # and prediction, which prevents subtle bugs.
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
            )),
        ])

        self.pipeline.fit(x_train, y_train)

        accuracy = self.pipeline.score(x_test, y_test)
        self.target_names = list(iris.target_names)
        self.accuracy = round(accuracy, 4)
        self.trained_at = datetime.now(timezone.utc)
        self.model_version = self.trained_at.strftime("%Y%m%d-%H%M%S")

        self._save()

        return TrainingResponse(
            message="Model trained successfully",
            accuracy=self.accuracy,
            samples_trained=len(x_train),
            samples_tested=len(x_test),
            trained_at=self.trained_at,
        )

    def predict(self, features: list[float]) -> tuple[str, float]:
        """Vorhersage der Iris-Spezies für gegebene Merkmale.

        Diese Methode führt die Vorhersage der Iris-Spezies für gegebene Merkmale durch:
        1. Überprüft, ob das Modell trainiert ist
        2. Wandelt die Merkmale in ein numpy-Array um
        3. Führt die Vorhersage durch
        4. Gibt die Vorhersage und die Konfidenzscore zurück
        """
        if not self.is_trained:
            raise ValueError("Modell wurde noch nicht trainiert. Rufen Sie /train auf.")

        feature_array = np.array(features).reshape(1, -1)
        prediction = self.pipeline.predict(feature_array)[0]
        probabilities = self.pipeline.predict_proba(feature_array)[0]
        confidence = float(probabilities[prediction])

        species = self.target_names[prediction]

        return species, round(confidence, 4)

    def _save(self) -> None:
        """Persist model and metadata to disk."""
        joblib.dump(self.pipeline, self._model_path)
        joblib.dump(
            {
                "target_names": self.target_names,
                "model_version": self.model_version,
                "trained_at": self.trained_at,
                "accuracy": self.accuracy,
            },
            self._metadata_path,
        )

    def _try_load_existing(self) -> None:
        # Hier wird versucht, ein zuvor trainiertes Modell zu laden, wenn es verfügbar ist
        if not self._model_path.exists():
            return

        self.pipeline = joblib.load(self._model_path)
        metadata = joblib.load(self._metadata_path)

        self.target_names = metadata["target_names"]
        self.model_version = metadata["model_version"]
        self.trained_at = metadata["trained_at"]
        self.accuracy = metadata["accuracy"]
