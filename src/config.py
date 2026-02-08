# Import der benötigten Module
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ------------------------------------------------------------
# Settings-Klasse
class Settings(BaseSettings):
    # Hier werden die Anwendungseinstellungen aus den Umgebungsvariablen geladen
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Iris ML API"
    debug: bool = False
    model_dir: Path = Path("trained_models")
    # Must match X-API-Key header; set in .env as API_KEY=your-secret
    api_key: str = Field(default="change-me-in-production", validation_alias="API_KEY")

# ------------------------------------------------------------
# Settings-Instanz
settings = Settings()