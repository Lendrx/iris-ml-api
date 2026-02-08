# --- Stage 1: Abhängigkeiten bauen ---
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Nur Abhängigkeitsdateien zuerst kopieren für besseres Layer-Caching.
# Docker cached jede Schicht; wenn pyproject.toml unverändert ist,
# wird die Abhängigkeits-Schicht wiederverwendet statt neu installiert.
COPY pyproject.toml .

RUN uv sync --no-dev --no-install-project

# --- Stage 2: Laufzeit ---
FROM python:3.12-slim

WORKDIR /app

# Virtuelle Umgebung aus dem Builder-Stage übernehmen.
# Das hält das finale Image klein, da Build-Tools nicht enthalten sind.
COPY --from=builder /app/.venv /app/.venv

COPY src/ ./src/

# Virtuelle Umgebung ins PATH aufnehmen
ENV PATH="/app/.venv/bin:$PATH"

# Verzeichnis für trainierte Modelle anlegen
RUN mkdir -p /app/trained_models

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
