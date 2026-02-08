# --- Stage 1: Build dependencies ---
    FROM python:3.12-slim AS builder

    COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
    
    WORKDIR /app
    
    # Copy only dependency files first for better layer caching.
    # Docker caches each layer. If pyproject.toml hasn't changed,
    # Docker reuses the cached dependency layer instead of reinstalling.
    COPY pyproject.toml .
    
    RUN uv sync --no-dev --no-install-project
    
    # --- Stage 2: Runtime ---
    FROM python:3.12-slim
    
    WORKDIR /app
    
    # Copy the virtual environment from the builder stage.
    # This keeps the final image small because build tools are not included.
    COPY --from=builder /app/.venv /app/.venv
    
    COPY src/ ./src/
    
    # Make sure the virtual environment is on the PATH
    ENV PATH="/app/.venv/bin:$PATH"
    
    # Create directory for trained models
    RUN mkdir -p /app/trained_models
    
    EXPOSE 8000
    
    CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]