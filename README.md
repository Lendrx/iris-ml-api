# Iris ML API

Übungsprojekt: REST-API für einen Iris-Klassifikator (Training, Vorhersage) mit FastAPI und Docker.

## Start

```bash
cp .env.example .env   # API_KEY setzen
docker compose up --build
```

→ **http://localhost:8000** · Docs: **http://localhost:8000/docs**

`/model/train` erfordert Header `X-API-Key` (Wert aus `API_KEY` in `.env`).
