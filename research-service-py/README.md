# Research Service (Python/FastAPI)

FastAPI service providing Research Mode: `/research/*` APIs, a web research pipeline, and CRM integration.

## Quickstart

```bash
cd research-service-py
cp .env.example .env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
```

## APIs
- POST /research/run
- POST /research/save
- POST /research/ideas-to-plan
- POST /research/subscribe

See `openapi/research.yaml`.
