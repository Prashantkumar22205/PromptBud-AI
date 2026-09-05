# PromptOptAI

> An Intelligent Prompt Optimization and Evaluation Framework for Large Language Models.

---

## Project Structure

```
PromptOptAI/
├── frontend/        ← Next.js 16 + React 19 + TypeScript
└── backend/         ← Python 3.11 + FastAPI (local prompt analysis)
```

---

## Frontend

Next.js application — existing PromptBud-AI UI.

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
```

---

## Backend

Python FastAPI — deterministic local prompt analysis (no LLM calls).

```bash
cd backend
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

- API:     http://localhost:8000
- Swagger: http://localhost:8000/docs
- Health:  http://localhost:8000/health

---

## API

| Method | URL | Description |
|--------|-----|-------------|
| GET  | `/health` | Health check |
| POST | `/api/analyze` | Local prompt analysis (Module 1 + 2) |

See [`backend/README.md`](backend/README.md) for full documentation.
