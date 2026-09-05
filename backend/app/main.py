"""
app/main.py

FastAPI application entry point for PromptOptAI Analysis API.

Run with:
    uvicorn app.main:app --reload --port 8000

Swagger docs available at:
    http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import logging.config
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes.analyze import router as analyze_router
from app.api.routes.semantic import router as semantic_router
from app.core.config import settings


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "%s v%s starting — frontend origin: %s, max prompt: %d chars",
        settings.app_name,
        settings.app_version,
        settings.frontend_url,
        settings.max_prompt_length,
    )
    yield
    logger.info("%s shutting down.", settings.app_name)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Deterministic / local prompt analysis API for PromptOptAI.  "
        "Module 1: User Prompt ingestion.  "
        "Module 2: Local Prompt Analysis (token count, text stats, repetition detection, structural analysis).  "
        "Module 3: Semantic Analysis (intent, context, requirements, constraints, ambiguity).  "
        "No LLM API is called."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
# Allow the Next.js frontend origin during local development.
# The allowed origin is configured via FRONTEND_URL in .env.
# Do not use allow_origins=["*"] in production with credentials.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)


# ---------------------------------------------------------------------------
# Global exception handler — never expose internal tracebacks to clients
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"success": False, "error": {"code": "INTERNAL_ERROR", "message": "An unexpected error occurred."}},
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    summary="Health check",
    description="Returns HTTP 200 and `{'status': 'ok'}` if the server is running.",
    tags=["System"],
)
async def health_check() -> dict:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# API routers
# ---------------------------------------------------------------------------

app.include_router(analyze_router, prefix="/api", tags=["Analysis"])
app.include_router(semantic_router, prefix="/api", tags=["Semantic Analysis"])

