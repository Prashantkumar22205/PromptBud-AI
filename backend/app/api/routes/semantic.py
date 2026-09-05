"""api/routes/semantic.py

POST /api/semantic-analyze — Module 3 (Semantic Analysis)

Security rules applied here:
  1. Prompt is validated as a non-empty string via Pydantic (SemanticAnalyzeRequest).
  2. Leading/trailing whitespace is stripped before analysis.
  3. Prompts exceeding settings.max_prompt_length are rejected with HTTP 413.
  4. Prompt content is NEVER executed as code.
  5. Full prompt is not logged (only its sanitised length).
  6. Internal exceptions are caught; clients receive only a generic message.
  7. No LLM API is called anywhere in this route or the modules it invokes.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.semantic.semantic_analyzer import analyze_semantics
from app.semantic.types import SemanticAnalyzeRequest, SemanticAnalyzeResponse
from app.analysis.types import ErrorResponse
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/semantic-analyze",
    response_model=SemanticAnalyzeResponse,
    summary="Semantic analysis of a prompt locally",
    description=(
        "Accepts a prompt string and returns structured semantic analysis "
        "(intent classification, context extraction, requirements, constraints, ambiguity detection). "
        "Does NOT rewrite or modify the prompt. No LLM API is called."
    ),
    responses={
        200: {"model": SemanticAnalyzeResponse, "description": "Successful semantic analysis"},
        400: {"model": ErrorResponse, "description": "Invalid or empty prompt"},
        413: {"model": ErrorResponse, "description": "Prompt exceeds maximum length"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def semantic_analyze(body: SemanticAnalyzeRequest) -> SemanticAnalyzeResponse:
    """Module 3 — Semantic Analysis endpoint.

    Steps:
      1. Pydantic validates the request body (non-empty string guaranteed).
      2. Strip whitespace (normalise input).
      3. Reject prompts that exceed the configured maximum length.
      4. Delegate to semantic_analyzer (no LLM calls).
      5. Return structured JSON.
    """
    # ── Sanitise ──────────────────────────────────────────────────────────────
    prompt = body.prompt.strip()

    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "EMPTY_PROMPT", "message": "Prompt cannot be empty or whitespace only."},
        )

    if len(prompt) > settings.max_prompt_length:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "PROMPT_TOO_LARGE",
                "message": (
                    f"Prompt length ({len(prompt)} chars) exceeds the maximum "
                    f"allowed length of {settings.max_prompt_length} characters."
                ),
            },
        )

    # Log sanitised metadata only — never the full prompt content
    logger.info("Performing semantic analysis on prompt: length=%d chars", len(prompt))

    # ── Analyse ───────────────────────────────────────────────────────────────
    try:
        analysis = analyze_semantics(prompt)
    except Exception:
        logger.exception("Unexpected error during semantic analysis")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "SEMANTIC_ANALYSIS_ERROR", "message": "An internal error occurred during semantic analysis."},
        )

    return SemanticAnalyzeResponse(success=True, analysis=analysis)
