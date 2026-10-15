"""POST /api/optimize — Module 4 LLM-based prompt generation."""

from __future__ import annotations

import logging

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.optimization.optimizer import PromptOptimizer
from app.optimization.types import (
    ErrorDetail,
    ErrorResponse,
    OptimizationError,
    OptimizationRequest,
    OptimizationResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()
optimizer = PromptOptimizer()


@router.post(
    "/optimize",
    response_model=OptimizationResponse,
    summary="Optimize a prompt using a configured LLM",
    description=(
        "Module 4 supports DIRECT generation from the original prompt and GUIDED "
        "generation using a supplied authoritative Module 3 semantic profile. "
        "This endpoint does not run Module 3 or calculate evaluation metrics."
    ),
    responses={
        200: {"model": OptimizationResponse},
        400: {"model": ErrorResponse, "description": "Unsupported model"},
        413: {"model": ErrorResponse, "description": "Prompt exceeds the configured limit"},
        422: {"description": "Invalid request or guided mode missing semantic profile"},
        502: {"model": ErrorResponse, "description": "Model provider or response error"},
        503: {"model": ErrorResponse, "description": "Missing provider credentials or LiteLLM"},
        504: {"model": ErrorResponse, "description": "Model provider timed out"},
    },
)
async def optimize_prompt(body: OptimizationRequest) -> OptimizationResponse | JSONResponse:
    try:
        return await optimizer.optimize(body)
    except OptimizationError as exc:
        logger.warning("Prompt optimization failed: code=%s", exc.code)
        error = ErrorResponse(
            success=False,
            error=ErrorDetail(code=exc.code, message=exc.message),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=error.model_dump(),
        )
