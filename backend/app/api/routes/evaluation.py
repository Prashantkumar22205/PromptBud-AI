"""POST /api/evaluate — independent Module 5 prompt evaluation."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.evaluation.evaluator import PromptEvaluator
from app.evaluation.response_evaluator import ResponseEvaluator
from app.evaluation.types import EvaluationError, EvaluationRequest, EvaluationResponse
from app.evaluation.response_types import (
    ResponseEvaluationError,
    ResponseEvaluationRequest,
    ResponseEvaluationResult,
)

router = APIRouter()
evaluator = PromptEvaluator()
response_evaluator = ResponseEvaluator()


@router.post(
    "/evaluate",
    response_model=EvaluationResponse,
    summary="Evaluate an original and optimized prompt",
    description=(
        "Computes token, embedding similarity, requirement, and constraint metrics. "
        "This endpoint does not call the optimizer or an LLM judge."
    ),
    responses={
        200: {"model": EvaluationResponse},
        422: {"description": "Invalid request or tokenizer"},
        503: {"description": "Tokenizer or embedding model unavailable"},
    },
)
def evaluate_prompts(body: EvaluationRequest) -> EvaluationResponse | JSONResponse:
    try:
        return evaluator.evaluate(body)
    except EvaluationError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"success": False, "error": {"code": exc.code, "message": exc.message}},
        )


@router.post(
    "/evaluate-response",
    response_model=ResponseEvaluationResult,
    summary="Evaluate one generated response against its original prompt",
    description=(
        "Combines deterministic response metrics with an independent LLM evaluator. "
        "The evaluator receives no optimization variant or target-model metadata."
    ),
    responses={
        200: {"model": ResponseEvaluationResult},
        422: {"description": "Invalid request or token metadata"},
        502: {"description": "Evaluator unavailable or invalid evaluator output"},
        503: {"description": "Independent evaluator is not configured"},
    },
)
async def evaluate_response(
    body: ResponseEvaluationRequest,
) -> ResponseEvaluationResult | JSONResponse:
    try:
        return await response_evaluator.evaluate(body)
    except ResponseEvaluationError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"success": False, "error": {"code": exc.code, "message": exc.message}},
        )
