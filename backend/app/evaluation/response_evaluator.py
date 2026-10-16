"""Independent LLM-judge orchestration for downstream response evaluation."""

from __future__ import annotations

import json
from time import perf_counter
from uuid import uuid4

from pydantic import ValidationError

from app.evaluation.response_config import ResponseEvaluationSettings, get_response_evaluation_settings
from app.evaluation.response_metrics import (
    calculate_deterministic_instruction_checks,
    calculate_target_model_cost,
    calculate_token_usage,
)
from app.evaluation.response_prompts import build_response_evaluator_messages
from app.evaluation.response_types import (
    ResponseEvaluationError,
    ResponseEvaluationMetrics,
    ResponseEvaluationRequest,
    ResponseEvaluationResult,
    ResponseEvaluatorInfo,
    ResponseQualityScores,
)
from app.optimization.llm_client import LLMClient, LiteLLMClient


class ResponseEvaluator:
    """Evaluate one response against its original prompt, without variant metadata."""

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        config: ResponseEvaluationSettings | None = None,
        instruction_check_calculator=calculate_deterministic_instruction_checks,
    ):
        # Reuse only the shared provider adapter; Module 4's optimizer is never imported or called.
        self._llm_client = llm_client or LiteLLMClient()
        self._config = config
        self._instruction_check_calculator = instruction_check_calculator

    async def evaluate(self, request: ResponseEvaluationRequest) -> ResponseEvaluationResult:
        try:
            config = self._config or get_response_evaluation_settings()
        except ValueError as exc:
            raise ResponseEvaluationError(
                "EVALUATOR_CONFIGURATION_INVALID",
                "Response evaluator configuration is invalid.",
                503,
            ) from exc
        evaluator_model = config.configured_evaluator_model()
        if not evaluator_model:
            raise ResponseEvaluationError(
                "EVALUATOR_NOT_CONFIGURED",
                "Set RESPONSE_EVALUATOR_MODEL to an independent evaluator model before evaluating responses.",
                503,
            )

        checks = self._instruction_check_calculator(request.original_prompt, request.response)
        messages = build_response_evaluator_messages(request.original_prompt, request.response, checks)
        started = perf_counter()
        try:
            completion = await self._llm_client.complete(
                model=evaluator_model,
                messages=messages,
                temperature=config.response_evaluator_temperature,
                max_tokens=config.response_evaluator_max_output_tokens,
                timeout=config.response_evaluator_timeout_seconds,
            )
        except Exception as exc:
            # Adapter/provider exceptions may contain credentials or request details; do not expose them.
            raise ResponseEvaluationError(
                "EVALUATOR_UNAVAILABLE",
                "The independent response evaluator could not complete the request.",
                502,
            ) from exc

        try:
            parsed = json.loads(completion.text)
            quality = ResponseQualityScores.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            raise ResponseEvaluationError(
                "INVALID_EVALUATOR_OUTPUT",
                "The evaluator response was not valid JSON matching the required score schema.",
                502,
            ) from exc

        evaluator_latency_ms = max(0.0, (perf_counter() - started) * 1000.0)
        return ResponseEvaluationResult(
            evaluation_id=uuid4(),
            original_prompt=request.original_prompt,
            response=request.response,
            variant=request.variant,
            target_model=request.target_model,
            generation_temperature=request.generation_temperature,
            max_output_tokens=request.max_output_tokens,
            metrics=ResponseEvaluationMetrics(
                token_usage=calculate_token_usage(request),
                target_model_latency_ms=request.latency_ms,
                cost=calculate_target_model_cost(request, config),
                deterministic_instruction_checks=checks,
                response_quality=quality,
            ),
            evaluator=ResponseEvaluatorInfo(
                method="independent_llm_judge",
                model=evaluator_model,
                latency_ms=evaluator_latency_ms,
            ),
        )
