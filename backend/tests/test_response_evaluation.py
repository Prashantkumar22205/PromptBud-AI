"""Tests for Module 5B downstream response evaluation; no live LLM calls."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.evaluation.response_config import ResponseEvaluationSettings
from app.evaluation.response_evaluator import ResponseEvaluator
from app.evaluation.response_metrics import (
    calculate_deterministic_instruction_checks,
    calculate_target_model_cost,
    calculate_token_usage,
)
from app.evaluation.response_types import ResponseEvaluationRequest, ResponseQualityScores
from app.main import app
from app.optimization.types import LLMCompletion


QUALITY = {
    "relevance": {"score": 4, "reason": "The response addresses the requested task."},
    "completeness": {"score": 4, "reason": "It covers the main requested parts."},
    "correctness": {"score": 5, "reason": "The included statements are accurate."},
    "instruction_following": {"score": 4, "reason": "The requested format is mostly followed."},
}
QUALITY_JSON = json.dumps(QUALITY)


class FakeJudgeClient:
    def __init__(self, text: str = QUALITY_JSON, error: Exception | None = None):
        self.text = text
        self.error = error
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return LLMCompletion(text=self.text)


def config(
    *,
    evaluator_model: str | None = "judge/independent-model",
    pricing: str | None = None,
) -> ResponseEvaluationSettings:
    return ResponseEvaluationSettings(
        response_evaluator_model=evaluator_model,
        response_evaluator_temperature=0.0,
        response_evaluator_max_output_tokens=1200,
        response_evaluator_timeout_seconds=30.0,
        target_model_pricing_json=pricing,
    )


def request(**overrides) -> ResponseEvaluationRequest:
    values = {
        "original_prompt": "Explain binary search for a beginner.",
        "response": "Binary search finds a value in a sorted list by halving the search range.",
        "variant": "original",
        "target_model": "target/model-v1",
    }
    values.update(overrides)
    return ResponseEvaluationRequest(**values)


def fixed_checks(original_prompt: str, response: str):
    return []


@pytest.mark.asyncio
async def test_valid_request_runs_independent_evaluator():
    client = FakeJudgeClient()
    result = await ResponseEvaluator(llm_client=client, config=config(), instruction_check_calculator=fixed_checks).evaluate(request())
    assert result.variant == "original"
    assert result.evaluator.model == "judge/independent-model"
    assert result.evaluator.method == "independent_llm_judge"
    assert result.metrics.response_quality.relevance.score == 4
    assert len(client.calls) == 1


@pytest.mark.parametrize("field", ["original_prompt", "response"])
def test_empty_original_prompt_or_response_is_rejected(field):
    with pytest.raises(ValidationError):
        request(**{field: "   "})


def test_invalid_variant_is_rejected():
    with pytest.raises(ValidationError):
        request(variant="guided-optimized")


def test_request_rejects_negative_or_inconsistent_token_usage():
    with pytest.raises(ValidationError):
        request(input_tokens=-1)
    with pytest.raises(ValidationError, match="total_tokens must equal"):
        request(input_tokens=10, output_tokens=4, total_tokens=15)


def test_token_usage_calculates_total_only_when_both_components_are_known():
    usage = calculate_token_usage(request(input_tokens=100, output_tokens=25))
    assert usage.input_tokens == 100
    assert usage.output_tokens == 25
    assert usage.total_tokens == 125


def test_missing_token_metadata_remains_null():
    usage = calculate_token_usage(request())
    assert usage.model_dump() == {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    partial = calculate_token_usage(request(input_tokens=100))
    assert partial.input_tokens == 100
    assert partial.output_tokens is None
    assert partial.total_tokens is None


@pytest.mark.asyncio
async def test_target_model_latency_is_reported_as_supplied():
    result = await ResponseEvaluator(
        llm_client=FakeJudgeClient(), config=config(), instruction_check_calculator=fixed_checks
    ).evaluate(request(latency_ms=1234.5))
    assert result.metrics.target_model_latency_ms == 1234.5


@pytest.mark.asyncio
async def test_missing_target_model_latency_remains_null():
    result = await ResponseEvaluator(
        llm_client=FakeJudgeClient(), config=config(), instruction_check_calculator=fixed_checks
    ).evaluate(request())
    assert result.metrics.target_model_latency_ms is None


def test_cost_is_computed_from_external_target_model_pricing():
    settings = config(
        pricing=json.dumps(
            {"target/model-v1": {"input_price_per_million": 1.0, "output_price_per_million": 2.0}}
        )
    )
    cost = calculate_target_model_cost(request(input_tokens=1000, output_tokens=500), settings)
    assert cost.estimated_cost == pytest.approx(0.002)
    assert cost.status == "estimated"


def test_missing_pricing_or_usage_returns_null_cost_with_reason_code():
    no_pricing = calculate_target_model_cost(request(input_tokens=10, output_tokens=5), config())
    assert no_pricing.estimated_cost is None
    assert no_pricing.status == "unavailable"
    assert no_pricing.reason_code == "TARGET_MODEL_PRICING_NOT_CONFIGURED"

    pricing = json.dumps({"target/model-v1": {"input_price_per_million": 1.0, "output_price_per_million": 2.0}})
    no_usage = calculate_target_model_cost(request(), config(pricing=pricing))
    assert no_usage.estimated_cost is None
    assert no_usage.reason_code == "TOKEN_USAGE_INCOMPLETE"


def test_caller_supplied_estimated_cost_is_preserved_without_recalculation():
    cost = calculate_target_model_cost(request(estimated_cost=0.0123), config())
    assert cost.estimated_cost == pytest.approx(0.0123)
    assert cost.status == "provided"


def test_deterministic_explicit_word_limit_check():
    checks = calculate_deterministic_instruction_checks(
        "Keep the response under 5 words.", "One two three four."
    )
    assert checks[0].status == "passed"
    assert checks[0].observed == "4 words"
    failed = calculate_deterministic_instruction_checks(
        "Keep the response under 5 words.", "One two three four five."
    )
    assert failed[0].status == "failed"


def test_deterministic_example_quantity_check_is_conservative():
    passed = calculate_deterministic_instruction_checks(
        "Give 3 examples.", "Examples:\n1. Example one\n2. Example two\n3. Example three"
    )
    assert passed[0].status == "passed"
    failed = calculate_deterministic_instruction_checks(
        "Give 3 examples.", "Examples:\n1. Example one\n2. Example two"
    )
    assert failed[0].status == "failed"
    uncertain = calculate_deterministic_instruction_checks("Give 3 examples.", "Here are some examples in prose.")
    assert uncertain[0].status == "not_evaluated"
    # Numbered steps are not necessarily examples, so do not claim a quantity result.
    numbered_steps = calculate_deterministic_instruction_checks(
        "Give 3 examples.", "1. First step\n2. Second step\n3. Third step"
    )
    assert numbered_steps[0].status == "not_evaluated"


def test_deterministic_format_check_supports_markdown_table():
    passed = calculate_deterministic_instruction_checks(
        "Answer in a table.", "| Term | Meaning |\n|---|---|\n| Search | Find a value |"
    )
    assert passed[0].status == "passed"
    failed = calculate_deterministic_instruction_checks("Answer in a table.", "A plain prose response.")
    assert failed[0].status == "failed"


def test_unsupported_language_and_restriction_checks_are_not_evaluated():
    checks = calculate_deterministic_instruction_checks(
        "Answer in French and do not use recursion.", "Some response text."
    )
    assert checks
    assert all(check.status == "not_evaluated" for check in checks)


def test_evaluator_score_schema_accepts_exact_four_dimensions_and_rejects_invalid_scores():
    scores = ResponseQualityScores.model_validate(QUALITY)
    assert scores.correctness.score == 5
    invalid = {**QUALITY, "correctness": {"score": 6, "reason": "Out of scale"}}
    with pytest.raises(ValidationError):
        ResponseQualityScores.model_validate(invalid)
    with pytest.raises(ValidationError):
        ResponseQualityScores.model_validate({**QUALITY, "extra_dimension": {"score": 3, "reason": "extra"}})


@pytest.mark.asyncio
async def test_invalid_evaluator_json_returns_structured_failure():
    with pytest.raises(Exception) as exc:
        await ResponseEvaluator(
            llm_client=FakeJudgeClient(text="not json"),
            config=config(),
            instruction_check_calculator=fixed_checks,
        ).evaluate(request())
    assert getattr(exc.value, "code", None) == "INVALID_EVALUATOR_OUTPUT"


@pytest.mark.asyncio
async def test_evaluator_score_outside_range_returns_structured_failure():
    invalid = {**QUALITY, "relevance": {"score": 0, "reason": "Invalid"}}
    with pytest.raises(Exception) as exc:
        await ResponseEvaluator(
            llm_client=FakeJudgeClient(text=json.dumps(invalid)),
            config=config(),
            instruction_check_calculator=fixed_checks,
        ).evaluate(request())
    assert getattr(exc.value, "code", None) == "INVALID_EVALUATOR_OUTPUT"


@pytest.mark.asyncio
async def test_missing_evaluator_configuration_does_not_fall_back_to_target_model():
    client = FakeJudgeClient()
    with pytest.raises(Exception) as exc:
        await ResponseEvaluator(
            llm_client=client,
            config=config(evaluator_model=None),
            instruction_check_calculator=fixed_checks,
        ).evaluate(request(target_model="target/must-not-be-used-as-judge"))
    assert getattr(exc.value, "code", None) == "EVALUATOR_NOT_CONFIGURED"
    assert client.calls == []


@pytest.mark.asyncio
async def test_mocked_evaluator_failure_is_safe_and_structured():
    client = FakeJudgeClient(error=RuntimeError("secret provider credential detail"))
    with pytest.raises(Exception) as exc:
        await ResponseEvaluator(
            llm_client=client, config=config(), instruction_check_calculator=fixed_checks
        ).evaluate(request())
    assert getattr(exc.value, "code", None) == "EVALUATOR_UNAVAILABLE"
    assert "secret" not in str(exc.value)


@pytest.mark.asyncio
async def test_evaluator_prompt_excludes_variant_target_model_and_optimizer_context():
    client = FakeJudgeClient()
    await ResponseEvaluator(
        llm_client=client, config=config(), instruction_check_calculator=fixed_checks
    ).evaluate(
        request(
            original_prompt="Explain binary search.",
            response="Binary search repeatedly halves a sorted search interval.",
            variant="guided",
            target_model="TARGET_MODEL_SECRET",
            generation_temperature=0.8,
            max_output_tokens=777,
        )
    )
    call = client.calls[0]
    assert call["model"] == "judge/independent-model"
    user_content = call["messages"][1]["content"]
    assert "Explain binary search." in user_content
    assert "Binary search repeatedly halves" in user_content
    for forbidden in ("guided", "direct", "TARGET_MODEL_SECRET", "optimizer_model", "Module 3", "reasoning"):
        assert forbidden not in user_content.casefold()
    assert '"variant"' not in user_content.casefold()
    assert "Do not reward or penalize the response based on its source." in call["messages"][0]["content"]


@pytest.mark.asyncio
async def test_original_direct_guided_variants_share_judge_configuration_and_prompt_policy():
    client = FakeJudgeClient()
    evaluator = ResponseEvaluator(llm_client=client, config=config(), instruction_check_calculator=fixed_checks)
    for variant, response in (
        ("original", "Response O"),
        ("direct", "Response D"),
        ("guided", "Response G"),
    ):
        await evaluator.evaluate(
            request(original_prompt="Same research prompt", response=response, variant=variant)
        )

    assert len(client.calls) == 3
    assert all(call["model"] == "judge/independent-model" for call in client.calls)
    assert all(call["temperature"] == 0.0 for call in client.calls)
    assert all(call["max_tokens"] == 1200 for call in client.calls)
    assert all(call["timeout"] == 30.0 for call in client.calls)
    system_prompts = [call["messages"][0]["content"] for call in client.calls]
    assert len(set(system_prompts)) == 1
    user_prompts = [call["messages"][1]["content"] for call in client.calls]
    def extract_sections(prompt: str):
        original, response = prompt.split(
            "\n\nRESPONSE TO EVALUATE (JSON string):\n", maxsplit=1
        )
        response, checks = response.split("\n\nOPTIONAL OBJECTIVE CHECKS (JSON):\n", maxsplit=1)
        original = original.split("ORIGINAL USER PROMPT (JSON string):\n", maxsplit=1)[1]
        return json.loads(original), json.loads(response), json.loads(checks)

    sections = [extract_sections(prompt) for prompt in user_prompts]
    assert [section[0] for section in sections] == ["Same research prompt"] * 3
    assert [section[1] for section in sections] == ["Response O", "Response D", "Response G"]
    assert sections[0][2] == sections[1][2] == sections[2][2]
    assert all('"variant"' not in prompt.casefold() for prompt in user_prompts)


@pytest.mark.asyncio
async def test_response_evaluation_api_returns_structured_result(monkeypatch):
    import app.api.routes.evaluation as evaluation_route

    monkeypatch.setattr(
        evaluation_route,
        "response_evaluator",
        ResponseEvaluator(
            llm_client=FakeJudgeClient(), config=config(), instruction_check_calculator=fixed_checks
        ),
    )
    response = TestClient(app).post(
        "/api/evaluate-response",
        json={
            "original_prompt": "Explain binary search.",
            "response": "Binary search halves a sorted range.",
            "variant": "direct",
            "target_model": "target/model-v1",
            "input_tokens": 10,
            "output_tokens": 20,
            "latency_ms": 100,
            "generation_temperature": 0.2,
            "max_output_tokens": 500,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["variant"] == "direct"
    assert body["metrics"]["token_usage"] == {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
    assert body["metrics"]["target_model_latency_ms"] == 100
    assert body["generation_temperature"] == 0.2
    assert body["max_output_tokens"] == 500
    assert body["metrics"]["cost"]["estimated_cost"] is None
    assert body["evaluator"]["model"] == "judge/independent-model"
    assert set(body["metrics"]["response_quality"]) == {
        "relevance", "completeness", "correctness", "instruction_following"
    }


def test_response_evaluation_api_validates_empty_prompt_response_and_variant():
    client = TestClient(app)
    common = {"original_prompt": "Prompt", "response": "Answer", "variant": "original", "target_model": "target/model"}
    for field, value in (("original_prompt", ""), ("response", ""), ("variant", "bad")):
        invalid = {**common, field: value}
        response = client.post("/api/evaluate-response", json=invalid)
        assert response.status_code == 422


def test_module5a_evaluate_api_behavior_remains_unchanged(monkeypatch):
    import app.api.routes.evaluation as evaluation_route

    class LegacyEvaluator:
        def evaluate(self, body):
            from app.evaluation.types import (
                ConstraintPreservationMetric,
                EvaluationMetrics,
                EvaluationResponse,
                RequirementPreservationMetric,
                SemanticSimilarityMetric,
                TokenMetrics,
            )

            return EvaluationResponse(
                evaluation_id="00000000-0000-0000-0000-000000000001",
                original_prompt=body.original_prompt,
                optimized_prompt=body.optimized_prompt,
                mode=body.mode,
                optimizer_model=body.optimizer_model,
                tokenizer="tiktoken/cl100k_base",
                embedding_model="all-MiniLM-L6-v2",
                metrics=EvaluationMetrics(
                    token_metrics=TokenMetrics(
                        original_tokens=1, optimized_tokens=1, token_delta=0, token_reduction_percent=0
                    ),
                    semantic_similarity=SemanticSimilarityMetric(score=1),
                    requirement_preservation=RequirementPreservationMetric(
                        original_requirements=[], optimized_requirements=[], preserved_count=0,
                        original_count=0, preservation_score=1
                    ),
                    constraint_preservation=ConstraintPreservationMetric(
                        original_constraints=[], optimized_constraints=[], preserved_count=0,
                        original_count=0, preservation_score=1
                    ),
                ),
                evaluation_latency_ms=0,
            )

    response_evaluator = AsyncMock(side_effect=AssertionError("5B must not run for /api/evaluate"))
    monkeypatch.setattr(evaluation_route, "evaluator", LegacyEvaluator())
    monkeypatch.setattr(evaluation_route, "response_evaluator", response_evaluator)
    response = TestClient(app).post(
        "/api/evaluate",
        json={
            "original_prompt": "Original",
            "optimized_prompt": "Optimized",
            "mode": "direct",
            "optimizer_model": "optimizer/model",
        },
    )
    assert response.status_code == 200
    assert set(response.json()) == {
        "evaluation_id", "original_prompt", "optimized_prompt", "mode", "optimizer_model",
        "tokenizer", "embedding_model", "metrics", "evaluation_latency_ms",
    }
    response_evaluator.assert_not_awaited()


@pytest.mark.asyncio
async def test_response_evaluation_does_not_call_module4_optimizer(monkeypatch):
    import app.api.routes.optimization as optimization_route

    optimizer_call = AsyncMock(side_effect=AssertionError("Module 4 optimizer must not be called"))
    monkeypatch.setattr(optimization_route.optimizer, "optimize", optimizer_call)
    evaluator = ResponseEvaluator(
        llm_client=FakeJudgeClient(), config=config(), instruction_check_calculator=fixed_checks
    )
    await evaluator.evaluate(request())
    optimizer_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_evaluator_failure_uses_structured_api_error(monkeypatch):
    import app.api.routes.evaluation as evaluation_route

    monkeypatch.setattr(
        evaluation_route,
        "response_evaluator",
        ResponseEvaluator(
            llm_client=FakeJudgeClient(error=RuntimeError("provider secret")),
            config=config(),
            instruction_check_calculator=fixed_checks,
        ),
    )
    response = TestClient(app).post(
        "/api/evaluate-response",
        json={
            "original_prompt": "Prompt",
            "response": "Answer",
            "variant": "guided",
            "target_model": "target/model",
        },
    )
    assert response.status_code == 502
    assert response.json() == {
        "success": False,
        "error": {
            "code": "EVALUATOR_UNAVAILABLE",
            "message": "The independent response evaluator could not complete the request.",
        },
    }
