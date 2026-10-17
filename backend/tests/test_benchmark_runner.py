"""Benchmark orchestration tests using temporary fixtures and mocked services."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.evaluation.response_types import (
    CostMetric,
    ResponseEvaluationMetrics,
    ResponseEvaluationResult,
    ResponseEvaluatorInfo,
    ResponseQualityScores,
    TokenUsageMetric,
)
from app.evaluation.types import (
    ConstraintPreservationMetric,
    EvaluationMetrics,
    EvaluationResponse,
    RequirementPreservationMetric,
    SemanticSimilarityMetric,
    TokenMetrics,
)
from app.optimization.types import LLMCompletion, OptimizationMetadata, OptimizationResponse, TokenUsage
from app.semantic.types import ContextResult, IntentResult, SemanticAnalysis
from experiments.benchmark_runner import (
    BenchmarkConfig,
    BenchmarkConditionResult,
    BenchmarkError,
    BenchmarkPrompt,
    BenchmarkRunner,
    TargetModelRunner,
    expected_execution_counts,
    load_benchmark_dataset,
    resolve_dataset_path,
    select_evaluation_prompts,
    validate_frozen_dataset,
)


def dataset_records() -> list[dict]:
    records = []
    for idx in range(150):
        frozen = idx >= 30
        eval_idx = idx - 30
        category_number = (eval_idx // 12) if frozen else idx // 3
        if category_number >= 10:
            category_number = 9
        records.append(
            {
                "prompt_id": f"POAI-{idx + 1:03d}",
                "category": f"category_{category_number + 1}",
                "split": "frozen_evaluation" if frozen else "development",
                "length_band": ("short", "medium", "long")[eval_idx % 3] if frozen else "short",
                "prompt": f"Explain benchmark topic number {idx + 1} clearly.",
                "word_count": 7,
            }
        )
    return records


@pytest.fixture
def dataset_path(tmp_path: Path) -> Path:
    path = tmp_path / "promptoptai_benchmark_dataset_v1_1.json"
    path.write_text(
        json.dumps({"dataset_version": "1.1", "prompts": dataset_records()}), encoding="utf-8"
    )
    return path


def make_config(dataset_path: Path, **kwargs) -> BenchmarkConfig:
    values = {
        "dataset_path": dataset_path,
        "optimizer_model": "optimizer/mock-v1",
        "target_model": "target/mock-v1",
        "temperature": 0.2,
        "max_output_tokens": 256,
        "timeout": 12.0,
    }
    values.update(kwargs)
    return BenchmarkConfig(**values)


def fake_semantic_profile() -> SemanticAnalysis:
    return SemanticAnalysis(
        intent=IntentResult(primaryIntent="educational_explanation", confidence=0.9),
        context=ContextResult(),
        requirements=[],
        constraints=[],
        ambiguities=[],
    )


def fake_module5a(original: str, optimized: str, mode: str, model: str):
    return EvaluationResponse(
        evaluation_id="00000000-0000-0000-0000-000000000001",
        original_prompt=original,
        optimized_prompt=optimized,
        mode=mode,
        optimizer_model=model,
        tokenizer="mock-tokenizer",
        embedding_model="mock-embedding",
        metrics=EvaluationMetrics(
            token_metrics=TokenMetrics(
                original_tokens=10,
                optimized_tokens=7,
                token_delta=3,
                token_reduction_percent=30.0,
            ),
            semantic_similarity=SemanticSimilarityMetric(score=0.8),
            requirement_preservation=RequirementPreservationMetric(
                original_requirements=[], optimized_requirements=[], preserved_count=0,
                original_count=0, preservation_score=1.0
            ),
            constraint_preservation=ConstraintPreservationMetric(
                original_constraints=[], optimized_constraints=[], preserved_count=0,
                original_count=0, preservation_score=1.0
            ),
        ),
        evaluation_latency_ms=1,
    )


def fake_module5b(body):
    return ResponseEvaluationResult(
        evaluation_id="00000000-0000-0000-0000-000000000002",
        original_prompt=body.original_prompt,
        evaluation_prompt=body.evaluation_prompt,
        response=body.response,
        variant=body.variant,
        target_model=body.target_model,
        generation_temperature=body.generation_temperature,
        max_output_tokens=body.max_output_tokens,
        metrics=ResponseEvaluationMetrics(
            token_usage=TokenUsageMetric(
                input_tokens=body.input_tokens,
                output_tokens=body.output_tokens,
                total_tokens=body.total_tokens,
            ),
            target_model_latency_ms=body.latency_ms,
            cost=CostMetric(estimated_cost=None, status="unavailable", reason="not configured"),
            deterministic_instruction_checks=[],
            response_quality=ResponseQualityScores.model_validate(
                {
                    key: {"score": 4, "reason": f"Mock {key} evidence."}
                    for key in ("relevance", "completeness", "correctness", "instruction_following")
                }
            ),
        ),
        evaluator=ResponseEvaluatorInfo(model="judge/mock", latency_ms=1.0),
    )


class MockOptimizer:
    def __init__(self):
        self.requests = []

    async def optimize(self, request):
        self.requests.append(request)
        return OptimizationResponse(
            mode=request.mode,
            model=request.model,
            original_prompt=request.prompt,
            optimized_prompt=f"{request.mode.upper()} optimized: {request.prompt}",
            metadata=OptimizationMetadata(
                latency_ms=10,
                usage=TokenUsage(input_tokens=20, output_tokens=8, total_tokens=28),
            ),
        )


class MockTargetRunner:
    def __init__(self, fail_prompt_id: str | None = None):
        self.calls = []
        self.fail_prompt_id = fail_prompt_id

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_prompt_id and self.fail_prompt_id in kwargs["prompt"]:
            from experiments.benchmark_runner import TargetExecution

            return TargetExecution(None, None, None, None, None, kwargs["model"], "LLM_TIMEOUT", "Mock timeout")
        return SimpleNamespace(
            response_text=f"Answer for {kwargs['prompt']}",
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            latency_ms=None,
            model=kwargs["model"],
            error_code=None,
            error_message=None,
        )


def make_runner(tmp_path: Path, *, fail_prompt_id: str | None = None):
    optimizer = MockOptimizer()
    semantic = SimpleNamespace(analyze=AsyncMock(side_effect=AssertionError("analyze is sync")))
    semantic.analyze = lambda prompt: fake_semantic_profile()
    semantic.calls = []
    original_analyze = semantic.analyze

    def analyze(prompt):
        semantic.calls.append(prompt)
        return original_analyze(prompt)

    semantic.analyze = analyze
    target = MockTargetRunner(fail_prompt_id=fail_prompt_id)
    prompt_eval = SimpleNamespace(calls=[])

    def evaluate_prompt(request):
        prompt_eval.calls.append(request)
        return fake_module5a(
            request.original_prompt, request.optimized_prompt, request.mode, request.optimizer_model
        )

    prompt_eval.evaluate = evaluate_prompt
    response_eval = SimpleNamespace(evaluate=AsyncMock())
    response_eval.evaluate = AsyncMock(side_effect=lambda body: fake_module5b(body))
    runner = BenchmarkRunner(
        optimizer=optimizer,
        semantic_analyzer=semantic,
        target_runner=target,
        prompt_evaluator=prompt_eval,
        response_evaluator=response_eval,
        backend_root=tmp_path,
    )
    return runner, optimizer, semantic, target, prompt_eval, response_eval


def test_dataset_load_hash_and_frozen_records(dataset_path):
    prompts, dataset_version, dataset_hash = load_benchmark_dataset(dataset_path)
    assert len(prompts) == 150
    validate_frozen_dataset(prompts)
    assert dataset_version == "1.1"
    assert len(dataset_hash) == 64
    assert select_evaluation_prompts(prompts) and all(p.split == "frozen_evaluation" for p in select_evaluation_prompts(prompts))


def test_development_records_are_excluded(dataset_path):
    prompts, _, _ = load_benchmark_dataset(dataset_path)
    selected = select_evaluation_prompts(prompts)
    assert len(selected) == 120
    assert not any(prompt.prompt_id == "POAI-001" for prompt in selected)
    assert selected[0].prompt_id == "POAI-031"


def test_length_band_and_category_filters(dataset_path):
    prompts, _, _ = load_benchmark_dataset(dataset_path)
    short = select_evaluation_prompts(prompts, length_band="short")
    category = select_evaluation_prompts(prompts, categories=["category_1"])
    assert len(short) == 40 and all(prompt.length_band == "short" for prompt in short)
    assert len(category) == 12 and all(prompt.category == "category_1" for prompt in category)
    assert len(select_evaluation_prompts(prompts, limit=2)) == 2


def test_rejects_dataset_with_wrong_frozen_distribution(dataset_path):
    prompts, _, _ = load_benchmark_dataset(dataset_path)
    with pytest.raises(BenchmarkError, match="150 records"):
        validate_frozen_dataset(prompts[:149])


def test_missing_default_path_searches_repository_by_exact_frozen_filename(tmp_path):
    backend = tmp_path / "backend"
    found = tmp_path / "archive" / "promptoptai_benchmark_dataset_v1_1.json"
    found.parent.mkdir(parents=True)
    found.write_text("[]", encoding="utf-8")
    resolved = resolve_dataset_path(
        "backend/data/promptoptai_benchmark_dataset_v1_1.json", backend_root=backend
    )
    assert resolved == found.resolve()


def test_expected_execution_counts():
    assert expected_execution_counts(120) == {
        "prompts": 120,
        "target_model_executions": 360,
        "optimizer_calls": 240,
        "guided_module3_analyses": 120,
        "module5a_evaluations": 240,
        "module5b_evaluations": 360,
    }


@pytest.mark.asyncio
async def test_three_conditions_use_same_target_configuration_and_only_guided_calls_module3(dataset_path, tmp_path):
    runner, optimizer, semantic, target, prompt_eval, response_eval = make_runner(tmp_path)
    summary = await runner.run(
        make_config(dataset_path, limit=1),
        results_root=tmp_path / "results",
        checkpoints_root=tmp_path / "checkpoints",
        logs_root=tmp_path / "logs",
        experiment_id="control-test",
    )
    assert summary["condition_records_written"] == 3
    assert len(target.calls) == 3
    assert {call["model"] for call in target.calls} == {"target/mock-v1"}
    assert {call["temperature"] for call in target.calls} == {0.2}
    assert {call["max_output_tokens"] for call in target.calls} == {256}
    assert {call["timeout"] for call in target.calls} == {12.0}
    assert len({call["system_instruction"] for call in target.calls}) == 1
    assert len(semantic.calls) == 1
    assert [request.mode for request in optimizer.requests] == ["direct", "guided"]
    assert optimizer.requests[0].semantic_profile is None
    assert optimizer.requests[1].semantic_profile == fake_semantic_profile()
    assert len(prompt_eval.calls) == 2
    assert response_eval.evaluate.await_count == 3
    original_prompt = dataset_records()[30]["prompt"]
    evaluated = [call.args[0] for call in response_eval.evaluate.await_args_list]
    assert [(body.original_prompt, body.evaluation_prompt) for body in evaluated] == [
        (original_prompt, original_prompt),
        (original_prompt, f"DIRECT optimized: {original_prompt}"),
        (original_prompt, f"GUIDED optimized: {original_prompt}"),
    ]
    assert [call["prompt"] for call in target.calls] == [body.evaluation_prompt for body in evaluated]
    assert all("semantic_profile" not in call for call in target.calls)
    assert all(not hasattr(body, "semantic_profile") for body in evaluated)
    records = [
        json.loads(line)
        for line in (tmp_path / "results" / "control-test" / "raw_results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    by_variant = {record["variant"]: record for record in records}
    assert by_variant["original"]["semantic_profile"] is None
    assert by_variant["direct"]["semantic_profile"] is None
    assert by_variant["guided"]["semantic_profile"] == optimizer.requests[1].semantic_profile.model_dump(
        mode="json"
    )
    assert by_variant["guided"]["semantic_profile"] == fake_semantic_profile().model_dump(mode="json")


@pytest.mark.asyncio
async def test_result_schema_and_missing_usage_latency_remain_null(dataset_path, tmp_path):
    runner, *_ = make_runner(tmp_path)
    await runner.run(
        make_config(dataset_path, limit=1),
        results_root=tmp_path / "results",
        checkpoints_root=tmp_path / "checkpoints",
        logs_root=tmp_path / "logs",
        experiment_id="schema-test",
    )
    result_dir = tmp_path / "results" / "schema-test"
    records = [json.loads(line) for line in (result_dir / "raw_results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 3
    assert all(BenchmarkConditionResult.model_validate(record) for record in records)
    original = next(record for record in records if record["variant"] == "original")
    assert original["target_usage"] == {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    assert original["target_latency_ms"] is None
    assert original["module5b_metrics"]["cost"]["estimated_cost"] is None
    direct = next(record for record in records if record["variant"] == "direct")
    assert direct["direct_prompt_token_count"] == 7
    assert direct["direct_prompt_token_reduction"] == 30.0
    assert direct["direct_relevance"] == 4
    assert direct["direct_optimizer_usage"]["total_tokens"] == 28
    assert direct["target_usage"]["total_tokens"] is None


@pytest.mark.asyncio
async def test_failed_target_is_recorded_and_later_prompts_continue(dataset_path, tmp_path):
    runner, _, _, target, _, _ = make_runner(tmp_path, fail_prompt_id="number 31")
    summary = await runner.run(
        make_config(dataset_path, limit=2),
        results_root=tmp_path / "results",
        checkpoints_root=tmp_path / "checkpoints",
        logs_root=tmp_path / "logs",
        experiment_id="failure-test",
    )
    result_dir = tmp_path / "results" / "failure-test"
    records = [json.loads(line) for line in (result_dir / "raw_results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 6
    assert len(target.calls) == 6  # 2 prompts × 3 conditions (smoke execution count).
    assert {row["prompt_id"] for row in records} == {"POAI-031", "POAI-032"}
    assert all(
        sum(row["variant"] == variant and row["prompt_id"] == prompt_id for row in records) == 1
        for prompt_id in ("POAI-031", "POAI-032")
        for variant in ("original", "direct", "guided")
    )
    assert sum(row["status"] == "error" for row in records) == 3
    assert all(row["error_code"] == "LLM_TIMEOUT" for row in records if row["prompt_id"] == "POAI-031")
    assert any(row["prompt_id"] == "POAI-032" and row["status"] == "success" for row in records)
    assert summary["failed_count"] == 3
    assert (tmp_path / "results" / "raw_results.jsonl").is_file()
    assert (tmp_path / "results" / "raw_results.csv").is_file()


@pytest.mark.asyncio
async def test_checkpoint_written_and_resume_skips_completed_conditions(dataset_path, tmp_path):
    runner, optimizer, semantic, target, _, _ = make_runner(tmp_path)
    kwargs = {
        "results_root": tmp_path / "results",
        "checkpoints_root": tmp_path / "checkpoints",
        "logs_root": tmp_path / "logs",
        "experiment_id": "resume-test",
    }
    await runner.run(make_config(dataset_path, limit=1), **kwargs)
    checkpoint = tmp_path / "checkpoints" / "resume-test" / "condition_checkpoints.jsonl"
    assert checkpoint.is_file()
    assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == 3

    second, second_optimizer, second_semantic, second_target, _, second_response_eval = make_runner(tmp_path)
    summary = await second.run(make_config(dataset_path, limit=1), resume=True, **kwargs)
    assert summary["condition_records_written"] == 3
    assert second_optimizer.requests == []
    assert second_semantic.calls == []
    assert second_target.calls == []
    assert second_response_eval.evaluate.await_count == 0


@pytest.mark.asyncio
async def test_resume_retries_failed_condition_and_preserves_successful_records(dataset_path, tmp_path):
    runner, optimizer, semantic, target, _, response_eval = make_runner(tmp_path)
    kwargs = {
        "results_root": tmp_path / "results",
        "checkpoints_root": tmp_path / "checkpoints",
        "logs_root": tmp_path / "logs",
        "experiment_id": "retry-failed-test",
    }
    await runner.run(make_config(dataset_path, limit=1), **kwargs)

    result_path = tmp_path / "results" / "retry-failed-test" / "raw_results.jsonl"
    before = {
        record["variant"]: record
        for record in map(json.loads, result_path.read_text(encoding="utf-8").splitlines())
    }
    checkpoint = tmp_path / "checkpoints" / "retry-failed-test" / "condition_checkpoints.jsonl"
    previous_guided = next(record for record in before.values() if record["variant"] == "guided")
    failed_guided = {
        **previous_guided,
        "status": "error",
        "error_code": "MOCK_TRANSIENT_FAILURE",
        "error_message": "Mock transient failure.",
        "errors": [{"code": "MOCK_TRANSIENT_FAILURE", "message": "Mock transient failure."}],
    }
    with checkpoint.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(failed_guided) + "\n")

    second_semantic_calls = []
    semantic.analyze = lambda prompt: (second_semantic_calls.append(prompt) or fake_semantic_profile())
    target.calls.clear()
    response_eval.evaluate.reset_mock()
    summary = await runner.run(make_config(dataset_path, limit=1), resume=True, **kwargs)

    after = {
        record["variant"]: record
        for record in map(json.loads, result_path.read_text(encoding="utf-8").splitlines())
    }
    # Only Guided was failed, so only that condition is regenerated on resume.
    assert len(target.calls) == 1
    assert second_semantic_calls == [dataset_records()[30]["prompt"]]
    assert [request.mode for request in optimizer.requests] == ["direct", "guided", "guided"]
    assert response_eval.evaluate.await_count == 1
    assert after["original"] == before["original"]
    assert after["direct"] == before["direct"]
    assert after["guided"]["status"] == "success"
    assert summary["completed_count"] == 3
    history = [json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines()]
    guided_history = [record for record in history if record["variant"] == "guided"]
    assert [record["status"] for record in guided_history] == ["success", "error", "success"]
    assert guided_history[1]["error_code"] == "MOCK_TRANSIENT_FAILURE"


@pytest.mark.asyncio
async def test_dry_run_makes_no_llm_calls_and_validates_counts(dataset_path, tmp_path):
    runner, optimizer, semantic, target, _, response_eval = make_runner(tmp_path)
    result = await runner.run(
        make_config(dataset_path, limit=2),
        results_root=tmp_path / "results",
        checkpoints_root=tmp_path / "checkpoints",
        logs_root=tmp_path / "logs",
        dry_run=True,
        experiment_id="dry-run-test",
    )
    assert result["selected_prompt_count"] == 2
    assert result["expected_experiment_counts"]["target_model_executions"] == 6
    assert result["llm_calls_made"] == 0
    assert optimizer.requests == [] and semantic.calls == [] and target.calls == []
    response_eval.evaluate.assert_not_awaited()
    assert Path(result["results_directory"]).is_dir()
    assert Path(result["checkpoints_directory"]).is_dir()


@pytest.mark.asyncio
async def test_target_runner_preserves_null_usage_and_maps_provider_errors():
    class TargetClient:
        async def complete(self, **kwargs):
            assert kwargs["messages"][0]["content"] == "same system"
            return LLMCompletion(text="answer")

    runner = TargetModelRunner(client=TargetClient())
    result = await runner.run(
        model="target/mock",
        prompt="prompt",
        temperature=0.1,
        max_output_tokens=10,
        timeout=3,
        system_instruction="same system",
    )
    assert result.response_text == "answer"
    assert result.input_tokens is result.output_tokens is result.total_tokens is None
    assert result.latency_ms is not None

    class FailingClient:
        async def complete(self, **kwargs):
            raise RuntimeError("secret key detail")

    failed = await TargetModelRunner(client=FailingClient()).run(
        model="target/mock", prompt="prompt", temperature=0, max_output_tokens=1, timeout=1
    )
    assert failed.error_code == "TARGET_MODEL_ERROR"
    assert "secret" not in (failed.error_message or "")
