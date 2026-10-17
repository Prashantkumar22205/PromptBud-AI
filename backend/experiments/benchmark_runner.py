"""Controlled, resumable orchestration for the frozen PromptOptAI benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import platform
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.config import settings
from app.evaluation.evaluator import PromptEvaluator
from app.evaluation.response_config import ResponseEvaluationSettings
from app.evaluation.response_evaluator import ResponseEvaluator
from app.evaluation.response_types import ResponseEvaluationRequest
from app.optimization.llm_client import LLMClient, LiteLLMClient
from app.optimization.optimizer import PromptOptimizer
from app.optimization.types import OptimizationError, OptimizationRequest
from app.semantic.semantic_analyzer import SemanticAnalyzer

logger = logging.getLogger(__name__)
DATASET_FILENAME = "promptoptai_benchmark_dataset_v1_1.json"
DATASET_VERSION = "1.1"
DEFAULT_SYSTEM_INSTRUCTION = "You are a helpful assistant."
CONDITIONS = ("original", "direct", "guided")


class BenchmarkError(ValueError):
    """Safe setup or dataset error shown to CLI users."""


class BenchmarkPrompt(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    prompt_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    split: str
    length_band: str = Field(pattern="^(short|medium|long)$")
    prompt: str = Field(min_length=1)
    word_count: int = Field(ge=0)


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_path: Path
    optimizer_model: str = Field(min_length=1)
    target_model: str = Field(min_length=1)
    temperature: float = Field(ge=0, le=2)
    max_output_tokens: int = Field(gt=0)
    timeout: float = Field(gt=0)
    system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION
    categories: tuple[str, ...] = ()
    length_band: str | None = Field(default=None, pattern="^(short|medium|long)$")
    limit: int | None = Field(default=None, gt=0)


class BenchmarkConditionResult(BaseModel):
    """Typed per-prompt, per-condition raw observation; additional metrics stay extensible."""

    model_config = ConfigDict(extra="allow")

    experiment_id: str
    timestamp: str
    dataset_version: str
    prompt_id: str
    category: str
    split: str
    length_band: str
    word_count: int = Field(ge=0)
    original_prompt: str
    optimizer_model: str
    target_model: str
    variant: str = Field(pattern="^(original|direct|guided)$")
    status: str = Field(pattern="^(success|error)$")
    error_code: str | None = None
    error_message: str | None = None


@dataclass
class TargetExecution:
    response_text: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    latency_ms: float | None
    model: str
    error_code: str | None = None
    error_message: str | None = None


class TargetModelRunner:
    """Run one fixed target-model request through the shared LiteLLM adapter."""

    def __init__(self, client: LLMClient | None = None):
        self.client = client or LiteLLMClient()

    async def run(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
        timeout: float,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
    ) -> TargetExecution:
        from time import perf_counter

        started = perf_counter()
        try:
            completion = await self.client.complete(
                model=model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_output_tokens,
                timeout=timeout,
            )
            elapsed_ms = max(0.0, (perf_counter() - started) * 1000.0)
            total = completion.total_tokens
            if total is None and completion.input_tokens is not None and completion.output_tokens is not None:
                total = completion.input_tokens + completion.output_tokens
            return TargetExecution(
                response_text=completion.text,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                total_tokens=total,
                latency_ms=elapsed_ms,
                model=model,
            )
        except OptimizationError as exc:
            return TargetExecution(
                response_text=None,
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                latency_ms=max(0.0, (perf_counter() - started) * 1000.0),
                model=model,
                error_code=exc.code,
                error_message=exc.message,
            )
        except Exception:
            return TargetExecution(
                response_text=None,
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                latency_ms=max(0.0, (perf_counter() - started) * 1000.0),
                model=model,
                error_code="TARGET_MODEL_ERROR",
                error_message="The target model execution failed.",
            )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _word_count(text: str) -> int:
    return len(text.split())


def _dataset_records(document: Any) -> tuple[list[dict[str, Any]], str]:
    if isinstance(document, list):
        return document, DATASET_VERSION
    if not isinstance(document, dict):
        raise BenchmarkError("Dataset root must be an object or an array of records.")
    records = next(
        (document[key] for key in ("prompts", "records", "items", "samples", "data") if isinstance(document.get(key), list)),
        None,
    )
    if records is None:
        raise BenchmarkError("Dataset has no prompt record list (expected prompts/records/items/samples/data).")
    return records, str(document.get("dataset_version", document.get("version", DATASET_VERSION)))


def resolve_dataset_path(value: str | Path, *, backend_root: Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() and candidate.is_file():
        return candidate.resolve()
    candidates = [Path.cwd() / candidate, backend_root / candidate]
    parts = candidate.parts
    if parts and parts[0].casefold() == "backend":
        candidates.append(backend_root.parent.joinpath(*parts))
        candidates.append(backend_root.joinpath(*parts[1:]))
    candidates.append(backend_root / "data" / DATASET_FILENAME)
    for item in candidates:
        if item.is_file():
            return item.resolve()
    repository_root = backend_root.parent
    discovered: list[Path] = []
    excluded = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}
    for current, directories, filenames in os.walk(repository_root):
        directories[:] = [name for name in directories if name not in excluded]
        if DATASET_FILENAME in filenames:
            discovered.append(Path(current) / DATASET_FILENAME)
    if len(discovered) == 1:
        return discovered[0].resolve()
    if len(discovered) > 1:
        raise BenchmarkError(
            f"Found multiple copies of {DATASET_FILENAME} in the repository; pass the intended frozen dataset path."
        )
    raise BenchmarkError(
        f"Frozen benchmark dataset not found: {value}. Searched the supplied path and repository; "
        f"expected {DATASET_FILENAME}."
    )


def load_benchmark_dataset(path: str | Path) -> tuple[list[BenchmarkPrompt], str, str]:
    dataset_path = Path(path).resolve()
    if not dataset_path.is_file():
        raise BenchmarkError(f"Dataset file does not exist: {dataset_path}")
    raw_bytes = dataset_path.read_bytes()
    try:
        document = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError("Dataset must be valid UTF-8 JSON.") from exc
    raw_records, dataset_version = _dataset_records(document)
    prompts: list[BenchmarkPrompt] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(raw_records):
        if not isinstance(record, dict):
            raise BenchmarkError(f"Dataset record {index} is not an object.")
        prompt_id = record.get("prompt_id", record.get("id"))
        prompt = record.get("prompt", record.get("original_prompt", record.get("text")))
        length_band = record.get("length_band", record.get("length"))
        try:
            parsed = BenchmarkPrompt(
                prompt_id=prompt_id,
                category=record.get("category"),
                split=record.get("split"),
                length_band=length_band,
                prompt=prompt,
                word_count=record.get("word_count", _word_count(prompt) if isinstance(prompt, str) else None),
                **{key: value for key, value in record.items() if key not in {"prompt_id", "id", "prompt", "original_prompt", "text", "length_band", "length", "category", "split", "word_count"}},
            )
        except (ValidationError, TypeError) as exc:
            raise BenchmarkError(f"Invalid dataset record {index}: {exc}") from exc
        if parsed.prompt_id in seen_ids:
            raise BenchmarkError(f"Duplicate prompt_id in dataset: {parsed.prompt_id}")
        seen_ids.add(parsed.prompt_id)
        prompts.append(parsed)
    return prompts, dataset_version, hashlib.sha256(raw_bytes).hexdigest()


def select_evaluation_prompts(
    prompts: Iterable[BenchmarkPrompt],
    *,
    categories: Iterable[str] = (),
    length_band: str | None = None,
    limit: int | None = None,
) -> list[BenchmarkPrompt]:
    selected = [prompt for prompt in prompts if prompt.split == "frozen_evaluation"]
    category_set = set(categories)
    if category_set:
        selected = [prompt for prompt in selected if prompt.category in category_set]
    if length_band:
        selected = [prompt for prompt in selected if prompt.length_band == length_band]
    if limit is not None:
        selected = selected[:limit]
    return selected


def validate_frozen_dataset(prompts: list[BenchmarkPrompt]) -> None:
    if len(prompts) != 150:
        raise BenchmarkError(f"Frozen v1.1 dataset must have 150 records; found {len(prompts)}.")
    development = [item for item in prompts if item.split == "development"]
    frozen = [item for item in prompts if item.split == "frozen_evaluation"]
    if len(development) != 30 or len(frozen) != 120:
        raise BenchmarkError(
            f"Expected 30 development and 120 frozen_evaluation records; found {len(development)} and {len(frozen)}."
        )
    categories: dict[str, int] = {}
    for item in prompts:
        categories[item.category] = categories.get(item.category, 0) + 1
    if len(categories) != 10 or any(count != 15 for count in categories.values()):
        raise BenchmarkError("Frozen dataset must contain 10 categories with 15 prompts each.")
    by_band = {band: sum(item.length_band == band for item in frozen) for band in ("short", "medium", "long")}
    if by_band != {"short": 40, "medium": 40, "long": 40}:
        raise BenchmarkError(f"Frozen evaluation split must have 40 prompts per length band; found {by_band}.")


def expected_execution_counts(prompt_count: int) -> dict[str, int]:
    return {
        "prompts": prompt_count,
        "target_model_executions": prompt_count * 3,
        "optimizer_calls": prompt_count * 2,
        "guided_module3_analyses": prompt_count,
        "module5a_evaluations": prompt_count * 2,
        "module5b_evaluations": prompt_count * 3,
    }


def _package_versions() -> dict[str, str | None]:
    output: dict[str, str | None] = {}
    for package in ("fastapi", "pydantic", "litellm", "pytest"):
        try:
            output[package] = version(package)
        except PackageNotFoundError:
            output[package] = None
    return output


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, timeout=3, check=True
        ).stdout.strip()
    except Exception:
        return None


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(_json_dump(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BenchmarkError(f"Invalid JSONL at {path}:{line_number}.") from exc
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _condition_csv_row(record: dict[str, Any]) -> dict[str, Any]:
    row = dict(record)
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            row[key] = _json_dump(value)
    return row


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(_condition_csv_row(row) for row in records)


def _optimization_usage(response: Any) -> dict[str, int | None] | None:
    usage = response.metadata.usage
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }


class BenchmarkRunner:
    """Run three controlled conditions and persist every attempted condition."""

    def __init__(
        self,
        *,
        optimizer: Any | None = None,
        semantic_analyzer: Any | None = None,
        target_runner: Any | None = None,
        prompt_evaluator: Any | None = None,
        response_evaluator: Any | None = None,
        backend_root: Path | None = None,
        logger_: logging.Logger | None = None,
    ):
        # Defer model/service setup until a real run reaches that condition.
        # This keeps --dry-run side-effect free and avoids loading local models.
        self._optimizer = optimizer
        self._semantic_analyzer = semantic_analyzer
        self._target_runner = target_runner
        self._prompt_evaluator = prompt_evaluator
        self._response_evaluator = response_evaluator
        self.backend_root = (backend_root or Path(__file__).resolve().parent.parent).resolve()
        self.logger = logger_ or logger

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._optimizer = PromptOptimizer()
        return self._optimizer

    @property
    def semantic_analyzer(self):
        if self._semantic_analyzer is None:
            self._semantic_analyzer = SemanticAnalyzer()
        return self._semantic_analyzer

    @property
    def target_runner(self):
        if self._target_runner is None:
            self._target_runner = TargetModelRunner()
        return self._target_runner

    @property
    def prompt_evaluator(self):
        if self._prompt_evaluator is None:
            self._prompt_evaluator = PromptEvaluator()
        return self._prompt_evaluator

    @property
    def response_evaluator(self):
        if self._response_evaluator is None:
            self._response_evaluator = ResponseEvaluator()
        return self._response_evaluator

    async def _condition_record(
        self,
        *,
        experiment_id: str,
        prompt: BenchmarkPrompt,
        variant: str,
        config: BenchmarkConfig,
    ) -> dict[str, Any]:
        base: dict[str, Any] = {
            "experiment_id": experiment_id,
            "timestamp": _now(),
            "dataset_version": DATASET_VERSION,
            "prompt_id": prompt.prompt_id,
            "category": prompt.category,
            "split": prompt.split,
            "length_band": prompt.length_band,
            "word_count": prompt.word_count,
            "original_prompt": prompt.prompt,
            "optimizer_model": config.optimizer_model,
            "target_model": config.target_model,
            "variant": variant,
            "system_instruction": config.system_instruction,
            "temperature": config.temperature,
            "max_output_tokens": config.max_output_tokens,
            "timeout": config.timeout,
            "optimization_latency_ms": None,
            "optimization_usage": None,
            "optimized_prompt": None,
            "semantic_profile": None,
            "target_response": None,
            "target_usage": None,
            "target_latency_ms": None,
            "module5a_metrics": None,
            "module5b_metrics": None,
            "estimated_cost": None,
            "status": "success",
            "error_code": None,
            "error_message": None,
            "errors": [],
            "optimizer_call_attempted": False,
            "module3_analysis_attempted": variant == "guided",
            "module5a_evaluation_attempted": False,
            "module5b_evaluation_attempted": False,
            "target_call_attempted": False,
            "original_response": None,
            "direct_optimized_prompt": None,
            "guided_optimized_prompt": None,
            "direct_response": None,
            "guided_response": None,
            "original_target_latency_ms": None,
            "direct_target_latency_ms": None,
            "guided_target_latency_ms": None,
            "original_input_tokens": None,
            "original_output_tokens": None,
            "original_total_tokens": None,
            "direct_input_tokens": None,
            "direct_output_tokens": None,
            "direct_total_tokens": None,
            "guided_input_tokens": None,
            "guided_output_tokens": None,
            "guided_total_tokens": None,
            "original_prompt_token_count": None,
            "direct_prompt_token_count": None,
            "guided_prompt_token_count": None,
            "direct_prompt_token_reduction": None,
            "guided_prompt_token_reduction": None,
            "direct_semantic_similarity": None,
            "guided_semantic_similarity": None,
            "direct_requirement_preservation": None,
            "guided_requirement_preservation": None,
            "direct_constraint_preservation": None,
            "guided_constraint_preservation": None,
            "direct_relevance": None,
            "direct_completeness": None,
            "direct_correctness": None,
            "direct_instruction_following": None,
            "guided_relevance": None,
            "guided_completeness": None,
            "guided_correctness": None,
            "guided_instruction_following": None,
            "direct_estimated_cost": None,
            "guided_estimated_cost": None,
            "direct_optimizer_latency_ms": None,
            "guided_optimizer_latency_ms": None,
            "direct_optimizer_usage": None,
            "guided_optimizer_usage": None,
        }

        def add_error(code: str, message: str) -> None:
            error = {"code": code, "message": message}
            base["errors"].append(error)
            if base["error_code"] is None:
                base["error_code"] = code
                base["error_message"] = message
            base["status"] = "error"

        request_prompt = prompt.prompt
        if variant == "direct":
            try:
                base["optimizer_call_attempted"] = True
                optimized = await self.optimizer.optimize(
                    OptimizationRequest(prompt=prompt.prompt, mode="direct", model=config.optimizer_model)
                )
                request_prompt = optimized.optimized_prompt
                base["optimized_prompt"] = request_prompt
                base["optimization_latency_ms"] = optimized.metadata.latency_ms
                base["optimization_usage"] = _optimization_usage(optimized)
                base["direct_optimized_prompt"] = request_prompt
                base["direct_optimizer_latency_ms"] = optimized.metadata.latency_ms
                base["direct_optimizer_usage"] = _optimization_usage(optimized)
                try:
                    base["module5a_evaluation_attempted"] = True
                    metrics = self.prompt_evaluator.evaluate(
                        self._module5a_request(prompt.prompt, request_prompt, "direct", config.optimizer_model)
                    )
                    base["module5a_metrics"] = metrics.metrics.model_dump(mode="json")
                    base["original_prompt_token_count"] = metrics.metrics.token_metrics.original_tokens
                    base["direct_prompt_token_count"] = metrics.metrics.token_metrics.optimized_tokens
                    base["direct_prompt_token_reduction"] = metrics.metrics.token_metrics.token_reduction_percent
                    base["direct_semantic_similarity"] = metrics.metrics.semantic_similarity.score
                    base["direct_requirement_preservation"] = metrics.metrics.requirement_preservation.preservation_score
                    base["direct_constraint_preservation"] = metrics.metrics.constraint_preservation.preservation_score
                except Exception:
                    add_error("MODULE5A_EVALUATION_FAILED", "Module 5A prompt metrics could not be calculated.")
            except OptimizationError as exc:
                add_error(exc.code, exc.message)
                return base
            except Exception:
                add_error("OPTIMIZATION_FAILED", "Direct prompt optimization failed.")
                return base
        elif variant == "guided":
            try:
                semantic_profile = self.semantic_analyzer.analyze(prompt.prompt)
                base["semantic_profile"] = semantic_profile.model_dump(mode="json")
            except Exception:
                add_error("MODULE3_ANALYSIS_FAILED", "Guided semantic analysis failed.")
                return base
            try:
                base["optimizer_call_attempted"] = True
                optimized = await self.optimizer.optimize(
                    OptimizationRequest(
                        prompt=prompt.prompt,
                        mode="guided",
                        model=config.optimizer_model,
                        semantic_profile=semantic_profile,
                    )
                )
                request_prompt = optimized.optimized_prompt
                base["optimized_prompt"] = request_prompt
                base["optimization_latency_ms"] = optimized.metadata.latency_ms
                base["optimization_usage"] = _optimization_usage(optimized)
                base["guided_optimized_prompt"] = request_prompt
                base["guided_optimizer_latency_ms"] = optimized.metadata.latency_ms
                base["guided_optimizer_usage"] = _optimization_usage(optimized)
                try:
                    base["module5a_evaluation_attempted"] = True
                    metrics = self.prompt_evaluator.evaluate(
                        self._module5a_request(prompt.prompt, request_prompt, "guided", config.optimizer_model)
                    )
                    base["module5a_metrics"] = metrics.metrics.model_dump(mode="json")
                    base["original_prompt_token_count"] = metrics.metrics.token_metrics.original_tokens
                    base["guided_prompt_token_count"] = metrics.metrics.token_metrics.optimized_tokens
                    base["guided_prompt_token_reduction"] = metrics.metrics.token_metrics.token_reduction_percent
                    base["guided_semantic_similarity"] = metrics.metrics.semantic_similarity.score
                    base["guided_requirement_preservation"] = metrics.metrics.requirement_preservation.preservation_score
                    base["guided_constraint_preservation"] = metrics.metrics.constraint_preservation.preservation_score
                except Exception:
                    add_error("MODULE5A_EVALUATION_FAILED", "Module 5A prompt metrics could not be calculated.")
            except OptimizationError as exc:
                add_error(exc.code, exc.message)
                return base
            except Exception:
                add_error("OPTIMIZATION_FAILED", "Guided prompt optimization failed.")
                return base

        try:
            base["target_call_attempted"] = True
            target = await self.target_runner.run(
                model=config.target_model,
                prompt=request_prompt,
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                timeout=config.timeout,
                system_instruction=config.system_instruction,
            )
        except Exception:
            add_error("TARGET_MODEL_ERROR", "The target model execution failed.")
            return base

        base["target_response"] = target.response_text
        base["target_latency_ms"] = target.latency_ms
        base["target_usage"] = {
            "input_tokens": target.input_tokens,
            "output_tokens": target.output_tokens,
            "total_tokens": target.total_tokens,
        }
        prefix = "original" if variant == "original" else variant
        base[f"{prefix}_target_latency_ms"] = target.latency_ms
        base[f"{prefix}_input_tokens"] = target.input_tokens
        base[f"{prefix}_output_tokens"] = target.output_tokens
        base[f"{prefix}_total_tokens"] = target.total_tokens
        if variant == "original":
            base["original_response"] = target.response_text
        elif variant == "direct":
            base["direct_response"] = target.response_text
        else:
            base["guided_response"] = target.response_text
        if target.error_code:
            add_error(target.error_code, target.error_message or "The target model execution failed.")
            return base

        try:
            base["module5b_evaluation_attempted"] = True
            body = ResponseEvaluationRequest(
                original_prompt=prompt.prompt,
                evaluation_prompt=request_prompt,
                response=target.response_text or "",
                variant=variant,
                target_model=target.model,
                input_tokens=target.input_tokens,
                output_tokens=target.output_tokens,
                total_tokens=target.total_tokens,
                latency_ms=target.latency_ms,
                generation_temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
            )
            evaluation = await self.response_evaluator.evaluate(body)
            base["module5b_metrics"] = evaluation.metrics.model_dump(mode="json")
            base["estimated_cost"] = evaluation.metrics.cost.estimated_cost
            scores = evaluation.metrics.response_quality.model_dump(mode="json")
            for dimension in ("relevance", "completeness", "correctness", "instruction_following"):
                base[f"{variant}_{dimension}"] = scores[dimension]["score"]
            if variant in ("direct", "guided"):
                base[f"{variant}_estimated_cost"] = evaluation.metrics.cost.estimated_cost
        except Exception:
            add_error("MODULE5B_EVALUATION_FAILED", "Module 5B response evaluation failed.")
        return base

    @staticmethod
    def _module5a_request(original: str, optimized: str, mode: str, optimizer_model: str):
        from app.evaluation.types import EvaluationRequest

        return EvaluationRequest(
            original_prompt=original,
            optimized_prompt=optimized,
            mode=mode,
            optimizer_model=optimizer_model,
        )

    async def run(
        self,
        config: BenchmarkConfig,
        *,
        results_root: Path | None = None,
        checkpoints_root: Path | None = None,
        logs_root: Path | None = None,
        resume: bool = False,
        dry_run: bool = False,
        experiment_id: str | None = None,
    ) -> dict[str, Any]:
        dataset_path = resolve_dataset_path(config.dataset_path, backend_root=self.backend_root)
        prompts, dataset_version, dataset_hash = load_benchmark_dataset(dataset_path)
        validate_frozen_dataset(prompts)
        selected = select_evaluation_prompts(
            prompts, categories=config.categories, length_band=config.length_band, limit=config.limit
        )
        if not selected:
            raise BenchmarkError("No frozen_evaluation prompts match the selected filters.")
        if not config.optimizer_model.strip() or not config.target_model.strip():
            raise BenchmarkError("Optimizer and target model identifiers must be configured.")
        try:
            response_settings = ResponseEvaluationSettings()
        except ValidationError as exc:
            raise BenchmarkError("Response evaluator configuration is invalid.") from exc
        response_evaluator_model = response_settings.configured_evaluator_model()
        if self._response_evaluator is None and not response_evaluator_model:
            raise BenchmarkError("Set RESPONSE_EVALUATOR_MODEL before running the benchmark.")
        if response_evaluator_model and response_evaluator_model == config.optimizer_model.strip():
            raise BenchmarkError("The independent response evaluator must not use the optimizer model identifier.")

        default_root = self.backend_root / "experiments" / "benchmark"
        results_root = (results_root or default_root / "results").resolve()
        checkpoints_root = (checkpoints_root or default_root / "checkpoints").resolve()
        logs_root = (logs_root or default_root / "logs").resolve()
        results_root.mkdir(parents=True, exist_ok=True)
        checkpoints_root.mkdir(parents=True, exist_ok=True)
        logs_root.mkdir(parents=True, exist_ok=True)

        if resume:
            if not experiment_id:
                raise BenchmarkError("--resume requires an experiment id.")
        else:
            experiment_id = experiment_id or f"benchmark-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
        assert experiment_id is not None
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}", experiment_id):
            raise BenchmarkError("Experiment id may contain only letters, digits, dot, underscore, and hyphen.")
        run_results = results_root / experiment_id
        run_checkpoints = checkpoints_root / experiment_id
        run_logs = logs_root / experiment_id
        run_results.mkdir(parents=True, exist_ok=True)
        run_checkpoints.mkdir(parents=True, exist_ok=True)
        run_logs.mkdir(parents=True, exist_ok=True)
        metadata_path = run_results / "experiment_metadata.json"
        checkpoint_path = run_checkpoints / "condition_checkpoints.jsonl"

        expected = expected_execution_counts(len(selected))
        if dry_run:
            return {
                "dry_run": True,
                "experiment_id": experiment_id,
                "dataset_path": str(dataset_path),
                "dataset_version": dataset_version,
                "dataset_hash": dataset_hash,
                "selected_prompt_count": len(selected),
                "expected_experiment_counts": expected,
                "results_directory": str(run_results),
                "checkpoints_directory": str(run_checkpoints),
                "logs_directory": str(run_logs),
                "llm_calls_made": 0,
            }

        existing = _read_jsonl(checkpoint_path) if resume else []
        attempted = {(item.get("prompt_id"), item.get("variant")): item for item in existing}
        if not resume and any(run_results.iterdir()):
            raise BenchmarkError(f"Experiment output already exists: {run_results}. Use --resume or a new experiment id.")

        filters = {
            "categories": list(config.categories),
            "length_band": config.length_band,
            "limit": config.limit,
        }
        if resume and not metadata_path.exists():
            raise BenchmarkError("Cannot resume: the experiment metadata file is missing.")
        if not metadata_path.exists():
            metadata = {
                "experiment_id": experiment_id,
                "dataset_name": dataset_path.name,
                "dataset_version": dataset_version,
                "dataset_path": str(dataset_path),
                "dataset_hash": dataset_hash,
                "optimizer_model": config.optimizer_model,
                "target_model": config.target_model,
                "temperature": config.temperature,
                "max_output_tokens": config.max_output_tokens,
                "timeout": config.timeout,
                "system_instruction": config.system_instruction,
                "optimizer_temperature": settings.optimization_temperature,
                "optimizer_max_output_tokens": settings.optimization_max_output_tokens,
                "optimizer_timeout": settings.optimization_timeout_seconds,
                "response_evaluator_model": response_evaluator_model,
                "response_evaluator_temperature": response_settings.response_evaluator_temperature,
                "response_evaluator_max_output_tokens": response_settings.response_evaluator_max_output_tokens,
                "response_evaluator_timeout": response_settings.response_evaluator_timeout_seconds,
                "target_model_pricing_json": response_settings.target_model_pricing_json,
                "module5a_tokenizer": settings.default_tokenizer,
                "module5a_embedding_model": settings.evaluation_embedding_model,
                "module3_intent_classifier_type": settings.intent_classifier_type,
                "module3_intent_hybrid_alpha": settings.intent_hybrid_alpha,
                "python_version": platform.python_version(),
                "package_versions": _package_versions(),
                "git_commit": _git_commit(self.backend_root),
                "start_time": _now(),
                "expected_count": expected["target_model_executions"],
                "expected_experiment_counts": expected,
                "filters": filters,
            }
            _write_json(metadata_path, metadata)
            _write_json(results_root / "experiment_metadata.json", metadata)
        else:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("dataset_hash") != dataset_hash:
                raise BenchmarkError("Resume dataset hash differs from the original experiment; refusing to continue.")
            fixed_values = {
                "optimizer_model": config.optimizer_model,
                "target_model": config.target_model,
                "temperature": config.temperature,
                "max_output_tokens": config.max_output_tokens,
                "timeout": config.timeout,
                "system_instruction": config.system_instruction,
                "optimizer_temperature": settings.optimization_temperature,
                "optimizer_max_output_tokens": settings.optimization_max_output_tokens,
                "optimizer_timeout": settings.optimization_timeout_seconds,
                "response_evaluator_model": response_evaluator_model,
                "response_evaluator_temperature": response_settings.response_evaluator_temperature,
                "response_evaluator_max_output_tokens": response_settings.response_evaluator_max_output_tokens,
                "response_evaluator_timeout": response_settings.response_evaluator_timeout_seconds,
                "target_model_pricing_json": response_settings.target_model_pricing_json,
                "module5a_tokenizer": settings.default_tokenizer,
                "module5a_embedding_model": settings.evaluation_embedding_model,
                "module3_intent_classifier_type": settings.intent_classifier_type,
                "module3_intent_hybrid_alpha": settings.intent_hybrid_alpha,
                "python_version": platform.python_version(),
                "package_versions": _package_versions(),
                "git_commit": _git_commit(self.backend_root),
                "filters": filters,
            }
            if any(metadata.get(key) != value for key, value in fixed_values.items()):
                raise BenchmarkError("Resume configuration differs from the original experiment; refusing to continue.")

        condition_records = dict(attempted)

        def publish_current_results() -> list[dict[str, Any]]:
            current_rows = [
                condition_records[(prompt.prompt_id, variant)]
                for prompt in selected
                for variant in CONDITIONS
                if (prompt.prompt_id, variant) in condition_records
            ]
            for output_dir in (run_results, results_root):
                (output_dir / "raw_results.jsonl").write_text(
                    "".join(_json_dump(row) + "\n" for row in current_rows), encoding="utf-8"
                )
                _write_csv(output_dir / "raw_results.csv", current_rows)
            return current_rows

        publish_current_results()
        _write_json(results_root / "experiment_metadata.json", metadata)
        log_path = run_logs / "benchmark.log"
        file_logger = logging.FileHandler(log_path, encoding="utf-8")
        self.logger.addHandler(file_logger)
        try:
            for prompt in selected:
                for variant in CONDITIONS:
                    key = (prompt.prompt_id, variant)
                    if condition_records.get(key, {}).get("status") == "success":
                        continue
                    record = await self._condition_record(
                        experiment_id=experiment_id, prompt=prompt, variant=variant, config=config
                    )
                    record["dataset_version"] = dataset_version
                    record = BenchmarkConditionResult.model_validate(record).model_dump(mode="json")
                    _append_jsonl(checkpoint_path, record)
                    condition_records[key] = record
                    publish_current_results()
                    self.logger.info("prompt=%s variant=%s status=%s", prompt.prompt_id, variant, record["status"])
        finally:
            self.logger.removeHandler(file_logger)
            file_logger.close()

        rows = publish_current_results()

        failed = sum(record.get("status") == "error" for record in rows)
        complete_count = sum(record.get("status") == "success" for record in rows)
        summary = {
            "experiment_id": experiment_id,
            "dataset_version": dataset_version,
            "dataset_hash": dataset_hash,
            "selected_prompt_count": len(selected),
            "condition_records_written": len(rows),
            "completed_count": complete_count,
            "failed_count": failed,
            "expected_count": expected["target_model_executions"],
            "expected_experiment_counts": expected,
            "actual_counts": {
                "optimizer_calls": sum(bool(row.get("optimizer_call_attempted")) for row in rows),
                "guided_module3_analyses": sum(bool(row.get("module3_analysis_attempted")) for row in rows),
                "target_model_attempts": sum(bool(row.get("target_call_attempted")) for row in rows),
                "module5a_evaluations": sum(bool(row.get("module5a_evaluation_attempted")) for row in rows),
                "module5b_evaluations": sum(bool(row.get("module5b_evaluation_attempted")) for row in rows),
            },
            "end_time": _now(),
        }
        metadata.update({
            "end_time": summary["end_time"],
            "completed_count": complete_count,
            "failed_count": failed,
            "expected_count": expected["target_model_executions"],
        })
        _write_json(metadata_path, metadata)
        _write_json(run_results / "execution_summary.json", summary)
        _write_json(results_root / "experiment_metadata.json", metadata)
        _write_json(results_root / "execution_summary.json", summary)
        return summary


def make_config(
    *,
    dataset: str | Path,
    optimizer_model: str,
    target_model: str,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    timeout: float | None = None,
    categories: Iterable[str] = (),
    length_band: str | None = None,
    limit: int | None = None,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        dataset_path=Path(dataset),
        optimizer_model=optimizer_model,
        target_model=target_model,
        temperature=settings.optimization_temperature if temperature is None else temperature,
        max_output_tokens=settings.optimization_max_output_tokens if max_output_tokens is None else max_output_tokens,
        timeout=settings.optimization_timeout_seconds if timeout is None else timeout,
        categories=tuple(categories),
        length_band=length_band,
        limit=limit,
    )
