"""
Temporary Paper 1 downstream replay pilot.

Purpose:
    Reuse the EXACT Direct and Guided optimized prompts already produced in
    prompt_level_groq_pilot_results(2).json. Do NOT call Module 3, Module 4,
    or Module 5A again.

For each selected prompt:
    Original prompt -> target LLM -> Module 5B
    Stored Direct prompt -> target LLM -> Module 5B
    Stored Guided prompt -> target LLM -> Module 5B

The script deliberately does not modify the frozen benchmark runner.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.evaluation.response_config import ResponseEvaluationSettings
from app.evaluation.response_evaluator import ResponseEvaluator
from app.evaluation.response_types import ResponseEvaluationRequest
from experiments.benchmark_runner import TargetModelRunner


DATASET_VERSION = "1.1"
EXPECTED_DATASET_HASH = "69a714ce801110ce80df457beab55bfa4f7ad86d547b0fe8c78733f75eb8b7b9"
DEFAULT_SYSTEM_INSTRUCTION = "You are a helpful assistant."
BANDS = ("short", "medium", "long")
VARIANTS = ("original", "direct", "guided")


def load_results(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError("Input JSON does not contain a top-level 'results' list.")

    experiment = data.get("experiment", {})
    if experiment.get("dataset_version") != DATASET_VERSION:
        raise ValueError(
            f"Expected dataset version {DATASET_VERSION}, "
            f"found {experiment.get('dataset_version')!r}."
        )
    if experiment.get("dataset_hash") != EXPECTED_DATASET_HASH:
        raise ValueError("Frozen dataset SHA-256 does not match the expected hash.")

    return data


def eligible_records(data: dict[str, Any]) -> list[dict[str, Any]]:
    eligible = []

    for record in data["results"]:
        prompt = record.get("prompt", {})
        direct = record.get("direct", {})
        guided = record.get("guided", {})

        if (
            direct.get("optimization_success") is True
            and guided.get("optimization_success") is True
            and isinstance(direct.get("optimized_prompt"), str)
            and direct["optimized_prompt"].strip()
            and isinstance(guided.get("optimized_prompt"), str)
            and guided["optimized_prompt"].strip()
        ):
            eligible.append(record)

    return eligible


def select_pilot_records(
    records: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """
    Deterministic representative pilot:
      - one successful paired prompt per category;
      - cycles through short/medium/long across categories;
      - never selects a record lacking either optimized prompt.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}

    for record in records:
        category = record["prompt"]["category"]
        grouped.setdefault(category, []).append(record)

    selected: list[dict[str, Any]] = []

    for category_index, category in enumerate(sorted(grouped)):
        if len(selected) >= limit:
            break

        preferred_band = BANDS[category_index % len(BANDS)]
        candidates = [
            r
            for r in grouped[category]
            if r["prompt"]["length_band"] == preferred_band
        ]

        if not candidates:
            candidates = grouped[category]

        candidates = sorted(
            candidates, key=lambda r: r["prompt"]["prompt_id"]
        )
        selected.append(candidates[0])

    # If a caller asks for > number of categories, fill deterministically.
    if len(selected) < limit:
        selected_ids = {r["prompt"]["prompt_id"] for r in selected}
        remaining = sorted(
            (
                r
                for r in records
                if r["prompt"]["prompt_id"] not in selected_ids
            ),
            key=lambda r: r["prompt"]["prompt_id"],
        )
        selected.extend(remaining[: limit - len(selected)])

    return selected[:limit]


def validate_evaluator(optimizer_model: str) -> str:
    settings = ResponseEvaluationSettings()
    evaluator_model = settings.configured_evaluator_model()

    if not evaluator_model:
        raise ValueError(
            "RESPONSE_EVALUATOR_MODEL is not configured. "
            "Configure an independent Module 5B evaluator first."
        )

    if evaluator_model.strip() == optimizer_model.strip():
        raise ValueError(
            "The Module 5B evaluator must be independent from the optimizer model."
        )

    return evaluator_model


async def evaluate_variant(
    *,
    evaluator: ResponseEvaluator,
    target_runner: TargetModelRunner,
    prompt_id: str,
    original_prompt: str,
    evaluation_prompt: str,
    variant: str,
    target_model: str,
    temperature: float,
    max_output_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    target = await target_runner.run(
        model=target_model,
        prompt=evaluation_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        timeout=timeout,
        system_instruction=DEFAULT_SYSTEM_INSTRUCTION,
    )

    result: dict[str, Any] = {
        "prompt_id": prompt_id,
        "variant": variant,
        "evaluation_prompt": evaluation_prompt,
        "target_model": target_model,
        "target_response": target.response_text,
        "target_usage": {
            "input_tokens": target.input_tokens,
            "output_tokens": target.output_tokens,
            "total_tokens": target.total_tokens,
        },
        "target_latency_ms": target.latency_ms,
        "target_error_code": target.error_code,
        "target_error_message": target.error_message,
        "module5b_metrics": None,
        "module5b_error": None,
    }

    if target.error_code or target.response_text is None:
        return result

    try:
        request = ResponseEvaluationRequest(
            original_prompt=original_prompt,
            evaluation_prompt=evaluation_prompt,
            response=target.response_text,
            variant=variant,
            target_model=target.model,
            input_tokens=target.input_tokens,
            output_tokens=target.output_tokens,
            total_tokens=target.total_tokens,
            latency_ms=target.latency_ms,
            generation_temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        evaluation = await evaluator.evaluate(request)
        result["module5b_metrics"] = evaluation.metrics.model_dump(mode="json")
    except Exception as exc:
        result["module5b_error"] = f"{type(exc).__name__}: {exc}"

    return result


async def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.results_json).resolve()
    data = load_results(input_path)

    optimizer_model = data["experiment"]["optimizer_model"]
    evaluator_model = validate_evaluator(optimizer_model)

    target_model = args.target_model or os.getenv("TARGET_MODEL")
    if not target_model:
        raise ValueError(
            "Pass --target-model or set TARGET_MODEL in the environment."
        )

    all_eligible = eligible_records(data)
    if not all_eligible:
        raise ValueError("No records have both successful Direct and Guided prompts.")

    selected = select_pilot_records(all_eligible, args.limit)

    evaluator = ResponseEvaluator()
    target_runner = TargetModelRunner()

    experiment_id = (
        f"downstream-replay-pilot-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )

    output_records: list[dict[str, Any]] = []

    for record in selected:
        prompt = record["prompt"]
        direct = record["direct"]
        guided = record["guided"]

        prompt_id = prompt["prompt_id"]
        original_prompt = prompt["original_prompt"]

        prompts = {
            "original": original_prompt,
            "direct": direct["optimized_prompt"],
            "guided": guided["optimized_prompt"],
        }

        print(
            f"\n[{prompt_id}] {prompt['category']} / "
            f"{prompt['length_band']}"
        )

        record_output = {
            "prompt_id": prompt_id,
            "category": prompt["category"],
            "length_band": prompt["length_band"],
            "word_count": prompt["word_count"],
            "original_prompt": original_prompt,
            "optimizer_model_from_source_result": optimizer_model,
            "source_result_dataset_version": DATASET_VERSION,
            "source_result_dataset_hash": EXPECTED_DATASET_HASH,
            "variants": {},
        }

        for variant in VARIANTS:
            print(f"  -> {variant}")

            evaluated = await evaluate_variant(
                evaluator=evaluator,
                target_runner=target_runner,
                prompt_id=prompt_id,
                original_prompt=original_prompt,
                evaluation_prompt=prompts[variant],
                variant=variant,
                target_model=target_model,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                timeout=args.timeout,
            )

            # Do not duplicate large prompt/response text in the terminal.
            status = (
                "target_error"
                if evaluated["target_error_code"]
                else "module5b_error"
                if evaluated["module5b_error"]
                else "success"
            )
            print(f"     status={status}")

            record_output["variants"][variant] = evaluated

        output_records.append(record_output)

    output = {
        "experiment": {
            "name": "Paper 1 downstream replay pilot",
            "experiment_id": experiment_id,
            "source_results": str(input_path),
            "source_optimizer_model": optimizer_model,
            "target_model": target_model,
            "response_evaluator_model": evaluator_model,
            "dataset_version": DATASET_VERSION,
            "dataset_hash": EXPECTED_DATASET_HASH,
            "prompt_count": len(selected),
            "module3_used": False,
            "module4_used": False,
            "module5a_used": False,
            "module5b_used": True,
            "methodology": (
                "Reuse stored Original, Direct-optimized, and "
                "Guided-optimized prompts from the completed prompt-level "
                "experiment; execute all three through the same target LLM "
                "and evaluate responses with independent Module 5B."
            ),
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
            "timeout": args.timeout,
            "system_instruction": DEFAULT_SYSTEM_INSTRUCTION,
        },
        "selected_prompt_ids": [r["prompt_id"] for r in output_records],
        "results": output_records,
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {output_path}")
    print(f"Prompts: {len(output_records)}")
    print(f"Target executions: {len(output_records) * 3}")
    print(f"Module 5B evaluations attempted: {len(output_records) * 3}")

    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Temporary Paper 1 downstream replay pilot."
    )
    parser.add_argument(
        "--results-json",
        required=True,
        help="Existing prompt_level_groq_pilot_results(2).json",
    )
    parser.add_argument(
        "--target-model",
        default=None,
        help="Target LLM. If omitted, uses TARGET_MODEL.",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--output",
        default="experiments/prompt_level_downstream_replay_pilot_results.json",
    )

    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be >= 1")

    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
