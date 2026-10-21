from __future__ import annotations

import asyncio
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from app.evaluation.evaluator import PromptEvaluator
from app.evaluation.types import EvaluationRequest
from app.optimization.optimizer import OptimizationRequest, PromptOptimizer
from app.semantic.semantic_analyzer import SemanticAnalyzer
from experiments.benchmark_runner import (
    load_benchmark_dataset,
    select_evaluation_prompts,
)


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = Path("data/promptoptai_benchmark_dataset_v1_1.json")

OPTIMIZER_MODEL = "groq/openai/gpt-oss-120b"

PILOT_LIMIT = 120

OUTPUT_PATH = Path(
    "experiments/prompt_level_groq_pilot_results.json"
)


# ============================================================
# Helpers
# ============================================================

def module5a_request(
    original_prompt: str,
    optimized_prompt: str,
    mode: str,
) -> EvaluationRequest:
    """
    Create the exact Module 5A request used by benchmark_runner.py.
    """
    return EvaluationRequest(
        original_prompt=original_prompt,
        optimized_prompt=optimized_prompt,
        mode=mode,
        optimizer_model=OPTIMIZER_MODEL,
    )


def metric_value(metrics: Any, path: str, default=None):
    """
    Safely retrieve nested metric values.
    """
    current = metrics

    for part in path.split("."):
        if current is None:
            return default

        if isinstance(current, dict):
            current = current.get(part, default)
        else:
            current = getattr(current, part, default)

    return current


def safe_float(value):
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def average(values):
    values = [
        safe_float(value)
        for value in values
        if safe_float(value) is not None
    ]

    if not values:
        return None

    return mean(values)


# ============================================================
# Direct optimization
# ============================================================

async def run_direct(
    prompt_optimizer: PromptOptimizer,
    prompt_evaluator: PromptEvaluator,
    prompt,
) -> dict[str, Any]:

    started = perf_counter()

    result = {
        "prompt_id": prompt.prompt_id,
        "category": prompt.category,
        "length_band": prompt.length_band,
        "word_count": prompt.word_count,
        "original_prompt": prompt.prompt,

        "variant": "direct",

        "optimizer_model": OPTIMIZER_MODEL,

        "optimization_success": False,
        "evaluation_success": False,

        "optimized_prompt": None,

        "optimization_latency_ms": None,
        "optimization_usage": None,

        "module5a_metrics": None,

        "prompt_token_reduction_percent": None,
        "semantic_similarity": None,
        "requirement_preservation": None,
        "constraint_preservation": None,

        "error": None,
    }

    try:
        # ----------------------------------------------------
        # Module 4 - Direct optimization
        # ----------------------------------------------------

        optimization = await prompt_optimizer.optimize(
            OptimizationRequest(
                prompt=prompt.prompt,
                mode="direct",
                model=OPTIMIZER_MODEL,
            )
        )

        result["optimization_success"] = True

        result["optimized_prompt"] = optimization.optimized_prompt

        result["optimization_latency_ms"] = (
            optimization.metadata.latency_ms
        )

        usage = getattr(optimization.metadata, "usage", None)

        if usage is not None:
            if hasattr(usage, "model_dump"):
                result["optimization_usage"] = usage.model_dump(
                    mode="json"
                )
            else:
                result["optimization_usage"] = usage

        # ----------------------------------------------------
        # Module 5A - Prompt-level evaluation
        # ----------------------------------------------------

        evaluation_request = module5a_request(
            original_prompt=prompt.prompt,
            optimized_prompt=optimization.optimized_prompt,
            mode="direct",
        )

        metrics_result = prompt_evaluator.evaluate(
            evaluation_request
        )

        result["evaluation_success"] = True

        if hasattr(metrics_result, "metrics"):
            metrics = metrics_result.metrics
        else:
            metrics = metrics_result

        if hasattr(metrics, "model_dump"):
            result["module5a_metrics"] = metrics.model_dump(
                mode="json"
            )
        else:
            result["module5a_metrics"] = metrics

        # Same metric fields used by benchmark_runner.py
        result["prompt_token_reduction_percent"] = (
            metrics.token_metrics.token_reduction_percent
        )

        result["semantic_similarity"] = (
            metrics.semantic_similarity.score
        )

        result["requirement_preservation"] = (
            metrics.requirement_preservation.preservation_score
        )

        result["constraint_preservation"] = (
            metrics.constraint_preservation.preservation_score
        )

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["total_script_latency_ms"] = (
        perf_counter() - started
    ) * 1000

    return result


# ============================================================
# Guided optimization
# ============================================================

async def run_guided(
    prompt_optimizer: PromptOptimizer,
    prompt_evaluator: PromptEvaluator,
    semantic_analyzer: SemanticAnalyzer,
    prompt,
) -> dict[str, Any]:

    started = perf_counter()

    result = {
        "prompt_id": prompt.prompt_id,
        "category": prompt.category,
        "length_band": prompt.length_band,
        "word_count": prompt.word_count,
        "original_prompt": prompt.prompt,

        "variant": "guided",

        "optimizer_model": OPTIMIZER_MODEL,

        "module3_success": False,
        "optimization_success": False,
        "evaluation_success": False,

        "semantic_profile": None,

        "optimized_prompt": None,

        "optimization_latency_ms": None,
        "optimization_usage": None,

        "module5a_metrics": None,

        "prompt_token_reduction_percent": None,
        "semantic_similarity": None,
        "requirement_preservation": None,
        "constraint_preservation": None,

        "error": None,
    }

    try:
        # ----------------------------------------------------
        # Module 3 - Semantic Analysis
        # ----------------------------------------------------

        semantic_profile = semantic_analyzer.analyze(
            prompt.prompt
        )

        result["module3_success"] = True

        if hasattr(semantic_profile, "model_dump"):
            result["semantic_profile"] = (
                semantic_profile.model_dump(mode="json")
            )
        else:
            result["semantic_profile"] = semantic_profile

        # ----------------------------------------------------
        # Module 4 - Guided optimization
        # ----------------------------------------------------

        optimization = await prompt_optimizer.optimize(
            OptimizationRequest(
                prompt=prompt.prompt,
                mode="guided",
                model=OPTIMIZER_MODEL,
                semantic_profile=semantic_profile,
            )
        )

        result["optimization_success"] = True

        result["optimized_prompt"] = optimization.optimized_prompt

        result["optimization_latency_ms"] = (
            optimization.metadata.latency_ms
        )

        usage = getattr(optimization.metadata, "usage", None)

        if usage is not None:
            if hasattr(usage, "model_dump"):
                result["optimization_usage"] = usage.model_dump(
                    mode="json"
                )
            else:
                result["optimization_usage"] = usage

        # ----------------------------------------------------
        # Module 5A - Prompt-level evaluation
        # ----------------------------------------------------

        evaluation_request = module5a_request(
            original_prompt=prompt.prompt,
            optimized_prompt=optimization.optimized_prompt,
            mode="guided",
        )

        metrics_result = prompt_evaluator.evaluate(
            evaluation_request
        )

        result["evaluation_success"] = True

        if hasattr(metrics_result, "metrics"):
            metrics = metrics_result.metrics
        else:
            metrics = metrics_result

        if hasattr(metrics, "model_dump"):
            result["module5a_metrics"] = metrics.model_dump(
                mode="json"
            )
        else:
            result["module5a_metrics"] = metrics

        # Same metric fields used by benchmark_runner.py
        result["prompt_token_reduction_percent"] = (
            metrics.token_metrics.token_reduction_percent
        )

        result["semantic_similarity"] = (
            metrics.semantic_similarity.score
        )

        result["requirement_preservation"] = (
            metrics.requirement_preservation.preservation_score
        )

        result["constraint_preservation"] = (
            metrics.constraint_preservation.preservation_score
        )

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    result["total_script_latency_ms"] = (
        perf_counter() - started
    ) * 1000

    return result


# ============================================================
# Main
# ============================================================

async def main():

    print("=" * 90)
    print("PromptOptAI - 10 Prompt Prompt-Level Groq Pilot")
    print("=" * 90)
    print(f"Optimizer : {OPTIMIZER_MODEL}")
    print(f"Dataset   : {DATASET_PATH}")
    print(f"Prompts   : {PILOT_LIMIT}")
    print("=" * 90)

    # --------------------------------------------------------
    # Load frozen dataset using the project's existing loader
    # --------------------------------------------------------

    prompts, dataset_version, dataset_hash = (
        load_benchmark_dataset(DATASET_PATH)
    )

    selected_prompts = select_evaluation_prompts(
        prompts,
        limit=PILOT_LIMIT,
    )

    if not selected_prompts:
        raise RuntimeError(
            "No frozen_evaluation prompts found."
        )

    print()
    print(f"Dataset version : {dataset_version}")
    print(f"Dataset hash    : {dataset_hash}")
    print()

    print("Selected prompts:")

    for prompt in selected_prompts:
        print(
            f"  {prompt.prompt_id:10s} | "
            f"{prompt.category:25s} | "
            f"{prompt.length_band:6s} | "
            f"{prompt.word_count} words"
        )

    print()
    print("=" * 90)

    # --------------------------------------------------------
    # Create existing project components
    # --------------------------------------------------------

    prompt_optimizer = PromptOptimizer()

    prompt_evaluator = PromptEvaluator()

    semantic_analyzer = SemanticAnalyzer()

    results = []

    # ========================================================
    # Process prompts
    # ========================================================

    for index, prompt in enumerate(selected_prompts, start=1):

        print()
        print(
            f"[{index}/{len(selected_prompts)}] "
            f"{prompt.prompt_id} - {prompt.category}"
        )

        # ----------------------------------------------------
        # DIRECT
        # ----------------------------------------------------

        print("  → Direct optimization...")

        direct_result = await run_direct(
            prompt_optimizer=prompt_optimizer,
            prompt_evaluator=prompt_evaluator,
            prompt=prompt,
        )

        if direct_result["optimization_success"]:
            print(
                "    ✓ Direct optimization successful "
                f"({direct_result['optimization_latency_ms']:.0f} ms)"
            )
        else:
            print(
                "    ✗ Direct optimization failed: "
                f"{direct_result['error']}"
            )

        if direct_result["evaluation_success"]:
            print(
                "    ✓ Direct Module 5A evaluation successful"
            )

        # ----------------------------------------------------
        # GUIDED
        # ----------------------------------------------------

        print("  → Guided optimization...")

        guided_result = await run_guided(
            prompt_optimizer=prompt_optimizer,
            prompt_evaluator=prompt_evaluator,
            semantic_analyzer=semantic_analyzer,
            prompt=prompt,
        )

        if guided_result["module3_success"]:
            print("    ✓ Module 3 analysis successful")
        else:
            print("    ✗ Module 3 analysis failed")

        if guided_result["optimization_success"]:
            print(
                "    ✓ Guided optimization successful "
                f"({guided_result['optimization_latency_ms']:.0f} ms)"
            )
        else:
            print(
                "    ✗ Guided optimization failed: "
                f"{guided_result['error']}"
            )

        if guided_result["evaluation_success"]:
            print(
                "    ✓ Guided Module 5A evaluation successful"
            )

        # ----------------------------------------------------
        # Store both
        # ----------------------------------------------------

        results.append(
            {
                "prompt": {
                    "prompt_id": prompt.prompt_id,
                    "category": prompt.category,
                    "split": prompt.split,
                    "length_band": prompt.length_band,
                    "word_count": prompt.word_count,
                    "original_prompt": prompt.prompt,
                },
                "direct": direct_result,
                "guided": guided_result,
            }
        )

    # ========================================================
    # Calculate summary
    # ========================================================

    direct_results = [
        item["direct"]
        for item in results
    ]

    guided_results = [
        item["guided"]
        for item in results
    ]

    direct_success = [
        item
        for item in direct_results
        if item["evaluation_success"]
    ]

    guided_success = [
        item
        for item in guided_results
        if item["evaluation_success"]
    ]

    summary = {
        "total_prompts": len(selected_prompts),

        "direct": {
            "optimization_success_count": sum(
                item["optimization_success"]
                for item in direct_results
            ),
            "module5a_success_count": len(direct_success),

            "average_token_reduction_percent": average(
                item["prompt_token_reduction_percent"]
                for item in direct_success
            ),

            "average_semantic_similarity": average(
                item["semantic_similarity"]
                for item in direct_success
            ),

            "average_requirement_preservation": average(
                item["requirement_preservation"]
                for item in direct_success
            ),

            "average_constraint_preservation": average(
                item["constraint_preservation"]
                for item in direct_success
            ),

            "average_optimization_latency_ms": average(
                item["optimization_latency_ms"]
                for item in direct_results
                if item["optimization_success"]
            ),
        },

        "guided": {
            "module3_success_count": sum(
                item["module3_success"]
                for item in guided_results
            ),
            "optimization_success_count": sum(
                item["optimization_success"]
                for item in guided_results
            ),
            "module5a_success_count": len(guided_success),

            "average_token_reduction_percent": average(
                item["prompt_token_reduction_percent"]
                for item in guided_success
            ),

            "average_semantic_similarity": average(
                item["semantic_similarity"]
                for item in guided_success
            ),

            "average_requirement_preservation": average(
                item["requirement_preservation"]
                for item in guided_success
            ),

            "average_constraint_preservation": average(
                item["constraint_preservation"]
                for item in guided_success
            ),

            "average_optimization_latency_ms": average(
                item["optimization_latency_ms"]
                for item in guided_results
                if item["optimization_success"]
            ),
        },
    }

    # ========================================================
    # Print results table
    # ========================================================

    print()
    print()
    print("=" * 110)
    print("PROMPT-LEVEL PILOT RESULTS")
    print("=" * 110)

    print(
        f"{'Prompt':<10}"
        f"{'Direct Red.':>14}"
        f"{'Guided Red.':>14}"
        f"{'Direct Sim.':>14}"
        f"{'Guided Sim.':>14}"
        f"{'Direct Req.':>14}"
        f"{'Guided Req.':>14}"
    )

    print("-" * 110)

    for item in results:

        direct = item["direct"]
        guided = item["guided"]

        def fmt(value):
            if value is None:
                return "N/A"

            return f"{float(value):.2f}"

        print(
            f"{item['prompt']['prompt_id']:<10}"
            f"{fmt(direct['prompt_token_reduction_percent']):>14}"
            f"{fmt(guided['prompt_token_reduction_percent']):>14}"
            f"{fmt(direct['semantic_similarity']):>14}"
            f"{fmt(guided['semantic_similarity']):>14}"
            f"{fmt(direct['requirement_preservation']):>14}"
            f"{fmt(guided['requirement_preservation']):>14}"
        )

    print("-" * 110)

    print()
    print("AVERAGES")
    print()

    print(
        "Direct:"
    )

    print(
        f"  Token reduction        : "
        f"{summary['direct']['average_token_reduction_percent']}"
    )

    print(
        f"  Semantic similarity    : "
        f"{summary['direct']['average_semantic_similarity']}"
    )

    print(
        f"  Requirement preservation: "
        f"{summary['direct']['average_requirement_preservation']}"
    )

    print(
        f"  Constraint preservation : "
        f"{summary['direct']['average_constraint_preservation']}"
    )

    print(
        f"  Optimization latency    : "
        f"{summary['direct']['average_optimization_latency_ms']} ms"
    )

    print()

    print(
        "Guided:"
    )

    print(
        f"  Token reduction        : "
        f"{summary['guided']['average_token_reduction_percent']}"
    )

    print(
        f"  Semantic similarity    : "
        f"{summary['guided']['average_semantic_similarity']}"
    )

    print(
        f"  Requirement preservation: "
        f"{summary['guided']['average_requirement_preservation']}"
    )

    print(
        f"  Constraint preservation : "
        f"{summary['guided']['average_constraint_preservation']}"
    )

    print(
        f"  Optimization latency    : "
        f"{summary['guided']['average_optimization_latency_ms']} ms"
    )

    # ========================================================
    # Save results
    # ========================================================

    output = {
        "experiment": {
            "name": "Prompt-level Groq pilot",
            "optimizer_model": OPTIMIZER_MODEL,
            "dataset_path": str(DATASET_PATH),
            "dataset_version": dataset_version,
            "dataset_hash": dataset_hash,
            "prompt_count": len(selected_prompts),

            "module5b_used": False,
            "target_model_used": False,

            "methodology": (
                "Direct: Original Prompt -> Groq Optimizer -> Module 5A\n"
                "Guided: Original Prompt -> Module 3 -> "
                "Groq Optimizer -> Module 5A"
            ),
        },

        "summary": summary,

        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 110)
    print("Pilot completed.")
    print(f"Results saved to: {OUTPUT_PATH}")
    print("=" * 110)


if __name__ == "__main__":
    asyncio.run(main())




# import asyncio
# import json
# from pathlib import Path

# from app.optimization.optimizer import PromptOptimizer
# from app.optimization.types import OptimizationRequest, OptimizationError
# from app.semantic.semantic_analyzer import SemanticAnalyzer
# from app.evaluation.evaluator import PromptEvaluator
# from experiments.benchmark_runner import load_benchmark_dataset


# DATASET = Path("data/promptoptai_benchmark_dataset_v1_1.json")
# PROMPT_ID = "POAI-004"
# OPTIMIZER_MODEL = "groq/openai/gpt-oss-120b"


# async def main():
#     prompts, version, dataset_hash = load_benchmark_dataset(DATASET)

#     prompt_record = next(
#         p for p in prompts
#         if p.prompt_id == PROMPT_ID
#     )

#     original_prompt = prompt_record.prompt

#     print("=" * 80)
#     print("PromptOptAI - Prompt-Level Groq Diagnostic")
#     print("=" * 80)
#     print(f"Prompt ID       : {PROMPT_ID}")
#     print(f"Optimizer       : {OPTIMIZER_MODEL}")
#     print(f"Dataset version : {version}")
#     print(f"Dataset hash    : {dataset_hash}")
#     print()

#     print("ORIGINAL PROMPT")
#     print("-" * 80)
#     print(original_prompt)
#     print()

#     optimizer = PromptOptimizer()
#     semantic_analyzer = SemanticAnalyzer()
#     prompt_evaluator = PromptEvaluator()

#     # ------------------------------------------------------------------
#     # DIRECT
#     # ------------------------------------------------------------------
#     print("=" * 80)
#     print("1. DIRECT OPTIMIZATION")
#     print("=" * 80)

#     try:
#         direct = await optimizer.optimize(
#             OptimizationRequest(
#                 prompt=original_prompt,
#                 mode="direct",
#                 model=OPTIMIZER_MODEL,
#             )
#         )

#         direct_prompt = direct.optimized_prompt

#         print("STATUS: SUCCESS")
#         print(f"Latency: {direct.metadata.latency_ms} ms")

#         if direct.metadata.usage:
#             print("Usage:")
#             print(json.dumps(
#                 direct.metadata.usage.model_dump(mode="json"),
#                 indent=2
#             ))

#         print()
#         print("DIRECT OPTIMIZED PROMPT")
#         print("-" * 80)
#         print(direct_prompt)
#         print()

#         print("DIRECT - MODULE 5A")
#         print("-" * 80)

#         direct_eval = prompt_evaluator.evaluate(
#             # Same request structure used by the frozen benchmark.
#             __import__("app.evaluation.types", fromlist=["EvaluationRequest"]).EvaluationRequest(
#                 original_prompt=original_prompt,
#                 optimized_prompt=direct_prompt,
#                 mode="direct",
#                 optimizer_model=OPTIMIZER_MODEL,
#             )
#         )

#         print(json.dumps(
#             direct_eval.metrics.model_dump(mode="json"),
#             indent=2
#         ))

#     except OptimizationError as exc:
#         print("DIRECT FAILED")
#         print(f"Code    : {exc.code}")
#         print(f"Message : {exc.message}")
#         return
#     except Exception as exc:
#         print("DIRECT FAILED")
#         print(f"Error: {type(exc).__name__}: {exc}")
#         return

#     # ------------------------------------------------------------------
#     # GUIDED
#     # ------------------------------------------------------------------
#     print()
#     print("=" * 80)
#     print("2. GUIDED OPTIMIZATION")
#     print("=" * 80)

#     try:
#         semantic_profile = semantic_analyzer.analyze(original_prompt)

#         print("MODULE 3 SEMANTIC PROFILE")
#         print("-" * 80)
#         print(json.dumps(
#             semantic_profile.model_dump(mode="json"),
#             indent=2,
#             ensure_ascii=False
#         ))
#         print()

#         guided = await optimizer.optimize(
#             OptimizationRequest(
#                 prompt=original_prompt,
#                 mode="guided",
#                 model=OPTIMIZER_MODEL,
#                 semantic_profile=semantic_profile,
#             )
#         )

#         guided_prompt = guided.optimized_prompt

#         print("STATUS: SUCCESS")
#         print(f"Latency: {guided.metadata.latency_ms} ms")

#         if guided.metadata.usage:
#             print("Usage:")
#             print(json.dumps(
#                 guided.metadata.usage.model_dump(mode="json"),
#                 indent=2
#             ))

#         print()
#         print("GUIDED OPTIMIZED PROMPT")
#         print("-" * 80)
#         print(guided_prompt)
#         print()

#         print("GUIDED - MODULE 5A")
#         print("-" * 80)

#         guided_eval = prompt_evaluator.evaluate(
#             __import__("app.evaluation.types", fromlist=["EvaluationRequest"]).EvaluationRequest(
#                 original_prompt=original_prompt,
#                 optimized_prompt=guided_prompt,
#                 mode="guided",
#                 optimizer_model=OPTIMIZER_MODEL,
#             )
#         )

#         print(json.dumps(
#             guided_eval.metrics.model_dump(mode="json"),
#             indent=2
#         ))

#     except OptimizationError as exc:
#         print("GUIDED FAILED")
#         print(f"Code    : {exc.code}")
#         print(f"Message : {exc.message}")
#         return
#     except Exception as exc:
#         print("GUIDED FAILED")
#         print(f"Error: {type(exc).__name__}: {exc}")
#         return

#     print()
#     print("=" * 80)
#     print("DIAGNOSTIC COMPLETE")
#     print("=" * 80)
#     print("Target model      : NOT USED")
#     print("Module 5B         : NOT USED")
#     print("Gemini evaluator  : NOT USED")
#     print("Groq optimizer    : TESTED")
#     print("Module 3          : TESTED")
#     print("Module 5A         : TESTED")
#     print("=" * 80)


# if __name__ == "__main__":
#     asyncio.run(main())
