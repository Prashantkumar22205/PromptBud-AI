"""CLI for the frozen PromptOptAI research benchmark runner."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from experiments.benchmark_runner import (
    BenchmarkError,
    BenchmarkRunner,
    make_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PromptOptAI frozen response benchmark.")
    parser.add_argument("--dataset", default="backend/data/promptoptai_benchmark_dataset_v1_1.json")
    parser.add_argument("--optimizer-model", default=os.getenv("OPTIMIZER_MODEL"))
    parser.add_argument("--target-model", default=os.getenv("TARGET_MODEL"))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("BENCHMARK_TEMPERATURE", "0.2")))
    parser.add_argument("--max-output-tokens", type=int, default=int(os.getenv("BENCHMARK_MAX_OUTPUT_TOKENS", "2048")))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("BENCHMARK_TIMEOUT_SECONDS", "60")))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--categories", nargs="*", default=[])
    parser.add_argument("--length-band", choices=("short", "medium", "long"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--experiment-id")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _find_resume_experiment(checkpoints_root: Path) -> str:
    candidates = sorted(
        (path.parent.name for path in checkpoints_root.glob("*/condition_checkpoints.jsonl")),
        reverse=True,
    )
    if not candidates:
        raise BenchmarkError("--resume requested but no benchmark checkpoint was found.")
    return candidates[0]


async def _run(args: argparse.Namespace) -> dict:
    if not args.optimizer_model or not args.target_model:
        raise BenchmarkError("Set OPTIMIZER_MODEL and TARGET_MODEL or pass both model flags.")
    backend_root = Path(__file__).resolve().parent.parent
    benchmark_root = backend_root / "experiments" / "benchmark"
    experiment_id = args.experiment_id
    if args.resume and not experiment_id:
        experiment_id = _find_resume_experiment(benchmark_root / "checkpoints")
    config = make_config(
        dataset=args.dataset,
        optimizer_model=args.optimizer_model,
        target_model=args.target_model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        timeout=args.timeout,
        categories=args.categories,
        length_band=args.length_band,
        limit=args.limit,
    )
    return await BenchmarkRunner(backend_root=backend_root).run(
        config,
        resume=args.resume,
        dry_run=args.dry_run,
        experiment_id=experiment_id,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parser().parse_args(argv)
    try:
        result = asyncio.run(_run(args))
    except (BenchmarkError, ValueError) as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
