"""Module 4 generation service for controlled DIRECT and GUIDED optimization."""

from __future__ import annotations

import asyncio
from time import perf_counter

from app.core.config import settings
from app.optimization.llm_client import LLMClient, LiteLLMClient
from app.optimization.prompts import SYSTEM_INSTRUCTION, build_direct_input, build_guided_input
from app.optimization.types import (
    LLMCompletion,
    OptimizationError,
    OptimizationMetadata,
    OptimizationRequest,
    OptimizationResponse,
    TokenUsage,
)


class PromptOptimizer:
    """Generate one optimized prompt without running Module 3 or evaluation."""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client or LiteLLMClient()

    async def optimize(self, request: OptimizationRequest) -> OptimizationResponse:
        if len(request.prompt) > settings.max_prompt_length:
            raise OptimizationError(
                "PROMPT_TOO_LARGE",
                f"Prompt exceeds the maximum allowed length of {settings.max_prompt_length} characters.",
                413,
            )

        if request.mode == "guided":
            # Pydantic validates this invariant at the API boundary.
            assert request.semantic_profile is not None
            user_content = build_guided_input(request.prompt, request.semantic_profile)
        else:
            # Intentionally ignore a supplied profile in DIRECT mode.
            user_content = build_direct_input(request.prompt)

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_content},
        ]
        timeout = settings.optimization_timeout_seconds
        started = perf_counter()
        try:
            completion: LLMCompletion = await asyncio.wait_for(
                self.llm_client.complete(
                    model=request.model,
                    messages=messages,
                    temperature=settings.optimization_temperature,
                    max_tokens=settings.optimization_max_output_tokens,
                    timeout=timeout,
                ),
                timeout=timeout,
            )
        except OptimizationError:
            raise
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise OptimizationError(
                "LLM_TIMEOUT", "The model did not respond before the configured timeout.", 504
            ) from exc
        except Exception as exc:
            raise OptimizationError(
                "PROVIDER_ERROR", "The model provider failed to complete the optimization request.", 502
            ) from exc

        optimized = completion.text.strip()
        if not optimized:
            raise OptimizationError(
                "EMPTY_OPTIMIZED_PROMPT", "The model returned an empty optimized prompt.", 502
            )

        usage = None
        if any(value is not None for value in (completion.input_tokens, completion.output_tokens, completion.total_tokens)):
            usage = TokenUsage(
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                total_tokens=completion.total_tokens,
            )

        return OptimizationResponse(
            mode=request.mode,
            model=request.model,
            original_prompt=request.prompt,
            optimized_prompt=optimized,
            metadata=OptimizationMetadata(
                latency_ms=max(0, int((perf_counter() - started) * 1000)),
                usage=usage,
            ),
        )
