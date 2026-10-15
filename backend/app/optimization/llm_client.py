"""LiteLLM adapter used by Module 4 for provider-independent chat completions."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Protocol

from app.optimization.types import LLMCompletion, OptimizationError


class LLMClient(Protocol):
    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> LLMCompletion: ...


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _token_usage(response: Any) -> tuple[int | None, int | None, int | None]:
    usage = _get(response, "usage")
    if usage is None:
        return None, None, None
    return (
        _get(usage, "prompt_tokens"),
        _get(usage, "completion_tokens"),
        _get(usage, "total_tokens"),
    )


class LiteLLMClient:
    """Thin async adapter; provider selection and credentials are handled by LiteLLM."""

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        timeout: float,
    ) -> LLMCompletion:
        try:
            import litellm
        except ImportError as exc:
            raise OptimizationError(
                "LLM_CLIENT_UNAVAILABLE",
                "LiteLLM is not installed. Install backend requirements before optimizing.",
                503,
            ) from exc

        try:
            response = await asyncio.wait_for(
                litellm.acompletion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                ),
                timeout=timeout,
            )
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise OptimizationError(
                "LLM_TIMEOUT", "The model did not respond before the configured timeout.", 504
            ) from exc
        except Exception as exc:
            raise self._map_provider_error(exc) from exc

        choices = _get(response, "choices")
        if not isinstance(choices, (list, tuple)) or not choices:
            raise OptimizationError(
                "MALFORMED_LLM_RESPONSE", "The model returned a malformed response.", 502
            )
        message = _get(choices[0], "message")
        if message is None:
            raise OptimizationError(
                "MALFORMED_LLM_RESPONSE", "The model returned a malformed response.", 502
            )
        content = _get(message, "content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise OptimizationError(
                "MALFORMED_LLM_RESPONSE", "The model returned a malformed response.", 502
            )

        input_tokens, output_tokens, total_tokens = _token_usage(response)
        return LLMCompletion(
            text=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    @staticmethod
    def _map_provider_error(exc: Exception) -> OptimizationError:
        error_name = type(exc).__name__.lower()
        message = str(exc).lower()

        if "timeout" in error_name or "timed out" in message:
            return OptimizationError(
                "LLM_TIMEOUT", "The model did not respond before the configured timeout.", 504
            )
        if any(
            term in message
            for term in (
                "api key is required",
                "api_key must",
                "api_key client option must be set",
                "api key not provided",
                "api key was not provided",
                "missing api key",
            )
        ):
            return OptimizationError(
                "MISSING_API_KEY", "No API key is configured for the selected model provider.", 503
            )
        if "authentication" in error_name or "permission" in error_name:
            return OptimizationError(
                "AUTHENTICATION_FAILED", "The model provider rejected its configured credentials.", 502
            )
        if any(
            term in message
            for term in (
                "unsupported model",
                "model not supported",
                "model not found",
                "model does not exist",
                "provider not found",
                "provider not provided",
                "provider not specified",
                "unknown model",
                "not a valid model",
            )
        ):
            return OptimizationError(
                "UNSUPPORTED_MODEL", "The selected model identifier is unsupported by LiteLLM.", 400
            )
        return OptimizationError(
            "PROVIDER_ERROR", "The model provider failed to complete the optimization request.", 502
        )
