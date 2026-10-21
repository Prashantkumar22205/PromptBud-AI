"""LiteLLM adapter used by Module 4 for provider-independent chat completions."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any, Protocol

from app.core.config import settings
from app.optimization.types import LLMCompletion, OptimizationError

logger = logging.getLogger(__name__)
MAX_ATTEMPTS = 3
RETRY_DELAYS_SECONDS = (2, 5)
NVIDIA_NIM_PREFIX = "nvidia_nim/"

_TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_PERMANENT_STATUS_CODES = {400, 401, 403, 404, 422}



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

        response = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                api_key = None

                if model.startswith("gemini/"):
                    api_key = settings.gemini_api_key
                elif model.startswith("groq/"):
                    api_key = settings.groq_api_key
                elif model.startswith(NVIDIA_NIM_PREFIX):
                    api_key = settings.nvidia_nim_api_key

                completion_kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timeout": timeout,
                }

                if api_key:
                    completion_kwargs["api_key"] = api_key

                if model.startswith(NVIDIA_NIM_PREFIX):
                    completion_kwargs["extra_body"] = {
                        "chat_template_kwargs": {
                            "enable_thinking": False,
                        }
                    }

                response = await asyncio.wait_for(
                    litellm.acompletion(**completion_kwargs),
                    timeout=timeout,
                )
                break

            except Exception as exc:
                if self._is_transient_provider_error(exc) and attempt < MAX_ATTEMPTS - 1:
                    status_code = self._status_code(exc)
                    logger.warning(
                        "Transient LiteLLM failure (type=%s, status=%s); retry %d/%d after %ss.",
                        type(exc).__name__,
                        status_code,
                        attempt + 1,
                        MAX_ATTEMPTS - 1,
                        RETRY_DELAYS_SECONDS[attempt],
                    )
                    await asyncio.sleep(RETRY_DELAYS_SECONDS[attempt])
                    continue

                mapped = self._map_provider_error(exc)

                # Keep the original provider exception chained for internal diagnostics;
                # only the safe, fixed message is exposed through OptimizationError.
                mapped.provider_error_type = type(exc).__name__
                mapped.provider_status_code = self._status_code(exc)
                raise mapped from exc

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
    def _status_code(exc: Exception) -> int | None:
        candidates = (exc, _get(exc, "response"))
        for candidate in candidates:
            if candidate is None:
                continue
            for key in ("status_code", "status"):
                value = _get(candidate, key)
                try:
                    if value is not None:
                        return int(value)
                except (TypeError, ValueError):
                    continue
        return None

    @classmethod
    def _is_transient_provider_error(cls, exc: Exception) -> bool:
        mapped = cls._map_provider_error(exc)
        if mapped.code in {"MISSING_API_KEY", "AUTHENTICATION_FAILED", "UNSUPPORTED_MODEL"}:
            return False

        status_code = cls._status_code(exc)
        if status_code in _PERMANENT_STATUS_CODES:
            return False
        if status_code in _TRANSIENT_STATUS_CODES:
            return True

        name = type(exc).__name__.casefold()
        message = str(exc).casefold()
        if mapped.code == "LLM_TIMEOUT":
            return True
        if any(
            marker in name
            for marker in (
                "ratelimit",
                "serviceunavailable",
                "internalserver",
                "apiconnection",
                "connectionerror",
                "connecterror",
                "readtimeout",
                "connecttimeout",
            )
        ):
            return True
        return any(
            marker in message
            for marker in (
                "rate limit",
                "rate_limit",
                "too many requests",
                "resource exhausted",
                "service unavailable",
                "temporarily unavailable",
                "temporary provider unavailable",
                "high demand",
                "temporarily overloaded",
                "connection reset",
                "connection refused",
                "connection aborted",
                "temporary connection",
                "request timed out",
                "timed out",
            )
        )

    @staticmethod
    def _map_provider_error(exc: Exception) -> OptimizationError:
        error_name = type(exc).__name__.lower()
        message = str(exc).lower()
        status_code = LiteLLMClient._status_code(exc)

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
        if (
            "authentication" in error_name
            or "permission" in error_name
            or "unauthorized" in error_name
            or status_code in {401, 403}
            or any(term in message for term in ("invalid api key", "unauthorized", "permission denied"))
        ):
            return OptimizationError(
                "AUTHENTICATION_FAILED", "The model provider rejected its configured credentials.", 502
            )
        if "timeout" in error_name or "timed out" in message:
            return OptimizationError(
                "LLM_TIMEOUT", "The model did not respond before the configured timeout.", 504
            )
        if any(
            marker in error_name
            for marker in ("badrequest", "invalidrequest", "notfound", "unsupportedmodel")
        ) or any(
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
        ) or status_code == 404:
            return OptimizationError(
                "UNSUPPORTED_MODEL", "The selected model identifier is unsupported by LiteLLM.", 400
            )
        if status_code in _TRANSIENT_STATUS_CODES:
            return OptimizationError(
                "PROVIDER_ERROR", f"The model provider returned a transient HTTP {status_code} error.", status_code
            )
        return OptimizationError(
            "PROVIDER_ERROR", "The model provider failed to complete the optimization request.", 502
        )
