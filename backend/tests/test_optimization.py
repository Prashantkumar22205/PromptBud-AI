"""Unit and API tests for Module 4; all LLM completions are mocked."""

from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.optimization.llm_client import LiteLLMClient
from app.optimization.optimizer import PromptOptimizer
from app.optimization.prompts import SYSTEM_INSTRUCTION
from app.optimization.types import (
    LLMCompletion,
    OptimizationError,
    OptimizationRequest,
)
from app.semantic.types import SemanticAnalysis


def semantic_profile() -> SemanticAnalysis:
    return SemanticAnalysis.model_validate(
        {
            "intent": {"primaryIntent": "educational_explanation", "confidence": 0.9},
            "context": {
                "topic": {
                    "value": "test profile marker",
                    "confidence": 0.9,
                    "method": "test",
                }
            },
            "requirements": [
                {
                    "action": "explain",
                    "object": "binary search",
                    "confidence": 0.9,
                    "method": "test",
                }
            ],
            "constraints": [],
            "ambiguities": [],
        }
    )


def synthetic_semantic_profile() -> SemanticAnalysis:
    return SemanticAnalysis.model_validate(
        {
            "intent": {"primaryIntent": "TEST_INTENT_123", "confidence": 0.9},
            "context": {
                "topic": {"value": "TEST_TOPIC_456", "confidence": 0.9, "method": "test"},
                "audience": {"value": "TEST_AUDIENCE_789", "confidence": 0.9, "method": "test"},
            },
            "requirements": [
                {"action": "TEST_REQUIREMENT_001", "confidence": 0.9, "method": "test"}
            ],
            "constraints": [
                {
                    "type": "scope",
                    "description": "TEST_CONSTRAINT_002",
                    "confidence": 0.9,
                    "method": "test",
                }
            ],
            "ambiguities": [
                {"type": "TEST_AMBIGUITY_003", "description": "Preserve marker", "severity": "low"}
            ],
        }
    )


class FakeLLMClient:
    def __init__(self, text="Optimized prompt", error=None):
        self.text = text
        self.error = error
        self.calls = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return LLMCompletion(text=self.text, input_tokens=12, output_tokens=8, total_tokens=20)


def request(mode="direct", model="openai/model-a", profile=None, prompt="Explain binary search."):
    return OptimizationRequest(
        prompt=prompt,
        mode=mode,
        model=model,
        semantic_profile=profile,
    )


@pytest.mark.asyncio
async def test_direct_request_returns_optimized_prompt():
    client = FakeLLMClient("Explain binary search clearly.")
    result = await PromptOptimizer(client).optimize(request())
    assert result.optimized_prompt == "Explain binary search clearly."
    assert result.original_prompt == "Explain binary search."
    assert result.mode == "direct"
    assert client.calls[0]["messages"][1]["content"] == "ORIGINAL PROMPT:\nExplain binary search."


@pytest.mark.asyncio
async def test_guided_request_passes_authoritative_semantic_profile():
    profile = semantic_profile()
    client = FakeLLMClient()
    await PromptOptimizer(client).optimize(request("guided", profile=profile))
    guided_input = client.calls[0]["messages"][1]["content"]
    assert guided_input.startswith("ORIGINAL PROMPT:\n")
    assert "Explain binary search." in guided_input
    assert "AUTHORITATIVE MODULE 3 SEMANTIC PROFILE" in guided_input
    assert "test profile marker" in guided_input
    assert "binary search" in guided_input
    assert "authoritative" in SYSTEM_INSTRUCTION.lower()
    assert "do not guess" in SYSTEM_INSTRUCTION.lower()


@pytest.mark.asyncio
async def test_direct_mode_ignores_even_a_supplied_semantic_profile():
    client = FakeLLMClient()
    await PromptOptimizer(client).optimize(
        request("direct", profile=semantic_profile(), prompt="Original only.")
    )
    assert client.calls[0]["messages"][1]["content"] == "ORIGINAL PROMPT:\nOriginal only."
    assert "test profile marker" not in str(client.calls[0]["messages"])


@pytest.mark.asyncio
async def test_direct_and_guided_share_rules_and_generation_settings():
    client = FakeLLMClient()
    optimizer = PromptOptimizer(client)
    await optimizer.optimize(request())
    await optimizer.optimize(request("guided", profile=semantic_profile()))
    direct, guided = client.calls
    assert direct["messages"][0] == guided["messages"][0]
    assert direct["temperature"] == guided["temperature"]
    assert direct["max_tokens"] == guided["max_tokens"]
    assert direct["timeout"] == guided["timeout"]


@pytest.mark.asyncio
async def test_direct_and_guided_captured_requests_differ_only_by_profile_section():
    marker_prompt = "TEST_ORIGINAL_PROMPT_999"
    profile = synthetic_semantic_profile()
    client = FakeLLMClient()
    optimizer = PromptOptimizer(client)
    await optimizer.optimize(request("direct", prompt=marker_prompt))
    await optimizer.optimize(request("guided", profile=profile, prompt=marker_prompt))
    direct, guided = client.calls

    direct_user = direct["messages"][1]["content"]
    guided_user = guided["messages"][1]["content"]
    profile_section = guided_user.split("\n\nAUTHORITATIVE MODULE 3 SEMANTIC PROFILE:\n", 1)
    assert len(profile_section) == 2
    assert profile_section[0] == direct_user
    assert guided_user == direct_user + "\n\nAUTHORITATIVE MODULE 3 SEMANTIC PROFILE:\n" + profile_section[1]
    assert marker_prompt in direct_user
    assert marker_prompt in guided_user

    profile_markers = (
        "TEST_INTENT_123",
        "TEST_TOPIC_456",
        "TEST_AUDIENCE_789",
        "TEST_REQUIREMENT_001",
        "TEST_CONSTRAINT_002",
        "TEST_AMBIGUITY_003",
    )
    for marker in profile_markers:
        assert marker not in direct_user
        assert marker in guided_user

    assert direct["messages"][0] == guided["messages"][0]
    for setting in ("model", "temperature", "max_tokens", "timeout"):
        assert direct[setting] == guided[setting]


@pytest.mark.asyncio
async def test_direct_and_guided_use_identical_original_prompt_framing():
    client = FakeLLMClient()
    optimizer = PromptOptimizer(client)
    prompt = "TEST_ORIGINAL_PROMPT_999"
    await optimizer.optimize(request("direct", prompt=prompt))
    await optimizer.optimize(request("guided", profile=synthetic_semantic_profile(), prompt=prompt))

    direct_user = client.calls[0]["messages"][1]["content"]
    guided_user = client.calls[1]["messages"][1]["content"]
    assert direct_user == f"ORIGINAL PROMPT:\n{prompt}"
    assert guided_user.startswith(direct_user + "\n\nAUTHORITATIVE MODULE 3 SEMANTIC PROFILE:\n")


def test_guided_mode_requires_semantic_profile():
    with pytest.raises(ValueError, match="semantic_profile is required"):
        request("guided")


@pytest.mark.parametrize("prompt", ["", "   "])
def test_empty_prompt_is_rejected(prompt):
    with pytest.raises(ValueError):
        request(prompt=prompt)


@pytest.mark.asyncio
async def test_provider_failure_is_a_structured_service_error():
    client = FakeLLMClient(error=OptimizationError("PROVIDER_ERROR", "Provider failed", 502))
    with pytest.raises(OptimizationError) as exc:
        await PromptOptimizer(client).optimize(request())
    assert exc.value.code == "PROVIDER_ERROR"
    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_empty_model_output_is_a_structured_service_error():
    with pytest.raises(OptimizationError) as exc:
        await PromptOptimizer(FakeLLMClient("  ")).optimize(request())
    assert exc.value.code == "EMPTY_OPTIMIZED_PROMPT"
    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_model_identifier_is_configurable_without_changing_logic():
    client = FakeLLMClient()
    optimizer = PromptOptimizer(client)
    first = await optimizer.optimize(request(model="provider-a/model-one"))
    second = await optimizer.optimize(request(model="provider-b/model-two"))
    assert [call["model"] for call in client.calls] == ["provider-a/model-one", "provider-b/model-two"]
    assert first.optimized_prompt == second.optimized_prompt
    assert client.calls[0]["messages"] == client.calls[1]["messages"]


def test_response_has_no_module5_evaluation_fields():
    response_fields = set(PromptOptimizer.__annotations__)
    assert not response_fields.intersection(
        {"token_reduction", "semantic_similarity", "preservation_score", "quality_score", "ranking"}
    )
    assert set(__import__("app.optimization.types", fromlist=["OptimizationResponse"]).OptimizationResponse.model_fields) == {
        "mode", "model", "original_prompt", "optimized_prompt", "metadata"
    }


@pytest.mark.asyncio
async def test_litellm_adapter_calls_unified_async_completion(monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Optimized"))],
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2, total_tokens=6),
    )
    completion = AsyncMock(return_value=response)
    fake_module = ModuleType("litellm")
    fake_module.acompletion = completion
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake_module)

    result = await LiteLLMClient().complete(
        model="provider/model",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.2,
        max_tokens=256,
        timeout=5,
    )
    assert result.text == "Optimized"
    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (4, 2, 6)
    completion.assert_awaited_once()
    assert completion.await_args.kwargs["model"] == "provider/model"
    assert completion.await_args.kwargs["max_tokens"] == 256


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error,expected_code,expected_status",
    [
        (RuntimeError("api_key client option must be set"), "MISSING_API_KEY", 503),
        (RuntimeError("model not found for this provider"), "UNSUPPORTED_MODEL", 400),
        (TimeoutError("request timed out"), "LLM_TIMEOUT", 504),
    ],
)
async def test_litellm_provider_errors_are_mapped(monkeypatch, error, expected_code, expected_status):
    fake_module = ModuleType("litellm")
    fake_module.acompletion = AsyncMock(side_effect=error)
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake_module)
    with pytest.raises(OptimizationError) as exc:
        await LiteLLMClient().complete(
            model="provider/model",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.2,
            max_tokens=256,
            timeout=5,
        )
    assert exc.value.code == expected_code
    assert exc.value.status_code == expected_status


@pytest.mark.asyncio
async def test_litellm_malformed_response_is_rejected(monkeypatch):
    fake_module = ModuleType("litellm")
    fake_module.acompletion = AsyncMock(return_value={"choices": []})
    monkeypatch.setitem(__import__("sys").modules, "litellm", fake_module)
    with pytest.raises(OptimizationError) as exc:
        await LiteLLMClient().complete(
            model="provider/model",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.2,
            max_tokens=256,
            timeout=5,
        )
    assert exc.value.code == "MALFORMED_LLM_RESPONSE"


class RouteFakeLLMClient(FakeLLMClient):
    pass


@pytest.fixture
def api_client(monkeypatch):
    import app.api.routes.optimization as optimization_route

    monkeypatch.setattr(optimization_route, "optimizer", PromptOptimizer(RouteFakeLLMClient()))
    return TestClient(app)


def test_api_guided_missing_profile_returns_validation_error(api_client):
    response = api_client.post(
        "/api/optimize",
        json={"prompt": "Explain this", "mode": "guided", "model": "provider/model"},
    )
    assert response.status_code == 422


def test_api_empty_prompt_returns_validation_error(api_client):
    response = api_client.post(
        "/api/optimize",
        json={"prompt": "  ", "mode": "direct", "model": "provider/model"},
    )
    assert response.status_code == 422


def test_api_returns_generation_result_and_module3_schema(api_client):
    response = api_client.post(
        "/api/optimize",
        json={"prompt": "Explain binary search", "mode": "direct", "model": "provider/model"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["optimized_prompt"] == "Optimized prompt"
    assert data["mode"] == "direct"
    assert data["model"] == "provider/model"
    assert set(data) == {"mode", "model", "original_prompt", "optimized_prompt", "metadata"}


def test_api_returns_structured_error_for_provider_failure(monkeypatch):
    import app.api.routes.optimization as optimization_route

    monkeypatch.setattr(
        optimization_route,
        "optimizer",
        PromptOptimizer(RouteFakeLLMClient(error=OptimizationError("PROVIDER_ERROR", "Provider failed", 502))),
    )
    response = TestClient(app).post(
        "/api/optimize",
        json={"prompt": "Explain binary search", "mode": "direct", "model": "provider/model"},
    )
    assert response.status_code == 502
    assert response.json() == {
        "success": False,
        "error": {"code": "PROVIDER_ERROR", "message": "Provider failed"},
    }


def test_api_returns_structured_error_for_empty_llm_output(monkeypatch):
    import app.api.routes.optimization as optimization_route

    monkeypatch.setattr(optimization_route, "optimizer", PromptOptimizer(RouteFakeLLMClient(text="")))
    response = TestClient(app).post(
        "/api/optimize",
        json={"prompt": "Explain binary search", "mode": "direct", "model": "provider/model"},
    )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "EMPTY_OPTIMIZED_PROMPT"
