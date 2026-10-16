"""Independent model, generation, timeout, and target pricing configuration."""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelPricing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_price_per_million: float = Field(..., ge=0)
    output_price_per_million: float = Field(..., ge=0)


class ResponseEvaluationSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    response_evaluator_model: str | None = None
    response_evaluator_temperature: float = Field(default=0.0, ge=0, le=2)
    response_evaluator_max_output_tokens: int = Field(default=1200, gt=0)
    response_evaluator_timeout_seconds: float = Field(default=90.0, gt=0)
    target_model_pricing_json: str | None = None

    def configured_evaluator_model(self) -> str | None:
        model = self.response_evaluator_model
        return model.strip() if model and model.strip() else None

    def pricing_for(self, target_model: str) -> tuple[ModelPricing | None, str | None]:
        if not self.target_model_pricing_json:
            return None, "Target-model pricing is not configured."
        try:
            import json

            pricing_data = json.loads(self.target_model_pricing_json)
            if not isinstance(pricing_data, dict):
                raise ValueError("Pricing configuration must be a JSON object keyed by model identifier.")
            model_data = pricing_data.get(target_model)
            if model_data is None:
                return None, f"No pricing is configured for target model '{target_model}'."
            return ModelPricing.model_validate(model_data), None
        except Exception:
            return None, "Target-model pricing configuration is invalid."


@lru_cache(maxsize=1)
def get_response_evaluation_settings() -> ResponseEvaluationSettings:
    try:
        return ResponseEvaluationSettings()
    except ValidationError as exc:
        raise ValueError("Response evaluation settings are invalid.") from exc
