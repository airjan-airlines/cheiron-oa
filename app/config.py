"""Runtime settings, read from environment variables and `.env`."""

from functools import lru_cache

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Company constraint: only these OpenAI models may be used (docs/openai-rules.md).
# tests/test_config.py keeps this list in sync with that file.
ALLOWED_MODELS: tuple[str, ...] = (
    "gpt-4o-mini",
    "gpt-4o-2024-08-06",
    "gpt-4.1-nano",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-5-mini",
    "gpt-5",
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.4-nano",
    "gpt-5.4-mini",
    "gpt-5.4",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Optional at startup so the service and offline tests run without a key;
    # the planner fails with a clear error if a request needs it and it is unset.
    openai_api_key: SecretStr | None = None
    # gpt-4.1-mini: non-reasoning (fast), supports Structured Outputs and temperature=0.
    openai_model: str = "gpt-4.1-mini"
    openai_timeout_s: float = 30.0

    ctgov_base_url: str = "https://clinicaltrials.gov/api/v2"
    ctgov_timeout_s: float = 30.0
    ctgov_max_retries: int = 3
    ctgov_cache_ttl_s: int = 900

    @field_validator("openai_model")
    @classmethod
    def _model_allowed(cls, value: str) -> str:
        if value not in ALLOWED_MODELS:
            raise ValueError(f"OPENAI_MODEL={value!r} is not allowed; choose one of {', '.join(ALLOWED_MODELS)}")
        return value

    @property
    def model_is_reasoning(self) -> bool:
        """gpt-5 family models reject `temperature`, so the planner omits it for them."""
        return self.openai_model.startswith("gpt-5")


@lru_cache
def get_settings() -> Settings:
    return Settings()
