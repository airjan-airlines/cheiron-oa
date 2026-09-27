"""Calls the LLM to turn a question into a `QueryPlanLLM`. Nothing else talks to the model."""

from typing import Protocol

import openai

from app.config import Settings
from app.errors import PlannerUnavailable
from app.planner.llm_schema import QueryPlanLLM

Message = dict[str, str]


class PlannerLLM(Protocol):
    async def plan(self, messages: list[Message]) -> QueryPlanLLM: ...


class OpenAIPlanner:
    def __init__(self, settings: Settings, client: openai.AsyncOpenAI | None = None) -> None:
        if client is None and settings.openai_api_key is None:
            raise PlannerUnavailable("OPENAI_API_KEY is not set; natural-language planning is unavailable.")
        self._client = client or openai.AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(), timeout=settings.openai_timeout_s, max_retries=2
        )
        self._model = settings.openai_model
        # gpt-5 family models reject `temperature`; others get 0 for repeatable plans.
        self._extra = {} if settings.model_is_reasoning else {"temperature": 0}

    async def plan(self, messages: list[Message]) -> QueryPlanLLM:
        try:
            completion = await self._client.chat.completions.parse(
                model=self._model, messages=messages, response_format=QueryPlanLLM, **self._extra
            )
        except (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.AuthenticationError, openai.InternalServerError) as exc:
            raise PlannerUnavailable(f"OpenAI request failed: {type(exc).__name__}") from exc
        message = completion.choices[0].message
        if message.parsed is not None:
            return message.parsed
        if message.refusal:
            return unsupported_plan(f"The planner declined this question: {message.refusal}")
        raise PlannerUnavailable("The model returned an empty response.")


def unsupported_plan(reason: str) -> QueryPlanLLM:
    empty_filters = {
        "condition": None, "intervention": None, "sponsor": None, "location": None,
        "phases": [], "statuses": [], "start_year_min": None, "start_year_max": None,
    }
    return QueryPlanLLM(
        is_about_clinical_trials=False, unsupported_reason=reason, analysis=None, group_by=None, metric=None,
        filters=empty_filters, compare=[], network=None, top_n=None, chart_type_suggestion=None, chart_rationale=None,
    )
