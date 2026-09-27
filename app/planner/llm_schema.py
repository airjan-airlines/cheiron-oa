"""The schema the LLM fills in (OpenAI Structured Outputs, strict mode).

Strict mode needs every field present and does not enforce numeric or length limits,
so this mirrors `QueryPlan` with nullable fields and no constraints. The real
constraints are applied when it is converted to `QueryPlan` in `validator.py`.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

from app.schemas.enums import Analysis, Dimension, Metric, Phase, Status, VizType


class FiltersLLM(BaseModel):
    condition: str | None = Field(description="Disease or condition in English, e.g. 'gastric cancer'. Null if not mentioned.")
    intervention: str | None = Field(description="Drug or intervention, generic English name, e.g. 'pembrolizumab'. Null if not mentioned.")
    sponsor: str | None = Field(description="Lead sponsor organization. Null if not mentioned.")
    location: str | None = Field(description="Country in English, e.g. 'South Korea'. Null if not mentioned.")
    phases: list[Phase] = Field(description="Only if the question restricts to specific phases; otherwise empty.")
    statuses: list[Status] = Field(description="Only if the question restricts status, e.g. RECRUITING for 'recruiting' or 'currently enrolling'; otherwise empty.")
    start_year_min: int | None = Field(description="Earliest trial start year, e.g. 2015 for 'since 2015'.")
    start_year_max: int | None = Field(description="Latest trial start year.")


class Unanswered(StrEnum):
    """Parts of a question the plan cannot answer. A fixed list, so notes are templated, not model prose."""

    SECOND_QUESTION = "second_question"  # "...and which countries?" alongside another question
    STATISTIC = "statistic"  # averages, medians, percentages, rates
    REGION = "region"  # "Europe", "Asia": only countries are searchable
    TWO_LEVEL_BREAKDOWN = "two_level_breakdown"  # e.g. phase mix over time
    OTHER = "other"


class CohortLLM(BaseModel):
    label: str = Field(description="Short display name, e.g. 'Semaglutide'.")
    filters: FiltersLLM = Field(description="Only the filters that distinguish this cohort; shared filters go in the plan's filters.")


class NetworkLLM(BaseModel):
    source: Dimension
    target: Dimension


class QueryPlanLLM(BaseModel):
    is_about_clinical_trials: bool = Field(description="False if the question cannot be answered from a clinical trial registry.")
    unsupported_reason: str | None = Field(description="Why the question is unsupported, when is_about_clinical_trials is false.")
    analysis: Analysis | None
    group_by: Dimension | None
    metric: Metric | None
    filters: FiltersLLM = Field(description="Filters shared by the whole question.")
    compare: list[CohortLLM] = Field(description="2-4 cohorts for a comparison; empty otherwise.")
    network: NetworkLLM | None
    top_n: int | None = Field(description="Only if the question asks for a specific number, e.g. 'top 10 countries'.")
    chart_type_suggestion: VizType | None
    chart_rationale: str | None = Field(description="One sentence on why the suggested chart fits. Never state results or numbers.")
    # Same shape as `filters`, but only what the question text itself says, ignoring the caller's
    # structured fields. Lets code (not the model) detect when a structured field overrides the question.
    question_mentions: FiltersLLM = Field(description="What the question itself asks for, ignoring the caller's structured fields.")
    unanswered: list[Unanswered] = Field(description="Parts of the question this plan does not answer; empty if it answers everything.")
