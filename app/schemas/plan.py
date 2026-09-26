"""The typed query plan: the only thing the LLM is allowed to produce (D1).

This model holds *structural* constraints only (types, ranges, list sizes).
Cross-field meaning (e.g. "a time trend must group by start year") lives in
`app/planner/validator.py`, which returns readable errors the LLM can repair.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.enums import Analysis, Dimension, Metric, Phase, Status, VizType

MIN_YEAR = 1900
MAX_YEAR = 2100


class Filters(BaseModel):
    """Search filters, translated to ClinicalTrials.gov query params by `ctgov.query_builder`."""

    model_config = ConfigDict(extra="forbid")

    condition: str | None = Field(None, description="Disease or condition, sent as `query.cond`.")
    intervention: str | None = Field(None, description="Drug or intervention, sent as `query.intr`.")
    sponsor: str | None = Field(None, description="Lead sponsor, sent as `query.lead`.")
    location: str | None = Field(None, description="Country or place, sent as `query.locn`.")
    phases: list[Phase] = Field(default_factory=list)
    statuses: list[Status] = Field(default_factory=list)
    start_year_min: int | None = Field(None, ge=MIN_YEAR, le=MAX_YEAR)
    start_year_max: int | None = Field(None, ge=MIN_YEAR, le=MAX_YEAR)

    @model_validator(mode="after")
    def _year_order(self) -> "Filters":
        if (
            self.start_year_min is not None
            and self.start_year_max is not None
            and self.start_year_min > self.start_year_max
        ):
            raise ValueError("start_year_min must be <= start_year_max")
        return self

    def is_empty(self) -> bool:
        return self == Filters()


class Cohort(BaseModel):
    """One side of a comparison, e.g. {"label": "Semaglutide", "filters": {"intervention": "semaglutide"}}.

    Cohort filters are layered on top of the plan's shared filters.
    """

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=100)
    filters: Filters


class NetworkSpec(BaseModel):
    """Which entity types become nodes. source == target means co-occurrence (drug <-> drug)."""

    model_config = ConfigDict(extra="forbid")

    source: Dimension
    target: Dimension


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supported: bool = Field(True, description="False when the question is not about clinical trials.")
    unsupported_reason: str | None = None
    analysis: Analysis | None = None
    group_by: Dimension | None = None
    metric: Metric = Metric.TRIAL_COUNT
    filters: Filters = Field(default_factory=Filters)
    compare: list[Cohort] = Field(default_factory=list, max_length=4)
    network: NetworkSpec | None = None
    top_n: int = Field(15, ge=1, le=50)
    chart_type_suggestion: VizType | None = None
    chart_rationale: str | None = Field(None, max_length=500)
