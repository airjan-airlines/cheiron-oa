"""Request body for `POST /v1/visualize` (OA §3.1)."""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.enums import Phase, Status, VizType
from app.schemas.plan import MAX_YEAR, MIN_YEAR, Filters


class VisualizeRequest(BaseModel):
    # Unknown fields are rejected (422): a misspelled filter that silently does
    # nothing is worse than an error.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(
        min_length=1,
        max_length=1000,
        description="Natural-language question about clinical trials. Any language.",
        examples=["How has the number of trials for this drug changed over time?"],
    )
    drug_name: str | None = Field(None, min_length=1, max_length=200, description="Drug or intervention.")
    condition: str | None = Field(None, min_length=1, max_length=200, description="Disease or condition.")
    sponsor: str | None = Field(None, min_length=1, max_length=200, description="Lead sponsor organization.")
    country: str | None = Field(None, min_length=1, max_length=100, description="Country where the trial has a site.")
    trial_phases: list[Phase] | None = Field(None, description="Restrict to these phases.")
    statuses: list[Status] | None = Field(None, description="Restrict to these overall statuses.")
    start_year: int | None = Field(None, ge=MIN_YEAR, le=MAX_YEAR, description="Earliest trial start year (inclusive).")
    end_year: int | None = Field(None, ge=MIN_YEAR, le=MAX_YEAR, description="Latest trial start year (inclusive).")
    chart_type: VizType | None = Field(
        None, description="Preferred chart. Used only if compatible with the question; otherwise overridden and noted."
    )
    top_n: int = Field(15, ge=1, le=50, description="Maximum categories (bars, nodes) to show.")
    max_records: int = Field(5000, ge=100, le=5000, description="Maximum trial records fetched per query.")
    max_citations_per_datum: int = Field(3, ge=0, le=20, description="Citation excerpts attached to each datum.")

    @field_validator("trial_phases", "statuses")
    @classmethod
    def _dedupe(cls, values: list | None) -> list | None:
        return list(dict.fromkeys(values)) if values else None

    @model_validator(mode="after")
    def _year_order(self) -> "VisualizeRequest":
        if self.start_year is not None and self.end_year is not None and self.start_year > self.end_year:
            raise ValueError("start_year must be <= end_year")
        return self

    def structured_filters(self) -> Filters:
        """Filters stated explicitly by the caller. These override anything the LLM infers (D6)."""
        return Filters(
            condition=self.condition,
            intervention=self.drug_name,
            sponsor=self.sponsor,
            location=self.country,
            phases=self.trial_phases or [],
            statuses=self.statuses or [],
            start_year_min=self.start_year,
            start_year_max=self.end_year,
        )
