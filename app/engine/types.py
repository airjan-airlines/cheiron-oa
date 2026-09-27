from dataclasses import dataclass, field

from app.ctgov.extract import TrialRow
from app.schemas.plan import Filters
from app.schemas.response import Coverage, RenderHints, VisualizationRationale, Visualization


@dataclass
class FetchedQuery:
    """One executed search: what was asked, what the API reported, and the records analyzed."""

    label: str | None  # cohort label for comparisons
    filters: Filters  # effective filters (plan + cohort + structured request fields)
    api_params: dict[str, str]
    total_matching: int
    rows: list[TrialRow]

    @property
    def truncated(self) -> bool:
        return self.total_matching > len(self.rows)


@dataclass
class ExecutionResult:
    visualization: Visualization
    interpretation: str
    coverage: Coverage | None
    render: RenderHints
    rationale: VisualizationRationale
    assumptions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
