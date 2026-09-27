from dataclasses import dataclass, field

from app.ctgov.extract import TrialRow
from app.schemas.plan import Filters
from app.schemas.response import Coverage, RenderHints, VisualizationRationale, Visualization


@dataclass
class ExactBuckets:
    """Per-bucket registry counts for a cohort too large to fetch (see ctgov/bucket_counts.py)."""

    counts: dict[str, int]
    samples: dict[str, list[TrialRow]]  # a few trials per bucket, for citations
    unclassified: int = 0
    excluded: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class FetchedQuery:
    """One executed search: what was asked, what the API reported, and the records analyzed."""

    label: str | None  # cohort label for comparisons
    filters: Filters  # effective filters (plan + cohort + structured request fields)
    api_params: dict[str, str]
    total_matching: int
    rows: list[TrialRow]
    exact: ExactBuckets | None = None  # set when the record cap was hit and the dimension is countable

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
