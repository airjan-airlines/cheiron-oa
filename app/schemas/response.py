"""Response body for `POST /v1/visualize` (OA §3.2, §5).

Design rules a renderer can rely on:
- `visualization` is a discriminated union on `type`; each type has its own encoding shape.
- Every `encoding.*.field` names a key present in every `data` row (validated below).
- Categorical rows carry the raw registry value and a display label side by side
  (`"phase": "PHASE2"`, `"phase_label": "Phase 2"`); the channel's `label_field` says which key holds the label.
- Every datum carries its evidence: the full list of contributing NCT IDs plus capped citation excerpts.
"""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.enums import VizType
from app.schemas.plan import Filters, QueryPlan

SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Evidence (deep citations)
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    nct_id: str = Field(pattern=r"^NCT\d{8}$")
    field: str = Field(description="API field path the excerpt was read from, e.g. `protocolSection.designModule.phases`.")
    excerpt: str = Field(description="Exact value from the API response. Lists are JSON-encoded, e.g. `[\"PHASE1\", \"PHASE2\"]`.")
    title: str | None = Field(None, description="The trial's brief title, for context.")
    url: str = Field(description="Link to the trial record on ClinicalTrials.gov.")


class Evidence(BaseModel):
    """Traceability attached to every datum: bar, time bucket, histogram bin, node and edge."""

    supporting_nct_ids: list[str] = Field(
        default_factory=list,
        description="Trials that contributed to this datum. Complete unless `supporting_nct_ids_complete` is false "
        "(large cohorts counted with per-bucket registry counts, where only example trials are listed).",
    )
    supporting_nct_ids_complete: bool = Field(True, description="True when `supporting_nct_ids` lists every contributing trial.")
    citation_count: int = Field(0, ge=0, description="Number of contributing trials (equals the plotted count).")
    citations: list[Citation] = Field(
        default_factory=list, description="Excerpts for the first `max_citations_per_datum` supporting trials."
    )


class DataRow(Evidence):
    """One tabular datum. Chart-specific keys (e.g. `phase`, `trial_count`) are extra fields."""

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

FieldType = Literal["nominal", "ordinal", "quantitative", "temporal"]


class Channel(BaseModel):
    field: str = Field(description="Key in each data row that feeds this channel.")
    type: FieldType
    label: str = Field(description="Axis or legend title.")
    label_field: str | None = Field(None, description="Row key holding the display label for categorical values.")


class XYEncoding(BaseModel):
    x: Channel
    y: Channel


class SeriesEncoding(XYEncoding):
    series: Channel


class OptionalSeriesEncoding(XYEncoding):
    series: Channel | None = None


class HistogramEncoding(BaseModel):
    x: Channel = Field(description="Bin start (inclusive).")
    x2: Channel = Field(description="Bin end (inclusive).")
    y: Channel


class MetricEncoding(BaseModel):
    value: Channel


class NodeEncoding(BaseModel):
    id: str = "id"
    label: str = "label"
    size: str = Field("size", description="Node key used for node size (trials involving this entity).")
    color: str = Field("type", description="Node key used for color (entity type).")


class EdgeEncoding(BaseModel):
    source: str = "source"
    target: str = "target"
    weight: str = Field("weight", description="Edge key used for thickness (trials linking both endpoints).")


class NetworkEncoding(BaseModel):
    node: NodeEncoding = Field(default_factory=NodeEncoding)
    edge: EdgeEncoding = Field(default_factory=EdgeEncoding)


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


class _TabularViz(BaseModel):
    """Shared validation for charts whose `data` is a list of rows.

    Each subclass declares `type`, `title`, `encoding`, `data` in that order, so
    serialized JSON reads the way the spec presents it.
    """

    @model_validator(mode="after")
    def _encoded_fields_exist(self) -> "_TabularViz":
        channels = [c for c in self.encoding.__dict__.values() if isinstance(c, Channel)]  # type: ignore[attr-defined]
        for channel in channels:
            keys = [channel.field] + ([channel.label_field] if channel.label_field else [])
            for key in keys:
                for i, row in enumerate(self.data):
                    if key not in (row.model_extra or {}):
                        raise ValueError(f"data[{i}] is missing encoded field '{key}'")
        return self


class BarChart(_TabularViz):
    type: Literal[VizType.BAR_CHART] = VizType.BAR_CHART
    title: str
    encoding: XYEncoding
    data: list[DataRow]


class GroupedBarChart(_TabularViz):
    type: Literal[VizType.GROUPED_BAR_CHART] = VizType.GROUPED_BAR_CHART
    title: str
    encoding: SeriesEncoding
    data: list[DataRow]


class TimeSeries(_TabularViz):
    type: Literal[VizType.TIME_SERIES] = VizType.TIME_SERIES
    title: str
    encoding: OptionalSeriesEncoding
    data: list[DataRow]


class Histogram(_TabularViz):
    type: Literal[VizType.HISTOGRAM] = VizType.HISTOGRAM
    title: str
    encoding: HistogramEncoding
    data: list[DataRow]


class MetricViz(_TabularViz):
    """A single number: the question needs no chart (OA §1, "identify if a visualization is needed")."""

    type: Literal[VizType.METRIC] = VizType.METRIC
    title: str
    encoding: MetricEncoding
    data: list[DataRow]


class Node(Evidence):
    id: str
    label: str
    type: str = Field(description="Entity type, e.g. `sponsor` or `intervention`.")
    size: int = Field(ge=0, description="Trials involving this entity.")


class Edge(Evidence):
    source: str
    target: str
    weight: int = Field(ge=1, description="Trials linking both endpoints.")


class NetworkData(BaseModel):
    nodes: list[Node]
    edges: list[Edge]

    @model_validator(mode="after")
    def _edges_reference_nodes(self) -> "NetworkData":
        ids = {n.id for n in self.nodes}
        for e in self.edges:
            if e.source not in ids or e.target not in ids:
                raise ValueError(f"edge {e.source!r} -> {e.target!r} references a missing node")
        return self


class NetworkGraph(BaseModel):
    type: Literal[VizType.NETWORK_GRAPH] = VizType.NETWORK_GRAPH
    title: str
    encoding: NetworkEncoding = Field(default_factory=NetworkEncoding)
    data: NetworkData


Visualization = Annotated[
    BarChart | GroupedBarChart | TimeSeries | Histogram | MetricViz | NetworkGraph,
    Field(discriminator="type"),
]

# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------


class AppliedQuery(BaseModel):
    """One ClinicalTrials.gov search that fed the chart (a comparison has one per cohort)."""

    label: str | None = Field(None, description="Cohort label for comparisons; null otherwise.")
    filters: Filters
    api_params: dict[str, str] = Field(description="Exact query parameters sent to `/api/v2/studies`.")
    total_matching: int = Field(ge=0, description="Trials matching per the API's `totalCount`.")
    records_analyzed: int = Field(ge=0)
    truncated: bool = Field(description="True when total_matching > records_analyzed (record cap hit).")


class Exclusion(BaseModel):
    reason: str
    count: int = Field(ge=0)


class Coverage(BaseModel):
    """How the plotted numbers relate to the underlying trials (D15)."""

    aggregation_mode: Literal["all_records", "exact_counts", "sample"] = Field(
        "all_records",
        description="`all_records`: every matching trial was analyzed. `exact_counts`: too many trials to fetch, so each "
        "bucket's count came from the registry's own per-bucket total. `sample`: counts cover only the first "
        "`records_analyzed` trials (dimensions without a fixed value set, e.g. country); the y-axis label says so.",
    )
    groupby_semantics: Literal["partition", "overlapping"] | None = Field(
        None,
        description="`partition`: each trial is in exactly one bucket, so buckets sum to the trial count. "
        "`overlapping`: a trial can appear in several buckets (e.g. multi-country trials). Null for networks and metrics.",
    )
    bucket_sum: int | None = Field(None, description="Sum of plotted values.")
    unclassified_count: int = Field(0, ge=0, description="Trials that could not be placed in any bucket.")
    overlap_note: str | None = None
    excluded: list[Exclusion] = Field(default_factory=list, description="Trials left out of the chart, by reason.")
    buckets_not_shown: int = Field(0, ge=0, description="Categories beyond `top_n` that were counted but not plotted.")


class SortSpec(BaseModel):
    field: str
    order: Literal["ascending", "descending"]


class Pruning(BaseModel):
    top_n: int
    min_edge_weight: int | None = None
    nodes_dropped: int = 0
    edges_dropped: int = 0


class RenderHints(BaseModel):
    """Everything a frontend needs beyond `encoding` + `data` (OA §3.2: units, sorting, granularity, grouping)."""

    sort: SortSpec | None = None
    time_granularity: Literal["year"] | None = None
    units: dict[str, str] = Field(default_factory=dict, description="Unit per channel, e.g. {\"y\": \"trials\"}.")
    grouping: str | None = Field(None, description="Row key that splits series or bar groups, if any.")
    pruning: Pruning | None = None


class VisualizationRationale(BaseModel):
    chosen: VizType
    source: Literal["request", "llm", "rule_default"] = Field(description="Where the chosen type came from.")
    suggested: VizType | None = Field(None, description="Type the caller or LLM proposed, if it was overridden.")
    reason: str


class Meta(BaseModel):
    interpretation: str | None = Field(None, description="What was computed, in words. Generated from the plan, never an LLM claim about results.")
    queries: list[AppliedQuery] = Field(default_factory=list)
    coverage: Coverage | None = None
    render: RenderHints | None = None
    visualization_rationale: VisualizationRationale | None = None
    assumptions: list[str] = Field(default_factory=list, description="Interpretation choices a reader would otherwise have to guess.")
    notes: list[str] = Field(default_factory=list, description="Overrides, fallbacks and other things worth knowing about this run.")
    plan: QueryPlan | None = Field(None, description="The validated plan that was executed.")
    source: str = "ClinicalTrials.gov API v2"
    data_as_of: str | None = Field(None, description="The API's `dataTimestamp` at query time.")
    generated_at: datetime | None = None


class ErrorInfo(BaseModel):
    code: str
    message: str


class VisualizeResponse(BaseModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    status: Literal["ok", "no_data", "unsupported", "error"]
    visualization: Visualization | None = None
    meta: Meta = Field(default_factory=Meta)
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def _visualization_iff_ok(self) -> "VisualizeResponse":
        if (self.status == "ok") != (self.visualization is not None):
            raise ValueError("visualization must be present exactly when status is 'ok'")
        return self
