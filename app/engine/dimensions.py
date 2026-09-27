"""How each group-by dimension behaves in a chart.

`partition` is the key property: a partition dimension puts every trial in at most one
bucket, so bars sum to the trial count. An overlapping one (country, intervention,
condition) can place one trial in several bars, and the response says so.
"""

from dataclasses import dataclass
from typing import Literal

from app.schemas.enums import Dimension

FieldType = Literal["nominal", "ordinal", "quantitative", "temporal"]

PHASE_ORDER = (
    "EARLY_PHASE1",
    "PHASE1",
    "PHASE1/PHASE2",
    "PHASE2",
    "PHASE2/PHASE3",
    "PHASE3",
    "PHASE4",
    "NA",
    "MISSING",
)


@dataclass(frozen=True)
class DimSpec:
    field: str  # row key holding the raw value, e.g. "phase"
    axis_label: str
    field_type: FieldType
    partition: bool
    order: Literal["fixed", "count", "chronological"]
    fixed_order: tuple[str, ...] = ()
    assumption: str | None = None

    @property
    def label_field(self) -> str | None:
        return None if self.field_type == "temporal" else f"{self.field}_label"


DIMENSIONS: dict[Dimension, DimSpec] = {
    Dimension.PHASE: DimSpec(
        "phase",
        "Trial phase",
        "ordinal",
        partition=True,
        order="fixed",
        fixed_order=PHASE_ORDER,
        assumption=(
            "Trials registered with two phases (e.g. Phase 1/Phase 2) are counted once, in a combined bucket. "
            "'Not Applicable' (an explicit NA, typical of non-drug studies) and 'Not specified' (no phase recorded) "
            "are kept separate."
        ),
    ),
    Dimension.STATUS: DimSpec("status", "Overall status", "nominal", partition=True, order="count"),
    Dimension.START_YEAR: DimSpec(
        "start_year",
        "Start year",
        "temporal",
        partition=True,
        order="chronological",
        assumption="Start year comes from the registry start date, which is the anticipated date for trials that have not started yet.",
    ),
    Dimension.SPONSOR: DimSpec(
        "sponsor",
        "Lead sponsor",
        "nominal",
        partition=True,
        order="count",
        assumption="Sponsor means the lead sponsor; collaborators are not counted.",
    ),
    Dimension.SPONSOR_CLASS: DimSpec(
        "sponsor_class",
        "Lead sponsor type",
        "nominal",
        partition=True,
        order="count",
        assumption="Sponsor type is the registry's class for the lead sponsor (e.g. Industry, NIH, Other).",
    ),
    Dimension.COUNTRY: DimSpec(
        "country",
        "Country",
        "nominal",
        partition=False,
        order="count",
        assumption="A trial with sites in several countries is counted once in each of those countries.",
    ),
    Dimension.INTERVENTION: DimSpec(
        "intervention",
        "Intervention",
        "nominal",
        partition=False,
        order="count",
        assumption=(
            "Intervention names are grouped by the MeSH term the registry assigns when one matches, so name "
            "variants of the same drug count together. A trial is counted once per distinct intervention."
        ),
    ),
    Dimension.INTERVENTION_TYPE: DimSpec(
        "intervention_type",
        "Intervention type",
        "nominal",
        partition=False,
        order="count",
        assumption="A trial is counted once per distinct intervention type it uses.",
    ),
    Dimension.CONDITION: DimSpec(
        "condition",
        "Condition",
        "nominal",
        partition=False,
        order="count",
        assumption="Conditions use the registry's MeSH terms when available; a trial is counted once per condition.",
    ),
}
