"""Closed vocabularies shared by the request, plan and response schemas.

Registry enum values and their human labels come from the live
`GET /api/v2/studies/enums` endpoint (snapshot 2026-09-26). The labels are the
API's own `legacyValue` strings, so the text we display matches what
ClinicalTrials.gov shows on its website.
"""

from enum import StrEnum


class Phase(StrEnum):
    EARLY_PHASE1 = "EARLY_PHASE1"
    PHASE1 = "PHASE1"
    PHASE2 = "PHASE2"
    PHASE3 = "PHASE3"
    PHASE4 = "PHASE4"
    NA = "NA"


PHASE_LABELS: dict[str, str] = {
    "EARLY_PHASE1": "Early Phase 1",
    "PHASE1": "Phase 1",
    "PHASE2": "Phase 2",
    "PHASE3": "Phase 3",
    "PHASE4": "Phase 4",
    "NA": "Not Applicable",
}


class Status(StrEnum):
    RECRUITING = "RECRUITING"
    NOT_YET_RECRUITING = "NOT_YET_RECRUITING"
    ENROLLING_BY_INVITATION = "ENROLLING_BY_INVITATION"
    ACTIVE_NOT_RECRUITING = "ACTIVE_NOT_RECRUITING"
    SUSPENDED = "SUSPENDED"
    TERMINATED = "TERMINATED"
    COMPLETED = "COMPLETED"
    WITHDRAWN = "WITHDRAWN"
    AVAILABLE = "AVAILABLE"
    NO_LONGER_AVAILABLE = "NO_LONGER_AVAILABLE"
    TEMPORARILY_NOT_AVAILABLE = "TEMPORARILY_NOT_AVAILABLE"
    APPROVED_FOR_MARKETING = "APPROVED_FOR_MARKETING"
    WITHHELD = "WITHHELD"
    UNKNOWN = "UNKNOWN"


STATUS_LABELS: dict[str, str] = {
    "RECRUITING": "Recruiting",
    "NOT_YET_RECRUITING": "Not yet recruiting",
    "ENROLLING_BY_INVITATION": "Enrolling by invitation",
    "ACTIVE_NOT_RECRUITING": "Active, not recruiting",
    "SUSPENDED": "Suspended",
    "TERMINATED": "Terminated",
    "COMPLETED": "Completed",
    "WITHDRAWN": "Withdrawn",
    "AVAILABLE": "Available",
    "NO_LONGER_AVAILABLE": "No longer available",
    "TEMPORARILY_NOT_AVAILABLE": "Temporarily not available",
    "APPROVED_FOR_MARKETING": "Approved for marketing",
    "WITHHELD": "Withheld",
    "UNKNOWN": "Unknown status",
}

SPONSOR_CLASS_LABELS: dict[str, str] = {
    "INDUSTRY": "Industry",
    "NIH": "NIH",
    "FED": "U.S. Federal (non-NIH)",
    "OTHER_GOV": "Other government",
    "INDIV": "Individual",
    "NETWORK": "Network",
    "AMBIG": "Ambiguous",
    "OTHER": "Other (academic, hospital, nonprofit)",
    "UNKNOWN": "Unknown",
}

INTERVENTION_TYPE_LABELS: dict[str, str] = {
    "BEHAVIORAL": "Behavioral",
    "BIOLOGICAL": "Biological",
    "COMBINATION_PRODUCT": "Combination Product",
    "DEVICE": "Device",
    "DIAGNOSTIC_TEST": "Diagnostic Test",
    "DIETARY_SUPPLEMENT": "Dietary Supplement",
    "DRUG": "Drug",
    "GENETIC": "Genetic",
    "PROCEDURE": "Procedure",
    "RADIATION": "Radiation",
    "OTHER": "Other",
}


class Analysis(StrEnum):
    """The question class. The LLM picks one; code does the rest."""

    DISTRIBUTION = "distribution"  # "How are X trials distributed across phases?"
    TIME_TREND = "time_trend"  # "How has the number of trials changed per year?"
    COMPARISON = "comparison"  # "Compare phases for Drug A vs Drug B"
    GEOGRAPHIC = "geographic"  # "Which countries have the most recruiting trials?"
    NETWORK = "network"  # "Network of sponsors and drugs"
    NUMERIC_DISTRIBUTION = "numeric_distribution"  # "Distribution of enrollment sizes"
    COUNT = "count"  # "How many trials...?" -> a single number, no chart needed


class Dimension(StrEnum):
    """Fields a chart can group by (or connect, for networks)."""

    PHASE = "phase"
    STATUS = "status"
    START_YEAR = "start_year"
    SPONSOR = "sponsor"
    SPONSOR_CLASS = "sponsor_class"
    COUNTRY = "country"
    INTERVENTION = "intervention"
    INTERVENTION_TYPE = "intervention_type"
    CONDITION = "condition"


class Metric(StrEnum):
    TRIAL_COUNT = "trial_count"
    ENROLLMENT = "enrollment"


class VizType(StrEnum):
    BAR_CHART = "bar_chart"
    GROUPED_BAR_CHART = "grouped_bar_chart"
    TIME_SERIES = "time_series"
    HISTOGRAM = "histogram"
    NETWORK_GRAPH = "network_graph"
    METRIC = "metric"
