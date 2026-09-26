"""Translate plan `Filters` into ClinicalTrials.gov `/studies` query parameters.

Every syntax used here was checked against the live API (see docs/api_reference.md):
- phase filtering goes through `filter.advanced` (`filter.phase` does not exist);
- `query.lead` matches the lead sponsor only (`query.spons` also matches collaborators,
  which would disagree with grouping by lead sponsor: Pfizer 3,880 vs 6,087);
- the registry spells Korea "South Korea"; "Korea, Republic of" returns 0 trials.
"""

from app.schemas.plan import Filters

# Only the fields the engine reads. Keeps pages ~160 KB instead of several MB.
FIELDS: tuple[str, ...] = (
    "NCTId",
    "BriefTitle",
    "Phase",
    "OverallStatus",
    "StartDate",
    "PrimaryCompletionDate",
    "LeadSponsorName",
    "LeadSponsorClass",
    "InterventionName",
    "InterventionType",
    "InterventionMeshTerm",
    "Condition",
    "ConditionMeshTerm",
    "LocationCountry",
    "EnrollmentCount",
)

# Keys are casefolded; values are the registry's own spelling. Only aliases that
# measurably change results are listed (live `query.locn` totals, 2026-09-26):
# "USA" 729 vs "United States" 195,650; "UK" 735 vs "United Kingdom" 28,935;
# "Republic of Korea" 191 vs "South Korea" 17,735; Korean script matches 0.
# The API already resolves Russia/Czechia/Iran variants itself, so they are omitted.
COUNTRY_ALIASES: dict[str, str] = {
    "korea": "South Korea",
    "republic of korea": "South Korea",
    "korea, republic of": "South Korea",
    "s. korea": "South Korea",
    "한국": "South Korea",
    "대한민국": "South Korea",
    "usa": "United States",
    "us": "United States",
    "u.s.": "United States",
    "u.s.a.": "United States",
    "united states of america": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "great britain": "United Kingdom",
}


def canonical_location(value: str) -> str:
    return COUNTRY_ALIASES.get(value.strip().casefold(), value.strip())


def normalize_filters(filters: Filters) -> tuple[Filters, list[str]]:
    """Apply registry spellings. Returns the normalized filters and a note per change."""
    notes: list[str] = []
    if filters.location:
        canonical = canonical_location(filters.location)
        if canonical != filters.location:
            notes.append(f"Location '{filters.location}' normalized to the registry spelling '{canonical}'.")
            filters = filters.model_copy(update={"location": canonical})
    return filters, notes


def _advanced_expression(filters: Filters) -> str | None:
    clauses: list[str] = []
    if filters.phases:
        values = [p.value for p in filters.phases]
        clauses.append(f"AREA[Phase]{values[0]}" if len(values) == 1 else f"AREA[Phase]({' OR '.join(values)})")
    if filters.start_year_min is not None or filters.start_year_max is not None:
        lo = f"{filters.start_year_min}-01-01" if filters.start_year_min is not None else "MIN"
        hi = f"{filters.start_year_max}-12-31" if filters.start_year_max is not None else "MAX"
        clauses.append(f"AREA[StartDate]RANGE[{lo},{hi}]")
    return " AND ".join(clauses) or None


def build_params(filters: Filters) -> dict[str, str]:
    """Search params for `GET /studies`, excluding paging (the client adds pageSize/pageToken)."""
    params: dict[str, str] = {}
    if filters.condition:
        params["query.cond"] = filters.condition
    if filters.intervention:
        params["query.intr"] = filters.intervention
    if filters.sponsor:
        params["query.lead"] = filters.sponsor
    if filters.location:
        params["query.locn"] = filters.location
    if filters.statuses:
        params["filter.overallStatus"] = ",".join(s.value for s in filters.statuses)
    advanced = _advanced_expression(filters)
    if advanced:
        params["filter.advanced"] = advanced
    params["fields"] = ",".join(FIELDS)
    params["countTotal"] = "true"
    return params
