"""Exact per-bucket counts for large cohorts (D20).

When a search matches more trials than the record cap, counting the fetched records would
undercount every bar (e.g. 237 vs 6,351 cancer trials starting in 2020). For dimensions with
a fixed set of values (phase, status, sponsor class, intervention type, start year), we can
instead ask the registry for each bucket's exact count: one `countTotal` request per bucket,
with that bucket's condition ANDed onto the search. The same request returns a few trials to
cite, so evidence stays real (just not exhaustive).

Every expression below was checked live: for "cancer" (123,589 trials) the nine phase
buckets sum to exactly 123,589.
"""

import asyncio
from datetime import date
from dataclasses import dataclass, field
from typing import Any

from app.ctgov.client import CTGovClient
from app.ctgov.extract import start_year
from app.schemas.enums import INTERVENTION_TYPE_LABELS, SPONSOR_CLASS_LABELS, STATUS_LABELS, Dimension

# Each key matches the bucket key `extract` produces for the same trials.
PHASE_EXPRESSIONS: dict[str, str] = {
    "EARLY_PHASE1": "AREA[Phase]EARLY_PHASE1",
    "PHASE1": "AREA[Phase]PHASE1 AND NOT AREA[Phase]PHASE2",
    "PHASE1/PHASE2": "AREA[Phase]PHASE1 AND AREA[Phase]PHASE2",
    "PHASE2": "AREA[Phase]PHASE2 AND NOT AREA[Phase]PHASE1 AND NOT AREA[Phase]PHASE3",
    "PHASE2/PHASE3": "AREA[Phase]PHASE2 AND AREA[Phase]PHASE3",
    "PHASE3": "AREA[Phase]PHASE3 AND NOT AREA[Phase]PHASE2",
    "PHASE4": "AREA[Phase]PHASE4",
    "NA": "AREA[Phase]NA",
    "MISSING": "AREA[Phase]MISSING",
}
ENUM_AREAS: dict[Dimension, tuple[str, list[str]]] = {
    Dimension.STATUS: ("OverallStatus", list(STATUS_LABELS)),
    Dimension.SPONSOR_CLASS: ("LeadSponsorClass", list(SPONSOR_CLASS_LABELS)),
    Dimension.INTERVENTION_TYPE: ("InterventionType", list(INTERVENTION_TYPE_LABELS)),
}
# Dimensions where "no value" is its own AREA[...]MISSING count rather than a bucket.
MISSING_AREAS: dict[Dimension, str] = {Dimension.START_YEAR: "StartDate", Dimension.INTERVENTION_TYPE: "InterventionType"}
EXACT_DIMENSIONS = {Dimension.PHASE, Dimension.START_YEAR, *ENUM_AREAS}
PARTITION_ENUMS = {Dimension.STATUS, Dimension.SPONSOR_CLASS}  # one value per trial
REMAINDER_KEY = "_OTHER_VALUES"  # never a registry value (sponsor class has a real "OTHER")
# Per response, across cohorts. The registry's rate limit is undocumented; during testing it
# returned 429s after roughly 40-60 requests in under a minute, so stay well under that.
MAX_BUCKET_REQUESTS = 20


@dataclass
class ExactCounts:
    counts: dict[str, int]  # bucket key -> exact number of matching trials (REMAINDER_KEY: all other values)
    samples: dict[str, list[dict[str, Any]]]  # bucket key -> a few raw studies to cite
    unclassified: int = 0  # trials with no value for the dimension
    excluded: list[tuple[str, int]] = field(default_factory=list)  # (reason, count) outside the counted window
    requests: int = 0


def _and(base: str | None, clause: str) -> str:
    return f"({base}) AND ({clause})" if base else clause


def _year_expr(lo: int | str, hi: int | str) -> str:
    lo = f"{lo}-01-01" if isinstance(lo, int) else lo
    hi = f"{hi}-12-31" if isinstance(hi, int) else hi
    return f"AREA[StartDate]RANGE[{lo},{hi}]"


async def _count(client: CTGovClient, params: dict[str, str], clause: str, sample_size: int) -> tuple[int, list[dict]]:
    p = {**params, "filter.advanced": _and(params.get("filter.advanced"), clause)}
    result = await client.search(p, max_records=sample_size)
    return result.total_count, result.studies


async def _earliest_year(client: CTGovClient, params: dict[str, str]) -> int | None:
    """Earliest start year among matching trials, via one sorted one-record search."""
    clause = _and(params.get("filter.advanced"), "NOT AREA[StartDate]MISSING")
    result = await client.search({**params, "filter.advanced": clause, "sort": "StartDate:asc"}, max_records=1)
    if not result.studies:
        return None
    return start_year(result.studies[0].get("protocolSection", {}).get("statusModule", {}).get("startDateStruct", {}).get("date"))


async def exact_counts(
    client: CTGovClient,
    params: dict[str, str],
    dim: Dimension,
    total: int,
    seen_keys: set[str],
    sample_size: int,
    year_range: tuple[int | None, int | None],
    request_budget: int,
    current_year: int | None = None,
) -> ExactCounts | None:
    """Exact bucket counts for `dim`, or None if it can't be counted exactly within budget.

    `seen_keys` are the values present in the fetched records of every cohort (their union, so
    all cohorts count the same values and a missing bar is a true zero). Enum dimensions count
    only those values; for partition dimensions the rest is an exact remainder bucket (total
    minus everything counted), so no trial goes missing.
    """
    sample_size = max(sample_size, 1)
    current_year = current_year or date.today().year
    excluded: list[tuple[str, int]] = []
    extra = 0
    if dim == Dimension.PHASE:
        clauses = dict(PHASE_EXPRESSIONS)
    elif dim in ENUM_AREAS:
        area, values = ENUM_AREAS[dim]
        clauses = {v: f"AREA[{area}]{v}" for v in values if v in seen_keys}
    elif dim == Dimension.START_YEAR:
        lo, hi = year_range
        if hi is None:
            # Anchor at the current year: anticipated future starts must not crowd real years out
            # of the window. They are counted once and reported, not plotted.
            hi = current_year
            later, _ = await _count(client, params, _year_expr(f"{hi + 1}-01-01", "MAX"), 1)
            extra += 1
            if later:
                excluded.append((f"planned to start after {hi} (anticipated dates)", later))
        if lo is None:
            lo = await _earliest_year(client, params)
            extra += 1
        if lo is None or lo > hi:
            return None
        max_years = request_budget - extra - 2  # room for the "missing" and "earlier" counts
        if max_years < 5:
            return None
        if hi - lo + 1 > max_years:
            cut = hi - max_years + 1
            earlier, _ = await _count(client, params, _year_expr("MIN", cut - 1), 1)
            extra += 1
            if earlier:
                excluded.append((f"started before {cut} (outside the {max_years}-year window counted exactly)", earlier))
            lo = cut
        clauses = {str(y): _year_expr(y, y) for y in range(lo, hi + 1)}
    else:
        return None

    missing_area = MISSING_AREAS.get(dim)
    needed = len(clauses) + (1 if missing_area else 0) + extra
    if needed > request_budget:
        return None

    keys = list(clauses)
    results = await asyncio.gather(*(_count(client, params, clauses[k], sample_size) for k in keys))
    unclassified = 0
    if missing_area:
        unclassified, _ = await _count(client, params, f"AREA[{missing_area}]MISSING", 1)
    counts = {k: n for k, (n, _) in zip(keys, results) if n or dim == Dimension.START_YEAR}
    samples = {k: studies for k, (_, studies) in zip(keys, results) if k in counts}
    if dim in PARTITION_ENUMS:
        remainder = total - sum(counts.values()) - unclassified
        if remainder > 0:
            counts[REMAINDER_KEY] = remainder
            samples[REMAINDER_KEY] = []
    return ExactCounts(counts, samples, unclassified, excluded, requests=needed)
