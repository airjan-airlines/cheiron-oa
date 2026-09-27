"""Deterministic grouping and counting, with evidence kept for every bucket."""

from collections.abc import Iterable
from dataclasses import dataclass, field

from app.ctgov.bucket_counts import REMAINDER_KEY
from app.ctgov.extract import MISSING_PHASE_KEY, MISSING_PHASE_LABEL, DimValue, TrialRow
from app.engine.dimensions import DIMENSIONS, DimSpec
from app.engine.types import ExactBuckets
from app.schemas.enums import INTERVENTION_TYPE_LABELS, PHASE_LABELS, SPONSOR_CLASS_LABELS, STATUS_LABELS, Dimension
from app.schemas.response import Citation

STUDY_URL = "https://clinicaltrials.gov/study/{}"


@dataclass
class Bucket:
    key: str
    label: str
    members: list[tuple[TrialRow, DimValue]] = field(default_factory=list)
    total: int | None = None  # exact registry count, when members are only example trials

    @property
    def count(self) -> int:
        return self.total if self.total is not None else len(self.members)


@dataclass
class Grouping:
    buckets: list[Bucket]  # in display order, already cut to top_n
    unclassified_count: int  # trials with no value for the dimension
    not_shown: int = 0  # buckets beyond top_n
    excluded: list[tuple[str, int]] = field(default_factory=list)  # (reason, count) left out of the chart


def evidence(members: Iterable[tuple[TrialRow, DimValue]], max_citations: int, total: int | None = None) -> dict:
    """Evidence fields for a datum: contributing NCT IDs plus capped excerpts.

    `total` is the exact count when `members` are only example trials (exact-count mode).
    """
    members = list(members)
    count = total if total is not None else len(members)
    return {
        "supporting_nct_ids": [row.nct_id for row, _ in members],
        "supporting_nct_ids_complete": count == len(members),
        "citation_count": count,
        "citations": [
            Citation(nct_id=row.nct_id, field=value.field, excerpt=value.excerpt, title=row.title, url=STUDY_URL.format(row.nct_id))
            for row, value in members[:max_citations]
        ],
    }


def collect(rows: Iterable[TrialRow], dim: Dimension) -> tuple[dict[str, Bucket], list[TrialRow]]:
    buckets: dict[str, Bucket] = {}
    unclassified: list[TrialRow] = []
    for row in rows:
        values = row.get(dim)
        if not values:
            unclassified.append(row)
        for value in values:
            buckets.setdefault(value.key, Bucket(value.key, value.label)).members.append((row, value))
    return buckets, unclassified


def order_keys(counts: dict[str, int], spec: DimSpec) -> list[str]:
    if spec.order == "fixed":
        rank = {k: i for i, k in enumerate(spec.fixed_order)}
        return sorted(counts, key=lambda k: (rank.get(k, len(rank)), k))
    if spec.order == "chronological":
        return sorted(counts, key=int)
    return sorted(counts, key=lambda k: (-counts[k], k))  # most trials first, ties alphabetical


def fill_years(buckets: dict[str, Bucket], lo: int | None, hi: int | None) -> dict[str, Bucket]:
    """Add empty buckets so a time series has no silent gaps."""
    years = [int(k) for k in buckets]
    lo = lo if lo is not None else min(years, default=None)
    hi = hi if hi is not None else max(years, default=None)
    if lo is None or hi is None:
        return buckets
    filled = {str(y): buckets.get(str(y), Bucket(str(y), str(y))) for y in range(lo, hi + 1)}
    return filled


def _cut(buckets: dict[str, Bucket], spec: DimSpec, top_n: int) -> tuple[list[Bucket], int]:
    keys = order_keys({k: b.count for k, b in buckets.items()}, spec)
    if REMAINDER_KEY in keys:  # the "other values" remainder always goes last
        keys = [k for k in keys if k != REMAINDER_KEY] + [REMAINDER_KEY]
    shown = keys[:top_n] if spec.order == "count" else keys
    return [buckets[k] for k in shown], len(keys) - len(shown)


def group(rows: list[TrialRow], dim: Dimension, top_n: int, year_range: tuple[int | None, int | None] = (None, None)) -> Grouping:
    """Group fetched records (every matching trial, or the first `max_records` of them)."""
    spec = DIMENSIONS[dim]
    buckets, unclassified = collect(rows, dim)
    if dim == Dimension.START_YEAR:
        buckets = fill_years(buckets, *year_range)
    shown, not_shown = _cut(buckets, spec, top_n)
    return Grouping(shown, len(unclassified), not_shown=not_shown)


def group_exact(exact: ExactBuckets, dim: Dimension, top_n: int) -> Grouping:
    """Group from per-bucket registry counts; members are example trials for citations."""
    spec = DIMENSIONS[dim]
    buckets: dict[str, Bucket] = {}
    for key, count in exact.counts.items():
        members = [(row, v) for row in exact.samples.get(key, []) for v in row.get(dim) if v.key == key]
        buckets[key] = Bucket(key, bucket_label(dim, key), members, total=count)
    shown, not_shown = _cut(buckets, spec, top_n)
    return Grouping(shown, exact.unclassified, not_shown=not_shown, excluded=list(exact.excluded))


def bucket_label(dim: Dimension, key: str) -> str:
    """Display label for a bucket key without needing a record that has it."""
    if key == REMAINDER_KEY:
        return "Other (values absent from the sampled records)"
    if dim == Dimension.PHASE:
        return MISSING_PHASE_LABEL if key == MISSING_PHASE_KEY else "/".join(PHASE_LABELS.get(p, p) for p in key.split("/"))
    labels = {Dimension.STATUS: STATUS_LABELS, Dimension.SPONSOR_CLASS: SPONSOR_CLASS_LABELS, Dimension.INTERVENTION_TYPE: INTERVENTION_TYPE_LABELS}
    return labels.get(dim, {}).get(key, key)


def overlap_note(rows: list[TrialRow], dim: Dimension) -> str | None:
    """For overlapping dimensions, say how many bucket memberships the trials produce."""
    if DIMENSIONS[dim].partition:
        return None
    classified = [r for r in rows if r.get(dim)]
    memberships = sum(len(r.get(dim)) for r in classified)
    if memberships == len(classified):
        return None
    return (
        f"{len(classified)} trials produce {memberships} {DIMENSIONS[dim].axis_label.lower()} memberships, "
        "so bars can sum to more than the number of trials."
    )
