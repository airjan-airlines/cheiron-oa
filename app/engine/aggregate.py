"""Deterministic grouping and counting, with evidence kept for every bucket."""

from collections.abc import Iterable
from dataclasses import dataclass, field

from app.ctgov.extract import DimValue, TrialRow
from app.engine.dimensions import DIMENSIONS, DimSpec
from app.schemas.enums import Dimension
from app.schemas.response import Citation

STUDY_URL = "https://clinicaltrials.gov/study/{}"


@dataclass
class Bucket:
    key: str
    label: str
    members: list[tuple[TrialRow, DimValue]] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.members)


@dataclass
class Grouping:
    buckets: list[Bucket]  # in display order, already cut to top_n
    unclassified: list[TrialRow]  # trials with no value for the dimension
    not_shown: int = 0  # buckets beyond top_n


def evidence(members: Iterable[tuple[TrialRow, DimValue]], max_citations: int) -> dict:
    """Evidence fields for a datum: every contributing NCT ID plus capped excerpts."""
    members = list(members)
    return {
        "supporting_nct_ids": [row.nct_id for row, _ in members],
        "citation_count": len(members),
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


def group(rows: list[TrialRow], dim: Dimension, top_n: int, year_range: tuple[int | None, int | None] = (None, None)) -> Grouping:
    spec = DIMENSIONS[dim]
    buckets, unclassified = collect(rows, dim)
    if dim == Dimension.START_YEAR:
        buckets = fill_years(buckets, *year_range)
    keys = order_keys({k: b.count for k, b in buckets.items()}, spec)
    shown = keys[:top_n] if spec.order == "count" else keys
    return Grouping([buckets[k] for k in shown], unclassified, not_shown=len(keys) - len(shown))


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
