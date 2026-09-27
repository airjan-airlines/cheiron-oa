"""Turn a validated plan plus fetched trials into a visualization. Fully deterministic.

Every analysis goes through one of five builders; group-by behavior comes from the
dimension registry, not per-question code, so a new dimension needs no new branches.
"""

from app.ctgov.extract import ENROLLMENT_PATH, DimValue, TrialRow
from app.engine.aggregate import Bucket, bucket_label, collect, evidence, fill_years, group, group_exact, order_keys, overlap_note
from app.engine.chart_rules import choose_chart
from app.engine.describe import capitalize, subject, year_span
from app.engine.dimensions import DIMENSIONS
from app.engine.network import build_network
from app.engine.types import ExecutionResult, FetchedQuery
from app.schemas.enums import Analysis, Dimension, VizType
from app.schemas.plan import QueryPlan
from app.schemas.response import (
    BarChart,
    Channel,
    Coverage,
    DataRow,
    Exclusion,
    GroupedBarChart,
    Histogram,
    HistogramEncoding,
    MetricEncoding,
    MetricViz,
    NetworkGraph,
    OptionalSeriesEncoding,
    RenderHints,
    SeriesEncoding,
    SortSpec,
    TimeSeries,
    VisualizationRationale,
    XYEncoding,
)

TITLE_PATH = "protocolSection.identificationModule.briefTitle"
TRIALS_Y = Channel(field="trial_count", type="quantitative", label="Number of trials")
OVERLAP_Y = Channel(field="trial_count", type="quantitative", label="Number of trials (a trial can appear in several bars)")

# Enrollment spans 0 to millions, so bins widen with size (equal-width bins would put
# nearly every trial in the first bin).
ENROLLMENT_BINS: list[tuple[int, int | None]] = [
    (0, 10), (11, 50), (51, 100), (101, 250), (251, 500), (501, 1000), (1001, 5000), (5001, None),
]


def execute(plan: QueryPlan, queries: list[FetchedQuery], requested_chart: VizType | None, max_citations: int) -> ExecutionResult:
    rationale = choose_chart(plan, requested_chart)
    match plan.analysis:
        case Analysis.COUNT:
            return _metric(queries[0], rationale, max_citations)
        case Analysis.NETWORK:
            return _network(plan, queries[0], rationale, max_citations)
        case Analysis.NUMERIC_DISTRIBUTION:
            return _histogram(queries[0], rationale, max_citations)
        case Analysis.COMPARISON:
            return _comparison(plan, queries, rationale, max_citations)
        case _:
            return _categorical(plan, queries[0], rationale, max_citations)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _title_value(row: TrialRow) -> DimValue:
    """Evidence for "this trial matched the search": cite its title."""
    return DimValue(row.nct_id, row.title or row.nct_id, TITLE_PATH, row.title or "")


def _x_channel(dim: Dimension, counted_years: list[str] | None = None) -> Channel:
    spec = DIMENSIONS[dim]
    label = spec.axis_label
    if counted_years:
        label += f" ({min(counted_years, key=int)}–{max(counted_years, key=int)} counted exactly)"
    return Channel(field=spec.field, type=spec.field_type, label=label, label_field=spec.label_field)


def _counted_years(queries: list[FetchedQuery]) -> set[str] | None:
    """Years counted in every exact-count cohort, or None when no cohort used exact counts."""
    sets = [set(q.exact.counts) for q in queries if q.exact is not None]
    return set.intersection(*sets) if sets else None


def _exclusion_notes(excluded: list[tuple[str, int]]) -> list[str]:
    return [f"Not plotted: {n:,} trials {reason}." for reason, n in excluded]


def _row_key(dim: Dimension, bucket: Bucket) -> dict:
    spec = DIMENSIONS[dim]
    fields = {spec.field: int(bucket.key) if dim == Dimension.START_YEAR else bucket.key}
    if spec.label_field:
        fields[spec.label_field] = bucket.label
    return fields


def _sort(dim: Dimension) -> SortSpec:
    spec = DIMENSIONS[dim]
    if spec.order == "count":
        return SortSpec(field="trial_count", order="descending")
    return SortSpec(field=spec.field, order="ascending")


def _exclusions(dim: Dimension, unclassified: int, extra: list[tuple[str, int]] = ()) -> list[Exclusion]:
    out = [Exclusion(reason=f"no {DIMENSIONS[dim].axis_label.lower()} recorded", count=unclassified)] if unclassified else []
    return out + [Exclusion(reason=reason, count=count) for reason, count in extra]


def _mode(queries: list[FetchedQuery]) -> str:
    """How the counts were obtained, across all searches feeding the chart."""
    if any(q.truncated and q.exact is None for q in queries):
        return "sample"
    return "exact_counts" if any(q.exact is not None for q in queries) else "all_records"


def _y_channel(partition: bool, queries: list[FetchedQuery]) -> Channel:
    base = TRIALS_Y if partition else OVERLAP_Y
    if _mode(queries) != "sample":
        return base
    # Counts cover only the fetched records: say so on the axis, where no reader can miss it.
    sampled = [q for q in queries if q.truncated and q.exact is None]
    detail = f"among the first {len(sampled[0].rows):,} of {sampled[0].total_matching:,} matching trials" if len(sampled) == 1 else "sampled: see meta.notes"
    return base.model_copy(update={"label": f"{base.label} ({detail})"})


def _year_range(queries: list[FetchedQuery]) -> tuple[int | None, int | None]:
    los = [q.filters.start_year_min for q in queries if q.filters.start_year_min is not None]
    his = [q.filters.start_year_max for q in queries if q.filters.start_year_max is not None]
    return (min(los) if los else None, max(his) if his else None)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def _categorical(plan: QueryPlan, q: FetchedQuery, rationale: VisualizationRationale, max_citations: int) -> ExecutionResult:
    """distribution, time_trend, geographic: one search, one group-by."""
    dim = plan.group_by
    spec = DIMENSIONS[dim]
    grouping = group_exact(q.exact, dim, plan.top_n) if q.exact else group(q.rows, dim, plan.top_n, _year_range([q]))
    rows = [DataRow(**_row_key(dim, b), trial_count=b.count, **evidence(b.members, max_citations, b.total)) for b in grouping.buckets]
    counted = [b.key for b in grouping.buckets] if dim == Dimension.START_YEAR and q.exact is not None else None
    encoding_args = {"x": _x_channel(dim, counted), "y": _y_channel(spec.partition, [q])}
    what = subject(q.filters)
    if rationale.chosen == VizType.TIME_SERIES:
        span = year_span(q.filters)
        title = f"{capitalize(subject(q.filters, include_years=False))} per year" + (f", {span}" if span else "")
        viz = TimeSeries(title=title, encoding=OptionalSeriesEncoding(**encoding_args), data=rows)
    else:
        viz = BarChart(title=f"{capitalize(what)} by {spec.axis_label.lower()}", encoding=XYEncoding(**encoding_args), data=rows)
    return ExecutionResult(
        visualization=viz,
        interpretation=f"Counted {what} in the registry, grouped by {spec.axis_label.lower()}.",
        coverage=Coverage(
            aggregation_mode=_mode([q]),
            groupby_semantics="partition" if spec.partition else "overlapping",
            bucket_sum=sum(b.count for b in grouping.buckets),
            unclassified_count=grouping.unclassified_count,
            overlap_note=overlap_note(q.rows, dim) if q.exact is None else (None if spec.partition else "A trial can be counted in several bars."),
            excluded=_exclusions(dim, grouping.unclassified_count, grouping.excluded),
            buckets_not_shown=grouping.not_shown,
        ),
        render=RenderHints(
            sort=_sort(dim),
            time_granularity="year" if dim == Dimension.START_YEAR else None,
            units={"y": "trials"},
        ),
        rationale=rationale,
        assumptions=[spec.assumption] if spec.assumption else [],
        notes=_exclusion_notes(grouping.excluded),
    )


def _comparison(plan: QueryPlan, queries: list[FetchedQuery], rationale: VisualizationRationale, max_citations: int) -> ExecutionResult:
    labels = [q.label for q in queries]
    versus = " vs ".join(labels)
    assumptions = ["Each cohort is a separate search; a trial matching more than one cohort is counted in each."]

    if plan.group_by is None:  # "how many trials: A vs B" -> one bar per cohort
        rows = [
            DataRow(cohort=q.label, trial_count=q.total_matching, **evidence(((r, _title_value(r)) for r in q.rows), max_citations, q.total_matching))
            for q in queries
        ]
        viz = BarChart(
            title=f"Number of trials: {versus}",
            encoding=XYEncoding(x=Channel(field="cohort", type="nominal", label="Cohort"), y=TRIALS_Y),
            data=rows,
        )
        return ExecutionResult(
            visualization=viz,
            interpretation=f"Compared the number of matching trials for {versus}.",
            coverage=Coverage(groupby_semantics="partition", bucket_sum=sum(q.total_matching for q in queries)),
            render=RenderHints(units={"y": "trials"}),
            rationale=rationale,
            assumptions=assumptions + ["Bar heights are the registry's total match counts for each search."],
        )

    dim = plan.group_by
    spec = DIMENSIONS[dim]
    per_cohort: list[dict[str, Bucket]] = []
    unclassified = 0
    excluded: list[tuple[str, int]] = []
    exclusion_notes: list[str] = []
    for q in queries:
        if q.exact is not None:
            buckets = {
                k: Bucket(k, bucket_label(dim, k), [(r, v) for r in q.exact.samples.get(k, []) for v in r.get(dim) if v.key == k], total=n)
                for k, n in q.exact.counts.items()
            }
            unclassified += q.exact.unclassified
            excluded += [(f"{q.label}: {reason}", n) for reason, n in q.exact.excluded]
            exclusion_notes += [f"Not plotted for {q.label}: {n:,} trials {reason}." for reason, n in q.exact.excluded]
        else:
            buckets, missing = collect(q.rows, dim)
            unclassified += len(missing)
        per_cohort.append(buckets)
    totals: dict[str, int] = {}
    for buckets in per_cohort:
        for key, b in buckets.items():
            totals[key] = totals.get(key, 0) + b.count
    window_notes: list[str] = []
    if dim == Dimension.START_YEAR:
        lo, hi = _year_range(queries)
        years = fill_years({k: Bucket(k, k) for k in totals}, lo, hi)
        counted = _counted_years(queries)
        if counted is not None:
            # Only plot years every cohort actually counted: a zero must mean zero trials, never "not counted".
            dropped = [y for y in years if y not in counted]
            years = {y: b for y, b in years.items() if y in counted}
            if dropped:
                window_notes.append(
                    f"Only {min(counted)}–{max(counted)} is plotted: those are the years counted exactly for every cohort "
                    f"within the request budget. {len(dropped)} other year(s) in range are not shown (see coverage.excluded)."
                )
        totals = {k: totals.get(k, 0) for k in years}
    keys = order_keys(totals, spec)
    shown = keys[: plan.top_n] if spec.order == "count" else keys
    labels_by_key = {k: b.label for buckets in per_cohort for k, b in buckets.items()}

    rows = []
    for key in shown:
        for q, buckets in zip(queries, per_cohort):
            b = buckets.get(key) or Bucket(key, labels_by_key.get(key, key))
            rows.append(DataRow(**_row_key(dim, b), cohort=q.label, trial_count=b.count, **evidence(b.members, max_citations, b.total)))

    series = Channel(field="cohort", type="nominal", label="Cohort")
    y = _y_channel(spec.partition, queries)
    title = f"{spec.axis_label}: {versus}"
    x = _x_channel(dim, shown if dim == Dimension.START_YEAR and _counted_years(queries) is not None else None)
    if rationale.chosen == VizType.TIME_SERIES:
        viz = TimeSeries(title=f"Trials per year: {versus}", encoding=OptionalSeriesEncoding(x=x, y=y, series=series), data=rows)
    else:
        viz = GroupedBarChart(title=title, encoding=SeriesEncoding(x=x, y=y, series=series), data=rows)
    if spec.assumption:
        assumptions.append(spec.assumption)
    return ExecutionResult(
        visualization=viz,
        interpretation=f"Counted trials for each of {versus}, grouped by {spec.axis_label.lower()}.",
        coverage=Coverage(
            groupby_semantics="partition" if spec.partition else "overlapping",
            bucket_sum=sum(r.model_extra["trial_count"] for r in rows),
            aggregation_mode=_mode(queries),
            unclassified_count=unclassified,
            overlap_note=None if spec.partition else f"Within a cohort, a trial can appear under several {spec.axis_label.lower()} values.",
            excluded=_exclusions(dim, unclassified, excluded),
            buckets_not_shown=len(keys) - len(shown),
        ),
        render=RenderHints(
            sort=_sort(dim),
            time_granularity="year" if dim == Dimension.START_YEAR else None,
            units={"y": "trials"},
            grouping="cohort",
        ),
        rationale=rationale,
        assumptions=assumptions,
        notes=window_notes + exclusion_notes,
    )


def _histogram(q: FetchedQuery, rationale: VisualizationRationale, max_citations: int) -> ExecutionResult:
    with_enrollment = [r for r in q.rows if r.enrollment is not None]
    bins: list[list[tuple[TrialRow, DimValue]]] = [[] for _ in ENROLLMENT_BINS]
    for r in with_enrollment:
        idx = next(i for i, (lo, hi) in enumerate(ENROLLMENT_BINS) if r.enrollment >= lo and (hi is None or r.enrollment <= hi))
        bins[idx].append((r, DimValue(str(r.enrollment), str(r.enrollment), ENROLLMENT_PATH, str(r.enrollment))))
    last = max((i for i, members in enumerate(bins) if members), default=-1)

    rows = []
    for (lo, hi), members in list(zip(ENROLLMENT_BINS, bins))[: last + 1]:
        end = hi if hi is not None else max(r.enrollment for r, _ in members)
        label = f"{lo:,}–{hi:,}" if hi is not None else f"{lo:,}+"
        rows.append(DataRow(bin_start=lo, bin_end=end, bin_label=label, trial_count=len(members), **evidence(members, max_citations)))

    what = subject(q.filters)
    missing = len(q.rows) - len(with_enrollment)
    viz = Histogram(
        title=f"{capitalize(what)}: enrollment sizes",
        encoding=HistogramEncoding(
            x=Channel(field="bin_start", type="quantitative", label="Enrollment (participants)", label_field="bin_label"),
            x2=Channel(field="bin_end", type="quantitative", label="Enrollment (participants)"),
            y=_y_channel(True, [q]),
        ),
        data=rows,
    )
    return ExecutionResult(
        visualization=viz,
        interpretation=f"Binned {what} by enrollment size.",
        coverage=Coverage(
            aggregation_mode=_mode([q]),
            groupby_semantics="partition",
            bucket_sum=len(with_enrollment),
            unclassified_count=missing,
            excluded=[Exclusion(reason="no enrollment count recorded", count=missing)] if missing else [],
        ),
        render=RenderHints(sort=SortSpec(field="bin_start", order="ascending"), units={"x": "participants", "y": "trials"}),
        rationale=rationale,
        assumptions=[
            "Enrollment is the registry's enrollment count: actual for completed trials, anticipated otherwise.",
            "Bins widen with size because enrollment spans several orders of magnitude.",
        ],
    )


def _network(plan: QueryPlan, q: FetchedQuery, rationale: VisualizationRationale, max_citations: int) -> ExecutionResult:
    source, target = plan.network.source, plan.network.target
    result = build_network(q.rows, source, target, plan.top_n, max_citations)
    src_label, tgt_label = DIMENSIONS[source].axis_label, DIMENSIONS[target].axis_label
    what = subject(q.filters)
    if source == target:
        title = f"{src_label} co-occurrence network: {what}"
        meaning = f"Two {src_label.lower()}s are linked when one trial lists both; edge weight is the number of such trials."
    else:
        title = f"{src_label}–{tgt_label.lower()} network: {what}"
        meaning = f"A {src_label.lower()} is linked to a {tgt_label.lower()} when they appear in the same trial; edge weight is the number of such trials."
    assumptions = [meaning]
    if Dimension.INTERVENTION in (source, target):
        assumptions.append(
            "Only drug, biological and combination-product interventions are nodes; placebo and standard-of-care "
            "comparators are left out because they would link to everything."
        )
    if source == target == Dimension.INTERVENTION:
        assumptions.append("Listing two drugs in one trial does not necessarily mean they were given together.")
    for dim in {source, target}:
        if dim != Dimension.INTERVENTION and DIMENSIONS[dim].assumption:
            assumptions.append(DIMENSIONS[dim].assumption)
    notes = [] if result.data.edges else ["No two entities appeared in the same trial, so the network has no edges."]
    return ExecutionResult(
        visualization=NetworkGraph(title=capitalize(title), data=result.data),
        interpretation=(
            f"Built a network of {src_label.lower()}s that appear together in {what}."
            if source == target
            else f"Built a network linking each {src_label.lower()} to the {tgt_label.lower()}s in its {what}."
        ),
        coverage=Coverage(aggregation_mode=_mode([q])),  # a network from a sample shows only the sampled trials' links
        render=RenderHints(units={"node.size": "trials", "edge.weight": "trials"}, pruning=result.pruning),
        rationale=rationale,
        assumptions=assumptions,
        notes=notes,
    )


def _metric(q: FetchedQuery, rationale: VisualizationRationale, max_citations: int) -> ExecutionResult:
    what = subject(q.filters)
    row = DataRow(trial_count=q.total_matching, **evidence(((r, _title_value(r)) for r in q.rows), max_citations, q.total_matching))
    return ExecutionResult(
        visualization=MetricViz(
            title=f"Number of {what}",
            encoding=MetricEncoding(value=Channel(field="trial_count", type="quantitative", label="Number of trials")),
            data=[row],
        ),
        interpretation=f"Counted {what}. A single number answers this, so no chart is needed.",
        coverage=None,
        render=RenderHints(units={"value": "trials"}),
        rationale=rationale,
        assumptions=["The value is the registry's own total match count for this search."],
    )
