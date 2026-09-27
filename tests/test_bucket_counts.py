"""Exact per-bucket counts for cohorts above the record cap (offline, stubbed registry)."""

import asyncio

from app.ctgov.bucket_counts import PHASE_EXPRESSIONS, REMAINDER_KEY, exact_counts
from app.ctgov.client import SearchResult
from app.engine import response_builder
from app.engine.executor import execute
from app.engine.types import ExactBuckets
from app.schemas.enums import Dimension as D
from app.schemas.plan import QueryPlan
from tests.helpers import fetched, trial

# Real breast-cancer numbers (16,853 trials), measured live on 2026-09-26.
BREAST_PHASES = {
    "EARLY_PHASE1": 232, "PHASE1": 1631, "PHASE1/PHASE2": 877, "PHASE2": 3557, "PHASE2/PHASE3": 147,
    "PHASE3": 1409, "PHASE4": 344, "NA": 4945, "MISSING": 3711,
}


class StubCounts:
    """Answers a count query by finding which known clause was ANDed onto the search."""

    def __init__(self, by_clause: dict[str, int], bounds: tuple[int, int] = (2015, 2020)):
        self.by_clause = by_clause
        self.bounds = bounds
        self.calls: list[dict] = []

    async def search(self, params, max_records):
        self.calls.append(params)
        adv = params.get("filter.advanced", "")
        if "sort" in params:  # earliest / latest start year probe
            year = self.bounds[0] if params["sort"].endswith("asc") else self.bounds[1]
            return SearchResult([trial(1, statusModule={"startDateStruct": {"date": f"{year}-01"}})], 1)
        for clause, n in self.by_clause.items():
            if adv.endswith(f"({clause})") or adv == clause:
                return SearchResult([trial(i, designModule={"phases": ["PHASE3"]}) for i in range(min(n, max_records))], n)
        return SearchResult([], 0)


def run(coro):
    return asyncio.run(coro)


def test_phase_buckets_reconcile_exactly_with_the_total():
    stub = StubCounts({PHASE_EXPRESSIONS[k]: n for k, n in BREAST_PHASES.items()})
    ec = run(exact_counts(stub, {"query.cond": "breast cancer"}, D.PHASE, 16853, set(), 2, (None, None), 24))
    assert sum(ec.counts.values()) == 16853
    assert ec.counts["PHASE1/PHASE2"] == 877
    assert all(len(v) <= 2 for v in ec.samples.values())  # only example trials are fetched
    assert ec.requests == 9


def test_bucket_clause_is_anded_onto_existing_filters():
    stub = StubCounts({PHASE_EXPRESSIONS["PHASE3"]: 5})
    run(exact_counts(stub, {"filter.advanced": "AREA[StartDate]RANGE[2015-01-01,MAX]"}, D.PHASE, 5, set(), 1, (2015, None), 24))
    assert all(c["filter.advanced"].startswith("(AREA[StartDate]RANGE[2015-01-01,MAX]) AND (") for c in stub.calls)


def test_partition_enum_counts_seen_values_and_an_exact_remainder():
    stub = StubCounts({"AREA[OverallStatus]RECRUITING": 700, "AREA[OverallStatus]COMPLETED": 250})
    ec = run(exact_counts(stub, {}, D.STATUS, 1000, {"RECRUITING", "COMPLETED"}, 1, (None, None), 24))
    assert ec.counts == {"RECRUITING": 700, "COMPLETED": 250, REMAINDER_KEY: 50}
    assert len(stub.calls) == 2  # unseen statuses cost no requests


def test_year_window_is_anchored_at_the_current_year_not_future_anticipated_starts():
    # Reviewer case: "cancer trials per year" counted 2015-2030, spending 4 of 16 slots on future
    # anticipated starts and pushing 2011-2014 into "excluded".
    by_clause = {f"AREA[StartDate]RANGE[{y}-01-01,{y}-12-31]": 10 for y in range(1960, 2031)}
    by_clause["AREA[StartDate]MISSING"] = 3
    by_clause["AREA[StartDate]RANGE[2027-01-01,MAX]"] = 40  # anticipated future starts
    by_clause["AREA[StartDate]RANGE[MIN,2006-12-31]"] = 470  # before the 20-year window
    stub = StubCounts(by_clause, bounds=(1960, 2030))
    ec = run(exact_counts(stub, {}, D.START_YEAR, 999, set(), 1, (None, None), 24, current_year=2026))
    years = sorted(int(k) for k in ec.counts)
    assert (years[0], years[-1], len(years)) == (2007, 2026, 20)
    assert ec.unclassified == 3
    assert ec.excluded == [
        ("planned to start after 2026 (anticipated dates)", 40),
        ("started before 2007 (outside the 20-year window counted exactly)", 470),
    ]


def test_dimension_that_cannot_fit_the_budget_falls_back_to_sampling():
    assert run(exact_counts(StubCounts({}), {}, D.PHASE, 10, set(), 1, (None, None), 5)) is None
    assert run(exact_counts(StubCounts({}), {}, D.COUNTRY, 10, set(), 1, (None, None), 24)) is None


def _executed(q, plan):
    result = execute(plan, [q], None, 2)
    return response_builder.ok(plan, [q], result, [], None).model_dump(mode="json")


def test_exact_mode_plots_registry_totals_and_says_evidence_is_partial():
    q = fetched("tirzepatide", limit=100)  # pretend the cap was hit at 100 of 290
    q.exact = ExactBuckets(counts={"PHASE3": 50, "PHASE2": 67}, samples={"PHASE3": [r for r in q.rows if [v.key for v in r.get(D.PHASE)] == ["PHASE3"]][:2]})
    out = _executed(q, QueryPlan(analysis="distribution", group_by="phase"))
    rows = {r["phase"]: r for r in out["visualization"]["data"]}
    assert rows["PHASE3"]["trial_count"] == rows["PHASE3"]["citation_count"] == 50
    assert rows["PHASE3"]["supporting_nct_ids_complete"] is False
    assert out["meta"]["coverage"]["aggregation_mode"] == "exact_counts"
    assert out["visualization"]["encoding"]["y"]["label"] == "Number of trials"
    assert any("exact total" in n for n in out["meta"]["notes"])


def test_sample_mode_is_labelled_on_the_axis_itself():
    q = fetched("tirzepatide", limit=100)
    out = _executed(q, QueryPlan(analysis="geographic", group_by="country"))
    assert out["meta"]["coverage"]["aggregation_mode"] == "sample"
    assert "among the first 100 of 290 matching trials" in out["visualization"]["encoding"]["y"]["label"]


def test_remainder_never_collides_with_a_real_value_called_other():
    # Sponsor class has a genuine "OTHER" (academic/hospital) value; the remainder must not overwrite it.
    stub = StubCounts({"AREA[LeadSponsorClass]OTHER": 9000, "AREA[LeadSponsorClass]INDUSTRY": 4000})
    ec = run(exact_counts(stub, {}, D.SPONSOR_CLASS, 13500, {"OTHER", "INDUSTRY"}, 1, (None, None), 24))
    assert ec.counts["OTHER"] == 9000
    assert ec.counts[REMAINDER_KEY] == 500
    assert sum(ec.counts.values()) == 13500


def test_comparison_never_plots_uncounted_years_as_zero():
    # Reviewer repro: lung vs breast cancer per year since 2018; the budget counted only 2021-2026,
    # and 2018-2020 were zero-filled and marked complete. Now they're not plotted, and that's stated.
    lung, breast = fetched("tirzepatide", "Lung", limit=100), fetched("kras_nsclc", "Breast", limit=100)
    for q, base in ((lung, 700), (breast, 500)):
        q.filters = q.filters.model_copy(update={"start_year_min": 2018})
        q.exact = ExactBuckets(
            counts={str(y): base + y - 2021 for y in range(2021, 2027)},
            samples={},
            excluded=[("started before 2021 (outside the 6-year window counted exactly)", 2100)],
        )
    plan = QueryPlan(analysis="comparison", group_by="start_year", compare=[{"label": "Lung", "filters": {"condition": "a"}}, {"label": "Breast", "filters": {"condition": "b"}}])
    result = execute(plan, [lung, breast], None, 1)
    out = response_builder.ok(plan, [lung, breast], result, [], None).model_dump(mode="json")
    years = sorted({r["start_year"] for r in out["visualization"]["data"]})
    assert years == list(range(2021, 2027))
    assert all(r["trial_count"] > 0 for r in out["visualization"]["data"])
    assert out["visualization"]["encoding"]["x"]["label"] == "Start year (2021–2026 counted exactly)"
    assert any(n.startswith("Only 2021–2026 is plotted") for n in out["meta"]["notes"])
    assert "Not plotted for Lung: 2,100 trials started before 2021 (outside the 6-year window counted exactly)." in out["meta"]["notes"]


def test_cohorts_count_the_union_of_values_so_missing_bars_are_true_zeros():
    # Reviewer repro: breast-cancer FED showed 0 because FED only appeared in the lung-cancer sample.
    from app.graph import _add_exact_counts

    a, b = fetched("kras_nsclc", "A", limit=100), fetched("tirzepatide", "B", limit=100)
    a_classes = {v.key for r in a.rows for v in r.get(D.SPONSOR_CLASS)}
    b_classes = {v.key for r in b.rows for v in r.get(D.SPONSOR_CLASS)}
    only_b = b_classes - a_classes
    assert only_b  # a class seen in just one cohort's sample
    stub = StubCounts({f"AREA[LeadSponsorClass]{c}": 3 for c in a_classes | b_classes})
    plan = QueryPlan(analysis="comparison", group_by="sponsor_class", compare=[{"label": "A", "filters": {"condition": "a"}}, {"label": "B", "filters": {"condition": "b"}}])
    run(_add_exact_counts(stub, plan, [a, b], 1, 5000))
    assert a.exact is not None and all(a.exact.counts[c] == 3 for c in only_b)  # counted for A too
