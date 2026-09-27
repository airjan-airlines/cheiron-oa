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


def test_long_year_ranges_count_the_latest_window_and_report_the_rest():
    by_clause = {f"AREA[StartDate]RANGE[{y}-01-01,{y}-12-31]": 10 for y in range(1960, 2031)}
    by_clause["AREA[StartDate]MISSING"] = 3
    by_clause["AREA[StartDate]RANGE[MIN,2010-12-31]"] = 510  # a 24-request budget leaves a 2011-2030 window
    stub = StubCounts(by_clause, bounds=(1960, 2030))
    ec = run(exact_counts(stub, {}, D.START_YEAR, 999, set(), 1, (None, None), 24))
    years = sorted(int(k) for k in ec.counts)
    assert years[-1] == 2030 and len(years) == 24 - 2 - 2
    assert ec.unclassified == 3
    assert years[0] == 2011
    assert ec.excluded == [("started before 2011 (outside the 20-year window counted exactly)", 510)]


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
