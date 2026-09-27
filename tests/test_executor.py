"""End-to-end engine behavior on recorded registry data (no LLM, no network)."""

import json

from app.engine import response_builder
from app.engine.executor import execute
from app.schemas.plan import QueryPlan
from app.schemas.response import VisualizeResponse
from tests.helpers import fetched

COHORTS = [{"label": "Tirzepatide", "filters": {}}, {"label": "KRAS NSCLC", "filters": {}}]


def _ok(plan, queries, chart=None, cites=3) -> dict:
    result = execute(plan, queries, chart, cites)
    resp = response_builder.ok(plan, queries, result, [], "2026-09-25T09:00:04")
    # Round-trip through the public schema, exactly as the API would serialize it.
    return VisualizeResponse.model_validate(resp.model_dump(mode="json")).model_dump(mode="json")


def test_phase_distribution_reconciles_with_the_api_total():
    q = fetched("korea_gastric_recruiting")
    out = _ok(QueryPlan(analysis="distribution", group_by="phase"), [q])
    viz, meta = out["visualization"], out["meta"]
    assert viz["type"] == "bar_chart"
    assert viz["encoding"]["x"] == {"field": "phase", "type": "ordinal", "label": "Trial phase", "label_field": "phase_label"}
    assert sum(r["trial_count"] for r in viz["data"]) == q.total_matching == meta["coverage"]["bucket_sum"]
    assert meta["coverage"]["groupby_semantics"] == "partition"
    assert meta["queries"][0]["truncated"] is False
    assert all(r["citation_count"] == r["trial_count"] == len(r["supporting_nct_ids"]) for r in viz["data"])


def test_every_citation_quotes_the_value_that_placed_the_trial_in_its_bucket():
    out = _ok(QueryPlan(analysis="distribution", group_by="phase"), [fetched("korea_gastric_recruiting")])
    for r in out["visualization"]["data"]:
        for c in r["citations"]:
            if r["phase"] == "MISSING":
                assert c["excerpt"] == "(field not present in record)"
            else:
                assert "/".join(json.loads(c["excerpt"])) == r["phase"]


def test_time_trend_is_a_gapless_yearly_series():
    out = _ok(QueryPlan(analysis="time_trend", group_by="start_year"), [fetched("tirzepatide")])
    years = [r["start_year"] for r in out["visualization"]["data"]]
    assert out["visualization"]["type"] == "time_series"
    assert years == list(range(years[0], years[-1] + 1))
    assert out["meta"]["render"]["time_granularity"] == "year"


def test_geographic_counts_flag_overlap():
    out = _ok(QueryPlan(analysis="geographic", group_by="country", top_n=5), [fetched("tirzepatide")])
    cov = out["meta"]["coverage"]
    assert cov["groupby_semantics"] == "overlapping" and cov["overlap_note"]
    assert cov["buckets_not_shown"] > 0
    assert out["meta"]["render"]["sort"] == {"field": "trial_count", "order": "descending"}


def test_comparison_rows_are_long_format_with_zeros_filled():
    plan = QueryPlan(analysis="comparison", group_by="phase", compare=COHORTS)
    queries = [fetched("tirzepatide", "Tirzepatide"), fetched("kras_nsclc", "KRAS NSCLC")]
    out = _ok(plan, queries)
    viz = out["visualization"]
    assert viz["type"] == "grouped_bar_chart"
    assert viz["encoding"]["series"]["field"] == "cohort"
    phases = {r["phase"] for r in viz["data"]}
    assert len(viz["data"]) == 2 * len(phases)  # every (phase, cohort) pair present
    for q in queries:
        assert sum(r["trial_count"] for r in viz["data"] if r["cohort"] == q.label) == q.total_matching
    assert out["meta"]["render"]["grouping"] == "cohort"
    assert len(out["meta"]["queries"]) == 2


def test_histogram_bins_are_contiguous_and_reconcile():
    q = fetched("kras_nsclc")
    out = _ok(QueryPlan(analysis="numeric_distribution", metric="enrollment"), [q])
    rows = out["visualization"]["data"]
    assert all(rows[i]["bin_end"] + 1 == rows[i + 1]["bin_start"] for i in range(len(rows) - 1))
    missing = out["meta"]["coverage"]["unclassified_count"]
    assert sum(r["trial_count"] for r in rows) + missing == len(q.rows)


def test_network_output_validates_against_the_public_schema():
    plan = QueryPlan(analysis="network", network={"source": "sponsor", "target": "intervention"}, top_n=6)
    out = _ok(plan, [fetched("kras_nsclc")])
    assert out["visualization"]["type"] == "network_graph"
    assert out["visualization"]["data"]["edges"]
    assert out["meta"]["render"]["pruning"]["top_n"] == 6


def test_count_question_returns_a_metric_from_the_api_total():
    q = fetched("korea_gastric_recruiting", limit=10)  # simulate the record cap
    out = _ok(QueryPlan(analysis="count"), [q])
    assert out["visualization"]["type"] == "metric"
    assert out["visualization"]["data"][0]["trial_count"] == q.total_matching  # exact, not the 10 analyzed


def test_truncation_is_disclosed():
    q = fetched("tirzepatide", limit=100)
    out = _ok(QueryPlan(analysis="distribution", group_by="phase"), [q])
    assert out["meta"]["queries"][0]["truncated"] is True
    assert any("record cap" in n for n in out["meta"]["notes"])
