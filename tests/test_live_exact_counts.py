"""Opt-in live checks of exact-count mode against the real registry (pytest -m live).

The stub tests prove the logic; these prove it matches ClinicalTrials.gov. Each plotted count
is re-checked with an independent search built by the normal query builder. Uses ~40 registry
requests, so it is paced and kept out of the default run.
"""

import asyncio

import pytest

from app.ctgov.client import CTGovClient
from app.ctgov.query_builder import build_params
from app.graph import build_graph
from app.schemas.enums import Phase
from app.schemas.plan import Filters
from app.schemas.request import VisualizeRequest
from tests.stubs import ScriptedLLM, draft

pytestmark = pytest.mark.live


async def _total(client: CTGovClient, filters: Filters) -> int:
    return (await client.search(build_params(filters), max_records=1)).total_count


def _run(llm, request):
    async def go():
        client = CTGovClient()
        try:
            return (await build_graph(llm, client).ainvoke({"request": request}))["response"]
        finally:
            await client.aclose()

    return asyncio.run(go())


def test_large_cohort_phase_buckets_reconcile_with_the_registry_total():
    llm = ScriptedLLM(draft(analysis="distribution", group_by="phase", filters={"condition": "cancer"}))
    resp = _run(llm, VisualizeRequest(query="cancer trials by phase"))
    assert resp.meta.coverage.aggregation_mode == "exact_counts"
    total = resp.meta.queries[0].total_matching
    assert total > 100_000
    assert resp.meta.coverage.bucket_sum + resp.meta.coverage.unclassified_count == total


def test_comparison_years_match_independent_counts_and_are_never_fake_zeros():
    # The reviewer's failing case: lung vs breast cancer per year since 2018.
    llm = ScriptedLLM(
        draft(
            analysis="comparison",
            group_by="start_year",
            filters={"start_year_min": 2018},
            compare=[{"label": "Lung", "filters": {"condition": "lung cancer"}}, {"label": "Breast", "filters": {"condition": "breast cancer"}}],
        )
    )
    resp = _run(llm, VisualizeRequest(query="lung vs breast cancer trials per year since 2018"))
    rows = resp.visualization.data
    assert rows and all(r.model_extra["trial_count"] > 0 for r in rows)

    async def check():
        client = CTGovClient()
        try:
            for r in rows[::3]:  # a spread of (year, cohort) cells, to stay polite
                year = r.model_extra["start_year"]
                condition = "lung cancer" if r.model_extra["cohort"] == "Lung" else "breast cancer"
                expected = await _total(client, Filters(condition=condition, start_year_min=year, start_year_max=year))
                assert r.model_extra["trial_count"] == expected, (year, condition)
        finally:
            await client.aclose()

    asyncio.run(check())


def test_phase_filter_count_matches_the_registry():
    llm = ScriptedLLM(draft(analysis="count", filters={"condition": "pancreatic cancer", "phases": ["PHASE3"], "statuses": ["RECRUITING"]}))
    resp = _run(llm, VisualizeRequest(query="how many recruiting phase 3 pancreatic cancer trials?"))

    async def expected():
        client = CTGovClient()
        try:
            return await _total(client, Filters(condition="pancreatic cancer", phases=[Phase.PHASE3], statuses=["RECRUITING"]))
        finally:
            await client.aclose()

    assert resp.visualization.data[0].model_extra["trial_count"] == asyncio.run(expected())
