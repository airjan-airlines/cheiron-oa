"""The full pipeline with a scripted LLM and recorded registry data."""

import asyncio
import json
import re

from app.graph import build_graph
from app.schemas.request import VisualizeRequest
from app.schemas.response import VisualizeResponse
from tests.conftest import load_fixture
from tests.stubs import FixtureRegistry, ScriptedLLM, draft

KOREA = {"condition": "gastric cancer", "location": "South Korea", "statuses": ["RECRUITING"]}


def run(llm, registry, **request):
    graph = build_graph(llm, registry)
    final = asyncio.run(graph.ainvoke({"request": VisualizeRequest(**request)}))
    return final["response"]


def test_happy_path_produces_a_cited_chart():
    llm = ScriptedLLM(draft(analysis="distribution", group_by="phase", filters=KOREA, chart_type_suggestion="bar_chart"))
    resp = run(llm, FixtureRegistry("korea_gastric_recruiting"), query="한국에서 모집 중인 위암 임상시험은 단계별로 어떻게 분포되어 있나요?")
    assert resp.status == "ok"
    assert resp.visualization.type == "bar_chart"
    assert sum(r.model_extra["trial_count"] for r in resp.visualization.data) == 74
    assert resp.meta.visualization_rationale.source == "llm"
    assert len(llm.calls) == 1


def test_korean_location_alias_is_normalized_before_searching():
    llm = ScriptedLLM(draft(analysis="distribution", group_by="phase", filters={**KOREA, "location": "Korea, Republic of"}))
    registry = FixtureRegistry("korea_gastric_recruiting")
    resp = run(llm, registry, query="q")
    assert registry.searches[0]["query.locn"] == "South Korea"
    assert resp.status == "ok"
    assert any("normalized" in n for n in resp.meta.notes)


def test_invalid_draft_is_repaired_once_with_the_validation_errors():
    llm = ScriptedLLM(
        draft(analysis="distribution", filters=KOREA),  # missing group_by
        draft(analysis="distribution", group_by="phase", filters=KOREA),
    )
    resp = run(llm, FixtureRegistry("korea_gastric_recruiting"), query="q")
    assert resp.status == "ok"
    assert len(llm.calls) == 2
    repair_turn = llm.calls[1][-1]["content"]
    assert "failed validation" in repair_turn and "group_by" in repair_turn


def test_still_invalid_after_repair_returns_an_error_not_a_guess():
    llm = ScriptedLLM(draft(analysis="distribution"), draft(analysis="distribution"))
    registry = FixtureRegistry()
    resp = run(llm, registry, query="q")
    assert resp.status == "error" and resp.error.code == "invalid_plan"
    assert resp.visualization is None
    assert registry.searches == []  # never fetched


def test_unsupported_question_stops_before_fetching():
    registry = FixtureRegistry()
    resp = run(ScriptedLLM(draft(is_about_clinical_trials=False, unsupported_reason="Not about trials.")), registry, query="Will it rain?")
    assert resp.status == "unsupported"
    assert registry.searches == []


def test_zero_matches_returns_no_data_with_the_search_shown():
    llm = ScriptedLLM(draft(analysis="distribution", group_by="phase", filters={"intervention": "pembrulizumab"}))
    resp = run(llm, FixtureRegistry(), query="q")
    assert resp.status == "no_data"
    assert resp.meta.queries[0].api_params["query.intr"] == "pembrulizumab"


def test_comparison_runs_one_search_per_cohort():
    llm = ScriptedLLM(
        draft(
            analysis="comparison",
            group_by="phase",
            compare=[{"label": "Tirzepatide", "filters": {"intervention": "tirzepatide"}}, {"label": "KRAS NSCLC", "filters": {"condition": "KRAS non-small cell lung cancer"}}],
        )
    )
    registry = FixtureRegistry("tirzepatide", "kras_nsclc")
    resp = run(llm, registry, query="q")
    assert resp.status == "ok" and resp.visualization.type == "grouped_bar_chart"
    assert len(registry.searches) == 2


def test_no_trial_data_reaches_the_llm():
    """D18: the model plans from the question alone; it never sees registry records, so it cannot invent from them."""
    llm = ScriptedLLM(
        draft(analysis="distribution", filters=KOREA),  # forces a repair turn too
        draft(analysis="distribution", group_by="phase", filters=KOREA),
    )
    resp = run(llm, FixtureRegistry("korea_gastric_recruiting"), query="How are recruiting gastric cancer trials in Korea split by phase?")
    assert resp.status == "ok"
    sent = json.dumps(llm.calls, ensure_ascii=False)
    assert not re.search(r"NCT\d{8}", sent)
    for study in load_fixture("korea_gastric_recruiting")["studies"]:
        title = study["protocolSection"]["identificationModule"]["briefTitle"]
        assert title not in sent
    assert "74" not in sent  # not even the match count


def test_caller_contradiction_fails_fast_without_repair_or_fetch():
    llm = ScriptedLLM(
        draft(
            analysis="comparison",
            group_by="phase",
            compare=[{"label": "Semaglutide", "filters": {"intervention": "semaglutide"}}, {"label": "Tirzepatide", "filters": {"intervention": "tirzepatide"}}],
        )
    )
    registry = FixtureRegistry()
    resp = run(llm, registry, query="Compare semaglutide vs tirzepatide", drug_name="pembrolizumab")
    assert resp.status == "error" and resp.error.code == "conflicting_constraints"
    assert "drug_name" in resp.error.message
    assert len(llm.calls) == 1 and registry.searches == []


def test_count_with_zero_matches_is_a_metric_of_zero():
    llm = ScriptedLLM(draft(analysis="count", filters={"intervention": "pembrolizumab", "location": "Iceland"}))
    resp = run(llm, FixtureRegistry(), query="How many?")
    assert resp.status == "ok" and resp.visualization.type == "metric"
    assert resp.visualization.data[0].model_extra["trial_count"] == 0
