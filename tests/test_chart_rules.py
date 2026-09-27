import pytest

from app.engine.chart_rules import allowed_charts, choose_chart
from app.schemas.enums import VizType as V
from app.schemas.plan import QueryPlan

COHORTS = [{"label": "A", "filters": {"intervention": "a"}}, {"label": "B", "filters": {"intervention": "b"}}]


@pytest.mark.parametrize(
    "plan, expected_default",
    [
        (QueryPlan(analysis="time_trend", group_by="start_year"), V.TIME_SERIES),
        (QueryPlan(analysis="distribution", group_by="phase"), V.BAR_CHART),
        (QueryPlan(analysis="geographic", group_by="country"), V.BAR_CHART),
        (QueryPlan(analysis="comparison", group_by="phase", compare=COHORTS), V.GROUPED_BAR_CHART),
        (QueryPlan(analysis="comparison", group_by="start_year", compare=COHORTS), V.TIME_SERIES),
        (QueryPlan(analysis="comparison", compare=COHORTS), V.BAR_CHART),
        (QueryPlan(analysis="numeric_distribution", metric="enrollment"), V.HISTOGRAM),
        (QueryPlan(analysis="network", network={"source": "sponsor", "target": "intervention"}), V.NETWORK_GRAPH),
        (QueryPlan(analysis="count"), V.METRIC),
    ],
)
def test_every_analysis_has_a_default_chart(plan, expected_default):
    assert allowed_charts(plan)[0] == expected_default


def test_compatible_llm_suggestion_is_used():
    plan = QueryPlan(analysis="time_trend", group_by="start_year", chart_type_suggestion="bar_chart", chart_rationale="Few years")
    r = choose_chart(plan, None)
    assert (r.chosen, r.source, r.reason) == (V.BAR_CHART, "llm", "Few years")


def test_incompatible_llm_suggestion_is_overridden_and_reported():
    plan = QueryPlan(analysis="distribution", group_by="phase", chart_type_suggestion="network_graph")
    r = choose_chart(plan, None)
    assert (r.chosen, r.source, r.suggested) == (V.BAR_CHART, "rule_default", V.NETWORK_GRAPH)
    assert "does not fit" in r.reason


def test_request_chart_type_beats_llm_suggestion():
    plan = QueryPlan(analysis="time_trend", group_by="start_year", chart_type_suggestion="time_series")
    assert choose_chart(plan, V.BAR_CHART).source == "request"


def test_incompatible_request_is_overridden():
    r = choose_chart(QueryPlan(analysis="count"), V.BAR_CHART)
    assert (r.chosen, r.suggested) == (V.METRIC, V.BAR_CHART)
