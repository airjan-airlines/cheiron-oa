from app.planner.validator import finalize, searches
from app.schemas.enums import Analysis, Dimension, Metric, Phase
from app.schemas.request import VisualizeRequest
from tests.stubs import draft

Q = VisualizeRequest(query="question")


def test_unsupported_question_becomes_an_unsupported_plan():
    check = finalize(draft(is_about_clinical_trials=False, unsupported_reason="Weather is not in the registry."), Q)
    assert check.plan.supported is False
    assert check.plan.unsupported_reason == "Weather is not in the registry."


def test_structured_fields_override_the_llm_and_the_override_is_noted():
    req = VisualizeRequest(query="trials for this drug by phase", drug_name="Pembrolizumab", trial_phases=["PHASE3"])
    check = finalize(draft(analysis="distribution", group_by="phase", filters={"intervention": "nivolumab"}), req)
    assert check.plan.filters.intervention == "Pembrolizumab"
    assert check.plan.filters.phases == [Phase.PHASE3]
    assert any("drug_name" in n and "nivolumab" in n for n in check.notes)


def test_case_only_differences_are_not_reported_as_overrides():
    req = VisualizeRequest(query="q", drug_name="Pembrolizumab")
    check = finalize(draft(analysis="distribution", group_by="phase", filters={"intervention": "pembrolizumab"}), req)
    assert check.notes == []


def test_time_trend_is_normalized_to_group_by_start_year():
    check = finalize(draft(analysis="time_trend", group_by="phase"), Q)
    assert check.plan.group_by == Dimension.START_YEAR
    assert check.notes


def test_numeric_distribution_forces_enrollment_metric():
    check = finalize(draft(analysis="numeric_distribution"), Q)
    assert check.plan.metric == Metric.ENROLLMENT


def test_distribution_without_group_by_is_an_error_for_repair():
    check = finalize(draft(analysis="distribution"), Q)
    assert check.plan is None
    assert check.errors == ["distribution needs group_by (which category to split trials by)."]


def test_comparison_needs_two_distinct_non_empty_cohorts():
    one = finalize(draft(analysis="comparison", group_by="phase", compare=[{"label": "A", "filters": {"intervention": "a"}}]), Q)
    assert any("2-4 cohorts" in e for e in one.errors)
    empty = finalize(draft(analysis="comparison", compare=[{"label": "A", "filters": {"intervention": "a"}}, {"label": "B", "filters": {}}]), Q)
    assert any("no filters" in e for e in empty.errors)


def test_network_dimensions_are_restricted():
    check = finalize(draft(analysis="network", network={"source": "phase", "target": "intervention"}), Q)
    assert any("network dimensions" in e for e in check.errors)


def test_schema_violations_become_readable_errors():
    check = finalize(draft(analysis="distribution", group_by="phase", filters={"start_year_min": 2024, "start_year_max": 2015}), Q)
    assert check.plan is None
    assert any("start_year_min must be <= start_year_max" in e for e in check.errors)


def test_llm_top_n_used_unless_caller_sets_it():
    assert finalize(draft(analysis="geographic", top_n=10), Q).plan.top_n == 10
    req = VisualizeRequest(query="q", top_n=5)
    assert finalize(draft(analysis="geographic", top_n=10), req).plan.top_n == 5


def test_comparison_searches_layer_cohort_filters_on_shared_ones():
    check = finalize(
        draft(
            analysis="comparison",
            group_by="phase",
            filters={"condition": "obesity"},
            compare=[{"label": "Semaglutide", "filters": {"intervention": "semaglutide"}}, {"label": "Tirzepatide", "filters": {"intervention": "tirzepatide"}}],
        ),
        Q,
    )
    assert check.plan.analysis == Analysis.COMPARISON
    specs = searches(check.plan)
    assert [(label, f.condition, f.intervention) for label, f in specs] == [
        ("Semaglutide", "obesity", "semaglutide"),
        ("Tirzepatide", "obesity", "tirzepatide"),
    ]
