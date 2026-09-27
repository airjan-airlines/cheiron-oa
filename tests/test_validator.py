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


# --- comparisons vs structured fields (cohorts narrow shared filters, never contradict the caller) ---

SEMA = {"label": "Semaglutide", "filters": {"intervention": "semaglutide"}}
TIRZ = {"label": "Tirzepatide", "filters": {"intervention": "tirzepatide"}}


def _compare(*cohorts, **draft_fields):
    return draft(analysis="comparison", group_by="phase", compare=list(cohorts), **draft_fields)


def test_structured_field_contradicting_every_cohort_is_a_conflict_not_silently_dropped():
    check = finalize(_compare(SEMA, TIRZ), VisualizeRequest(query="q", drug_name="pembrolizumab"))
    assert check.plan is None and check.errors == []
    assert len(check.conflicts) == 2 and "drug_name" in check.conflicts[0]


def test_structured_field_that_is_one_side_of_the_comparison_is_allowed_and_noted():
    check = finalize(_compare(SEMA, TIRZ), VisualizeRequest(query="q", drug_name="Semaglutide"))
    assert check.plan is not None
    assert check.plan.filters.intervention is None  # no longer claimed to apply to every cohort
    assert [c.filters.intervention for c in check.plan.compare] == ["semaglutide", "tirzepatide"]
    assert any("one of the compared cohorts" in n for n in check.notes)


def test_structured_field_not_varied_by_cohorts_applies_to_every_cohort():
    cohorts = ({"label": "Lung", "filters": {"condition": "lung cancer"}}, {"label": "Breast", "filters": {"condition": "breast cancer"}})
    check = finalize(_compare(*cohorts), VisualizeRequest(query="q", country="Japan", trial_phases=["PHASE3"]))
    assert [(c.filters.condition, c.filters.location, c.filters.phases) for c in check.plan.compare] == [
        ("lung cancer", "Japan", [Phase.PHASE3]),
        ("breast cancer", "Japan", [Phase.PHASE3]),
    ]


def test_year_range_conflict_with_structured_fields_is_reported_not_a_500():
    # Reviewer repro: "Compare phases of melanoma trials that started before 2012 vs from 2012 onward" + start_year=2015
    before = {"label": "Before 2012", "filters": {"start_year_max": 2011}}
    after = {"label": "2012 onward", "filters": {"start_year_min": 2012}}
    check = finalize(_compare(before, after, filters={"condition": "melanoma"}), VisualizeRequest(query="q", start_year=2015))
    assert check.plan is None
    assert any("Before 2012" in c and "start_year" in c for c in check.conflicts)


def test_cohort_years_narrow_the_shared_range():
    cohorts = ({"label": "Early", "filters": {"start_year_max": 2018}}, {"label": "Late", "filters": {"start_year_min": 2019}})
    check = finalize(_compare(*cohorts, filters={"condition": "melanoma"}), VisualizeRequest(query="q", start_year=2015, end_year=2024))
    assert [(c.filters.start_year_min, c.filters.start_year_max) for c in check.plan.compare] == [(2015, 2018), (2019, 2024)]


def test_llm_inconsistent_year_range_goes_back_for_repair():
    cohorts = ({"label": "A", "filters": {"start_year_max": 2011}}, {"label": "B", "filters": {"start_year_min": 2012}})
    check = finalize(_compare(*cohorts, filters={"start_year_min": 2015}), Q)
    assert check.conflicts == [] and check.errors


def test_list_fields_intersect_and_empty_intersection_with_caller_is_a_conflict():
    p23 = {"label": "Phase 2/3", "filters": {"phases": ["PHASE2", "PHASE3"]}}
    p4 = {"label": "Phase 4", "filters": {"phases": ["PHASE4"]}}
    ok = finalize(_compare(p23, p4), VisualizeRequest(query="q", trial_phases=["PHASE3", "PHASE4"]))
    assert [c.filters.phases for c in ok.plan.compare] == [[Phase.PHASE3], [Phase.PHASE4]]
    p2 = {"label": "Phase 2", "filters": {"phases": ["PHASE2"]}}
    bad = finalize(_compare(p2, p4), VisualizeRequest(query="q", trial_phases=["PHASE3"]))
    assert bad.plan is None and any("no phases in common" in c for c in bad.conflicts)


def test_caller_phase_filter_is_never_widened_by_a_cohort():
    # trial_phases=[PHASE3] + "Phase 2 vs Phase 3": the Phase 2 cohort contradicts the caller.
    p2 = {"label": "Phase 2", "filters": {"phases": ["PHASE2"]}}
    p3 = {"label": "Phase 3", "filters": {"phases": ["PHASE3"]}}
    check = finalize(_compare(p2, p3), VisualizeRequest(query="q", trial_phases=["PHASE3"]))
    assert check.plan is None and len(check.conflicts) == 1 and "Phase 2" in check.conflicts[0]


def test_cohorts_identical_after_merging_go_back_for_repair():
    a = {"label": "A", "filters": {"phases": ["PHASE2", "PHASE3"]}}
    b = {"label": "B", "filters": {"phases": ["PHASE3"]}}
    check = finalize(_compare(a, b), VisualizeRequest(query="q", trial_phases=["PHASE3"]))
    assert check.plan is None and check.errors == ["cohorts must differ once shared filters are applied."]


def test_llm_shared_value_replaced_by_cohort_value_is_noted():
    check = finalize(_compare(SEMA, TIRZ, filters={"intervention": "pembrolizumab"}), Q)
    assert [c.filters.intervention for c in check.plan.compare] == ["semaglutide", "tirzepatide"]
    assert any("instead of the shared" in n for n in check.notes)


def test_geographic_question_grouped_by_something_else_keeps_that_grouping():
    # Reviewer repro: "which sponsors run recruiting breast cancer trials in Japan" came back grouped by country.
    check = finalize(draft(analysis="geographic", group_by="sponsor", filters={"location": "Japan"}), Q)
    assert (check.plan.analysis, check.plan.group_by, check.plan.filters.location) == (Analysis.DISTRIBUTION, Dimension.SPONSOR, "Japan")


def test_cohort_repeating_the_caller_value_does_not_strip_it_from_other_cohorts():
    # drug_name=semaglutide; cohorts differ by condition, one also names semaglutide: every cohort keeps the drug.
    a = {"label": "Obesity", "filters": {"condition": "obesity", "intervention": "semaglutide"}}
    b = {"label": "Diabetes", "filters": {"condition": "type 2 diabetes"}}
    check = finalize(_compare(a, b), VisualizeRequest(query="q", drug_name="semaglutide"))
    assert [c.filters.intervention for c in check.plan.compare] == ["semaglutide", "semaglutide"]
    assert check.plan.filters.intervention == "semaglutide"
    assert not any("one of the compared cohorts" in n for n in check.notes)
