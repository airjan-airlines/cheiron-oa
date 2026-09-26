from app.ctgov.query_builder import build_params, canonical_location, normalize_filters
from app.schemas.plan import Filters


def test_free_text_filters_map_to_query_params():
    params = build_params(Filters(condition="melanoma", intervention="pembrolizumab", sponsor="Merck", location="Japan"))
    assert params["query.cond"] == "melanoma"
    assert params["query.intr"] == "pembrolizumab"
    assert params["query.lead"] == "Merck"  # lead sponsor only, never query.spons
    assert "query.spons" not in params
    assert params["query.locn"] == "Japan"
    assert params["countTotal"] == "true"
    assert "NCTId" in params["fields"]


def test_phases_use_filter_advanced_because_filter_phase_does_not_exist():
    assert build_params(Filters(phases=["PHASE3"]))["filter.advanced"] == "AREA[Phase]PHASE3"
    assert build_params(Filters(phases=["PHASE2", "PHASE3"]))["filter.advanced"] == "AREA[Phase](PHASE2 OR PHASE3)"
    assert "filter.phase" not in build_params(Filters(phases=["PHASE3"]))


def test_year_bounds_become_a_start_date_range():
    assert build_params(Filters(start_year_min=2015))["filter.advanced"] == "AREA[StartDate]RANGE[2015-01-01,MAX]"
    assert build_params(Filters(start_year_max=2016))["filter.advanced"] == "AREA[StartDate]RANGE[MIN,2016-12-31]"


def test_phase_and_year_clauses_are_anded():
    params = build_params(Filters(phases=["PHASE3"], start_year_min=2015, start_year_max=2020))
    assert params["filter.advanced"] == "AREA[Phase]PHASE3 AND AREA[StartDate]RANGE[2015-01-01,2020-12-31]"


def test_statuses_are_comma_joined():
    assert build_params(Filters(statuses=["RECRUITING", "COMPLETED"]))["filter.overallStatus"] == "RECRUITING,COMPLETED"


def test_empty_filters_send_no_search_terms():
    params = build_params(Filters())
    assert not any(k.startswith(("query.", "filter.")) for k in params)


def test_country_aliases_use_registry_spelling():
    assert canonical_location("Korea, Republic of") == "South Korea"
    assert canonical_location(" USA ") == "United States"
    assert canonical_location("한국") == "South Korea"
    assert canonical_location("Japan") == "Japan"


def test_normalize_filters_reports_each_change():
    filters, notes = normalize_filters(Filters(location="Republic of Korea"))
    assert filters.location == "South Korea"
    assert notes == ["Location 'Republic of Korea' normalized to the registry spelling 'South Korea'."]
    unchanged, no_notes = normalize_filters(Filters(location="Japan"))
    assert unchanged.location == "Japan" and no_notes == []
