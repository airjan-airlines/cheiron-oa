"""Test doubles for the two external services: the LLM and ClinicalTrials.gov."""

from app.ctgov.client import SearchResult
from app.planner.llm_schema import QueryPlanLLM
from tests.conftest import load_fixture

NO_FILTERS = {
    "condition": None, "intervention": None, "sponsor": None, "location": None,
    "phases": [], "statuses": [], "start_year_min": None, "start_year_max": None,
}


def draft(**overrides) -> QueryPlanLLM:
    """A QueryPlanLLM with every field defaulted, as strict structured output would return it."""
    fields = {
        "is_about_clinical_trials": True, "unsupported_reason": None, "analysis": None, "group_by": None,
        "metric": None, "filters": NO_FILTERS, "compare": [], "network": None, "top_n": None,
        "chart_type_suggestion": None, "chart_rationale": None,
    }
    filters = overrides.pop("filters", {})
    fields.update(overrides)
    fields["filters"] = {**NO_FILTERS, **filters}
    fields["compare"] = [{"label": c["label"], "filters": {**NO_FILTERS, **c["filters"]}} for c in fields["compare"]]
    return QueryPlanLLM(**fields)


class ScriptedLLM:
    """Returns the given drafts in order and records every message it was sent."""

    def __init__(self, *drafts: QueryPlanLLM):
        self.drafts = list(drafts)
        self.calls: list[list[dict]] = []

    async def plan(self, messages):
        self.calls.append([dict(m) for m in messages])
        return self.drafts.pop(0)


class FixtureRegistry:
    """Serves recorded fixtures by matching the search params; unknown searches match nothing."""

    def __init__(self, *fixture_names: str):
        self.fixtures = [load_fixture(n) for n in fixture_names]
        self.searches: list[dict] = []

    async def search(self, params, max_records):
        self.searches.append(params)
        search_keys = {k: v for k, v in params.items() if k.startswith(("query.", "filter."))}
        for fx in self.fixtures:
            if {k: v for k, v in fx["params"].items() if k.startswith(("query.", "filter."))} == search_keys:
                return SearchResult(studies=fx["studies"][:max_records], total_count=fx["totalCount"])
        return SearchResult(studies=[], total_count=0)

    async def data_timestamp(self):
        return "2026-09-25T09:00:04"
