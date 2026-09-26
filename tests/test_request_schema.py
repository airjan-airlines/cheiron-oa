import pytest
from pydantic import ValidationError

from app.schemas.enums import Phase, Status
from app.schemas.request import VisualizeRequest


def test_minimal_request_uses_defaults():
    req = VisualizeRequest(query="How many trials by phase?")
    assert req.top_n == 15
    assert req.max_records == 5000
    assert req.max_citations_per_datum == 3


def test_query_is_stripped_and_must_not_be_blank():
    assert VisualizeRequest(query="  trials by phase  ").query == "trials by phase"
    with pytest.raises(ValidationError):
        VisualizeRequest(query="   ")


def test_unknown_fields_are_rejected():
    # A typo'd filter must fail loudly instead of being silently ignored.
    with pytest.raises(ValidationError, match="drugname"):
        VisualizeRequest(query="trials", drugname="Pembrolizumab")


def test_start_year_must_not_exceed_end_year():
    with pytest.raises(ValidationError, match="start_year must be <= end_year"):
        VisualizeRequest(query="trials", start_year=2024, end_year=2015)


def test_phase_and_status_must_be_registry_enums():
    with pytest.raises(ValidationError):
        VisualizeRequest(query="trials", trial_phases=["Phase 3"])
    req = VisualizeRequest(query="trials", trial_phases=["PHASE3"], statuses=["RECRUITING"])
    assert req.trial_phases == [Phase.PHASE3]
    assert req.statuses == [Status.RECRUITING]


def test_duplicate_enum_values_are_collapsed():
    req = VisualizeRequest(query="trials", trial_phases=["PHASE2", "PHASE2", "PHASE3"])
    assert req.trial_phases == [Phase.PHASE2, Phase.PHASE3]


def test_structured_fields_map_to_plan_filters():
    req = VisualizeRequest(
        query="trials",
        drug_name="Pembrolizumab",
        condition="melanoma",
        sponsor="Merck Sharp & Dohme LLC",
        country="South Korea",
        trial_phases=["PHASE3"],
        start_year=2015,
    )
    f = req.structured_filters()
    assert f.intervention == "Pembrolizumab"
    assert f.condition == "melanoma"
    assert f.sponsor == "Merck Sharp & Dohme LLC"
    assert f.location == "South Korea"
    assert f.phases == [Phase.PHASE3]
    assert f.start_year_min == 2015 and f.start_year_max is None


def test_limits_are_bounded():
    with pytest.raises(ValidationError):
        VisualizeRequest(query="trials", max_records=50_000)
    with pytest.raises(ValidationError):
        VisualizeRequest(query="trials", top_n=0)
