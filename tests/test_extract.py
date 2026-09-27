import json

from app.ctgov.extract import ABSENT_EXCERPT, FIELD_PATHS, extract, start_year
from app.schemas.enums import Dimension as D
from tests.conftest import load_fixture


def _study(**protocol) -> dict:
    ps = {"identificationModule": {"nctId": "NCT00000001", "briefTitle": "A trial"}}
    ps.update(protocol)
    return {"protocolSection": ps}


def test_multi_phase_trial_gets_one_combined_bucket_with_raw_excerpt():
    row = extract(_study(designModule={"phases": ["PHASE1", "PHASE2"]}))
    (value,) = row.get(D.PHASE)
    assert value.key == "PHASE1/PHASE2"
    assert value.label == "Phase 1/Phase 2"
    assert json.loads(value.excerpt) == ["PHASE1", "PHASE2"]
    assert value.field == FIELD_PATHS[D.PHASE]


def test_missing_phase_and_not_applicable_are_distinct_buckets():
    missing = extract(_study()).get(D.PHASE)[0]
    na = extract(_study(designModule={"phases": ["NA"]})).get(D.PHASE)[0]
    assert (missing.key, missing.label, missing.excerpt) == ("MISSING", "Not specified", ABSENT_EXCERPT)
    assert (na.key, na.label) == ("NA", "Not Applicable")


def test_enum_fields_carry_registry_labels():
    row = extract(
        _study(
            statusModule={"overallStatus": "ACTIVE_NOT_RECRUITING"},
            sponsorCollaboratorsModule={"leadSponsor": {"name": "Merck Sharp & Dohme LLC", "class": "INDUSTRY"}},
        )
    )
    assert row.get(D.STATUS)[0].label == "Active, not recruiting"
    assert row.get(D.SPONSOR_CLASS)[0].label == "Industry"
    sponsor = row.get(D.SPONSOR)[0]
    assert sponsor.key == "merck sharp & dohme llc" and sponsor.label == "Merck Sharp & Dohme LLC"


def test_start_year_parses_partial_dates_and_keeps_raw_date_as_excerpt():
    assert start_year("2017") == 2017
    assert start_year("2017-10") == 2017
    assert start_year("2017-10-03") == 2017
    assert start_year(None) is None
    row = extract(_study(statusModule={"startDateStruct": {"date": "2017-10"}}))
    assert (row.get(D.START_YEAR)[0].key, row.get(D.START_YEAR)[0].excerpt) == ("2017", "2017-10")
    assert extract(_study()).get(D.START_YEAR) == ()


def test_countries_are_distinct_per_trial():
    row = extract(_study(contactsLocationsModule={"locations": [{"country": "South Korea"}, {"country": "South Korea"}, {"country": "Japan"}]}))
    assert [v.key for v in row.get(D.COUNTRY)] == ["South Korea", "Japan"]


def test_intervention_names_resolve_to_mesh_terms_and_split_combinations():
    study = _study(
        armsInterventionsModule={
            "interventions": [
                {"type": "DRUG", "name": "Pembrolizumab 200 mg IV"},
                {"type": "DRUG", "name": "Docetaxel and capecitabine"},
                {"type": "DRUG", "name": "IV"},
                {"type": "BEHAVIORAL", "name": "Exercise"},
            ]
        }
    )
    study["derivedSection"] = {"interventionBrowseModule": {"meshes": [{"term": "pembrolizumab"}, {"term": "Docetaxel"}, {"term": "Capecitabine"}, {"term": "Ivermectin"}]}}
    values = extract(study).get(D.INTERVENTION)
    by_label = {v.label: v for v in values}
    assert by_label["Pembrolizumab"].excerpt == "Pembrolizumab 200 mg IV"  # excerpt is always the raw name
    assert by_label["Docetaxel"].excerpt == by_label["Capecitabine"].excerpt == "Docetaxel and capecitabine"
    assert "Ivermectin" not in by_label  # "IV" must not match inside another word
    assert by_label["Exercise"].category == "BEHAVIORAL"
    types = extract(study).get(D.INTERVENTION_TYPE)
    assert [t.key for t in types] == ["DRUG", "BEHAVIORAL"]


def test_conditions_prefer_mesh_terms():
    study = _study(conditionsModule={"conditions": ["Stage IV Gastric Adenocarcinoma AJCC v8"]})
    study["derivedSection"] = {"conditionBrowseModule": {"meshes": [{"term": "Stomach Neoplasms"}]}}
    (value,) = extract(study).get(D.CONDITION)
    assert value.label == "Stomach Neoplasms"
    assert value.field == FIELD_PATHS[D.CONDITION]


def test_every_recorded_study_extracts_with_an_nct_id_and_phase():
    for name in ("korea_gastric_recruiting", "tirzepatide", "kras_nsclc"):
        for study in load_fixture(name)["studies"]:
            row = extract(study)
            assert row.nct_id.startswith("NCT")
            assert len(row.get(D.PHASE)) == 1  # phase is always exactly one bucket


def test_salt_or_ester_names_group_with_the_parent_drug():
    study = _study(armsInterventionsModule={"interventions": [{"type": "DRUG", "name": "Fludarabine phosphate"}]})
    study["derivedSection"] = {"interventionBrowseModule": {"meshes": [{"term": "fludarabine phosphate"}, {"term": "Fludarabine"}]}}
    assert [v.label for v in extract(study).get(D.INTERVENTION)] == ["Fludarabine"]
