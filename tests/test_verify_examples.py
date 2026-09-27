"""The example auditor must catch wrong excerpts, or its "0 problems" means nothing."""

import json

from scripts.verify_examples import excerpt_matches
from tests.conftest import load_fixture

STUDY = load_fixture("korea_gastric_recruiting")["studies"][0]
PS = STUDY["protocolSection"]
PHASES = "protocolSection.designModule.phases"
COUNTRIES = "protocolSection.contactsLocationsModule.locations[].country"
NAMES = "protocolSection.armsInterventionsModule.interventions[].name"


def test_accepts_real_values():
    assert excerpt_matches(STUDY, PHASES, json.dumps(PS["designModule"]["phases"]))
    assert excerpt_matches(STUDY, COUNTRIES, PS["contactsLocationsModule"]["locations"][0]["country"])


def test_rejects_wrong_values():
    assert not excerpt_matches(STUDY, PHASES, '["PHASE4"]')
    assert not excerpt_matches(STUDY, COUNTRIES, "Narnia")
    assert not excerpt_matches(STUDY, NAMES, '"Aspirin" | "Nonexistent"')
