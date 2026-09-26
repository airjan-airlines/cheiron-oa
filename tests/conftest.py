import json
from functools import cache
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


@cache
def load_fixture(name: str) -> dict:
    """Recorded ClinicalTrials.gov responses (see scripts/record_fixtures.py)."""
    return json.loads((FIXTURES / f"{name}.json").read_text())
