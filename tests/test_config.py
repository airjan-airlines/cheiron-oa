import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import ALLOWED_MODELS, Settings

RULES_FILE = Path(__file__).parent.parent / "docs" / "openai-rules.md"


def test_allowlist_matches_company_rules_file():
    listed = [line.strip() for line in RULES_FILE.read_text().splitlines() if re.fullmatch(r"gpt-[\w.-]+", line.strip())]
    assert tuple(listed) == ALLOWED_MODELS


def test_default_model_is_allowed():
    assert Settings(_env_file=None).openai_model in ALLOWED_MODELS


def test_disallowed_model_is_rejected():
    with pytest.raises(ValidationError, match="not allowed"):
        Settings(_env_file=None, openai_model="gpt-4-turbo")


def test_reasoning_models_are_flagged():
    assert Settings(_env_file=None, openai_model="gpt-5-mini").model_is_reasoning
    assert not Settings(_env_file=None, openai_model="gpt-4.1-mini").model_is_reasoning
