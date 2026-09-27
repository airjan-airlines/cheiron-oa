"""Plain-language descriptions of a search, built from the plan (never from an LLM)."""

from app.schemas.enums import PHASE_LABELS, STATUS_LABELS
from app.schemas.plan import Filters


def year_span(filters: Filters) -> str | None:
    """e.g. "2015–2020", "2015 onward", "through 2016"; None when unbounded."""
    lo, hi = filters.start_year_min, filters.start_year_max
    if lo is not None and hi is not None:
        return f"{lo}–{hi}" if lo != hi else str(lo)
    if lo is not None:
        return f"{lo} onward"
    if hi is not None:
        return f"through {hi}"
    return None


def subject(filters: Filters, include_years: bool = True) -> str:
    """e.g. "recruiting gastric cancer trials in South Korea", "pembrolizumab trials starting 2015 or later"."""
    words: list[str] = []
    if len(filters.statuses) == 1:
        words.append(STATUS_LABELS[filters.statuses[0]].lower())
    if filters.phases:
        words.append(" or ".join(PHASE_LABELS[p] for p in filters.phases))
    if filters.intervention:
        words.append(filters.intervention)
    if filters.condition:
        words.append(("for " if filters.intervention else "") + filters.condition)
    words.append("trials")
    if filters.sponsor:
        words.append(f"led by {filters.sponsor}")
    if filters.location:
        words.append(f"in {filters.location}")
    if len(filters.statuses) > 1:
        words.append("(status: " + ", ".join(STATUS_LABELS[s].lower() for s in filters.statuses) + ")")
    lo, hi = (filters.start_year_min, filters.start_year_max) if include_years else (None, None)
    if lo is not None and hi is not None:
        words.append(f"starting {lo}–{hi}" if lo != hi else f"starting in {lo}")
    elif lo is not None:
        words.append(f"starting {lo} or later")
    elif hi is not None:
        words.append(f"starting {hi} or earlier")
    text = " ".join(words)
    return text if text != "trials" else "all registered trials"


def capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]
