"""Exceptions that cross module boundaries."""


class UpstreamUnavailable(Exception):
    """ClinicalTrials.gov could not be reached or kept failing after retries (HTTP 502)."""


class UpstreamRejectedQuery(Exception):
    """ClinicalTrials.gov answered 400: the search terms could not be parsed. Not retried."""


class PlannerUnavailable(Exception):
    """The LLM planner could not be called, e.g. missing API key or provider outage (HTTP 502)."""
