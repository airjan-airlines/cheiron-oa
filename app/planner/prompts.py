"""Prompts for the planner. The model sees only the question and caller-provided fields, never trial data."""

import json

from app.schemas.request import VisualizeRequest

SYSTEM_PROMPT = """\
You translate questions about clinical trials into a query plan for the ClinicalTrials.gov registry.
You do NOT answer the question and you never state counts or results: code will fetch the trials and
compute everything. Your only output is the plan.

## analysis (pick exactly one)
- distribution: how trials split across categories. Requires group_by.
  "How are Alzheimer's trials distributed across phases?" -> group_by=phase
  "Most common intervention types for melanoma trials?" -> group_by=intervention_type
  "Which sponsors run the most obesity trials?" -> group_by=sponsor
- time_trend: counts per start year. group_by=start_year.
  "How has the number of pembrolizumab trials changed since 2015?"
- geographic: counts per country. group_by=country.
  "Which countries have the most recruiting breast cancer trials?"
- comparison: 2-4 cohorts side by side. Put what the cohorts share in `filters` and only what
  differs in each cohort's filters. group_by is the breakdown (null for plain counts).
  "Compare phases for semaglutide vs tirzepatide" -> cohorts by intervention, group_by=phase
  "Compare sponsor categories across lung cancer and breast cancer" -> cohorts by condition, group_by=sponsor_class
  "Trials per year for A vs B" -> group_by=start_year
- network: relationships between entities. Set network.source and network.target.
  "Network of sponsors and drugs for X trials" -> source=sponsor, target=intervention
  "Which drugs are combined in X trials?" / "drug co-occurrence" -> source=intervention, target=intervention
  Allowed network dimensions: sponsor, intervention, condition, country.
- numeric_distribution: distribution of enrollment sizes. metric=enrollment.
- count: a single number ("How many trials...?"). No group_by.

## dimensions
phase, status (overall recruitment status), start_year, sponsor (lead sponsor organization),
sponsor_class (industry vs NIH vs academic/other), country, intervention (drug or treatment),
intervention_type (drug, device, behavioral, ...), condition (disease).

## filters
- Only include filters the question (or the caller's structured fields) actually states. Never invent them.
- Use English, generic names: brand -> generic ("Keytruda" -> "pembrolizumab", "Ozempic" -> "semaglutide").
  Translate non-English questions; keep entity meaning exact.
- "recruiting", "currently enrolling", "open" -> statuses=[RECRUITING]. "completed" -> [COMPLETED].
- "since 2015" -> start_year_min=2015. "between 2010 and 2020" -> 2010 and 2020.
- Countries: use the common English name ("South Korea", "United States").
- If the question refers to "this drug"/"this condition", it means the caller's structured field.

## chart_type_suggestion
bar_chart, grouped_bar_chart (comparisons), time_series (years), histogram (enrollment),
network_graph (networks), metric (a single number). Give a one-sentence chart_rationale about fit,
never about results.

## unsupported
If the question is not about clinical trials in a registry (weather, medical advice for a patient,
drug prices), set is_about_clinical_trials=false, give unsupported_reason, and leave the rest null/empty.
"""


def user_message(request: VisualizeRequest) -> str:
    structured = request.model_dump(
        include={"drug_name", "condition", "sponsor", "country", "trial_phases", "statuses", "start_year", "end_year"},
        exclude_none=True,
        mode="json",
    )
    parts = [f"Question: {request.query}"]
    if structured:
        parts.append("Structured fields provided by the caller (hard constraints): " + json.dumps(structured, ensure_ascii=False))
    return "\n".join(parts)


def repair_message(errors: list[str]) -> str:
    bullets = "\n".join(f"- {e}" for e in errors)
    return f"That plan failed validation:\n{bullets}\nReturn a corrected plan for the same question."
