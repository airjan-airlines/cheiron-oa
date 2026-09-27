"""Turn the LLM's draft into a validated `QueryPlan` (D5, D6).

Three kinds of outcome:
- errors: the draft is unusable as-is; they are sent back to the LLM for one repair attempt;
- normalizations: unambiguous slips (a time trend grouped by phase) are fixed deterministically
  and reported in `notes`, rather than spending an LLM round trip;
- overrides: the caller's structured fields always win over what the LLM inferred.
"""

from dataclasses import dataclass, field

from pydantic import ValidationError

from app.planner.llm_schema import QueryPlanLLM
from app.schemas.enums import Analysis, Dimension, Metric
from app.schemas.plan import Cohort, Filters, NetworkSpec, QueryPlan
from app.schemas.request import VisualizeRequest

NETWORK_DIMENSIONS = {Dimension.SPONSOR, Dimension.INTERVENTION, Dimension.CONDITION, Dimension.COUNTRY}
REQUEST_FIELD_NAMES = {
    "condition": "condition",
    "intervention": "drug_name",
    "sponsor": "sponsor",
    "location": "country",
    "phases": "trial_phases",
    "statuses": "statuses",
    "start_year_min": "start_year",
    "start_year_max": "end_year",
}


@dataclass
class PlanCheck:
    plan: QueryPlan | None
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _empty(value) -> bool:
    return value is None or value == []


def _same(a, b) -> bool:
    """Equal, ignoring case and surrounding space for text ("Pembrolizumab" == "pembrolizumab")."""
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().casefold() == b.strip().casefold()
    return a == b


def _apply_structured_fields(inferred: dict, request: VisualizeRequest, notes: list[str]) -> dict:
    forced = request.structured_filters().model_dump()
    merged = dict(inferred)
    for name, value in forced.items():
        if _empty(value):
            continue
        if not _empty(inferred.get(name)) and not _same(inferred[name], value):
            notes.append(
                f"Request field '{REQUEST_FIELD_NAMES[name]}' ({value}) overrides '{inferred[name]}' inferred from the question."
            )
        merged[name] = value
    return merged


def _format_errors(exc: ValidationError) -> list[str]:
    return [f"{'.'.join(str(p) for p in err['loc']) or 'plan'}: {err['msg']}" for err in exc.errors()]


def finalize(draft: QueryPlanLLM, request: VisualizeRequest) -> PlanCheck:
    if not draft.is_about_clinical_trials:
        reason = draft.unsupported_reason or "The question is not about clinical trials in the registry."
        return PlanCheck(QueryPlan(supported=False, unsupported_reason=reason))

    notes: list[str] = []
    top_n = request.top_n if "top_n" in request.model_fields_set else (draft.top_n or 15)
    if not 1 <= top_n <= 50:
        notes.append(f"top_n {top_n} is outside 1–50; clamped.")
        top_n = min(max(top_n, 1), 50)
    try:
        plan = QueryPlan(
            analysis=draft.analysis,
            group_by=draft.group_by,
            metric=draft.metric or Metric.TRIAL_COUNT,
            filters=Filters(**_apply_structured_fields(draft.filters.model_dump(), request, notes)),
            compare=[Cohort(label=c.label, filters=Filters(**c.filters.model_dump())) for c in draft.compare],
            network=NetworkSpec(**draft.network.model_dump()) if draft.network else None,
            top_n=top_n,
            chart_type_suggestion=draft.chart_type_suggestion,
            chart_rationale=draft.chart_rationale,
        )
    except ValidationError as exc:
        return PlanCheck(None, errors=_format_errors(exc), notes=notes)

    plan, errors = _check_semantics(plan, notes)
    return PlanCheck(None if errors else plan, errors=errors, notes=notes)


def _check_semantics(plan: QueryPlan, notes: list[str]) -> tuple[QueryPlan, list[str]]:
    errors: list[str] = []
    update: dict = {}
    a = plan.analysis

    if a is None:
        return plan, ["analysis is required for a clinical-trial question."]

    fixed_group_by = {Analysis.TIME_TREND: Dimension.START_YEAR, Analysis.GEOGRAPHIC: Dimension.COUNTRY}
    if a in fixed_group_by and plan.group_by != fixed_group_by[a]:
        if plan.group_by is not None:
            notes.append(f"A {a.value} question groups by {fixed_group_by[a].value}; ignored group_by={plan.group_by.value}.")
        update["group_by"] = fixed_group_by[a]
    if a == Analysis.DISTRIBUTION and plan.group_by is None:
        errors.append("distribution needs group_by (which category to split trials by).")
    if a in (Analysis.COUNT, Analysis.NUMERIC_DISTRIBUTION, Analysis.NETWORK) and plan.group_by is not None:
        update["group_by"] = None

    if a == Analysis.NUMERIC_DISTRIBUTION:
        update["metric"] = Metric.ENROLLMENT
    elif plan.metric == Metric.ENROLLMENT:
        notes.append("Enrollment is only supported as a distribution (histogram); counting trials instead.")
        update["metric"] = Metric.TRIAL_COUNT

    if a == Analysis.COMPARISON:
        if not 2 <= len(plan.compare) <= 4:
            errors.append(f"comparison needs 2-4 cohorts; got {len(plan.compare)}.")
        for c in plan.compare:
            if c.filters.is_empty():
                errors.append(f"cohort '{c.label}' has no filters, so it would match the whole registry.")
        if len({c.label for c in plan.compare}) != len(plan.compare):
            errors.append("cohort labels must be distinct.")
        if len({c.filters.model_dump_json() for c in plan.compare}) != len(plan.compare):
            errors.append("cohorts must differ in at least one filter.")
    elif plan.compare:
        notes.append(f"Ignored cohorts: a {a.value} question is a single search.")
        update["compare"] = []

    if a == Analysis.NETWORK:
        if plan.network is None:
            errors.append("network needs network.source and network.target.")
        else:
            bad = {plan.network.source, plan.network.target} - NETWORK_DIMENSIONS
            if bad:
                allowed = ", ".join(sorted(d.value for d in NETWORK_DIMENSIONS))
                errors.append(f"network dimensions must be among {allowed}; got {', '.join(sorted(d.value for d in bad))}.")
    elif plan.network is not None:
        update["network"] = None

    return plan.model_copy(update=update), errors


def searches(plan: QueryPlan) -> list[tuple[str | None, Filters]]:
    """The registry searches a plan needs: one per cohort for comparisons, else one."""
    if plan.analysis != Analysis.COMPARISON:
        return [(None, plan.filters)]
    out = []
    for cohort in plan.compare:
        overlay = {k: v for k, v in cohort.filters.model_dump().items() if not _empty(v)}
        out.append((cohort.label, plan.filters.model_copy(update=overlay)))
    return out
