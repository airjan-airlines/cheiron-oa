"""Turn the LLM's draft into a validated `QueryPlan` (D5, D6).

Three kinds of outcome:
- errors: the draft is unusable as-is; they are sent back to the LLM for one repair attempt;
- normalizations: unambiguous slips (a time trend grouped by phase) are fixed deterministically
  and reported in `notes`, rather than spending an LLM round trip;
- overrides: the caller's structured fields always win over what the LLM inferred;
- conflicts: the caller's own input contradicts itself (e.g. `drug_name` names a drug that none
  of the compared cohorts use). No repair can fix that, so the request fails without fetching.
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


TEXT_FIELDS = ("condition", "intervention", "sponsor", "location")
LIST_FIELDS = ("phases", "statuses")


@dataclass
class PlanCheck:
    plan: QueryPlan | None
    errors: list[str] = field(default_factory=list)  # sent back to the LLM for repair
    notes: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)  # contradictions in the caller's input; not repairable


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
    conflicts: list[str] = []
    if not errors and plan.analysis == Analysis.COMPARISON:
        plan, errors, conflicts = _merge_cohorts(plan, request.structured_filters(), notes)
    ok = not errors and not conflicts
    return PlanCheck(plan if ok else None, errors=errors, notes=notes, conflicts=conflicts)


def _check_semantics(plan: QueryPlan, notes: list[str]) -> tuple[QueryPlan, list[str]]:
    errors: list[str] = []
    update: dict = {}
    a = plan.analysis

    if a is None:
        return plan, ["analysis is required for a clinical-trial question."]

    if a == Analysis.GEOGRAPHIC and plan.group_by not in (None, Dimension.COUNTRY):
        # "Which sponsors run trials in Japan?": the country is a filter, the grouping is the question.
        notes.append(f"Treated as a breakdown by {plan.group_by.value}; the location is a filter, not the grouping.")
        a = Analysis.DISTRIBUTION
        update["analysis"] = a
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


def _fmt(value) -> str:
    return ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)


def _merge_cohorts(plan: QueryPlan, structured: Filters, notes: list[str]) -> tuple[QueryPlan, list[str], list[str]]:
    """Combine each cohort's filters with the shared ones. Cohorts narrow shared filters; they may
    not contradict the caller's structured fields.

    Returns the plan with fully merged cohort filters, repairable errors, and caller conflicts.
    """
    shared = plan.filters.model_dump()
    forced = structured.model_dump()
    labels = [c.label for c in plan.compare]
    cohorts = [c.filters.model_dump() for c in plan.compare]
    merged = [dict(shared) for _ in cohorts]
    shared_update: dict = {}
    errors: list[str] = []
    conflicts: list[str] = []

    for name in TEXT_FIELDS + LIST_FIELDS:
        values = [c[name] for c in cohorts]
        if all(_empty(v) for v in values):
            continue  # cohorts don't vary this field: the shared value applies to all of them
        common = shared[name]
        if _empty(common):
            for m, v in zip(merged, values):
                if not _empty(v):
                    m[name] = v
            continue
        from_request = not _empty(forced[name])
        field_name = REQUEST_FIELD_NAMES[name] if from_request else f"filters.{name}"
        set_values = [v for v in values if not _empty(v)]
        one_side = any(_same(v, common) for v in set_values) and any(not _same(v, common) for v in set_values)
        if from_request and name in TEXT_FIELDS and one_side:
            # e.g. drug_name=semaglutide + "compare semaglutide vs tirzepatide": the caller's field is one
            # side of the comparison (typical when a UI passes the drug being viewed). Lists never take
            # this path: they always intersect, so a cohort can't widen the caller's phases or statuses.
            for m, v in zip(merged, values):
                if not _empty(v):
                    m[name] = v
            shared_update[name] = [] if name in LIST_FIELDS else None
            notes.append(f"Request field '{field_name}' ({_fmt(common)}) is one of the compared cohorts; the other cohorts use their own value.")
            continue
        for label, m, v in zip(labels, merged, values):
            if _empty(v):
                continue  # this cohort inherits the shared value
            if name in LIST_FIELDS:
                narrowed = [x for x in v if x in common]
                if narrowed:
                    m[name] = narrowed
                    continue
                problem = f"{field_name} ({_fmt(common)}) and cohort '{label}' ({_fmt(v)}) have no {name} in common."
                (conflicts if from_request else errors).append(problem)
            elif not _same(v, common):
                if from_request:
                    conflicts.append(f"Request field '{field_name}' ({common}) contradicts cohort '{label}' ({v}).")
                else:
                    m[name] = v
                    notes.append(f"Cohort '{label}' uses {name}='{v}' instead of the shared '{common}'.")

    # Year bounds intersect: a cohort can only narrow the shared range.
    for label, m, c in zip(labels, merged, cohorts):
        lows = [y for y in (shared["start_year_min"], c["start_year_min"]) if y is not None]
        highs = [y for y in (shared["start_year_max"], c["start_year_max"]) if y is not None]
        lo, hi = (max(lows) if lows else None), (min(highs) if highs else None)
        m["start_year_min"], m["start_year_max"] = lo, hi
        if lo is not None and hi is not None and lo > hi:
            problem = f"Cohort '{label}' covers start years that the shared range excludes (would need {lo} to {hi})."
            from_request = forced["start_year_min"] is not None or forced["start_year_max"] is not None
            (conflicts if from_request else errors).append(problem + (" The range comes from start_year/end_year." if from_request else ""))

    if errors or conflicts:
        return plan, errors, conflicts
    try:
        final = [Cohort(label=label, filters=Filters(**m)) for label, m in zip(labels, merged)]
    except ValidationError as exc:
        return plan, _format_errors(exc), []
    if len({c.filters.model_dump_json() for c in final}) != len(final):
        return plan, ["cohorts must differ once shared filters are applied."], []
    return plan.model_copy(update={"compare": final, "filters": plan.filters.model_copy(update=shared_update)}), [], []


def searches(plan: QueryPlan) -> list[tuple[str | None, Filters]]:
    """The registry searches a plan needs: one per cohort for comparisons, else one.

    Cohort filters are already merged with the shared ones by `finalize`.
    """
    if plan.analysis != Analysis.COMPARISON:
        return [(None, plan.filters)]
    return [(cohort.label, cohort.filters) for cohort in plan.compare]
