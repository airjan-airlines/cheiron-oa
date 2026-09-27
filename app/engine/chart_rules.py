"""Which chart types fit which questions (D4): the LLM suggests, these rules decide.

Precedence: the caller's `chart_type` > the LLM's suggestion > the rule default. A
suggestion that doesn't fit the data shape is overridden, and the override is reported
in `meta.visualization_rationale`, so the planner can never produce an unrenderable spec.
"""

from app.schemas.enums import Analysis, Dimension, VizType
from app.schemas.plan import QueryPlan
from app.schemas.response import VisualizationRationale

V = VizType


def allowed_charts(plan: QueryPlan) -> tuple[VizType, ...]:
    """Compatible types for this plan, default first."""
    by_year = plan.group_by == Dimension.START_YEAR
    match plan.analysis:
        case Analysis.TIME_TREND:
            return (V.TIME_SERIES, V.BAR_CHART)
        case Analysis.DISTRIBUTION:
            return (V.TIME_SERIES, V.BAR_CHART) if by_year else (V.BAR_CHART,)
        case Analysis.GEOGRAPHIC:
            return (V.BAR_CHART,)
        case Analysis.COMPARISON:
            if plan.group_by is None:
                return (V.BAR_CHART,)  # one bar per cohort
            return (V.TIME_SERIES, V.GROUPED_BAR_CHART) if by_year else (V.GROUPED_BAR_CHART,)
        case Analysis.NUMERIC_DISTRIBUTION:
            return (V.HISTOGRAM,)
        case Analysis.NETWORK:
            return (V.NETWORK_GRAPH,)
        case Analysis.COUNT:
            return (V.METRIC,)
    raise ValueError(f"no chart rules for analysis {plan.analysis!r}")


def choose_chart(plan: QueryPlan, requested: VizType | None) -> VisualizationRationale:
    allowed = allowed_charts(plan)
    default = allowed[0]
    requested = VizType(requested) if requested is not None else None
    fits = ", ".join(v.value for v in allowed)
    if requested is not None:
        if requested in allowed:
            return VisualizationRationale(chosen=requested, source="request", reason="Chart type requested by the caller.")
        return VisualizationRationale(
            chosen=default,
            source="rule_default",
            suggested=requested,
            reason=f"Requested '{requested.value}' does not fit a {plan.analysis.value} question (fits: {fits}); used '{default.value}'.",
        )
    suggestion = plan.chart_type_suggestion
    if suggestion is not None and suggestion in allowed:
        return VisualizationRationale(chosen=suggestion, source="llm", reason=plan.chart_rationale or "Suggested by the planner.")
    if suggestion is not None:
        return VisualizationRationale(
            chosen=default,
            source="rule_default",
            suggested=suggestion,
            reason=f"Planner suggested '{suggestion.value}', which does not fit a {plan.analysis.value} question (fits: {fits}); used '{default.value}'.",
        )
    return VisualizationRationale(chosen=default, source="rule_default", reason=f"Default chart for a {plan.analysis.value} question.")
