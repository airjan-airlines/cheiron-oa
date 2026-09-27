"""Assemble the response envelope for every pipeline outcome."""

from datetime import UTC, datetime

from app.engine.types import ExecutionResult, FetchedQuery
from app.schemas.plan import QueryPlan
from app.schemas.response import AppliedQuery, ErrorInfo, Meta, VisualizeResponse


def applied_queries(queries: list[FetchedQuery]) -> list[AppliedQuery]:
    return [
        AppliedQuery(
            label=q.label,
            filters=q.filters,
            api_params={k: v for k, v in q.api_params.items() if k not in ("fields", "countTotal")},
            total_matching=q.total_matching,
            records_analyzed=len(q.rows),
            truncated=q.truncated,
        )
        for q in queries
    ]


def truncation_notes(queries: list[FetchedQuery]) -> list[str]:
    notes = []
    for q in queries:
        if q.truncated:
            name = f"'{q.label}': " if q.label else ""
            notes.append(
                f"{name}{q.total_matching:,} trials match, but only the first {len(q.rows):,} (in the API's default "
                "order) were analyzed because of the record cap. Bucket counts cover those records only."
            )
    return notes


def ok(
    plan: QueryPlan,
    queries: list[FetchedQuery],
    result: ExecutionResult,
    notes: list[str],
    data_as_of: str | None,
) -> VisualizeResponse:
    return VisualizeResponse(
        status="ok",
        visualization=result.visualization,
        meta=Meta(
            interpretation=result.interpretation,
            queries=applied_queries(queries),
            coverage=result.coverage,
            render=result.render,
            visualization_rationale=result.rationale,
            assumptions=result.assumptions,
            notes=notes + truncation_notes(queries) + result.notes,
            plan=plan,
            data_as_of=data_as_of,
            generated_at=datetime.now(UTC),
        ),
    )


def no_data(plan: QueryPlan, queries: list[FetchedQuery], notes: list[str], data_as_of: str | None) -> VisualizeResponse:
    return VisualizeResponse(
        status="no_data",
        meta=Meta(
            interpretation="No trials in the registry match this search.",
            queries=applied_queries(queries),
            notes=notes + ["Check spelling, or try a broader term (e.g. the generic drug name or a parent condition)."],
            plan=plan,
            data_as_of=data_as_of,
            generated_at=datetime.now(UTC),
        ),
    )


def unsupported(plan: QueryPlan | None, reason: str) -> VisualizeResponse:
    return VisualizeResponse(
        status="unsupported",
        meta=Meta(interpretation=reason, plan=plan, generated_at=datetime.now(UTC)),
    )


def error(code: str, message: str, notes: list[str] | None = None, plan: QueryPlan | None = None) -> VisualizeResponse:
    return VisualizeResponse(
        status="error",
        error=ErrorInfo(code=code, message=message),
        meta=Meta(notes=notes or [], plan=plan, generated_at=datetime.now(UTC)),
    )
