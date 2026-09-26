"""HTTP layer: validation, routing and error-to-status mapping. No business logic lives here."""

from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.errors import PlannerUnavailable, UpstreamUnavailable
from app.graph import run_pipeline
from app.schemas.request import VisualizeRequest
from app.schemas.response import ErrorInfo, VisualizeResponse

Pipeline = Callable[[VisualizeRequest], Awaitable[VisualizeResponse]]

app = FastAPI(
    title="ClinicalTrials.gov Query-to-Visualization API",
    version="1.0.0",
    description=(
        "Turns a natural-language question about clinical trials into a structured, "
        "citation-backed visualization spec. The LLM only plans the query; every number "
        "is computed from ClinicalTrials.gov records."
    ),
)


def get_pipeline() -> Pipeline:
    return run_pipeline


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/v1/visualize",
    response_model=VisualizeResponse,
    response_model_exclude_none=True,
    responses={
        422: {"description": "Request failed validation (unknown field, bad enum, start_year > end_year, ...)."},
        502: {"model": VisualizeResponse, "description": "ClinicalTrials.gov or the LLM provider was unreachable."},
    },
)
async def visualize(
    request: VisualizeRequest, pipeline: Annotated[Pipeline, Depends(get_pipeline)]
) -> VisualizeResponse:
    """Every pipeline outcome (`ok`, `no_data`, `unsupported`, `error`) returns HTTP 200 with a `status` field."""
    return await pipeline(request)


def _unavailable(code: str, message: str) -> JSONResponse:
    body = VisualizeResponse(status="error", error=ErrorInfo(code=code, message=message))
    return JSONResponse(status_code=502, content=body.model_dump(mode="json", exclude_none=True))


@app.exception_handler(UpstreamUnavailable)
async def _upstream_unavailable(_: Request, exc: UpstreamUnavailable) -> JSONResponse:
    return _unavailable("upstream_unavailable", str(exc) or "ClinicalTrials.gov is unavailable.")


@app.exception_handler(PlannerUnavailable)
async def _planner_unavailable(_: Request, exc: PlannerUnavailable) -> JSONResponse:
    return _unavailable("planner_unavailable", str(exc) or "The LLM planner is unavailable.")
