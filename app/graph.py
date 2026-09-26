"""Pipeline orchestration. Wired with LangGraph in build step 4; placeholder until then."""

from app.schemas.request import VisualizeRequest
from app.schemas.response import ErrorInfo, VisualizeResponse


async def run_pipeline(request: VisualizeRequest) -> VisualizeResponse:
    return VisualizeResponse(
        status="error",
        error=ErrorInfo(code="not_implemented", message="The planning pipeline is not wired up yet."),
    )
