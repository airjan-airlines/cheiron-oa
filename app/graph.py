"""The agent pipeline as an explicit LangGraph state machine (D2).

    plan ──► validate ──► fetch ──► execute ──► END
      ▲         │           │
      └─repair──┘           └──► END (no_data / query rejected)
    plan ──► END (unsupported)      validate ──► END (still invalid after one repair)

Each node is a thin wrapper around a plain function in `planner/`, `ctgov/` or `engine/`,
which are unit-tested without LangGraph. Only the `plan` node talks to the LLM, and it
sees only the question and the caller's fields: trial data never flows back into it.
"""

import asyncio
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.ctgov.client import CTGovClient
from app.ctgov.extract import extract
from app.ctgov.query_builder import build_params, normalize_filters
from app.engine import response_builder
from app.engine.executor import execute
from app.engine.types import FetchedQuery
from app.errors import UpstreamRejectedQuery
from app.planner.llm_planner import Message, OpenAIPlanner, PlannerLLM
from app.planner.llm_schema import QueryPlanLLM
from app.planner.prompts import SYSTEM_PROMPT, repair_message, user_message
from app.planner.validator import finalize, searches
from app.schemas.enums import Analysis
from app.schemas.plan import QueryPlan
from app.schemas.request import VisualizeRequest
from app.schemas.response import VisualizeResponse

MAX_PLAN_ATTEMPTS = 2  # the first draft plus one repair


class PipelineState(TypedDict, total=False):
    request: VisualizeRequest
    messages: list[Message]  # the LLM conversation (question, drafts, validation errors)
    draft: QueryPlanLLM
    attempts: int
    errors: list[str]
    plan: QueryPlan
    notes: list[str]
    queries: list[FetchedQuery]
    data_as_of: str | None
    response: VisualizeResponse


def build_graph(llm: PlannerLLM, client: CTGovClient):
    async def plan_node(state: PipelineState) -> PipelineState:
        messages = state.get("messages") or [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(state["request"])},
        ]
        if state.get("errors"):  # repair turn: show the model its draft and what was wrong
            messages = messages + [
                {"role": "assistant", "content": state["draft"].model_dump_json()},
                {"role": "user", "content": repair_message(state["errors"])},
            ]
        draft = await llm.plan(messages)
        update: PipelineState = {"messages": messages, "draft": draft, "attempts": state.get("attempts", 0) + 1}
        if not draft.is_about_clinical_trials:
            check = finalize(draft, state["request"])
            update["response"] = response_builder.unsupported(check.plan, check.plan.unsupported_reason)
        return update

    def validate_node(state: PipelineState) -> PipelineState:
        check = finalize(state["draft"], state["request"])
        if check.plan is not None:
            return {"plan": check.plan, "errors": [], "notes": check.notes}
        if check.conflicts:  # the caller's own input contradicts itself; re-asking the LLM can't fix that
            message = "The request's structured fields contradict the question: " + " ".join(check.conflicts)
            return {"errors": check.conflicts, "response": response_builder.error("conflicting_constraints", message, check.notes)}
        if state["attempts"] >= MAX_PLAN_ATTEMPTS:
            message = "The question could not be turned into a valid query plan: " + "; ".join(check.errors)
            return {"errors": check.errors, "response": response_builder.error("invalid_plan", message, check.notes)}
        return {"errors": check.errors}

    async def fetch_node(state: PipelineState) -> PipelineState:
        request, plan = state["request"], state["plan"]
        notes = list(state.get("notes", []))
        # A count needs only the API's total; fetch just enough records to cite.
        cap = max(request.max_citations_per_datum, 1) if plan.analysis == Analysis.COUNT else request.max_records
        # (comparison cohorts arrive fully merged and validated from `finalize`)
        specs = []
        for label, filters in searches(plan):
            filters, changes = normalize_filters(filters)
            notes.extend(changes)
            specs.append((label, filters, build_params(filters)))
        try:
            results, data_as_of = await asyncio.gather(
                asyncio.gather(*(client.search(params, cap) for _, _, params in specs)), client.data_timestamp()
            )
        except UpstreamRejectedQuery as exc:
            return {"response": response_builder.error("query_rejected", f"ClinicalTrials.gov rejected the search: {exc}", notes, plan)}
        queries = [
            FetchedQuery(label, filters, params, result.total_count, [extract(s) for s in result.studies])
            for (label, filters, params), result in zip(specs, results)
        ]
        update: PipelineState = {"queries": queries, "notes": notes, "data_as_of": data_as_of}
        # A count of zero is an answer (metric 0), not a failed search.
        if plan.analysis != Analysis.COUNT and all(q.total_matching == 0 for q in queries):
            update["response"] = response_builder.no_data(plan, queries, notes, data_as_of)
        return update

    def execute_node(state: PipelineState) -> PipelineState:
        request, plan, queries = state["request"], state["plan"], state["queries"]
        result = execute(plan, queries, request.chart_type, request.max_citations_per_datum)
        return {"response": response_builder.ok(plan, queries, result, state.get("notes", []), state.get("data_as_of"))}

    def done_or(next_node: str):
        return lambda state: END if state.get("response") else next_node

    def after_validate(state: PipelineState) -> str:
        if state.get("response"):
            return END
        return "plan" if state.get("errors") else "fetch"

    graph = StateGraph(PipelineState)
    graph.add_node("plan", plan_node)
    graph.add_node("validate", validate_node)
    graph.add_node("fetch", fetch_node)
    graph.add_node("execute", execute_node)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", done_or("validate"), ["validate", END])
    graph.add_conditional_edges("validate", after_validate, ["plan", "fetch", END])
    graph.add_conditional_edges("fetch", done_or("execute"), ["execute", END])
    graph.add_edge("execute", END)
    return graph.compile()


_default_graph = None


def _graph():
    global _default_graph
    if _default_graph is None:
        settings = get_settings()
        client = CTGovClient(
            base_url=settings.ctgov_base_url,
            timeout_s=settings.ctgov_timeout_s,
            max_attempts=settings.ctgov_max_retries,
            cache_ttl_s=settings.ctgov_cache_ttl_s,
        )
        _default_graph = build_graph(OpenAIPlanner(settings), client)
    return _default_graph


async def run_pipeline(request: VisualizeRequest) -> VisualizeResponse:
    final = await _graph().ainvoke({"request": request})
    return final["response"]
