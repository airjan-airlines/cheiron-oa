# Implementation Plan — ClinicalTrials.gov Query-to-Visualization Agent (Cheiron OA)

## Context

Take-home for Cheiron (AI-native OS for drug programs; stack: Python/FastAPI/Postgres/LangGraph/OpenAI; values grounded, typed, citable agent outputs). Build a backend that turns a natural-language clinical-trial question (+ optional structured fields) into a **structured visualization spec** backed by ClinicalTrials.gov API v2 data, with spec-level deep citations. Due tomorrow (~24h).

Workspace (greenfield, not a git repo): `clinicaltrials-take-home-assignment.md` (the OA), `api_reference.md` (corrected against the live API: no `filter.phase` → use `filter.advanced=AREA[Phase]PHASE3`; `nextPageToken`; `countTotal=true`; plain-text errors), `openai-rules.md` (allowed models), `cheiron-extensions.md` (research), `.env` (`OPENAI_API_KEY`).

Grading: System Design 35%, AI/Agent Design 20%, Code Quality 20%, Query/Viz Coverage 15%, I/O Design 10%, bonus citations. Plan reviewed against the OA by a subagent; all must-fix findings incorporated below.

---

## Key Decisions

**D1. LLM plans, code computes.** The LLM emits only a typed `QueryPlan`. Deterministic Python fetches and aggregates. The output still carries every number the frontend needs. This avoids the "hallucination-prone step" the rubric warns about, and citations fall out of aggregation for free.

**D2. Orchestration: a thin LangGraph layer over plain node functions.** `app/graph.py` wires a `StateGraph` over a typed `PipelineState`: request, plan, validation_errors, repair_attempts, trials, response, status.
- Nodes (one module each, unit-tested without LangGraph):
  - `plan` (`planner/llm_planner.py`)
  - `validate` (`planner/validator.py`)
  - `fetch` (`ctgov/client.py`)
  - `execute` (`engine/executor.py`)
  - `respond` (`engine/response_builder.py`)
- Conditional edges:
  - `plan` → END(`unsupported`) when the query isn't about trials
  - `validate` → `plan` (one repair retry) | `fetch` | END(`error`)
  - `fetch` → END(`no_data`) | `execute`
- LangGraph is MIT-licensed and free. No LangSmith and no LangChain model wrappers; nodes call the OpenAI SDK directly.
- Fallback if it eats time: chain the same functions in `graph.py` without LangGraph (~10 min).

**D3. One generic QueryPlan, no per-question templates.**
```
QueryPlan {
  analysis: distribution | time_trend | comparison | geographic | network | numeric_distribution | count
  group_by: Dimension   # phase, status, start_year, sponsor, sponsor_class, country,
                        # intervention, intervention_type, condition
  metric: trial_count | enrollment
  filters: Filters      # condition, intervention, sponsor, location, phases[], statuses[],
                        # start_year_min, start_year_max
  compare: [{label, filters}]            # 2–4 cohorts (Drug A vs B; condition X vs Y)
  network: {source: Dimension, target: Dimension}
  top_n: int
  chart_type_suggestion, chart_rationale
}
```
All OA appendix queries are expressible. For example, "sponsor categories across two conditions" is `comparison` + `group_by=sponsor_class` + 2 cohorts, and "recruiting by country" is `geographic` + a `statuses` filter.

**D4. Chart type: the LLM suggests, rules validate.** `engine/chart_rules.py` maps `(analysis, dimension kind)` → {allowed types, default}:

| analysis | allowed types | default |
|---|---|---|
| `time_trend` | `time_series`, `bar_chart` | `time_series` |
| `distribution` (categorical) | `bar_chart` | `bar_chart` |
| `geographic` | `bar_chart` (sorted desc, `country` field ready for a map) | `bar_chart` |
| `comparison` | `grouped_bar_chart`, `time_series` (multi-series, when grouped by year) | per dimension |
| `numeric_distribution` | `histogram` | `histogram` |
| `network` | `network_graph` | `network_graph` |
| `count` | `metric` (no chart needed, OA §1.3) | `metric` |

Precedence is request `chart_type` > LLM suggestion > rule default, and every choice is validated. Overrides are recorded in `meta.visualization_rationale`.

**D5. Plan validation with one repair retry.** Uses OpenAI Structured Outputs with an LLM-facing model (`QueryPlanLLM`) that has only nullable fields and no numeric or length constraints, since strict mode doesn't honor those. That model is converted to the internal `QueryPlan`, which holds the real constraints:
- enums are valid
- the dimension is compatible with the analysis
- the year range is sane
- `compare` has 2–4 cohorts
- `network` has 2 dimensions

On failure the errors go back to the LLM once. If it still fails, the response is `status:error` and never a guessed chart.

**D6. Structured fields win over LLM extraction.** Explicit request fields override LLM-extracted values. Conflicts are recorded in `meta.notes`.

**D7. Fetch client-side, then aggregate.**
- Server-side filters: `query.cond/intr/locn`, **`query.lead`** for sponsor (lead sponsor only; `query.spons` also matches collaborators — Pfizer: 3,880 vs 6,087, verified live — and would disagree with lead-sponsor grouping; disclosed in `meta.assumptions`), `filter.overallStatus`, `filter.advanced` (phase and start-date range).
- `fields=` limited to about 15 verified fields (NCTId, BriefTitle, Phase, OverallStatus, StartDate, PrimaryCompletionDate, LeadSponsorName, LeadSponsorClass, InterventionName, InterventionType, InterventionMeshTerm, Condition, ConditionMeshTerm, LocationCountry, EnrollmentCount).
- `pageSize=1000`, follow `nextPageToken`, `countTotal=true`.
- `max_records` defaults to 5000.
- If `total_matching > max_records`, set `meta.truncated=true` and add a note that the sample is the API's default order and may be biased.
- `httpx` with timeout, exponential backoff on 429/5xx, and an in-memory TTL cache. (On this machine, Python's stdlib `urllib` fails SSL verification but the `certifi` bundle works; `httpx` uses `certifi` by default, so never use bare `urllib`.)
- `pageToken` values must be URL-encoded (a raw token silently returned an empty body in testing).
- Rejected alternative: one `countTotal` call per bucket. It can't produce citations or networks.

**D8. Rules for multi-valued fields** (stated in `meta.assumptions`):
- **Phase:** `["PHASE1","PHASE2"]` becomes one "Phase 1/Phase 2" bucket, so no double counting. **An explicit `NA` phase ("Not applicable", 237,522 trials registry-wide) and a missing phase field ("Not specified", 143,294) are separate buckets, never merged.**
- **Countries, interventions, conditions:** a trial is counted once per distinct value, so bars can sum to more than the trial count. The y-axis label says so.
- **Start year:** parsed from `YYYY[-MM[-DD]]`. Trials without one are excluded and counted in `meta.data_quality.excluded`.
- **Interventions:** prefer MeSH terms, fall back to raw names. Networks keep only `DRUG` and `BIOLOGICAL` types and drop a small stoplist (Placebo, Standard of care…).
- **Sponsors:** light normalization of names.
- **Countries:** a small alias map runs after planning, because the registry uses its own spellings: `query.locn="Korea, Republic of"` returns 0 while "South Korea" returns 74 recruiting gastric-cancer trials (verified). Aliases: Korea / Republic of Korea / Korea, Republic of → South Korea; USA / US / America → United States; UK → United Kingdom.
- **Comparison cohorts:** a trial that matches both cohorts is counted in both (assumption noted).

**D9. Visualization spec: a custom typed discriminated union.**
- Envelope follows the OA example shape: `visualization {type, title, encoding, data}` + `meta`.
- **Data** is always tidy (long) rows. Missing x×series combinations and missing years are filled with 0.
- **Encoding** channels are `{field, type (nominal|ordinal|quantitative|temporal), label}` for `x`, `y`, and `series` (grouped and multi-series charts).
- **`network_graph`:**
  - `data = {nodes:[{id,label,type,size,citations…}], edges:[{source,target,weight,citations…}]}`
  - `encoding = {node:{id,label,size,color:"type"}, edge:{source,target,weight}}`
- **`histogram`:** rows are `{bin_start, bin_end, count}`. The bin width goes in meta.
- **`metric`:** `{value, label}`.
- **`meta.render` tells the frontend how to draw it:** `{sort:{field,order}, time_granularity:"year"|null, units:{y:"trials"}, grouping:{field}|null, pruning:{top_n, min_edge_weight}|null}`.
- The JSON Schema is auto-published at `/openapi.json`.

**D10. Deep citations at the spec's level, with complete traceability.**
- Every datum (bar, time bucket, bin, node, edge) carries:
  - `supporting_nct_ids`: the **full** list, so every contributing record is referenced (OA §5)
  - `citations: [{nct_id, field, excerpt, title}]`, capped at `max_citations_per_datum` (default 3)
  - `citation_count`
- Excerpts are the **raw** API value, e.g. `["PHASE1","PHASE2"]` for a derived "Phase 1/Phase 2" bucket, or `"2017-10-03"`.
- Edge excerpts quote both endpoints, e.g. `"Pembrolizumab" | "Carboplatin"`.
- Citation excerpts are capped because payload size grows with trials × data points; the README explains this.

**D11. Response envelope and HTTP semantics.**
- `status: ok | no_data | unsupported | error` is always returned with `meta`, which holds interpretation, filters_applied, assumptions, notes, data_quality, visualization_rationale, render, records_analyzed, total_matching, truncated, source, data_as_of, and the echoed plan.
- HTTP status codes:
  - 422: request validation failed
  - 200: every envelope status
  - 502: ClinicalTrials.gov is unreachable after retries

**D12. LLM: OpenAI with Structured Outputs.**
- Only models from `docs/openai-rules.md` are allowed. Default `OPENAI_MODEL=gpt-4.1-mini`: it isn't a reasoning model (low latency), supports Structured Outputs and `temperature=0`, and planning is a small extraction task.
- `config.py` checks the model against the allowlist at startup. gpt-5* models also work, but their calls leave out `temperature`.
- The `openai` SDK version is pinned, and calls use `client.chat.completions.parse(response_format=QueryPlanLLM)`.
- The key is read from the existing `.env`. Tests stub the client, so they run offline.

**D13. Korean.** Queries in any language are accepted, and entities are normalized to English for the API. One Korean example run is included. There's no `response_language` field.

**D14. No frontend demo.** Example JSON plus the `/docs` OpenAPI page cover it. A renderer is listed as future work.

---

## Additions after reviewing prior art

These came from studying how other implementations of the same problem handled the ClinicalTrials.gov API's edge cases. No code was reused.

**D15. `groupby_semantics` in `meta.coverage`.** Every response says whether its buckets are a `partition` (each trial in exactly one bucket, so bars sum to the total: phase with combined buckets, status, sponsor class, start year) or `overlapping` (a trial can appear in several bars: country, intervention, condition). It also includes `bucket_sum`, `unclassified_count` (e.g. trials with no start date) and a one-line `overlap_note`. Share or percentage fields are only emitted under `partition`.

**D16. Raw key plus human label on every categorical row.** For example `{"phase": "PHASE2", "phase_label": "Phase 2", "trial_count": 4581}`. The renderer plots the key and prints the label, and citations quote the raw key.

**D17. Unknown request fields rejected.** `VisualizeRequest` uses `extra="forbid"`, so a misspelled filter returns 422 instead of being silently ignored.

**D18. Provable LLM data isolation.** `test_no_trial_data_reaches_llm` captures every message sent to the (stubbed) OpenAI client across a full run and asserts none contains an NCT ID or any fetched field value. The README links this test next to the claim.

**D19. Commit history as the build narrative.** One commit per build step, and the message says *why*, not just what (e.g. "Split NA and missing phase buckets: 237k vs 143k trials, merging them hides a real category"). OA §8 rewards "evidence of thoughtful construction, testing, and iteration."

**D20. Exact-count mode for large cohorts (OPTIONAL, ~2h; decided: build only after step 5, once the core, saved examples and README are done).** When `total_matching > max_records` and the group-by dimension is closed (phase, status, sponsor class, intervention type, start year within a bounded range), send one `countTotal=true&pageSize=K` request per bucket with that bucket's filter added (`filter.advanced=AREA[Phase]PHASE3`). Each response returns the exact count *and* K citation records in the same call. Combined phase buckets use `AND` queries (e.g. `AREA[Phase]PHASE1 AND AREA[Phase]PHASE2`), and single-phase buckets subtract them. Verified live: for breast cancer (16,853 trials), 9 queries give exact partition buckets summing to exactly 16,853. Open dimensions (country, sponsor name, drug) keep fetch mode with truncation disclosed. `meta.coverage.aggregation_mode` is `records` or `bucket_counts`.

---

## Request Schema (`VisualizeRequest`, OA §3.1)

| field | type | req | validation | maps to |
|---|---|---|---|---|
| `query` | str | ✅ | 1–1000 chars, non-blank | LLM planner |
| `drug_name` | str | – | 1–200 chars | `filters.intervention` |
| `condition` | str | – | 1–200 chars | `filters.condition` |
| `sponsor` | str | – | 1–200 chars | `filters.sponsor` |
| `country` | str | – | 1–100 chars | `filters.location` |
| `trial_phases` | list[Phase enum] | – | EARLY_PHASE1, PHASE1–4, NA | `filters.phases` |
| `statuses` | list[Status enum] | – | API status values | `filters.statuses` |
| `start_year` / `end_year` | int | – | 1900–2100, start ≤ end (registry has trials from the 1970s) | `filters.start_year_min/max` |
| `chart_type` | VizType enum | – | must be compatible (D4) | chart override |
| `top_n` | int | – | 1–50, default 15 | plan `top_n` |
| `max_records` | int | – | 100–5000, default 5000 | fetch cap |
| `max_citations_per_datum` | int | – | 0–20, default 3 | citation cap |

Unknown fields → 422 (D17). `country` goes through the alias map (D8). `sponsor` matches the lead sponsor only (D7).

---

## Project Layout
```
cheiron-oa/
  app/
    main.py                 # FastAPI: POST /v1/visualize, GET /health
    config.py               # env settings, model allowlist
    graph.py                # LangGraph StateGraph (D2)
    schemas/  request.py · plan.py · response.py
    planner/  llm_planner.py · prompts.py · validator.py
    ctgov/    client.py · query_builder.py · extract.py (record → TrialRow)
    engine/   executor.py · aggregate.py · network.py · chart_rules.py · response_builder.py
  examples/  run_examples.py · outputs/*.json
  tests/     test_extract · test_aggregate · test_network · test_chart_rules · test_validator
             · test_graph (stubbed LLM + recorded fixtures) · test_live_smoke (marked, opt-in)
             · fixtures/*.json
  docs/      clinicaltrials-take-home-assignment.md · api_reference.md · openai-rules.md
             · cheiron-extensions.md · implementation-plan.md
  README.md · requirements.txt · .env.example · .gitignore · make_submission.sh
  .env       # existing; excluded from the zip
```

## Build Order (time-boxed)
0. **Docs reorganization (~5 min, first action after approval).**
   - `mkdir docs` and move the four .md files into it.
   - Save this plan as `docs/implementation-plan.md`.
   - `.env` stays at the root.
   - Add `.gitignore`.
1. **Skeleton and schemas (~2.5h).** Request, plan and response models; FastAPI endpoint; config; start README (schemas section).
2. **CT.gov client and extract (~2h).** Pagination, fields, retries, cache. `TrialRow` normalization. Record fixtures with live calls and test against them.
3. **Engine (~4h).** Aggregate (count_by, year buckets with 0-fill, cohorts, bins), chart rules, network builder, citations, response builder. Unit tests on fixtures (no LLM).
4. **Planner and graph (~3h).** Prompt, structured output, validator with repair, LangGraph wiring. Stubbed-LLM graph tests covering ok, repair, error, unsupported and no_data.
5. **Examples (~1h, run right after step 4 in case the API is flaky).** Five live queries, with the actual JSON saved.
6. **README (~2h total, written as I go).** Each section gets added as its component lands.
7. **Submission (~10 min).** `make_submission.sh`:
   - builds the zip with `-x '.env' '.venv/*' '*__pycache__*' '.pytest_cache/*' 'docs/cheiron-extensions.md'`; the implementation plan stays in the zip as evidence of deliberate design for OA §8
   - `unzip -l` confirms `.env` is absent

If time runs short, cut in this order: stretch types (`scatter`/`table`), then LangGraph (fall back to plain chaining).

## Example Runs (all under the record cap; actual outputs go in `examples/outputs/`)

| # | query | expected chart |
|---|---|---|
| 1 | "How has the number of trials for Pembrolizumab changed per year since 2015?" (~3k trials) | `time_series` |
| 2 | "How are Alzheimer's disease trials distributed across phases?" (check the count; add a recruiting/status filter if it's over the cap) | `bar_chart` |
| 3 | "Compare phases for trials involving Semaglutide vs Tirzepatide." | `grouped_bar_chart` |
| 4 | "Show a network of sponsors and drugs for KRAS-mutant non-small cell lung cancer trials." | `network_graph` |
| 5 | "한국에서 모집 중인 위암 임상시험은 단계별로 어떻게 분포되어 있나요?" ("How are recruiting gastric cancer trials in Korea distributed across phases?") | `bar_chart` |

## README (OA §6 and §8)
1. Overview and architecture diagram (the graph nodes and edges)
2. **How to run:**
   - install: `pip install -r requirements.txt`
   - configure: `.env` from `.env.example` (key, model from the allowlist)
   - start: `uvicorn app.main:app`
   - a curl example
   - run examples and tests
3. **Request schema:** the table above
4. **Response schema:** envelope, each viz `type` (encoding, data shape, example), citations, `meta` and `meta.render`, statuses and HTTP codes, and a pointer to `/openapi.json`
5. **Design decisions and tradeoffs:** D1–D14, condensed
6. **Limitations and improvements:**
   - noise from `query.intr` full-text matching
   - the record cap and sampling bias
   - no MeSH synonym expansion
   - no persistent cache
   - citation verification
   - competitive-intelligence views
   - an eval harness
   - a renderer
7. **Example runs:** 5 queries with actual JSON (abbreviated inline, full files linked)
8. **AI tools and integrity:**
   - tools used (Claude Code, OpenAI models)
   - how correctness was validated (fixture tests; bucket totals checked against `totalCount` when not truncated, including "Not specified"; manual NCT spot checks)
   - what was deliberately designed vs generated and adapted

## Verification
- `pytest`: offline unit and graph tests on recorded fixtures with a stubbed LLM, including the failure paths.
- Live check: run `uvicorn`, then `curl -X POST localhost:8000/v1/visualize` for each example.
  - For single-valued dimensions with `truncated=false`, the sum of the counts should equal `meta.total_matching`.
  - Open 2–3 cited NCT IDs on clinicaltrials.gov to confirm the excerpts.
- Every example output is checked with `VisualizeResponse.model_validate`.
- `make_submission.sh`, then `unzip -l`: no `.env`, no `.venv`.
