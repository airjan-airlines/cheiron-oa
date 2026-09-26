# ClinicalTrials.gov Query-to-Visualization Agent

A backend that turns a natural-language question about clinical trials into a **structured,
citation-backed visualization spec**, using live data from the
[ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-api/api).

**Core rule: the LLM plans, code computes.** The model's only job is to turn the question into a
typed, validated query plan. It never sees trial data and never produces a number. Fetching,
counting, chart selection and citations are deterministic Python, so every value in a response can be
traced to specific trial records.

> Status: under active development. Sections marked *(pending)* are filled in as each part lands;
> the commit history records the build step by step. Design notes: [`docs/implementation-plan.md`](docs/implementation-plan.md).

---

## 1. How to run

Requires Python 3.12+ and an OpenAI API key.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then set OPENAI_API_KEY
uvicorn app.main:app --reload
```

Interactive API docs: <http://localhost:8000/docs>. Machine-readable schema: `/openapi.json`.

```bash
curl -s -X POST localhost:8000/v1/visualize \
  -H 'content-type: application/json' \
  -d '{"query": "How has the number of trials for this drug changed over time?", "drug_name": "Pembrolizumab"}'
```

| Env var | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | – | Required for planning. |
| `OPENAI_MODEL` | `gpt-4.1-mini` | Must be one of the models in [`docs/openai-rules.md`](docs/openai-rules.md); anything else fails at startup. |

Tests run offline (no API key, no network):

```bash
pytest            # unit + contract tests
pytest -m live    # opt-in: calls the real ClinicalTrials.gov and OpenAI APIs
```

---

## 2. Request schema

`POST /v1/visualize` with a JSON body. Only `query` is required.

| Field | Type | Required | Validation | Effect |
|---|---|---|---|---|
| `query` | string | **yes** | 1–1000 chars after trimming | The question. Any language. |
| `drug_name` | string | no | 1–200 chars | Filters by intervention (`query.intr`). |
| `condition` | string | no | 1–200 chars | Filters by condition (`query.cond`). |
| `sponsor` | string | no | 1–200 chars | Filters by **lead** sponsor (`query.lead`). |
| `country` | string | no | 1–100 chars | Filters by site location (`query.locn`). Common aliases are normalized, e.g. "Korea" → "South Korea". |
| `trial_phases` | string[] | no | each one of `EARLY_PHASE1`, `PHASE1`, `PHASE2`, `PHASE3`, `PHASE4`, `NA` | Restricts to these phases. |
| `statuses` | string[] | no | each a registry status, e.g. `RECRUITING`, `COMPLETED` | Restricts to these overall statuses. |
| `start_year` | int | no | 1900–2100 | Earliest trial start year (inclusive). |
| `end_year` | int | no | 1900–2100, ≥ `start_year` | Latest trial start year (inclusive). |
| `chart_type` | string | no | a visualization `type` (§3) | Preferred chart. Used only if it fits the question; otherwise overridden and explained in `meta.visualization_rationale`. |
| `top_n` | int | no | 1–50, default 15 | Maximum categories or nodes shown. |
| `max_records` | int | no | 100–5000, default 5000 | Maximum trial records fetched per search. |
| `max_citations_per_datum` | int | no | 0–20, default 3 | Citation excerpts attached to each datum. |

Rules:
- **Unknown fields are rejected with 422.** A misspelled filter that silently does nothing is worse than an error.
- **Structured fields override the LLM.** If `drug_name` is set, it wins over whatever drug the model
  reads from `query`; any conflict is recorded in `meta.notes`.

---

## 3. Response schema

Every response has the same envelope, and every pipeline outcome returns HTTP 200:

```jsonc
{
  "schema_version": "1.0",
  "status": "ok",            // ok | no_data | unsupported | error
  "visualization": { ... },  // present only when status is "ok"
  "meta": { ... },           // always present
  "error": { "code": "...", "message": "..." }  // present only when status is "error"
}
```

| `status` | Meaning |
|---|---|
| `ok` | A visualization was produced. |
| `no_data` | The search ran but matched no trials; `meta.queries` shows exactly what was searched. |
| `unsupported` | The question is not about clinical trials. |
| `error` | The question could not be planned into a valid query. No guessed chart is returned. |

| HTTP | When |
|---|---|
| 200 | Any of the statuses above. |
| 422 | The request body failed validation. |
| 502 | ClinicalTrials.gov or the LLM provider was unreachable after retries (body is the envelope with `status: "error"`). |

### 3.1 `visualization`

`type` selects the shape. Every `encoding.*.field` names a key that exists in **every** `data` row;
the server validates this before responding, so a renderer needs only `encoding` + `data`.

| `type` | `encoding` channels | `data` |
|---|---|---|
| `bar_chart` | `x`, `y` | rows |
| `grouped_bar_chart` | `x`, `y`, `series` | rows in long format: one per (x, series) pair, zeros filled in |
| `time_series` | `x` (temporal), `y`, optional `series` | rows, one per year (per series); empty years filled with 0 |
| `histogram` | `x` (bin start), `x2` (bin end), `y` | rows, one per bin |
| `metric` | `value` | a single row: the question needs a number, not a chart |
| `network_graph` | `node` {`id`, `label`, `size`, `color`}, `edge` {`source`, `target`, `weight`} | `{ "nodes": [...], "edges": [...] }` |

A channel is `{field, type, label, label_field?}` where `type` is `nominal`, `ordinal`,
`quantitative` or `temporal`.

**Raw value + display label.** Categorical rows carry the registry's raw value and a display label
side by side; the channel's `label_field` names the label key. Plot the raw value, print the label:

```json
{ "phase": "PHASE2", "phase_label": "Phase 2", "trial_count": 4581 }
```

### 3.2 Deep citations

Every datum (bar, time bucket, histogram bin, node, edge) carries its evidence:

```jsonc
{
  "phase": "PHASE3", "phase_label": "Phase 3", "trial_count": 41,
  "supporting_nct_ids": ["NCT01234567", "..."],   // every contributing trial, never truncated
  "citation_count": 41,
  "citations": [                                   // first max_citations_per_datum trials
    {
      "nct_id": "NCT01234567",
      "field": "protocolSection.designModule.phases",
      "excerpt": "[\"PHASE3\"]",                     // exact value from the API response
      "title": "A Study of Pembrolizumab in ...",
      "url": "https://clinicaltrials.gov/study/NCT01234567"
    }
  ]
}
```

`excerpt` is the exact field value that placed the trial in this datum (lists are JSON-encoded). For a
derived bucket such as "Phase 1/Phase 2", the excerpt is the raw `["PHASE1", "PHASE2"]`.

### 3.3 `meta`

| Field | Meaning |
|---|---|
| `interpretation` | What was computed, in words. Generated from the plan, never an LLM statement about results. |
| `queries[]` | Each ClinicalTrials.gov search that fed the chart (one per cohort in a comparison): `label`, normalized `filters`, the exact `api_params` sent, `total_matching` (the API's own count), `records_analyzed`, `truncated`. |
| `coverage` | How the numbers relate to the trials: `groupby_semantics` (`partition` = each trial in exactly one bucket, so buckets sum to the total; `overlapping` = a trial can be in several, e.g. multi-country trials), `bucket_sum`, `unclassified_count`, `overlap_note`, `excluded[]` (trials left out, by reason), `buckets_not_shown` (counted but beyond `top_n`). |
| `render` | Drawing hints: `sort` {`field`, `order`}, `time_granularity`, `units` per channel, `grouping` (the series key), `pruning` (for networks). |
| `visualization_rationale` | The chosen `type`, where it came from (`request`, `llm` or `rule_default`), any overridden suggestion, and why. |
| `assumptions[]` | Interpretation choices a reader would otherwise have to guess. |
| `notes[]` | Overrides, fallbacks, truncation and other things worth knowing about this run. |
| `plan` | The validated query plan that was executed. |
| `source`, `data_as_of`, `generated_at` | Provenance: the API and its data timestamp. |

---

## 4. Example runs *(pending)*

## 5. How it works *(pending)*

## 6. Design decisions and tradeoffs *(pending)*

## 7. Limitations and what I would improve *(pending)*

## 8. AI tools, validation, and what was deliberate *(pending)*
