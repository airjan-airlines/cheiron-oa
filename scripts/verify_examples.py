"""Independently audit the saved example outputs against the live registry.

    python -m scripts.verify_examples

For every citation in every example, fetch the cited trial fresh from ClinicalTrials.gov
and check that the excerpt really is the value at the cited field path. Also checks that
each datum's counts agree with its evidence, and that partition charts reconcile with
the API's total. Writes examples/verification.json.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from app.ctgov.client import CTGovClient
from app.ctgov.extract import ABSENT_EXCERPT

EXAMPLES = Path(__file__).parent.parent / "examples" / "outputs"
REPORT = Path(__file__).parent.parent / "examples" / "verification.json"


def resolve(record: dict, path: str) -> Any:
    """Follow `a.b[].c` through a study record. A `[]` segment fans out over a list."""
    values: list[Any] = [record]
    fanned = False
    for part in path.split("."):
        is_list = part.endswith("[]")
        key = part.removesuffix("[]")
        nxt = []
        for v in values:
            item = v.get(key) if isinstance(v, dict) else None
            if is_list:
                nxt.extend(item or [])
            else:
                nxt.append(item)
        fanned = fanned or is_list
        values = nxt
    return values if fanned else values[0]


def excerpt_matches(record: dict, field: str, excerpt: str) -> bool:
    paths = field.split(" + ")
    if " | " in excerpt:  # network edge: one quoted value per endpoint
        parts = [json.loads(p) for p in excerpt.split(" | ")]
        paths = paths * len(parts) if len(paths) == 1 else paths
        return all(part in resolve(record, path) for part, path in zip(parts, paths))
    value = resolve(record, field)
    if excerpt == ABSENT_EXCERPT:
        return value in (None, [])
    if isinstance(value, list) and "[]" in field:
        return excerpt in [str(v) for v in value]
    if isinstance(value, list):
        return json.loads(excerpt) == value
    return str(value) == excerpt


def data_items(viz: dict) -> list[dict]:
    data = viz["data"]
    return data if isinstance(data, list) else data["nodes"] + data["edges"]


async def main() -> None:
    files = sorted(EXAMPLES.glob("*.json"))
    outputs = {f.name: json.loads(f.read_text())["response"] for f in files}
    cited = sorted({c["nct_id"] for r in outputs.values() if r.get("visualization") for d in data_items(r["visualization"]) for c in d["citations"]})

    client = CTGovClient()
    records: dict[str, dict] = {}
    try:
        for i in range(0, len(cited), 100):
            batch = cited[i : i + 100]
            result = await client.search({"filter.ids": ",".join(batch)}, max_records=len(batch))
            records.update({s["protocolSection"]["identificationModule"]["nctId"]: s for s in result.studies})
    finally:
        await client.aclose()

    report: dict[str, Any] = {"data_timestamp_checked": None, "examples": {}}
    failures = 0
    for name, resp in outputs.items():
        if not resp.get("visualization"):
            report["examples"][name] = {"status": resp["status"], "checked": "no visualization"}
            continue
        viz, meta = resp["visualization"], resp["meta"]
        items = data_items(viz)
        problems: list[str] = []
        n_citations = 0
        for d in items:
            if d["citation_count"] != len(d["supporting_nct_ids"]):
                problems.append(f"citation_count != len(supporting_nct_ids) in {d}")
            for c in d["citations"]:
                n_citations += 1
                record = records.get(c["nct_id"])
                if record is None:
                    problems.append(f"{c['nct_id']} not found in the registry")
                elif not excerpt_matches(record, c["field"], c["excerpt"]):
                    problems.append(f"{c['nct_id']}: excerpt {c['excerpt']!r} not found at {c['field']}")
        cov = meta.get("coverage") or {}
        queries = meta["queries"]
        reconciles = None
        if cov.get("groupby_semantics") == "partition" and len(queries) == 1 and not queries[0]["truncated"] and not cov.get("buckets_not_shown"):
            reconciles = cov["bucket_sum"] + cov["unclassified_count"] == queries[0]["total_matching"]
            if not reconciles:
                problems.append("partition buckets + unclassified != total_matching")
        failures += len(problems)
        report["examples"][name] = {
            "status": resp["status"],
            "type": viz["type"],
            "citations_checked": n_citations,
            "citations_verified": n_citations - sum("excerpt" in p or "not found" in p for p in problems),
            "partition_reconciles_with_api_total": reconciles,
            "problems": problems,
        }
        print(f"{name}: {n_citations} citations checked, {len(problems)} problems, reconciles={reconciles}")

    report["cited_trials_fetched"] = len(records)
    report["total_problems"] = failures
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n{len(records)} cited trials re-fetched; {failures} problems. Report: {REPORT}")


if __name__ == "__main__":
    asyncio.run(main())
