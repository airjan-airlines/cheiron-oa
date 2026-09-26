"""Record live ClinicalTrials.gov responses as offline test fixtures.

Uses the service's own query builder and client, so fixtures contain exactly what the
pipeline would fetch. Re-run after a registry refresh:

    python -m scripts.record_fixtures
"""

import asyncio
import json
from pathlib import Path

from app.ctgov.client import CTGovClient
from app.ctgov.query_builder import build_params
from app.schemas.plan import Filters

OUT = Path(__file__).parent.parent / "tests" / "fixtures"

FIXTURES: dict[str, Filters] = {
    # 74 trials: phase distribution, Korean example
    "korea_gastric_recruiting": Filters(condition="gastric cancer", location="South Korea", statuses=["RECRUITING"]),
    # ~290 trials: comparison cohort, time trend
    "tirzepatide": Filters(intervention="tirzepatide"),
    # ~238 trials: sponsor <-> drug network
    "kras_nsclc": Filters(condition="KRAS non-small cell lung cancer"),
}


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    client = CTGovClient()
    try:
        stamp = await client.data_timestamp()
        for name, filters in FIXTURES.items():
            params = build_params(filters)
            result = await client.search(params, max_records=5000)
            payload = {
                "filters": filters.model_dump(mode="json", exclude_defaults=True),
                "params": params,
                "data_timestamp": stamp,
                "totalCount": result.total_count,
                "studies": result.studies,
            }
            path = OUT / f"{name}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            print(f"{name}: {len(result.studies)}/{result.total_count} studies -> {path} ({path.stat().st_size // 1024} KB)")
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
