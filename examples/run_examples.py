"""Run the example queries through the real API and save the actual responses.

    python -m examples.run_examples

Requires OPENAI_API_KEY and network access. Each output file holds the request and the
exact JSON the API returned.
"""

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

OUT = Path(__file__).parent / "outputs"

EXAMPLES: list[tuple[str, dict]] = [
    ("01_time_trend_pembrolizumab", {"query": "How has the number of trials for this drug changed per year since 2015?", "drug_name": "Pembrolizumab"}),
    ("02_geographic_breast_cancer", {"query": "Which countries have the most recruiting trials for breast cancer?", "top_n": 10}),
    ("03_comparison_semaglutide_tirzepatide", {"query": "Compare phases for trials involving Semaglutide vs Tirzepatide."}),
    ("04_network_kras_nsclc", {"query": "Show a network of sponsors and drugs for KRAS-mutant non-small cell lung cancer trials."}),
    ("05_korean_gastric_cancer_phases", {"query": "한국에서 모집 중인 위암 임상시험은 단계별로 어떻게 분포되어 있나요?"}),
    # Extra coverage: remaining chart types and an out-of-scope question.
    ("06_extra_drug_cooccurrence_myeloma", {"query": "Which drugs frequently co-occur in combination studies for multiple myeloma trials started since 2022?"}),
    ("07_extra_enrollment_histogram_psoriasis", {"query": "What is the distribution of enrollment sizes for completed Phase 3 psoriasis trials?"}),
    ("08_extra_count_pfizer", {"query": "How many Phase 3 trials is Pfizer running that are currently recruiting?"}),
    ("09_extra_unsupported", {"query": "What's the weather in Seoul tomorrow?"}),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with TestClient(app) as client:
        for name, body in EXAMPLES:
            started = time.monotonic()
            resp = client.post("/v1/visualize", json=body)
            elapsed = time.monotonic() - started
            payload = resp.json()
            (OUT / f"{name}.json").write_text(
                json.dumps({"request": body, "http_status": resp.status_code, "response": payload}, ensure_ascii=False, indent=2)
            )
            viz = payload.get("visualization") or {}
            print(f"{name}: HTTP {resp.status_code} {payload['status']} {viz.get('type', '-')} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
