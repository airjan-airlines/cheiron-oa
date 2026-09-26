from fastapi.testclient import TestClient

from app.errors import UpstreamUnavailable
from app.main import app, get_pipeline
from app.schemas.response import VisualizeResponse

client = TestClient(app)


def _use_pipeline(fn):
    app.dependency_overrides[get_pipeline] = lambda: fn


def teardown_function():
    app.dependency_overrides.clear()


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_invalid_request_returns_422():
    resp = client.post("/v1/visualize", json={"query": "trials", "drugname": "x"})
    assert resp.status_code == 422


def test_pipeline_outcome_is_returned_with_200():
    async def fake(request):
        return VisualizeResponse(status="unsupported")

    _use_pipeline(fake)
    resp = client.post("/v1/visualize", json={"query": "What's the weather?"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "unsupported"


def test_upstream_outage_returns_502_with_envelope():
    async def failing(request):
        raise UpstreamUnavailable("ClinicalTrials.gov timed out after 3 attempts")

    _use_pipeline(failing)
    resp = client.post("/v1/visualize", json={"query": "trials by phase"})
    assert resp.status_code == 502
    body = resp.json()
    assert body["status"] == "error"
    assert body["error"]["code"] == "upstream_unavailable"
