"""CTGovClient against a fake transport: paging, retries, errors, cache."""

import asyncio

import httpx
import pytest

from app.ctgov.client import CTGovClient
from app.errors import UpstreamRejectedQuery, UpstreamUnavailable


def _study(i: int) -> dict:
    return {"protocolSection": {"identificationModule": {"nctId": f"NCT{i:08d}"}}}


class FakeRegistry:
    """Serves `total` studies in pages; can fail the first N calls."""

    def __init__(self, total: int, fail_first: int = 0, fail_status: int = 503):
        self.total = total
        self.fail_first = fail_first
        self.fail_status = fail_status
        self.calls: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if len(self.calls) <= self.fail_first:
            return httpx.Response(self.fail_status, text="try later")
        if request.url.path.endswith("/version"):
            return httpx.Response(200, json={"apiVersion": "2.0.5", "dataTimestamp": "2026-09-25T09:00:04"})
        size = int(request.url.params["pageSize"])
        start = int(request.url.params.get("pageToken", "0"))
        end = min(start + size, self.total)
        body = {"studies": [_study(i) for i in range(start, end)]}
        if start == 0:
            body["totalCount"] = self.total
        if end < self.total:
            body["nextPageToken"] = str(end)
        return httpx.Response(200, json=body)


def _client(registry, **kwargs) -> tuple[CTGovClient, list[float]]:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    return CTGovClient(transport=httpx.MockTransport(registry), sleep=fake_sleep, **kwargs), sleeps


def run(coro):
    return asyncio.run(coro)


def test_follows_page_tokens_until_exhausted():
    registry = FakeRegistry(total=2500)
    client, _ = _client(registry)
    result = run(client.search({"query.intr": "x"}, max_records=5000))
    assert result.total_count == 2500
    assert len(result.studies) == 2500
    assert [r.url.params["pageSize"] for r in registry.calls] == ["1000", "1000", "1000"]


def test_stops_at_record_cap_and_keeps_true_total():
    registry = FakeRegistry(total=2500)
    client, _ = _client(registry)
    result = run(client.search({"query.intr": "x"}, max_records=1200))
    assert len(result.studies) == 1200
    assert result.total_count == 2500  # truncation is visible to callers
    assert [r.url.params["pageSize"] for r in registry.calls] == ["1000", "200"]


def test_retries_transient_failures_with_backoff():
    registry = FakeRegistry(total=10, fail_first=2)
    client, sleeps = _client(registry, max_attempts=3)
    result = run(client.search({}, max_records=100))
    assert len(result.studies) == 10
    assert sleeps == [1, 2]


def test_gives_up_after_max_attempts():
    registry = FakeRegistry(total=10, fail_first=99)
    client, _ = _client(registry, max_attempts=3)
    with pytest.raises(UpstreamUnavailable, match="after 3 attempts"):
        run(client.search({}, max_records=100))
    assert len(registry.calls) == 3


def test_honors_retry_after_on_429():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200, json={"studies": [], "totalCount": 0})

    client, sleeps = _client(handler)
    run(client.search({}, max_records=10))
    assert sleeps == [7.0]


def test_bad_request_is_not_retried_and_keeps_upstream_message():
    registry = FakeRegistry(total=0, fail_first=99, fail_status=400)
    client, _ = _client(registry)
    with pytest.raises(UpstreamRejectedQuery, match="try later"):
        run(client.search({}, max_records=10))
    assert len(registry.calls) == 1


def test_identical_searches_are_served_from_cache():
    registry = FakeRegistry(total=5)
    client, _ = _client(registry)

    async def twice():
        await client.search({"query.cond": "x"}, max_records=10)
        await client.search({"query.cond": "x"}, max_records=10)

    run(twice())
    assert len(registry.calls) == 1


def test_data_timestamp_is_read_from_version_endpoint():
    client, _ = _client(FakeRegistry(total=0))
    assert run(client.data_timestamp()) == "2026-09-25T09:00:04"


@pytest.mark.live
def test_live_search_matches_api_total():
    from app.ctgov.query_builder import build_params
    from app.schemas.plan import Filters

    async def go():
        client = CTGovClient()
        try:
            return await client.search(build_params(Filters(condition="gastric cancer", location="South Korea", statuses=["RECRUITING"])), 5000)
        finally:
            await client.aclose()

    result = run(go())
    assert result.total_count > 0
    assert len(result.studies) == result.total_count
