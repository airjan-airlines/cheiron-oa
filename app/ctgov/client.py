"""Async client for the ClinicalTrials.gov API v2.

Handles paging (`nextPageToken`), retries with backoff on 429/5xx/network errors,
a small in-memory TTL cache, and a cap on concurrent requests (a comparison runs one
search per cohort in parallel). The API's rate limit is undocumented; ~50 req/min is
commonly cited, so we stay polite rather than fast.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from app.errors import UpstreamRejectedQuery, UpstreamUnavailable

PAGE_SIZE_MAX = 1000
CACHE_MAX_ENTRIES = 256
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class SearchResult:
    studies: list[dict[str, Any]]
    total_count: int


class CTGovClient:
    def __init__(
        self,
        base_url: str = "https://clinicaltrials.gov/api/v2",
        timeout_s: float = 30.0,
        max_attempts: int = 3,
        cache_ttl_s: float = 900.0,
        max_concurrency: int = 4,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout_s, transport=transport)
        self._max_attempts = max_attempts
        self._cache_ttl_s = cache_ttl_s
        self._cache: dict[tuple, tuple[float, Any]] = {}
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._sleep = sleep

    async def aclose(self) -> None:
        await self._http.aclose()

    async def search(self, params: dict[str, str], max_records: int) -> SearchResult:
        """Fetch up to `max_records` studies matching `params`, following page tokens."""
        key = ("search", tuple(sorted(params.items())), max_records)
        if (hit := self._cached(key)) is not None:
            return hit

        studies: list[dict[str, Any]] = []
        total = 0
        token: str | None = None
        while True:
            page_params = {**params, "pageSize": str(min(PAGE_SIZE_MAX, max_records - len(studies)))}
            if token:
                page_params["pageToken"] = token
            body = await self._get_json("/studies", page_params)
            if not token:
                total = int(body.get("totalCount", 0))
            studies.extend(body.get("studies", []))
            token = body.get("nextPageToken")
            if not token or len(studies) >= max_records:
                break

        result = SearchResult(studies=studies[:max_records], total_count=total)
        self._store(key, result)
        return result

    async def data_timestamp(self) -> str | None:
        """The registry snapshot time (`/version` -> `dataTimestamp`), for provenance."""
        key = ("version",)
        if (hit := self._cached(key)) is not None:
            return hit
        try:
            stamp = (await self._get_json("/version", {})).get("dataTimestamp")
        except UpstreamUnavailable:
            return None  # provenance is nice to have; never fail a request over it
        self._store(key, stamp)
        return stamp

    async def _get_json(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        last_problem = "no attempt made"
        for attempt in range(1, self._max_attempts + 1):
            try:
                async with self._semaphore:
                    resp = await self._http.get(path, params=params)
            except httpx.TransportError as exc:  # timeouts, connection errors
                last_problem = f"{type(exc).__name__}: {exc}"
                delay = 2 ** (attempt - 1)
            else:
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 400:
                    # The API returns plain text, e.g. "Invalid value in parameter `overallStatus`: `FOO`".
                    raise UpstreamRejectedQuery(resp.text.strip() or "ClinicalTrials.gov rejected the query.")
                if resp.status_code not in _RETRYABLE_STATUS:
                    raise UpstreamUnavailable(f"ClinicalTrials.gov returned HTTP {resp.status_code}: {resp.text[:200]}")
                last_problem = f"HTTP {resp.status_code}"
                delay = _retry_after(resp) or 2 ** (attempt - 1)
            if attempt < self._max_attempts:
                await self._sleep(delay)
        raise UpstreamUnavailable(f"ClinicalTrials.gov failed after {self._max_attempts} attempts ({last_problem}).")

    def _cached(self, key: tuple) -> Any | None:
        entry = self._cache.get(key)
        if entry and entry[0] > time.monotonic():
            return entry[1]
        return None

    def _store(self, key: tuple, value: Any) -> None:
        now = time.monotonic()
        for stale in [k for k, (expires, _) in self._cache.items() if expires <= now]:
            del self._cache[stale]
        while len(self._cache) >= CACHE_MAX_ENTRIES:
            del self._cache[next(iter(self._cache))]  # oldest insertion first
        self._cache[key] = (now + self._cache_ttl_s, value)


def _retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("Retry-After", "")
    return min(float(value), 30.0) if value.isdigit() else None
