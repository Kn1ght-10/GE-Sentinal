"""HTTP client for the OSRS Wiki real-time prices API.

Implements every documented route (/latest, /mapping, /5m, /1h, /timeseries)
with the acceptable-use rules baked in:
  * a descriptive User-Agent is mandatory (defaults like python-requests are
    pre-emptively blocked by the wiki),
  * bulk endpoints are preferred over per-item loops,
  * polite retry with backoff, never hammering on failure.
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from . import config

BLOCKED_UA_FRAGMENTS = ("python-requests", "python-urllib", "curl/", "java/")


class WikiPricesClient:
    def __init__(self, user_agent: str | None = None, base_url: str | None = None,
                 timeout: float = 20.0, max_retries: int = 3):
        ua = (user_agent or config.USER_AGENT).strip()
        if not ua or any(b in ua.lower() for b in BLOCKED_UA_FRAGMENTS):
            raise ValueError(
                "Set a descriptive User-Agent with contact info via GE_SENTINEL_UA; "
                "the wiki blocks default client UAs."
            )
        self.base_url = (base_url or config.API_BASE).rstrip("/")
        self.max_retries = max_retries
        self._client = httpx.Client(
            headers={"User-Agent": ua}, timeout=timeout, follow_redirects=True
        )

    # --- low level --------------------------------------------------------
    def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = self._client.get(url, params=params)
                r.raise_for_status()
                return r.json()
            except (httpx.HTTPError, ValueError) as exc:  # network or bad JSON
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"GET {url} failed after {self.max_retries} attempts") from last_exc

    # --- documented routes --------------------------------------------------
    def mapping(self) -> list[dict]:
        """Static item metadata: name, GE buy limit, alch values, members flag."""
        return self._get("mapping")

    def latest(self, item_id: int | None = None) -> dict:
        params = {"id": item_id} if item_id is not None else None
        return self._get("latest", params)["data"]

    def five_minute(self, timestamp: int | None = None) -> dict:
        """All items, one call. Returns {'data': {item_id: {...}}, 'timestamp': ...}."""
        params = {"timestamp": timestamp} if timestamp else None
        return self._get("5m", params)

    def one_hour(self, timestamp: int | None = None) -> dict:
        params = {"timestamp": timestamp} if timestamp else None
        return self._get("1h", params)

    def timeseries(self, item_id: int, timestep: str = "5m") -> dict:
        if timestep not in {"5m", "1h", "6h", "24h"}:
            raise ValueError("timestep must be one of 5m, 1h, 6h, 24h")
        return self._get("timeseries", {"id": item_id, "timestep": timestep})

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
