"""HTTP layer: auth header, JSON in and out, retries with backoff on transient failures, and error mapping."""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Any, Optional

import httpx

try:  # orjson encodes large request bodies several times faster; optional
    import orjson

    def _dumps(obj: Any) -> bytes:
        return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY)
except ImportError:  # pragma: no cover
    import json as jsonlib

    def _dumps(obj: Any) -> bytes:
        return jsonlib.dumps(obj, separators=(",", ":"), allow_nan=False).encode()


from ._errors import APIConnectionError, APITimeoutError, ConfigurationError, error_from_response
from ._progress import activity
from ._version import __version__

log = logging.getLogger("yhatlabs")
if os.environ.get(
    "YHAT_LOG"
):  # YHAT_LOG=debug|info: request lines on stderr (method, path, status, time, request id; never bodies or keys)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[yhatlabs] %(message)s"))
    log.addHandler(_h)
    log.setLevel(
        os.environ["YHAT_LOG"].upper() if os.environ["YHAT_LOG"].upper() in ("DEBUG", "INFO", "WARNING") else "INFO"
    )

_LABELS = {"/v1/forecast": "forecasting", "/v1/tabular/predict": "fitting and predicting", "/v1/fits": "fitting"}


def _label(method: str, path: str) -> str:
    if path.endswith("/predict") and path.startswith("/v1/fits/"):
        return "predicting"
    if path.endswith("/importance"):
        return "computing importance"
    return _LABELS.get(path, f"{method} {path}")


def _log(method: str, path: str, resp: Optional[httpx.Response], t0: float, attempt: int, note: str = "") -> None:
    if not log.isEnabledFor(logging.INFO):
        return
    ms = (time.monotonic() - t0) * 1000
    status = resp.status_code if resp is not None else "-"
    rid = resp.headers.get("X-Request-Id") if resp is not None else None
    log.info(
        f"{method} {path} -> {status} in {ms:.0f} ms{f' (attempt {attempt + 1})' if attempt else ''}{f' request_id={rid}' if rid else ''}{f' {note}' if note else ''}"
    )


DEFAULT_BASE_URL = "https://api.yhatlabs.com"
RETRY_STATUSES = {408, 409, 429, 500, 502, 503, 504}
USER_AGENT = f"yhatlabs-python/{__version__}"


def resolve_api_key(api_key: Optional[str]) -> str:
    key = api_key or os.environ.get("YHAT_API_KEY") or os.environ.get("YHATLABS_API_KEY")
    if not key:
        raise ConfigurationError(
            "No API key. Pass api_key=... or set the YHAT_API_KEY environment variable. "
            "Request access at https://yhatlabs.com/models/access."
        )
    return key


def resolve_base_url(base_url: Optional[str]) -> str:
    return (base_url or os.environ.get("YHAT_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def _retry_after(resp: Optional[httpx.Response]) -> Optional[float]:
    if resp is None:
        return None
    v = resp.headers.get("Retry-After")
    try:
        return min(float(v), 120.0) if v is not None else None
    except ValueError:
        return None


def _backoff(attempt: int, resp: Optional[httpx.Response]) -> float:
    ra = _retry_after(resp)
    if ra is not None:
        return ra
    if resp is not None and (resp.status_code == 503 or _streamed_error(_json(resp)) == 503):
        return min(60.0, 10.0 * 2**attempt)  # the model is starting up: that takes tens of seconds, not milliseconds
    return min(30.0, 0.5 * 2**attempt) * (0.5 + random.random())


# A long tabular fit answers 200 at once and streams keep-alive spaces until the result, so an error that happens later can
# only arrive in the body. The status it would have had comes from its type.
STREAMED_STATUS = {"model_unavailable": 503, "upstream_error": 502, "invalid_request_error": 422, "internal_error": 500}


def _streamed_error(body: Any) -> Optional[int]:
    if isinstance(body, dict) and isinstance(body.get("error"), dict) and len(body) == 1:
        return STREAMED_STATUS.get(body["error"].get("type"), 500)
    return None


def _json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return None


def _is_streamed(resp: httpx.Response) -> bool:
    return resp.status_code == 200 and _streamed_error(_json(resp)) is not None


def _should_retry(resp: httpx.Response, streamed_retries: int = 0) -> bool:
    status = resp.status_code
    if status == 200:
        status = _streamed_error(_json(resp)) or 200
        # an error after a long fit (possibly the 30-minute upstream timeout): one more try, not max_retries more fits
        if status != 200 and streamed_retries >= 1:
            return False
    if status not in RETRY_STATUSES:
        return False
    if status == 429:  # a used-up quota does not recover by retrying; a rate limit does
        return ((_json(resp) or {}).get("error") or {}).get("type") != "quota_exceeded"
    return True


def _decode(resp: httpx.Response) -> Any:
    rid = resp.headers.get("X-Request-Id")
    body = _json(resp)
    if resp.status_code >= 400:
        raise error_from_response(resp.status_code, body, rid)
    streamed = _streamed_error(body)
    if streamed is not None:
        raise error_from_response(streamed, body, rid)
    if body is None:
        raise error_from_response(
            502, {"error": {"type": "invalid_response", "message": "The API returned a non-JSON response."}}, rid
        )
    return body


class SyncTransport:
    def __init__(
        self,
        api_key: Optional[str],
        base_url: Optional[str],
        timeout: float,
        max_retries: int,
        http_client: Optional[httpx.Client],
    ):
        self.api_key = resolve_api_key(api_key)
        self.base_url = resolve_base_url(base_url)
        self.max_retries = max_retries
        self._own = http_client is None
        self.http = http_client or httpx.Client(timeout=timeout)
        self.timeout = timeout

    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def request(self, method: str, path: str, json: Any = None) -> Any:
        if method == "POST":
            with activity(_label(method, path)):
                return self._request(method, path, json)
        return self._request(method, path, json)

    def _request(self, method: str, path: str, json: Any = None) -> Any:
        url = f"{self.base_url}{path}"
        last_resp: Optional[httpx.Response] = None
        streamed_retries = 0
        for attempt in range(self.max_retries + 1):
            t0 = time.monotonic()
            try:
                resp = self.http.request(
                    method,
                    url,
                    content=None if json is None else _dumps(json),
                    headers=self.headers(),
                    timeout=self.timeout,
                )
            except httpx.TimeoutException as e:
                _log(method, path, None, t0, attempt, "timed out")
                if attempt < self.max_retries:
                    time.sleep(_backoff(attempt, None))
                    continue
                raise APITimeoutError(f"The request to {path} timed out after {self.timeout:.0f}s.") from e
            except httpx.TransportError as e:
                _log(method, path, None, t0, attempt, f"connection error: {type(e).__name__}")
                if attempt < self.max_retries:
                    time.sleep(_backoff(attempt, None))
                    continue
                raise APIConnectionError(f"Could not reach {self.base_url}: {e}") from e
            last_resp = resp
            _log(method, path, resp, t0, attempt)
            if attempt < self.max_retries and _should_retry(resp, streamed_retries):
                streamed_retries += _is_streamed(resp)
                time.sleep(_backoff(attempt, resp))
                continue
            return _decode(resp)
        return _decode(last_resp)  # pragma: no cover

    def close(self) -> None:
        if self._own:
            self.http.close()


class AsyncTransport:
    def __init__(
        self,
        api_key: Optional[str],
        base_url: Optional[str],
        timeout: float,
        max_retries: int,
        http_client: Optional[httpx.AsyncClient],
    ):
        self.api_key = resolve_api_key(api_key)
        self.base_url = resolve_base_url(base_url)
        self.max_retries = max_retries
        self._own = http_client is None
        self.http = http_client or httpx.AsyncClient(timeout=timeout)
        self.timeout = timeout

    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def request(self, method: str, path: str, json: Any = None) -> Any:
        if method == "POST":
            with activity(_label(method, path)):
                return await self._request(method, path, json)
        return await self._request(method, path, json)

    async def _request(self, method: str, path: str, json: Any = None) -> Any:
        url = f"{self.base_url}{path}"
        last_resp: Optional[httpx.Response] = None
        streamed_retries = 0
        for attempt in range(self.max_retries + 1):
            t0 = time.monotonic()
            try:
                resp = await self.http.request(
                    method,
                    url,
                    content=None if json is None else _dumps(json),
                    headers=self.headers(),
                    timeout=self.timeout,
                )
            except httpx.TimeoutException as e:
                _log(method, path, None, t0, attempt, "timed out")
                if attempt < self.max_retries:
                    await asyncio.sleep(_backoff(attempt, None))
                    continue
                raise APITimeoutError(f"The request to {path} timed out after {self.timeout:.0f}s.") from e
            except httpx.TransportError as e:
                _log(method, path, None, t0, attempt, f"connection error: {type(e).__name__}")
                if attempt < self.max_retries:
                    await asyncio.sleep(_backoff(attempt, None))
                    continue
                raise APIConnectionError(f"Could not reach {self.base_url}: {e}") from e
            last_resp = resp
            _log(method, path, resp, t0, attempt)
            if attempt < self.max_retries and _should_retry(resp, streamed_retries):
                streamed_retries += _is_streamed(resp)
                await asyncio.sleep(_backoff(attempt, resp))
                continue
            return _decode(resp)
        return _decode(last_resp)  # pragma: no cover

    async def aclose(self) -> None:
        if self._own:
            await self.http.aclose()
