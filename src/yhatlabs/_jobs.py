"""Background jobs: a table call too large for a direct answer runs on our side, and its result is read when it is done.

A fit or one-shot predict sent with background=True returns a Job at once. Without it, a call the API refuses as too large
for a direct answer (413 async_required) is resent as a job and waited for, and so is a direct call the API hands back as a
job because it ran long, so the caller gets the same result either way.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Iterator, Optional

from ._errors import APIError, error_from_response
from ._progress import activity

DONE = ("succeeded", "failed", "canceled")


def is_job(reply: Any) -> bool:
    return isinstance(reply, dict) and reply.get("object") == "job" and isinstance(reply.get("id"), str)


def async_required(e: APIError) -> bool:
    return e.status == 413 and e.type == "async_required"


WAIT_S = 2 * 3600.0  # a job runs at most an hour; past two, something is wrong


def _delays() -> Iterator[float]:
    """Seconds between status checks: quick at first, then every 15 s."""
    yield from (2, 3, 5, 8, 10)
    while True:
        yield 15


class JobFailedError(APIError):
    """A background job ended without a result: it failed, or it was canceled (`type == "canceled"`). `job_id` names the job."""

    def __init__(self, message: str, *, job_id: Optional[str] = None, **kw: Any):
        super().__init__(message, **kw)
        self.job_id = job_id


def _failure(state: dict) -> JobFailedError:
    err = state.get("error") if isinstance(state.get("error"), dict) else {}
    canceled = state.get("status") == "canceled"
    message = str(err.get("message") or ("The job was canceled." if canceled else "The job failed."))
    etype = err.get("type") or ("canceled" if canceled else "job_failed")
    return JobFailedError(
        message,
        job_id=state.get("id"),
        status=409 if canceled else 500,
        type=etype,
        request_id=err.get("request_id"),
        body=state,
    )


def _result(state: dict) -> Any:
    if state.get("status") == "succeeded":
        result = state.get("result")
        if result is None:
            raise error_from_response(
                502, {"error": {"type": "invalid_response", "message": "The job has no result."}}, state.get("id")
            )
        return result
    raise _failure(state)


class Job:
    """A table call running in the background. status() checks it, wait() (or result()) blocks until it is done and returns
    what the direct call would have returned, cancel() stops it.
    """

    def __init__(self, transport: Any, reply: dict, convert: Callable[[Any], Any] = lambda r: r):
        self._t = transport
        self._convert = convert
        self.id: str = reply["id"]
        self.kind: Optional[str] = reply.get("kind")
        self.model: Optional[str] = reply.get("model")
        self._state = reply

    def __repr__(self) -> str:
        return f"Job(id={self.id!r}, kind={self.kind!r}, status={self._state.get('status')!r})"

    def status(self) -> str:
        """queued, running, succeeded, failed or canceled."""
        self._state = self._t.request("GET", f"/v1/jobs/{self.id}")
        return str(self._state.get("status"))

    def wait(self, timeout: Optional[float] = WAIT_S) -> Any:
        """Blocks until the job is done and returns its result; raises JobFailedError if it failed or was canceled.

        Args:
            timeout: Seconds to wait at most (default two hours; None: as long as it takes); raises TimeoutError past it, the job
                still runs.
        """
        until = None if timeout is None else time.monotonic() + timeout
        delays = _delays()
        with activity("running in the background"):
            while self._state.get("status") not in DONE:
                pause = next(delays)
                if until is not None and time.monotonic() + pause > until:
                    raise TimeoutError(f"Job {self.id} is still {self._state.get('status')}; wait() again later.")
                time.sleep(pause)
                self.status()
        return self._convert(_result(self._state))

    def result(self, timeout: Optional[float] = WAIT_S) -> Any:
        """The same as wait()."""
        return self.wait(timeout)

    def cancel(self) -> bool:
        """Stops the job if it hasn't finished; True when it was canceled."""
        try:
            self._state = {**self._state, **self._t.request("DELETE", f"/v1/jobs/{self.id}")}
        except APIError as e:
            if e.status == 409:
                return False
            raise
        return self._state.get("status") == "canceled"


class AsyncJob(Job):
    """asyncio version of Job."""

    async def status(self) -> str:  # type: ignore[override]
        self._state = await self._t.request("GET", f"/v1/jobs/{self.id}")
        return str(self._state.get("status"))

    async def wait(self, timeout: Optional[float] = WAIT_S) -> Any:  # type: ignore[override]
        until = None if timeout is None else time.monotonic() + timeout
        delays = _delays()
        with activity("running in the background"):
            while self._state.get("status") not in DONE:
                pause = next(delays)
                if until is not None and time.monotonic() + pause > until:
                    raise TimeoutError(f"Job {self.id} is still {self._state.get('status')}; wait() again later.")
                await asyncio.sleep(pause)
                await self.status()
        return self._convert(_result(self._state))

    async def result(self, timeout: Optional[float] = WAIT_S) -> Any:  # type: ignore[override]
        return await self.wait(timeout)

    async def cancel(self) -> bool:  # type: ignore[override]
        try:
            self._state = {**self._state, **(await self._t.request("DELETE", f"/v1/jobs/{self.id}"))}
        except APIError as e:
            if e.status == 409:
                return False
            raise
        return self._state.get("status") == "canceled"


def call(transport: Any, path: str, body: dict, job: bool = False) -> Any:
    """A table call, waited for: as a job from the start when `job` (the table is past the direct size), else direct, becoming a
    job when the API asks for one. Returns the direct reply either way.
    """
    try:
        reply = transport.request("POST", path, {**body, "async": True} if job else body)
    except APIError as e:
        if not async_required(e):
            raise
        reply = transport.request("POST", path, {**body, "async": True})
    return Job(transport, reply).wait() if is_job(reply) else reply


async def acall(transport: Any, path: str, body: dict, job: bool = False) -> Any:
    try:
        reply = await transport.request("POST", path, {**body, "async": True} if job else body)
    except APIError as e:
        if not async_required(e):
            raise
        reply = await transport.request("POST", path, {**body, "async": True})
    return await AsyncJob(transport, reply).wait() if is_job(reply) else reply
