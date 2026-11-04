import asyncio
import json

import httpx
import pytest
from conftest import FakeAPI

from yhatlabs import Client, Job, Limits, Tuned
from yhatlabs._errors import DataError

TUNE = {
    "id": "tune_" + "a" * 24,
    "object": "tune",
    "model": "chakra-ts",
    "name": None,
    "horizon": 24,
    "freq": "h",
    "series": 2,
    "steps": 300,
    "used": True,
    "report": {
        "used": True,
        "holdout": {"window": 24, "zero_shot": {"loss": 1.0, "wape": 0.2}, "tuned": {"loss": 0.7, "wape": 0.13}},
    },
    "created_at": 1,
    "expires_at": 2,
    "usage": {"tokens": 10, "credits": 0.01},
}


class TuneAPI(FakeAPI):
    """FakeAPI plus tunes: POST /v1/series/tunes queues a job that succeeds on its first poll."""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        if path == "/v1/series/tunes" and method == "POST":
            self.requests.append((method, path, json.loads(request.content), dict(request.headers)))
            return httpx.Response(
                202,
                json={
                    "id": "job_" + "1" * 24,
                    "object": "job",
                    "kind": "tune",
                    "model": "chakra-ts",
                    "status": "queued",
                },
            )
        if path.startswith("/v1/jobs/"):
            self.requests.append((method, path, None, dict(request.headers)))
            return httpx.Response(
                200, json={"id": path.rsplit("/", 1)[-1], "object": "job", "status": "succeeded", "result": TUNE}
            )
        if path.startswith("/v1/series/tunes/"):
            self.requests.append((method, path, None, dict(request.headers)))
            return httpx.Response(
                200, json={"id": TUNE["id"], "object": "tune", "deleted": True} if method == "DELETE" else TUNE
            )
        if path == "/v1/series/tunes":
            return httpx.Response(200, json={"object": "list", "data": [TUNE]})
        return super().__call__(request)


@pytest.fixture
def api():
    return TuneAPI()


def test_tune_waits_for_the_job_and_forecasts_with_it(client, api, hourly):
    t = client.tune(hourly, 24, steps=200, name="stores")
    sent = next(b for m, p, b, _ in api.requests if p == "/v1/series/tunes")
    assert sent["steps"] == 200 and sent["name"] == "stores" and sent["horizon"] == 24 and len(sent["series"]) == 2
    assert isinstance(t, Tuned) and t.used and t.holdout["tuned"]["wape"] == 0.13
    assert "0.2 -> 0.13" in repr(t)
    fc = t.forecast(hourly)
    body = [b for m, p, b, _ in api.requests if p == "/v1/series/forecast"][-1]
    assert body["tune"] == TUNE["id"] and body["horizon"] == 24
    assert len(fc.df) == 48
    client.forecast(hourly, 12, tune=t.id, preset="pro")
    body = [b for m, p, b, _ in api.requests if p == "/v1/series/forecast"][-1]
    assert body["tune"] == TUNE["id"] and body["preset"] == "pro"


def test_background_returns_the_job(client, hourly):
    job = client.tune(hourly, 24, background=True)
    assert isinstance(job, Job) and job.kind == "tune"
    assert job.wait().id == TUNE["id"]


def test_refuses_bad_steps_preset_and_too_many_series(client, hourly):
    with pytest.raises(DataError):
        client.tune(hourly, 24, steps=10)
    with pytest.raises(DataError):
        client.forecast(hourly, 24, preset="medium")
    small = Client("test-key", base_url="https://fake.test", limits=Limits(max_series=1), http_client=client._t.http)
    with pytest.raises(DataError, match="one request"):
        small.tune(hourly, 24)


def test_get_list_delete(client, api):
    t = client.get_tune(TUNE["id"])
    assert t.id == TUNE["id"] and client.tunes()[0]["id"] == TUNE["id"]
    t.delete()
    assert ("DELETE", f"/v1/series/tunes/{TUNE['id']}") in [(m, p) for m, p, *_ in api.requests]


def test_async_tune(aclient, hourly):
    async def run():
        t = await aclient.tune(hourly, 24)
        fc = await t.forecast(hourly)
        await aclient.close()
        return t, fc

    t, fc = asyncio.run(run())
    assert t.used and len(fc.df) == 48
