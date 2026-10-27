import itertools
import json

import httpx
import numpy as np
import pandas as pd
import pytest
from conftest import FakeAPI

import yhatlabs._jobs as jobs
from yhatlabs import AsyncClient, Client, Comparison, Job, JobFailedError, Limits
from yhatlabs._errors import DataError


class JobsAPI(FakeAPI):
    """FakeAPI plus background jobs, /v1/table/compare and /v1/estimate. A direct table call with more than `direct_rows`
    training rows answers 413 async_required; a job reports `steps` (e.g. running, then succeeded) on its polls."""

    def __init__(self, direct_rows=1000, steps=("running", "succeeded"), long_direct=False):
        super().__init__()
        self.direct_rows = direct_rows
        self.steps = list(steps)
        self.long_direct = long_direct
        self.jobs: dict = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        body = json.loads(request.content) if request.content else None
        if path.startswith("/v1/jobs/"):
            self.requests.append((method, path, body, dict(request.headers)))
            jid = path.rsplit("/", 1)[-1]
            job = self.jobs[jid]
            if method == "DELETE":
                job["status"] = "canceled"
                return httpx.Response(200, json={"id": jid, "object": "job", "status": "canceled"})
            if job["status"] not in ("succeeded", "failed", "canceled"):
                job["status"] = job["steps"].pop(0)
            out = {"id": jid, "object": "job", "status": job["status"]}
            if job["status"] == "succeeded":
                out["result"] = job["result"]
            if job["status"] == "failed":
                out["error"] = {"type": "upstream_error", "message": "The model failed on this job."}
            return httpx.Response(200, json=out)
        if path == "/v1/table/compare":
            self.requests.append((method, path, body, dict(request.headers)))
            n = len(body["test"])
            ys = [r[body["target"]] for r in body["train"]]
            top = max(set(ys), key=ys.count)
            models = [
                {"model": m, "label": m.title(), "predictions": [top] * n, "fit_s": 0.1}
                for m in body.get("models") or ["linear"]
            ]
            models.append({"model": "xgboost", "label": "XGBoost", "error": "did not train on this table"})
            return httpx.Response(
                200,
                json={
                    "id": "req_cmp",
                    "problem_type": "binary",
                    "classes": ["0", "1"],
                    "models": models,
                    "baseline": {"model": "most_frequent", "label": "Most common class", "predictions": [top] * n},
                    "usage": {"rows": n, "tokens": 20000, "credits": 0.02},
                },
            )
        if path == "/v1/estimate":
            self.requests.append((method, path, body, dict(request.headers)))
            tokens = body["train_rows"] * body["features"] + body.get("rows", 0)
            return httpx.Response(
                200,
                json={"tokens": tokens, "credits": round(tokens / 1e6, 3), "credits_per_m_tokens": 1, "min_tokens": 0},
            )
        if path in ("/v1/table/predict", "/v1/table/fits") and method == "POST" and body is not None:
            train = body["train"] if isinstance(body["train"], list) else []
            if body.get("async"):
                self.requests.append((method, path, body, dict(request.headers)))
                direct = {k: v for k, v in body.items() if k != "async"}
                saved, self.requests = self.requests, []
                result = json.loads(super().__call__(_post(path, direct)).content)
                self.requests = saved
                jid = f"job_{len(self.jobs)}"
                self.jobs[jid] = {"status": "queued", "steps": list(self.steps), "result": result}
                return httpx.Response(202, json={"id": jid, "object": "job", "kind": "predict", "status": "queued"})
            if len(train) > self.direct_rows:
                self.requests.append((method, path, body, dict(request.headers)))
                return httpx.Response(413, json={"error": {"type": "async_required", "message": "Send it with async."}})
            if self.long_direct:
                self.requests.append((method, path, body, dict(request.headers)))
                result = json.loads(super().__call__(_post(path, body)).content)
                self.requests.pop()
                jid = f"job_{len(self.jobs)}"
                self.jobs[jid] = {"status": "running", "steps": list(self.steps), "result": result}
                return httpx.Response(200, json={"id": jid, "object": "job", "status": "running"})
        return super().__call__(request)


def _post(path, body):
    return httpx.Request("POST", f"https://fake.test{path}", json=body, headers={"authorization": "Bearer test-key"})


@pytest.fixture(autouse=True)
def quick_polls(monkeypatch):
    monkeypatch.setattr(jobs, "_delays", lambda: itertools.repeat(0))


def _client(api, limits=None):
    return Client(
        "test-key",
        base_url="https://fake.test",
        limits=limits or Limits(),
        http_client=httpx.Client(transport=httpx.MockTransport(api)),
    )


def _table(n):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "y": (rng.normal(size=n) > 0).astype(int)})


def test_async_required_runs_as_a_job_and_waits():
    api = JobsAPI(direct_rows=10)
    df = _table(40)
    res = _client(api).predict(df, "y", df.drop(columns="y").head(3))
    paths = [(m, p, (b or {}).get("async")) for m, p, b, _ in api.requests]
    assert paths[0] == ("POST", "/v1/table/predict", None)  # tried directly, refused as too large
    assert paths[1] == ("POST", "/v1/table/predict", True)  # resent as a job
    assert [p for m, p, a in paths[2:]] == ["/v1/jobs/job_0", "/v1/jobs/job_0"]  # running, then succeeded
    assert len(res.predictions) == 3


def test_a_table_past_the_published_direct_size_goes_straight_to_a_job():
    api = JobsAPI(direct_rows=10)
    lim = Limits(tabular_direct={"fast": {"rows": 10, "columns": 100}, "pro": {"rows": 5, "columns": 100}})
    df = _table(40)
    _client(api, lim).predict(df, "y", df.drop(columns="y").head(3))
    first = api.requests[0]
    assert first[1] == "/v1/table/predict" and first[2]["async"] is True  # no 413 round trip


def test_pro_reads_the_pro_direct_size():
    lim = Limits(tabular_direct={"fast": {"rows": 100, "columns": 100}, "pro": {"rows": 5, "columns": 100}})
    assert lim.direct_fits("fast", 50, 3) and not lim.direct_fits("pro", 50, 3) and not lim.direct_fits("full", 50, 3)
    assert Limits().direct_fits("pro", 10**9, 10**4)  # nothing published: direct, and the 413 decides


def test_a_long_direct_call_answered_as_a_job_is_polled():
    api = JobsAPI(long_direct=True, steps=("succeeded",))
    df = _table(20)
    res = _client(api).predict(df, "y", df.drop(columns="y").head(2))
    assert len(res.predictions) == 2 and api.requests[-1][1] == "/v1/jobs/job_0"


def test_background_returns_a_job_and_fit_results_come_back_as_a_model():
    api = JobsAPI(steps=("queued", "running", "succeeded"))
    c = _client(api)
    df = _table(20)
    job = c.fit(df, "y", background=True)
    assert isinstance(job, Job) and job.id == "job_0"
    assert job.status() == "queued"
    model = job.wait()
    assert model.id.startswith("fit_") and job.status() == "succeeded"
    with pytest.raises(DataError):
        c.fit(df.assign(z=df.y), ["y", "z"], background=True)


def test_a_failed_or_canceled_job_raises():
    api = JobsAPI(steps=("running", "failed"))
    df = _table(20)
    job = _client(api).predict(df, "y", df.drop(columns="y").head(2), background=True)
    with pytest.raises(JobFailedError) as e:
        job.wait()
    assert e.value.type == "upstream_error"
    api2 = JobsAPI(steps=("running", "running", "running"))
    job2 = _client(api2).predict(df, "y", df.drop(columns="y").head(2), background=True)
    assert job2.cancel() is True
    with pytest.raises(JobFailedError) as e2:
        job2.wait()
    assert e2.value.type == "canceled"


def test_wait_timeout_leaves_the_job_running():
    api = JobsAPI(steps=("running",) * 50)
    df = _table(20)
    job = _client(api).predict(df, "y", df.drop(columns="y").head(2), background=True)
    with pytest.raises(TimeoutError):
        job.wait(timeout=0)


def test_store_false_and_pro_are_sent():
    api = JobsAPI()
    c = _client(api)
    df = _table(20)
    c.predict(df, "y", df.drop(columns="y").head(2), preset="pro", store=False)
    body = [b for m, p, b, _ in api.requests if p == "/v1/table/predict"][-1]
    assert body["store"] is False and body["preset"] == "pro"
    c.fit(df, "y", store=False, ttl_days=60)
    fb = [b for m, p, b, _ in api.requests if p == "/v1/table/fits"][-1]
    assert fb["store"] is False and fb["ttl_days"] == 60.0 and "ttl_hours" not in fb
    c.predict(df, "y", df.drop(columns="y").head(2))
    assert "store" not in [b for m, p, b, _ in api.requests if p == "/v1/table/predict"][-1]


def test_store_false_on_a_forecast(client, api, hourly):
    client.forecast(hourly, h=4, store=False)
    assert all(b.get("store") is False for m, p, b, _ in api.requests if p == "/v1/series/forecast")


def test_compare_returns_each_model_and_scores_them_with_chakratab():
    api = JobsAPI()
    c = _client(api)
    df = _table(30)
    test = df.drop(columns="y").head(5)
    res = c.predict(df, "y", test, compare=["linear", "random_forest"])
    cmp_body = [b for m, p, b, _ in api.requests if p == "/v1/table/compare"][-1]
    assert cmp_body["models"] == ["linear", "random_forest"] and "preset" not in cmp_body
    cmp = res.comparison
    assert isinstance(cmp, Comparison) and cmp.baseline == "Most common class"
    assert cmp.errors == {"XGBoost": "did not train on this table"}
    assert list(cmp.predictions["Linear"].index) == list(test.index)
    board = cmp.score(df.y.head(5), ChakraTab=res)
    assert {"ChakraTab", "Linear", "Random_Forest", "Most common class", "XGBoost"} <= set(board.model)
    assert board.loc[board.model == "ChakraTab", "accuracy"].between(0, 1).all()
    assert cmp.usage["credits"] == 0.02
    with pytest.raises(DataError):
        c.compare(df, "y", test, models=["catboost"])
    alone = c.compare(df, "y", test)
    assert set(alone.predictions) == {"Linear", "Most common class"}


def test_estimate_asks_the_api_and_credits_are_decimals():
    api = JobsAPI()
    est = _client(api).estimate_table(train_rows=200_000, features=10, predict_rows=1000, preset="pro", task="regress")
    calls = [b for m, p, b, _ in api.requests if p == "/v1/estimate"]
    assert [b["call"] for b in calls] == ["fit", "predict"]
    assert calls[0]["preset"] == "pro" and calls[0]["regression"] is True and calls[1]["rows"] == 1000
    assert est == {"fit": 2.0, "predict": 2.001, "total": 4.001, "tokens": 4_001_000, "unit": "credits"}
    assert isinstance(est["total"], float)


def test_forecast_usage_adds_up_credits_across_batches(client, api, hourly, monkeypatch):
    orig = api.__call__

    def with_credits(request):
        r = orig(request)
        if request.url.path == "/v1/series/forecast" and r.status_code == 200:
            j = json.loads(r.content)
            j["usage"]["credits"] = 0.001
            return httpx.Response(200, json=j)
        return r

    c = Client(
        "test-key",
        base_url="https://fake.test",
        limits=Limits(max_series=1),
        http_client=httpx.Client(transport=httpx.MockTransport(with_credits)),
    )
    fc = c.forecast(hourly, h=4)
    assert fc.usage["credits"] == pytest.approx(0.002) and fc.usage["requests"] == 2


@pytest.mark.asyncio
async def test_async_client_jobs_and_compare():
    api = JobsAPI(direct_rows=10)
    c = AsyncClient(
        "test-key",
        base_url="https://fake.test",
        limits=Limits(),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(api)),
    )
    async with c:
        df = _table(40)
        res = await c.predict(df, "y", df.drop(columns="y").head(3), compare=True)
        assert len(res.predictions) == 3 and isinstance(res.comparison, Comparison)
        job = await c.fit(_table(20), "y", background=True)
        model = await job.wait()
        assert model.id.startswith("fit_")
