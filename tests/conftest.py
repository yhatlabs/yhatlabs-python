import json

import httpx
import numpy as np
import pandas as pd
import pytest

import yhatlabs._transport as transport
from yhatlabs import AsyncClient, Client, Limits

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _deciles(last: float, h: int) -> list:
    return [[last + (q - 0.5) * 10 for q in LEVELS] for _ in range(h)]


def _last(vals: list) -> float:
    return next(v for v in reversed(vals) if v is not None)


class FakeAPI:
    """Mimics the gateway: forecasts are the last observed value with a fixed spread; tabular returns the train mode."""

    def __init__(self):
        self.requests: list = []
        self.script: list = []  # queued (status, body) replies served before the normal behaviour
        self.fits: dict = {}
        self.no_fits = False  # behave like a gateway without /v1/fits
        self.fail_batches = set()  # forecast requests (by order) that answer 500
        self.capabilities: dict = {}  # what /v1/limits advertises

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body, dict(request.headers)))
        if self.script:
            status, reply = self.script.pop(0)
            return httpx.Response(status, json=reply, headers={"X-Request-Id": "req_err"})
        if request.headers.get("authorization") != "Bearer test-key":
            return httpx.Response(
                401, json={"error": {"type": "authentication_error", "message": "bad key", "request_id": "r1"}}
            )
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(200, json={"models": [{"id": "chakra-ts", "kind": "timeseries", "description": "x"}]})
        if path == "/v1/usage":
            return httpx.Response(200, json={"unit": "series", "used": {"monthly": 1, "total": 1}})
        if path == "/v1/limits":
            return httpx.Response(
                200,
                json={
                    "forecast": {
                        "max_series": 2,
                        "max_points_per_series": 500,
                        "max_total_points": 10_000,
                        "max_horizon": 48,
                    },
                    "tabular": {"max_rows": 1000, "max_bytes": 10_000_000},
                    "capabilities": self.capabilities,
                },
            )
        if path.startswith("/v1/fits"):
            return self.fit_route(request.method, path, body)
        if path == "/v1/forecast":
            n_forecast = sum(1 for r in self.requests if r[1] == "/v1/forecast")
            if n_forecast in self.fail_batches:
                return httpx.Response(500, json={"error": {"type": "internal_error", "message": "boom"}})
            h = body["horizon"]
            levels = body.get("quantile_levels") or LEVELS
            q = lambda last: [[last + (lv - 0.5) * 10 for lv in levels] for _ in range(h)]  # noqa: E731
            fcs = []
            for s in body["series"]:
                if "targets" in s:
                    f = {"id": s["id"], "targets": {k: q(_last(v)) for k, v in s["targets"].items()}}
                else:
                    f = {"id": s["id"], "quantiles": q(_last(s["values"]))}
                    last = _last(s["values"])
                    if body.get("samples"):
                        f["samples"] = [[last + k for _ in range(h)] for k in range(body["samples"])]
                    if body.get("history"):
                        T = len(s["values"])
                        f["history"] = [
                            {"cutoff": T - 2 * h, "quantiles": q(last)},
                            {"cutoff": T - h, "quantiles": q(last)},
                        ]
                fcs.append(f)
            n = len(body["series"])
            return httpx.Response(
                200,
                json={
                    "id": f"req_{len(self.requests)}",
                    "model": body["model"],
                    "horizon": h,
                    "quantile_levels": levels,
                    "forecasts": fcs,
                    "usage": {"series": n},
                },
            )
        if path == "/v1/tabular/predict":
            return self.tabular(body["train"], body["test"], body["target"], body)
        return httpx.Response(404, json={"error": {"type": "not_found", "message": "nope"}})

    def fit_route(self, method: str, path: str, body):
        if self.no_fits:
            return httpx.Response(404, json={"error": {"type": "not_found", "message": "Not found."}})
        parts = path.strip("/").split("/")  # v1 fits [id] [op]
        if method == "POST" and len(parts) == 2:
            self.n_fits = getattr(self, "n_fits", 0) + 1
            fid = f"fit_{self.n_fits:024x}"
            ys = [r[body["target"]] for r in body["train"]]
            pt = body.get("problem_type")
            reg = pt == "regression" or (pt is None and all(isinstance(y, float) and not y.is_integer() for y in ys))
            classes = (
                None if reg else sorted({str(int(y) if isinstance(y, float) and y.is_integer() else y) for y in ys})
            )
            self.fits[fid] = body
            return httpx.Response(
                200,
                json={
                    "id": fid,
                    "object": "fit",
                    "model": body.get("model"),
                    "target": body["target"],
                    "created_at": 1,
                    "expires_at": 2,
                    "fit": {
                        "n_rows": len(ys),
                        "problem_type": "regression" if reg else "binary",
                        "preset": body.get("preset", "fast"),
                    },
                    "classes": classes,
                    "intervals": reg,
                },
            )
        if method == "GET" and len(parts) == 2:
            return httpx.Response(200, json={"object": "list", "data": [{"id": k} for k in self.fits]})
        fid = parts[2]
        if fid not in self.fits:
            return httpx.Response(404, json={"error": {"type": "fit_not_found", "message": "No fit."}})
        fb = self.fits[fid]
        if method == "DELETE":
            del self.fits[fid]
            return httpx.Response(200, json={"id": fid, "object": "fit", "deleted": True})
        if method == "GET":
            return httpx.Response(
                200,
                json={
                    "id": fid,
                    "object": "fit",
                    "model": fb.get("model"),
                    "target": fb["target"],
                    "fit": {},
                    "classes": None,
                    "intervals": True,
                    "created_at": 1,
                    "expires_at": 2,
                },
            )
        if parts[3] == "predict":
            r = self.tabular(
                fb["train"],
                body["test"],
                fb["target"],
                {**fb, "quantiles": body.get("quantiles"), "set_coverage": body.get("set_coverage")},
            )
            j = json.loads(r.content)
            j.pop("fit", None)
            j["fit_id"] = fid
            return httpx.Response(r.status_code, json=j)
        if parts[3] == "importance":
            feats = [k for k in fb["train"][0] if k != fb["target"]]
            return httpx.Response(
                200,
                json={
                    "fit_id": fid,
                    "metric": "rmse",
                    "source": "data" if body.get("data") else "validation",
                    "features": [
                        {"feature": f, "importance": 1.0 / (i + 1), "stddev": 0.01} for i, f in enumerate(feats)
                    ],
                },
            )
        return httpx.Response(404, json={"error": {"type": "not_found", "message": "nope"}})

    def tabular(self, train, test, target, body):
        # the Worker's JSON.parse/stringify turns 1.0 into 1, so the server sees integer labels
        ys = [
            int(y) if isinstance(y, float) and y.is_integer() and body.get("problem_type") != "regression" else y
            for y in (r[target] for r in train)
        ]
        fit = {
            "n_rows": len(train),
            "n_features": len(train[0]) - 1,
            "fit_s": 0.1,
            "problem_type": body.get("problem_type"),
            "eval_metric": None,
            "preset": body.get("preset", "fast"),
            "time_limit": None,
        }
        n = len(test) if isinstance(test, list) else 3
        pt = body.get("problem_type")
        if pt == "regression" or (pt is None and all(isinstance(y, float) and not float(y).is_integer() for y in ys)):
            mean = float(np.mean(ys))
            out = {
                "id": "req_tab",
                "model": body.get("model"),
                "predictions": [mean] * n,
                "fit": fit,
                "usage": {"rows": n},
            }
            if body.get("quantiles"):
                out.update(
                    quantile_levels=body["quantiles"],
                    quantiles=[[mean + (q - 0.5) * 10 for q in body["quantiles"]]] * n,
                    interval_method="conformal",
                )
            return httpx.Response(200, json=out)
        classes = sorted({str(y) for y in ys})
        counts = [sum(str(y) == c for y in ys) / len(ys) for c in classes]
        top = classes[int(np.argmax(counts))]
        out = {
            "id": "req_tab",
            "model": body.get("model"),
            "predictions": [top] * n,
            "probabilities": [counts] * n,
            "classes": classes,
            "fit": fit,
            "usage": {"rows": n},
        }
        if body.get("set_coverage"):
            out.update(prediction_sets=[classes] * n, set_coverage=body["set_coverage"])
        return httpx.Response(200, json=out)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(transport, "_backoff", lambda attempt, resp: 0.0)


@pytest.fixture
def api():
    return FakeAPI()


@pytest.fixture
def client(api):
    c = Client(
        "test-key",
        base_url="https://fake.test",
        limits=Limits(),
        http_client=httpx.Client(transport=httpx.MockTransport(api)),
    )
    yield c
    c.close()


@pytest.fixture
def aclient(api):
    return AsyncClient(
        "test-key",
        base_url="https://fake.test",
        limits=Limits(),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(api)),
    )


@pytest.fixture
def hourly():
    ts = pd.date_range("2026-01-01", periods=96, freq="h")
    rows = []
    for uid, base in (("a", 10.0), ("b", 100.0)):
        y = base + np.sin(np.arange(96) * 2 * np.pi / 24) + np.random.default_rng(int(base)).normal(0, 0.1, 96)
        rows.append(pd.DataFrame({"unique_id": uid, "ds": ts, "y": y}))
    return pd.concat(rows, ignore_index=True)
