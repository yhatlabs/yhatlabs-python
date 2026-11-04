import json

import httpx
import numpy as np
import pandas as pd
import pytest
from test_jobs import JobsAPI, _client
from test_tunes import TUNE, TuneAPI

import yhatlabs
import yhatlabs._client as client_mod
from yhatlabs import Client, Limits
from yhatlabs._cli import main
from yhatlabs._errors import DataError


def test_check_finds_a_date_column_by_its_values():
    df = pd.DataFrame(
        {"Month": pd.date_range("1949-01", periods=48, freq="MS").strftime("%Y-%m"), "Passengers": range(48)}
    )
    rep = yhatlabs.check(df, h=12)
    assert rep.kind == "timeseries"
    with pytest.raises(DataError, match="time_col="):
        yhatlabs.check(pd.DataFrame({"a": [1, 2, 3], "b": [3.0, 4.0, 5.0]}))
    # numbers are never taken for dates, and a table with target= stays a table
    assert (
        yhatlabs.check(pd.DataFrame({"code": ["2001", "2002", "2003"] * 10, "y": [0, 1] * 15}), target="y").kind
        == "tabular"
    )


def test_horizon_error_says_what_to_do(client, hourly):
    with pytest.raises(DataError, match="coarser frequency"):
        client.forecast(hourly, 5000)


def test_estimate_guesses_regression_and_prices_quantiles():
    api = JobsAPI()
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"x": rng.normal(size=200), "price": rng.normal(100, 10, 200)})
    _client(api).estimate_table(train, target="price", test=train.drop(columns="price"), quantiles=[0.1, 0.9])
    fit, pred = [b for m, p, b, _ in api.requests if p == "/v1/estimate"]
    assert fit["regression"] is True and pred["ranges"] is True and pred["rows"] == 200
    api.requests.clear()
    labels = pd.DataFrame({"x": rng.normal(size=200), "churn": rng.integers(0, 2, 200)})
    _client(api).estimate_table(labels, target="churn", quantiles=[0.1, 0.9])
    (fit,) = [b for m, p, b, _ in api.requests if p == "/v1/estimate"]
    assert "regression" not in fit


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and capsys.readouterr().out.strip() == f"yhatlabs {yhatlabs.__version__}"


def test_cli_tune_and_forecast_with_it(tmp_path, capsys, monkeypatch, hourly):
    api = TuneAPI()

    def made(**kw):
        return Client(
            "test-key",
            base_url="https://fake.test",
            limits=Limits(),
            http_client=httpx.Client(transport=httpx.MockTransport(api)),
        )

    monkeypatch.setattr(client_mod, "Client", made)
    path = tmp_path / "data.csv"
    hourly.to_csv(path, index=False)
    assert main(["--json", "tune", str(path), "--h", "24", "--steps", "100"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["id"] == TUNE["id"] and out["used"] is True
    sent = next(b for m, p, b, _ in api.requests if p == "/v1/series/tunes")
    assert sent["steps"] == 100
    assert (
        main(
            [
                "forecast",
                str(path),
                "--h",
                "24",
                "--tune",
                TUNE["id"],
                "--preset",
                "pro",
                "-o",
                str(tmp_path / "fc.csv"),
            ]
        )
        == 0
    )
    body = [b for m, p, b, _ in api.requests if p == "/v1/series/forecast"][-1]
    assert body["tune"] == TUNE["id"] and body["preset"] == "pro"
    assert len(pd.read_csv(tmp_path / "fc.csv")) == 48
    assert main(["check", str(path), "--target-col", "y", "--h", "24"]) == 0
