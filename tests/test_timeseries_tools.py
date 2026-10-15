import numpy as np
import pandas as pd
import pytest

from yhatlabs import DataError
from yhatlabs.features import add_calendar, make_future


def test_level_alias(client, hourly):
    fc = client.forecast(hourly, h=2, level=[80])
    assert fc.quantile_levels == [0.1, 0.5, 0.9]
    with pytest.raises(DataError):
        client.forecast(hourly, h=2, level=[80], quantiles=[0.5])


def test_static_covariates(client, api, hourly):
    df = hourly.assign(region=hourly.unique_id.map({"a": "north", "b": "south"}))
    client.forecast(df, h=3, static=["region"])
    s = api.requests[-1][2]["series"][0]
    assert s["known_covariates"]["region"] == ["north"] * 99
    with pytest.raises(DataError, match="changes within series"):
        client.forecast(hourly.assign(region=np.arange(len(hourly)) % 2), h=3, static=["region"])


def test_partial_failures(api, hourly):
    import httpx

    from yhatlabs import Client, InternalServerError, Limits

    many = pd.concat([hourly.assign(unique_id=hourly.unique_id + str(i)) for i in range(3)], ignore_index=True)
    c = Client(
        "test-key",
        base_url="https://fake.test",
        limits=Limits(max_series=2),
        max_concurrency=1,
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(api)),
    )
    api.fail_batches = {1}
    with pytest.warns(UserWarning, match="series failed"):
        fc = c.forecast(many, h=2, on_error="warn")
    assert len(fc.errors) == 1 and len(fc.errors[0]["ids"]) == 2
    assert fc.df.unique_id.nunique() == 4
    with pytest.raises(InternalServerError):
        api.requests.clear()
        c.forecast(many, h=2)


def test_scenarios(client, hourly):
    df = hourly.assign(price=1.0)
    fut = make_future(df, 4).assign(price=1.0)
    out = client.scenarios(df, 4, {"base": fut, "price up": fut.assign(price=1.1)}, known=["price"])
    assert set(out.scenario) == {"base", "price up"} and len(out) == 2 * 2 * 4


def test_features():
    pytest.importorskip("holidays")
    df = pd.DataFrame({"ds": pd.date_range("2026-12-24", periods=72, freq="h"), "y": 1.0})
    out = add_calendar(df, holidays="US")
    assert {"hour", "dayofweek", "month", "is_weekend", "is_holiday"} <= set(out.columns)
    assert (
        out.loc[out.ds.dt.day == 25, "is_holiday"].eq(1).all()
        and out.loc[out.ds.dt.day == 24, "is_holiday"].eq(0).all()
    )
    fut = make_future(df, 5)
    assert list(fut.columns) == ["unique_id", "ds"] and fut.ds.iloc[0] == pd.Timestamp("2026-12-27 00:00")
    with pytest.raises(DataError):
        add_calendar(df, ["fortnight"])


def test_detect_anomalies(client, hourly):
    df = hourly.copy()
    spike = df.index[(df.unique_id == "a")][-5]
    df.loc[spike, "y"] += 50
    out = client.detect_anomalies(df, h=12, windows=2)
    assert len(out) == 2 * 24 and {"score", "anomaly", "median", "q10", "q90"} <= set(out.columns)
    flagged = out[out.anomaly]
    assert len(flagged) >= 1 and (flagged.unique_id == "a").all()
    assert df.loc[spike, "ds"] in set(flagged.ds)


def test_fill_gaps(client, api, hourly):
    df = hourly.drop(index=[40, 41, 42]).copy()  # missing timestamps
    df.loc[60, "y"] = np.nan  # a missing value
    out = client.fill_gaps(df)
    a = out[out.unique_id == "a"].reset_index(drop=True)
    assert len(a) == 96 and a.y.notna().all() and a.y_filled.sum() == 4
    assert np.isclose(a.loc[40, "y"], hourly.loc[39, "y"])  # the fake forecasts the last value seen before the gap
    hs = sorted(b["horizon"] for _, p, b, _ in api.requests if p == "/v1/forecast")
    assert hs == [1, 3]  # one call per gap length


def test_evaluate_compare(client, hourly):
    fc = hourly.groupby("unique_id").tail(36).rename(columns={"y": "forecast"})
    fc["forecast"] = fc["forecast"] + 0.5
    ev = client.evaluate(hourly, h=12, windows=3, compare={"current": fc})
    assert "current" in set(ev.scores.model)
    assert np.isclose(ev.scores[ev.scores.model == "current"].mae, 0.5).all()
    assert np.isnan(ev.scores[ev.scores.model == "current"]["coverage@0.8"]).all()


async def test_async_evaluate(aclient, hourly):
    async with aclient as c:
        ev = await c.evaluate(hourly, h=12, windows=2)
        assert set(ev.scores.model) == {"chakra-ts", "seasonal_naive"}


def test_plot_smoke(client, hourly):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    fig = client.forecast(hourly, h=12).plot(history=hourly)
    assert len(fig.axes) == 2


def test_cli(api, hourly, tmp_path, monkeypatch, capsys):
    import httpx

    import yhatlabs._transport as t
    from yhatlabs._cli import main

    orig = t.SyncTransport.__init__
    monkeypatch.setattr(
        t.SyncTransport,
        "__init__",
        lambda self, k, b, timeout, mr, hc: orig(
            self, "test-key", "https://fake.test", timeout, mr, httpx.Client(transport=httpx.MockTransport(api))
        ),
    )
    src = tmp_path / "in.csv"
    hourly.to_csv(src, index=False)
    out = tmp_path / "out.csv"
    assert main(["forecast", str(src), "--h", "6", "--level", "80", "-o", str(out)]) == 0
    assert len(pd.read_csv(out)) == 12
    assert main(["anomalies", str(src), "--h", "12", "--windows", "1"]) == 0
    assert "score" in capsys.readouterr().out
    assert main(["models"]) == 0 and "chakra-ts" in capsys.readouterr().out
    assert main(["forecast", str(src), "--h", "0"]) == 1


def test_holidays_missing_gives_clear_error(monkeypatch):
    import builtins

    from yhatlabs import ConfigurationError

    real = builtins.__import__
    monkeypatch.setattr(
        builtins,
        "__import__",
        lambda n, *a, **k: (_ for _ in ()).throw(ImportError(n)) if n == "holidays" else real(n, *a, **k),
    )
    with pytest.raises(ConfigurationError, match="pip install holidays"):
        add_calendar(pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=3, freq="D")}), holidays="US")
