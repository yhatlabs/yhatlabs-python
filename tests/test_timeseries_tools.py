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
    hs = sorted(b["horizon"] for _, p, b, _ in api.requests if p == "/v1/series/forecast")
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


def test_calendar_with_custom_column_names(client, api):
    df = pd.DataFrame({"Date": pd.date_range("2026-01-01", periods=60, freq="D"), "Temp": np.arange(60.0)})
    fc = client.forecast(df, h=5, time_col="Date", target_col="Temp", calendar=["dayofweek", "month"])
    s = api.requests[-1][2]["series"][0]
    assert set(s["known_covariates"]) == {"dayofweek", "month"} and len(s["known_covariates"]["month"]) == 65
    assert list(fc.df.columns)[:2] == ["unique_id", "Date"]
    client.evaluate(df, h=5, windows=1, time_col="Date", target_col="Temp", calendar=True)


def test_compare_single_series_without_id(client):
    df = pd.DataFrame({"Date": pd.date_range("2026-01-01", periods=60, freq="D"), "Temp": np.arange(60.0)})
    mine = df.tail(10)[["Date"]].assign(forecast=df["Temp"].tail(10).to_numpy() + 1)
    ev = client.evaluate(df, h=5, windows=2, time_col="Date", target_col="Temp", compare={"current": mine})
    assert np.isclose(ev.scores[ev.scores.model == "current"].mae, 1.0).all()


def test_cli_warnings_are_one_line(api, tmp_path, monkeypatch):
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
    df = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=60, freq="D"), "y": np.arange(60.0)}).drop(index=[5])
    src = tmp_path / "d.csv"
    df.to_csv(src, index=False)
    import warnings

    monkeypatch.setattr(warnings, "formatwarning", warnings.formatwarning)  # restored after the test
    assert main(["forecast", str(src), "--h", "3"]) == 0
    line = warnings.formatwarning("Filled 1 missing timestamps", UserWarning, "x.py", 1)
    assert line == "yhatlabs: warning: Filled 1 missing timestamps\n"


def test_anomaly_stream_scores_only_new_points(client, api, hourly):
    stream = client.anomaly_stream(window=50)
    first = stream.update(hourly[hourly.ds < hourly.ds[72]])
    assert first.empty and not any(r[1] == "/v1/series/forecast" for r in api.requests)  # nothing before them yet
    new = hourly[(hourly.ds >= hourly.ds[72]) & (hourly.ds < hourly.ds[84])].copy()
    spike = new.index[new.unique_id == "a"][5]
    new.loc[spike, "y"] += 50
    n = len(api.requests)
    out = stream.update(new)
    assert len(api.requests) == n + 1  # one call for every series
    assert len(out) == 24 and set(out.ds) == set(new.ds)
    assert out[out.anomaly].ds.tolist() == [new.loc[spike, "ds"]] and (out[out.anomaly].unique_id == "a").all()
    assert (stream.history.groupby("unique_id").size() == 50).all()
    assert len(stream.update(hourly[hourly.ds >= hourly.ds[84]])) == 24
    with pytest.raises(DataError, match="window"):
        client.anomaly_stream(window=3)


def test_anomaly_stream_one_series_without_an_id(client, hourly):
    stream = client.anomaly_stream()
    a = hourly[hourly.unique_id == "a"].drop(columns="unique_id")
    stream.update(a.iloc[:80])
    out = stream.update(a.iloc[80:])
    assert len(out) == 16 and (out.unique_id == "series_0").all()


def test_explain_forecast_neutralises_one_covariate_at_a_time(client, monkeypatch):
    from yhatlabs import ForecastResult

    ts = pd.date_range("2026-01-01", periods=30, freq="D")
    hist = pd.concat(
        [
            pd.DataFrame(
                {"unique_id": u, "ds": ts, "y": 1.0, "price": p + np.arange(30.0), "promo": ["on"] * 20 + ["off"] * 10}
            )
            for u, p in (("a", 0.0), ("b", 100.0))
        ],
        ignore_index=True,
    )
    fut = make_future(hist, 3).assign(price=lambda f: f.unique_id.map({"a": 50.0, "b": 200.0}))
    calls = []

    def fake(data, h, *, known, past, future, **kw):
        calls.append((data, future))
        g = future.groupby("unique_id")
        on = data.groupby("unique_id").promo.agg(lambda s: (s == "on").mean())
        df = future[["unique_id", "ds"]].copy()
        df["median"] = g.price.transform("mean") + 10 * df.unique_id.map(on)
        return ForecastResult(df, "chakra-ts", "D", [0.5], [0.5])

    monkeypatch.setattr(client, "forecast", fake)
    out = client.explain_forecast(hist, 3, known=["price"], past=["promo"], future=fut)
    assert len(calls) == 3  # one forecast, then one per covariate
    neutral = calls[1][0].groupby("unique_id").price.unique()
    mean_a = np.mean(np.r_[np.arange(30.0), [50.0] * 3])
    assert np.allclose(neutral["a"], [mean_a]) and (calls[1][1].price[calls[1][1].unique_id == "a"] == mean_a).all()
    assert (calls[2][0].promo == "on").all()  # the most common value
    eff = out.set_index(["covariate", "unique_id"]).effect
    assert np.allclose(eff.loc[("price", "a")], 50.0 - mean_a)
    assert np.allclose(eff.loc[("promo", "a")], 10 * (20 / 30) - 10)
    assert list(out.columns) == ["unique_id", "ds", "covariate", "median", "median_without", "effect"]
    s = out.attrs["summary"]
    assert list(s.columns) == ["covariate", "mean_abs_effect"] and s.mean_abs_effect.is_monotonic_decreasing
    with pytest.raises(DataError, match="known"):
        client.explain_forecast(hist, 3)


def test_explain_forecast_end_to_end(client, api):
    ts = pd.date_range("2026-01-01", periods=40, freq="D")
    df = pd.DataFrame({"ds": ts, "y": np.arange(40.0), "temp": np.sin(np.arange(40.0))})
    fut = make_future(df, 2).assign(temp=0.5)
    out = client.explain_forecast(df, 2, known=["temp"], future=fut)
    assert len(out) == 2 and (out.covariate == "temp").all()
    assert sum(r[1] == "/v1/series/forecast" for r in api.requests) == 2


async def test_async_explain_forecast(aclient, api):
    df = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=40, freq="D"), "y": 1.0, "temp": 2.0, "load": 3.0})
    async with aclient as c:
        out = await c.explain_forecast(df, 2, known=["temp"], past=["load"], future=make_future(df, 2).assign(temp=1.0))
    assert set(out.covariate) == {"temp", "load"} and len(out) == 4
    assert sum(r[1] == "/v1/series/forecast" for r in api.requests) == 3
