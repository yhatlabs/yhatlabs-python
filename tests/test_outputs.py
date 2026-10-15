import asyncio
import json

import numpy as np
import pandas as pd
import pytest

from yhatlabs import DataError, FitGroup, GroupPrediction, TableEvaluation, load
from yhatlabs._forecast import cdf_at, mean_from_quantiles

LV = np.arange(1, 10) / 10


def test_mean_from_quantiles_is_exact_on_symmetric_and_close_on_skewed():
    stats = pytest.importorskip("scipy.stats")

    assert np.isclose(mean_from_quantiles(stats.norm(5, 2).ppf(LV)[None], LV)[0], 5)
    assert abs(mean_from_quantiles(stats.expon().ppf(LV)[None], LV)[0] - 1) < 0.01
    assert abs(mean_from_quantiles(stats.gamma(2).ppf(LV)[None], LV)[0] - 2) < 0.02


def test_cdf_at_and_point_mass():
    stats = pytest.importorskip("scipy.stats")

    d = stats.norm(5, 2)
    x = np.array([3.0, 5.0, 8.0])
    assert np.allclose(cdf_at(np.tile(d.ppf(LV), (3, 1)), LV, x), d.cdf(x), atol=0.01)
    zeros = np.array([[0, 0, 0, 0, 0, 1, 2, 3, 5.0]])
    assert cdf_at(zeros, LV, np.array([0.0]))[0] == 0.5 and cdf_at(zeros, LV, np.array([-1.0]))[0] == 0


def test_mean_intervals_wide_and_prob(client, hourly):
    fc = client.forecast(hourly, h=6, quantiles="all", mean=True)
    assert "mean" in fc.df.columns and "mean" in fc.derived
    assert set(fc.to_long().output_type) == {"quantile", "mean"}
    iv = fc.intervals([60, 80])
    assert list(iv.columns) == ["unique_id", "ds", "median", "lower_60", "upper_60", "lower_80", "upper_80"]
    assert (iv.lower_80 <= iv.lower_60).all() and (iv.upper_80 >= iv.upper_60).all()
    with pytest.raises(DataError, match="level=\\[50\\]"):
        fc.intervals(50)
    wide = fc.to_wide()
    assert list(wide.columns) == ["a", "b"] and len(wide) == 6
    p = fc.prob(above=fc.df["median"].to_numpy())
    assert np.allclose(p["prob"], 0.5)
    assert (fc.prob(below=-1e9)["prob"] < 1e-6).all()
    with pytest.raises(DataError, match="exactly one"):
        fc.prob()


def test_plots(client, hourly):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from yhatlabs import plots

    fc = client.forecast(hourly, h=6, level=[80])
    assert plots.forecast(fc, history=hourly) is not None
    assert client.evaluate(hourly, h=6, windows=2).plot() is not None
    assert plots.anomalies(client.detect_anomalies(hourly)) is not None


def _table(n=120, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"a": rng.normal(size=n), "line": rng.choice(["A", "B"], n), "machine": np.repeat(np.arange(12), n // 12)}
    )
    df["t"] = pd.date_range("2026-01-01", periods=n)
    df["cls"] = (df.a > 0).astype(int)
    df["y"] = 2 * df.a + rng.normal(size=n) / 3 + 0.5
    return df


def test_evaluate_table_splits(client, api):
    df = _table()
    ev = client.evaluate_table(df.drop(columns=["y", "t"]), "cls", groups="machine", cv=4)
    assert isinstance(ev, TableEvaluation) and ev.task == "classify" and ev.split == "grouped"
    assert len(ev.scores) == 4 and {"accuracy", "log_loss", "auc", "brier", "ece"} <= set(ev.summary)
    assert ev.reliability is not None and ev.predictions.index.equals(df.index)
    tab = [r for r in api.requests if r[1] == "/v1/tabular/predict"]
    for _, _, body, _ in tab:  # a group never sits on both sides of a split, and the group column is not a feature
        assert "machine" not in body["train"][0]
    ev = client.evaluate_table(df.drop(columns=["cls"]), "y", time_col="t", cv=3)
    assert ev.task == "regress" and ev.split == "time-ordered" and ev.coverage is not None and ev.pit is not None
    assert {"mae", "rmse", "r2", "crps", "coverage_90"} <= set(ev.summary)
    assert ev.predictions.index.equals(df.index[30:])  # the first block only ever trains
    with pytest.raises(DataError, match="at least 2"):
        client.evaluate_table(df, "cls", cv=1)
    with pytest.raises(DataError, match="not both"):
        client.evaluate_table(df, "cls", groups="machine", time_col="t")


def test_evaluate_table_async(aclient):
    df = _table()
    ev = asyncio.run(aclient.evaluate_table(df.drop(columns=["y", "t", "machine"]), "cls", cv=3))
    assert len(ev.scores) == 3 and len(ev.predictions) == len(df)


def test_out_of_range_and_novelty_survive_save_and_load(client, tmp_path):
    rng = np.random.default_rng(0)
    tr = pd.DataFrame({"temp": rng.normal(20, 3, 400), "line": rng.choice(["A", "B"], 400)})
    tr["y"] = tr.temp * 2.5
    te = pd.DataFrame({"temp": [20.0, 45.0, 21.0], "line": ["A", "B", "C"]})
    res = client.predict(tr, "y", test=te)
    assert res.out_of_range.tolist() == [[], ["temp"], ["line"]]
    assert res.novelty.iloc[0] < 0.9 and res.novelty.iloc[1] > 0.99
    m = client.fit(tr, "y")
    m.save(tmp_path / "fit.json")
    back = load(tmp_path / "fit.json", client=client)
    assert back.predict(te).out_of_range.tolist() == [[], ["temp"], ["line"]]
    assert client.fit(tr, "y", profile=False).predict(te).out_of_range is None


def test_prob_on_regression_intervals(client):
    tr = pd.DataFrame({"a": np.arange(40.0), "y": np.arange(40.0) + 0.5})
    res = client.predict(tr, "y", test=tr[["a"]].head(3), quantiles=[0.1, 0.5, 0.9])
    assert np.allclose(res.prob(above=res.quantiles["median"].to_numpy()), 0.5)
    with pytest.raises(DataError, match="quantiles"):
        client.predict(tr, "y", test=tr[["a"]].head(3)).prob(above=1)


def test_effect(client):
    rng = np.random.default_rng(0)
    tr = pd.DataFrame({"temp": rng.normal(20, 3, 300), "line": rng.choice(["A", "B"], 300)})
    tr["y"] = tr.temp * 2.5
    m = client.fit(tr, "y")
    e = m.effect("temp", grid=5, rows=50)
    assert list(e.columns) == ["temp", "average", "low", "high"] and len(e) == 5 and e.temp.is_monotonic_increasing
    assert e.attrs["predicted_rows"] == 250
    assert list(m.effect("line").line) == ["A", "B"] or list(m.effect("line").line) == ["B", "A"]
    tr["c"] = (tr.temp > 20).astype(int)
    e = client.fit(tr.drop(columns="y"), "c").effect("temp", grid=3)
    assert "output" in e.columns and len(e) == 6
    with pytest.raises(DataError, match="not a feature"):
        m.effect("nope")


def test_fit_group(client, api, tmp_path):
    df = _table().drop(columns=["t", "machine"])
    g = client.fit(df, ["cls", "y"], task={"cls": "classify", "y": "regress"})
    assert isinstance(g, FitGroup) and g.targets == ["cls", "y"]
    for _, path, body, _ in api.requests:
        if path == "/v1/fits":  # each target is fitted without the other target
            assert {"cls", "y"} & set(body["train"][0]) == {body["target"]}
    p = g.predict(df.drop(columns=["cls", "y"]).head(4), quantiles=[0.1, 0.9])
    assert isinstance(p, GroupPrediction) and p["y"].quantiles is not None and p["cls"].probabilities is not None
    assert set(p.df.columns.get_level_values(0)) == {"cls", "y"}
    assert set(g.importance().target) == {"cls", "y"}
    g.save(tmp_path / "g.json")
    assert json.loads((tmp_path / "g.json").read_text())["format"] == "yhatlabs.fitgroup/1"
    back = load(tmp_path / "g.json", client=client)
    assert isinstance(back, FitGroup) and back.targets == ["cls", "y"]
    g.delete()
    assert api.fits == {}
    with pytest.raises(DataError, match="not columns"):
        client.fit(df, ["cls", "nope"])


def test_fit_group_cleans_up_when_one_target_fails(client, api):
    df = _table().drop(columns=["t", "machine"])
    df["bad"] = 1  # one class: refused before sending
    with pytest.raises(DataError):
        client.fit(df, ["y", "bad"])
    assert api.fits == {}


def test_fit_group_async(aclient):
    from yhatlabs import AsyncFitGroup

    df = _table().drop(columns=["t", "machine"])

    async def run():
        g = await aclient.fit(df, ["cls", "y"])
        p = await g.predict(df.drop(columns=["cls", "y"]).head(2))
        await g.delete()
        return g, p

    g, p = asyncio.run(run())
    assert isinstance(g, AsyncFitGroup) and set(p.results) == {"cls", "y"}


def test_estimate_table_reads_the_published_rates(client):
    from yhatlabs import Limits, UnsupportedError

    rates = {
        "unit": "credits",
        "fit_cells_per_credit": 1_000_000,
        "preset_multiplier": {"fast": 1, "medium": 3, "full": 8},
        "predict_rows_per_credit": 10_000,
        "min_credits": 1,
    }
    client.limits = Limits(pricing={"tabular": rates})
    df = pd.DataFrame({"a": range(100_000), "b": 1.0, "y": 0.0})
    est = client.estimate_table(df, target="y", test=df.head(25_000), preset="medium")
    assert est == {"fit": 3, "predict": 3, "total": 6, "unit": "credits"}  # 200k cells -> 1 x 3 folds; 25k rows -> 3
    assert client.estimate_table(train_rows=200_000, features=100, preset="full")["fit"] == 160
    client.limits = Limits()
    with pytest.raises(UnsupportedError):
        client.estimate_table(train_rows=10)


def test_groups_time_col_and_prediction_sets(client, api):
    df = _table().drop(columns=["y"])
    test = df.drop(columns=["cls", "machine"]).head(3)  # the groups column is not a feature: test may lack it
    res = client.predict(df, "cls", test=test, groups="machine", set_coverage=0.9)
    body = [r for r in api.requests if r[1] == "/v1/tabular/predict"][-1][2]
    assert body["groups"] == "machine" and body["set_coverage"] == 0.9 and "machine" not in body["test"][0]
    assert res.set_coverage == 0.9 and res.prediction_sets.tolist() == [[0, 1]] * 3  # labels in their own type
    assert "prediction_set" in res.df.columns
    client.predict(df.drop(columns="machine"), "cls", test=test, time_col="t")  # time_col stays a feature
    assert [r for r in api.requests if r[1] == "/v1/tabular/predict"][-1][2]["time_col"] == "t"
    with pytest.raises(DataError, match="not both"):
        client.predict(df, "cls", test=test, groups="machine", time_col="t")
    with pytest.raises(DataError, match="must name a column"):
        client.predict(df, "cls", test=test, groups="nope")
    with pytest.raises(DataError, match="classification"):
        client.predict(_table().drop(columns="cls"), "y", test=test, task="regress", set_coverage=0.9)
    m = client.fit(df, "cls", groups="machine")
    assert [r for r in api.requests if r[1] == "/v1/fits"][-1][2]["groups"] == "machine"
    assert "machine" not in m.features
    assert m.predict(test, set_coverage=0.8).set_coverage == 0.8
