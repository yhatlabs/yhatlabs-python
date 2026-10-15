import pickle

import numpy as np
import pandas as pd
import pytest

from yhatlabs import DataError, FittedModel, UnsupportedError


@pytest.fixture
def table():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"age": rng.integers(20, 70, 60), "city": rng.choice(["NY", "SF"], 60)})
    return X.assign(churn=[1.0, 0.0] * 30, spend=rng.gamma(2, 50, 60))


def test_fit_once_predict_many(client, api, table):
    m = client.fit(table.drop(columns="spend"), "churn", task="classify", preset="medium", ttl_hours=2)
    assert isinstance(m, FittedModel) and m.id.startswith("fit_") and m.classes == [0.0, 1.0]
    body = api.requests[-1][2]
    assert (
        "test" not in body
        and body["ttl_hours"] == 2.0
        and body["preset"] == "medium"
        and body["problem_type"] == "binary"
    )
    n_before = len(api.requests)
    for _ in range(3):
        res = m.predict(table[["age", "city"]].iloc[:5])
        assert res.fit_id == m.id and list(res.probabilities.columns) == [0.0, 1.0]
    sent = [r for r in api.requests[n_before:]]
    assert all(r[1] == f"/v1/fits/{m.id}/predict" and "train" not in r[2] for r in sent)  # only the rows to score
    assert isinstance(res.predictions.iloc[0], float)  # labels restored to the training type
    with pytest.raises(DataError, match="missing feature columns"):
        m.predict(table[["age"]])


def test_regression_intervals_and_importance(client, api, table):
    m = client.fit(table.drop(columns="churn"), "spend", task="regress")
    res = m.predict(table[["age", "city"]], quantiles=[0.05, 0.95])
    assert list(res.quantiles.columns) == ["q5", "q95"]
    assert (res.quantiles["q5"] < res.predictions).all() and (res.predictions < res.quantiles["q95"]).all()
    imp = m.importance()
    assert list(imp.columns) == ["feature", "importance", "stddev"] and imp.attrs["source"] == "validation"
    imp2 = m.importance(table.drop(columns="churn").iloc[:20])
    assert imp2.attrs["source"] == "data" and api.requests[-1][2]["data"][0]["spend"] > 0
    with pytest.raises(DataError, match="regression only"):
        client.fit(table.drop(columns="spend"), "churn", task="classify").predict(
            table[["age", "city"]], quantiles=[0.1]
        )


def test_delete_context_manager_and_pickle(client, api, table):
    with client.fit(table.drop(columns="spend"), "churn") as m:
        fid = m.id
        assert fid in api.fits
    assert fid not in api.fits
    m2 = client.fit(table.drop(columns="spend"), "churn")
    restored = pickle.loads(pickle.dumps(m2))
    assert restored._client is None and restored.labels == m2.labels
    with pytest.raises(Exception, match="bind"):
        restored.predict(table[["age", "city"]])
    assert len(restored.bind(client).predict(table[["age", "city"]]).predictions) == 60
    assert any(f["id"] == m2.id for f in client.fits())
    assert client.get_fit(m2.id).id == m2.id


def test_old_gateway_without_fits(client, api, table, monkeypatch):
    api.no_fits = True
    with pytest.raises(UnsupportedError, match="does not offer fitted models"):
        client.fit(table.drop(columns="spend"), "churn")
    pytest.importorskip("sklearn")
    from yhatlabs.integrations import ChakraTabClassifier

    monkeypatch.setattr(ChakraTabClassifier, "_client", lambda self: client)
    clf = ChakraTabClassifier().fit(table[["age"]], table.churn)
    assert clf.fit_ is None and len(clf.predict(table[["age"]])) == 60  # falls back to one-shot calls
    assert api.requests[-1][1] == "/v1/tabular/predict"


def test_sklearn_uses_one_fit(client, api, table, monkeypatch):
    pytest.importorskip("sklearn")
    from yhatlabs.integrations import ChakraTabRegressor

    monkeypatch.setattr(ChakraTabRegressor, "_client", lambda self: client)
    reg = ChakraTabRegressor(ttl_hours=1).fit(table[["age"]], table.spend)
    n = len(api.requests)
    reg.predict(table[["age"]].iloc[:10])
    reg.predict(table[["age"]].iloc[10:20])
    q = reg.predict_quantiles(table[["age"]].iloc[10:14], (0.1, 0.9))
    assert list(q.columns) == ["q10", "q90"] and list(q.index) == [10, 11, 12, 13]
    assert [r[1] for r in api.requests[n:]] == [f"/v1/fits/{reg.fit_.id}/predict"] * 3
    assert list(reg.importance().feature) == ["age"]
    clone = pickle.loads(pickle.dumps(reg))
    assert clone.fit_.id == reg.fit_.id
    assert len(clone.predict(table[["age"]].iloc[:3])) == 3  # an unpickled estimator reattaches and predicts


def test_impute(client, api):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=40), "b": rng.choice(["x", "y"], 40), "c": rng.normal(size=40)})
    df.loc[::7, "a"] = np.nan
    df.loc[::9, "b"] = None
    out = client.impute(df)
    assert out["a"].notna().all() and out["b"].notna().all()
    assert out.attrs["filled"]["a"].sum() == df["a"].isna().sum()
    assert out.loc[~df["a"].isna(), "a"].equals(df.loc[~df["a"].isna(), "a"])


async def test_async_fit(aclient, table):
    async with aclient as c:
        m = await c.fit(table.drop(columns="churn"), "spend", task="regress")
        res = await m.predict(table[["age", "city"]].iloc[:3], quantiles=[0.1, 0.9])
        assert res.quantiles.shape == (3, 2)
        await m.delete()


def test_server_limits(client):
    lim = client.server_limits()
    assert (
        lim["forecast"]["max_series"] == 2 and client.limits.max_series == 2 and client.limits.max_tabular_rows == 1000
    )
