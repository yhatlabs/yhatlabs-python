import numpy as np
import pandas as pd
import pytest

from yhatlabs import DataError


@pytest.fixture
def clf_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "age": rng.integers(20, 70, 40),
            "city": rng.choice(["NY", "SF"], 40),
            "joined": pd.date_range("2025-01-01", periods=40, freq="D"),
        }
    )
    y = pd.Series([1] * 25 + [0] * 15, name="churn")
    return X, y


def test_classification(client, api, clf_data):
    X, y = clf_data
    train = X.iloc[:30].assign(churn=y.iloc[:30])
    test = X.iloc[30:]
    res = client.predict(train, "churn", test, task="classify", preset="medium")
    assert list(res.predictions.index) == list(test.index)
    assert res.classes == [0, 1] and res.predictions.iloc[0] == 1  # labels restored to int
    assert list(res.probabilities.columns) == [0, 1]
    assert np.allclose(res.probabilities.sum(axis=1), 1)
    body = api.requests[-1][2]
    assert body["problem_type"] == "binary" and body["preset"] == "medium"
    assert body["train"][0]["joined"].startswith("2025-01-01") and isinstance(body["train"][0]["age"], int)
    assert "churn" not in body["test"][0]
    assert list(res.df.columns) == ["prediction", "p_0", "p_1"]


def test_single_table_mode(client, clf_data):
    X, y = clf_data
    df = X.assign(churn=y.astype(float))
    df.loc[35:, "churn"] = np.nan
    res = client.predict(df, "churn")
    assert list(res.predictions.index) == list(range(35, 40))


def test_regression(client, api):
    df = pd.DataFrame({"x": np.arange(20.0), "price": np.arange(20.0) * 1.5})
    res = client.predict(df.iloc[:15], "price", df.iloc[15:].drop(columns="price"), task="regress")
    assert res.probabilities is None and res.predictions.dtype == float
    assert api.requests[-1][2]["problem_type"] == "regression"
    res = client.predict(
        df.iloc[:15], "price", df.iloc[15:].drop(columns="price"), task="regress", quantiles=[0.9, 0.05, 0.5]
    )
    assert list(res.quantiles.columns) == ["q5", "median", "q90"] and list(res.quantiles.index) == list(range(15, 20))
    assert api.requests[-1][2]["quantiles"] == [0.05, 0.5, 0.9]
    assert list(res.df.columns) == ["prediction", "q5", "median", "q90"]


def test_validation(client, clf_data):
    X, y = clf_data
    train = X.assign(churn=y)
    with pytest.raises(DataError, match="not a column"):
        client.predict(train, "nope", X)
    with pytest.raises(DataError, match="missing feature columns"):
        client.predict(train, "churn", X.drop(columns="age"))
    with pytest.raises(DataError, match="Pass test="):
        client.predict(train, "churn")
    with pytest.raises(DataError, match="preset"):
        client.predict(train, "churn", X, preset="huge")
    with pytest.warns(UserWarning, match="dropped before sending"):
        client.predict(train, "churn", train)


def test_sklearn_estimators(client, clf_data, monkeypatch):
    pytest.importorskip("sklearn")
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score

    from yhatlabs.integrations import ChakraTabClassifier, ChakraTabRegressor

    monkeypatch.setattr(ChakraTabClassifier, "_client", lambda self: client)
    monkeypatch.setattr(ChakraTabRegressor, "_client", lambda self: client)
    X, y = clf_data
    X = X[["age"]]
    clf = ChakraTabClassifier(preset="fast").fit(X, y)
    assert list(clf.classes_) == [0, 1] and clf.n_features_in_ == 1
    proba = clf.predict_proba(X.iloc[:5])
    assert proba.shape == (5, 2)
    assert list(clf.predict(X.iloc[:5])) == [1] * 5
    assert clone(clf).get_params()["preset"] == "fast"
    assert len(cross_val_score(ChakraTabClassifier(), X, y, cv=3)) == 3
    reg = ChakraTabRegressor().fit(np.arange(20.0).reshape(-1, 1), np.arange(20.0) + 0.5)
    assert reg.predict(np.array([[1.0], [2.0]])).shape == (2,)


def test_lazy_sklearn_import():
    import yhatlabs

    assert yhatlabs.ChakraTabClassifier.__name__ == "ChakraTabClassifier"


def test_float_labels_survive_json_round_trip(client, clf_data, monkeypatch):
    X, y = clf_data
    yf = y.astype(float)
    res = client.predict(X.iloc[:30].assign(churn=yf.iloc[:30]), "churn", X.iloc[30:], task="classify")
    assert res.classes == [0.0, 1.0] and isinstance(res.predictions.iloc[0], float)
    pytest.importorskip("sklearn")
    from yhatlabs.integrations import ChakraTabClassifier

    monkeypatch.setattr(ChakraTabClassifier, "_client", lambda self: client)
    clf = ChakraTabClassifier().fit(X[["age"]], yf.to_numpy())
    proba = clf.predict_proba(X[["age"]].iloc[:3])
    assert np.allclose(proba.sum(axis=1), 1) and proba[0, 1] > 0
    assert list(clf.predict(X[["age"]].iloc[:3])) == [1.0, 1.0, 1.0]


def test_label_keys():
    from yhatlabs._tabular import _key

    assert _key(1) == _key(1.0) == _key("1") == _key(np.int64(1)) == "1"
    assert _key(True) == "True" != _key(1) and _key(0.5) == "0.5"


def test_local_checks(client, clf_data):
    X, y = clf_data
    with pytest.raises(DataError, match="nothing to learn"):
        client.predict(X.assign(churn=1), "churn", X)
    with pytest.raises(DataError, match="not a metric name"):
        client.predict(X.assign(churn=y), "churn", X, eval_metric="F1 score")
    client.predict(X.assign(churn=y), "churn", X, eval_metric="f1")  # digits are fine now
    dup = pd.DataFrame([[1, 2, 0], [3, 4, 1], [5, 6, 0]], columns=[1, "1", "t"])
    with pytest.raises(DataError, match="duplicate column names"):
        client.predict(dup, "t", dup.drop(columns="t"))


def test_body_size_limit(api, clf_data):
    import httpx

    from yhatlabs import Client, Limits

    c = Client(
        "test-key",
        base_url="https://fake.test",
        limits=Limits(max_tabular_bytes=1000),
        http_client=httpx.Client(transport=httpx.MockTransport(api)),
    )
    X, y = clf_data
    with pytest.raises(DataError, match="at most"):
        c.predict(X.assign(churn=y), "churn", X)
    assert api.requests == []


def test_cells():
    from yhatlabs._tabular import _cell

    assert _cell(2**60) == str(2**60) and _cell(5) == 5
    assert _cell(pd.Timestamp("2026-01-01 12:00", tz="America/Chicago")) == "2026-01-01T18:00:00+00:00"


def test_intervals_missing_from_old_api(client, api, monkeypatch):
    from yhatlabs import UnsupportedError

    orig = api.tabular
    monkeypatch.setattr(api, "tabular", lambda tr, te, t, b: orig(tr, te, t, {**b, "quantiles": None}))
    df = pd.DataFrame({"x": np.arange(20.0), "price": np.arange(20.0) * 1.5})
    with pytest.raises(UnsupportedError, match="prediction intervals"):
        client.predict(df.iloc[:15], "price", df.iloc[15:].drop(columns="price"), task="regress", quantiles=[0.1, 0.9])


def test_task_accepts_the_api_names():
    from yhatlabs import _tabular

    y = pd.Series([1.5, 2.0, 3.25, 4.0])
    assert _tabular.problem_type(y, "regression") == _tabular.problem_type(y, "regress")
    assert _tabular.norm_task("classification") == "classify"
    assert _tabular.norm_task("binary") == "classify"


def test_outliers_new_rows_and_leave_one_out():
    import yhatlabs

    rng = np.random.default_rng(0)
    train = pd.DataFrame({"x": rng.normal(size=300), "z": rng.normal(size=300), "kind": rng.choice(["a", "b"], 300)})
    new = pd.DataFrame({"x": [0.0, 9.0], "z": [0.0, 0.0], "kind": ["a", "c"]}, index=[10, 11])
    out = yhatlabs.outliers(train, new)
    assert list(out.index) == [10, 11] and list(out.columns) == ["novelty", "outlier", "out_of_range"]
    assert out.outlier.tolist() == [False, True] and out.novelty[11] == 1.0
    assert out.out_of_range[10] == [] and out.out_of_range[11] == ["x", "kind"]
    far = train.copy()
    far.loc[7, "x"] = 50.0
    loo = yhatlabs.outliers(far, threshold=0.99)
    assert len(loo) == 300 and loo.outlier[7] and loo.out_of_range[7] == ["x"]
    assert loo.outlier.mean() <= 0.02  # about 1% of ordinary rows past the 99th percentile
    assert yhatlabs.outliers(train.drop(columns="kind"), new).out_of_range[11] == ["x"]  # only shared columns
    with pytest.raises(DataError, match="threshold"):
        yhatlabs.outliers(train, threshold=99)


def test_outliers_sample_larger_tables():
    import yhatlabs

    rng = np.random.default_rng(1)
    train = pd.DataFrame({"x": rng.normal(size=2500)})
    train.loc[0, "x"] = 40.0
    loo = yhatlabs.outliers(train)
    assert loo.outlier[0] and 0.003 < loo.outlier.mean() < 0.03
