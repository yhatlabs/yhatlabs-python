import warnings

import numpy as np
import pandas as pd
import pytest

from yhatlabs import DataError, Limits


def _daily(n=30, **cols):
    return pd.DataFrame(
        {"ds": pd.date_range("2026-01-01", periods=n, freq="D"), "y": np.arange(n, dtype=float), **cols}
    )


def test_empty_table(client):
    with pytest.raises(DataError, match="no rows"):
        client.forecast(pd.DataFrame({"ds": [], "y": []}), h=3)


def test_irregular_timestamps_refused(client):
    rng = np.random.default_rng(0)
    ds = pd.to_datetime("2026-01-01") + pd.to_timedelta(np.sort(rng.choice(5000, 50, replace=False)), unit="h")
    with pytest.raises(DataError, match="irregular"):
        client.forecast(pd.DataFrame({"ds": ds, "y": np.arange(50.0)}), h=3)


def test_a_few_gaps_still_fill(client):
    df = _daily(30).drop(index=[5, 6, 20])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fc = client.forecast(df, h=3)
    assert len(fc.df) == 3 and any("Filled 3" in str(x.message) for x in w)


def test_integer_steps(client, api):
    df = pd.DataFrame({"t": np.arange(10, 50, 2), "y": np.arange(20.0)})
    with pytest.raises(DataError, match="integer steps"):
        client.forecast(df, h=3, time_col="t")
    fc = client.forecast(df.drop(index=[3]), h=3, time_col="t", freq="D")
    assert fc.df["t"].tolist() == [50, 52, 54]
    assert len(api.requests[-1][2]["series"][0]["values"]) == 20  # the dropped step came back as a gap


def test_integer_steps_with_known_future(client, api):
    df = pd.DataFrame({"t": np.arange(20), "y": np.arange(20.0), "temp": np.arange(20.0)})
    fut = pd.DataFrame({"t": [20, 21], "temp": [1.0, 2.0]})
    client.forecast(df, h=2, time_col="t", freq="h", known=["temp"], future=fut)
    assert api.requests[-1][2]["series"][0]["known_covariates"]["temp"][-2:] == [1.0, 2.0]


def test_missing_ids_refused(client):
    df = pd.concat([_daily(20, unique_id="a"), _daily(20, unique_id=None)])
    with pytest.raises(DataError, match="no series id"):
        client.forecast(df, h=3)


def test_too_many_covariates_checked_locally(client, api):
    df = _daily(30, **{f"c{i}": 1.0 for i in range(5)})
    client.limits = Limits(max_columns=4)
    with pytest.raises(DataError, match="at most 4"):
        client.forecast(df, h=3, past=[f"c{i}" for i in range(5)])
    assert not [r for r in api.requests if r[1] == "/v1/forecast"]


def test_rows_are_series(client, api):
    fc = client.forecast([[1.0, 2, 3, 4], [5.0, 6, 7, 8, 9]], h=2, freq="D")
    assert sorted(fc.df.unique_id.unique()) == ["series_0", "series_1"]
    fc = client.forecast(np.arange(12.0).reshape(3, 4), h=2, freq="D")
    assert fc.df.unique_id.nunique() == 3
    fc = client.forecast(np.arange(12.0).reshape(12, 1), h=2, freq="D")
    assert fc.df.unique_id.nunique() == 1
    with pytest.raises(DataError, match="mixes"):
        client.forecast([1.0, [2.0, 3.0]], h=2, freq="D")


def test_inf_warned(client):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        client.forecast(_daily(30).assign(y=[np.inf] + [1.0] * 29), h=3)
    assert any("infinite" in str(x.message) for x in w)


def test_datetime_target_refused(client):
    train = pd.DataFrame({"a": range(20), "t": pd.date_range("2026-01-01", periods=20)})
    with pytest.raises(DataError, match="neither numbers nor classes"):
        client.predict(train, "t", test=train.drop(columns="t"))


def test_posts_carry_one_idempotency_key_across_retries(client, api):
    api.script = [(500, {"error": {"type": "server_error", "message": "boom"}})]
    client.forecast(_daily(30), h=3)
    keys = [r[3].get("idempotency-key") for r in api.requests if r[1] == "/v1/forecast"]
    assert len(keys) == 2 and keys[0] and keys[0] == keys[1]
    client.forecast(_daily(30), h=3)
    assert api.requests[-1][3]["idempotency-key"] != keys[0]


def test_capabilities_read_from_limits():
    lim = Limits.from_api({"forecast": {"max_columns": 64}, "capabilities": {"gzip_requests": True}})
    assert lim.max_columns == 64 and lim.supports("gzip_requests") and not lim.supports("samples")
    assert not Limits.from_api({}).supports("gzip_requests")
