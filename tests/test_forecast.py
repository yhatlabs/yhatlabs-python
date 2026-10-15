import numpy as np
import pandas as pd
import pytest

from yhatlabs import DataError, Limits, UnsupportedError


def test_long_frame(client, api, hourly):
    fc = client.forecast(hourly, h=12)
    assert list(fc.df.columns) == ["unique_id", "ds", "q10", "median", "q90"]
    assert len(fc.df) == 24
    a = fc.df[fc.df.unique_id == "a"]
    assert a.ds.iloc[0] == pd.Timestamp("2026-01-05 00:00")
    assert a.ds.is_monotonic_increasing
    assert np.allclose(a["median"], hourly[hourly.unique_id == "a"].y.iloc[-1])
    assert fc.derived == [] and fc.native_quantile_levels[0] == 0.1
    _, path, body, headers = api.requests[0]
    assert path == "/v1/series/forecast" and body["freq"].lower() == "h" and body["horizon"] == 12
    assert headers["user-agent"].startswith("yhatlabs-python/")
    assert fc.usage == {"series": 2, "requests": 1}


def test_custom_columns_and_unsorted(client, hourly):
    df = hourly.rename(columns={"unique_id": "store", "ds": "when", "y": "sales"}).sample(frac=1, random_state=0)
    fc = client.forecast(df, h=3, id_col="store", time_col="when", target_col="sales")
    assert list(fc.df.columns)[:2] == ["store", "when"]
    assert np.isclose(fc.df[fc.df.store == "b"]["median"].iloc[0], hourly[hourly.unique_id == "b"].y.iloc[-1])


def test_series_with_datetime_index(client):
    s = pd.Series(np.arange(30.0), index=pd.date_range("2026-01-01", periods=30, freq="D"), name="sales")
    fc = client.forecast(s, h=5)
    assert len(fc.df) == 5 and fc.freq == "D"
    assert fc.df["median"].iloc[0] == 29.0


def test_arrays_need_freq(client, api):
    with pytest.raises(DataError, match="freq"):
        client.forecast([1.0, 2.0, 3.0], h=2)
    fc = client.forecast({"x": [1, 2, 3, 4], "y": [5, 6, 7, 8]}, h=2, freq="D")
    assert set(fc.df.unique_id) == {"x", "y"}
    assert api.requests[-1][2]["series"][0]["values"] == [1.0, 2.0, 3.0, 4.0]


def test_gaps_filled_with_warning(client, api, hourly):
    df = hourly.drop(index=[5, 6, 7])
    with pytest.warns(UserWarning, match="Filled 3 missing"):
        client.forecast(df, h=2)
    vals = api.requests[-1][2]["series"][0]["values"]
    assert len(vals) == 96 and vals[5:8] == [None, None, None]


def test_duplicates_rejected(client, hourly):
    with pytest.raises(DataError, match="Duplicate"):
        client.forecast(pd.concat([hourly, hourly.head(2)]), h=2)


def test_mixed_frequencies_rejected(client, hourly):
    daily = pd.DataFrame({"unique_id": "d", "ds": pd.date_range("2026-01-01", periods=10, freq="D"), "y": 1.0})
    with pytest.raises(DataError, match="different frequencies"):
        client.forecast(pd.concat([hourly, daily]), h=2)


def test_known_covariates_aligned(client, api, hourly):
    df = hourly.assign(temp=np.arange(len(hourly), dtype=float), promo=1.0)
    fut = pd.DataFrame(
        {
            "unique_id": np.repeat(["a", "b"], 4),
            "ds": list(pd.date_range("2026-01-05", periods=4, freq="h")) * 2,
            "temp": -1.0,
        }
    )
    fc = client.forecast(df, h=4, known=["temp"], past=["promo"], future=fut)
    s = api.requests[-1][2]["series"][0]
    assert len(s["known_covariates"]["temp"]) == 100 and s["known_covariates"]["temp"][-4:] == [-1.0] * 4
    assert len(s["past_covariates"]["promo"]) == 96
    assert len(fc.df) == 8


def test_known_covariates_must_cover_horizon(client, hourly):
    df = hourly.assign(temp=1.0)
    fut = pd.DataFrame({"unique_id": ["a", "b"], "ds": [pd.Timestamp("2026-01-05")] * 2, "temp": 0.0})
    with pytest.raises(DataError, match="must cover"):
        client.forecast(df, h=3, known=["temp"], future=fut)
    with pytest.raises(DataError, match="future="):
        client.forecast(df, h=3, known=["temp"])


def test_ignored_columns_warn(client, hourly):
    with pytest.warns(UserWarning, match="Ignored columns"):
        client.forecast(hourly.assign(extra=1.0), h=1)


def test_batching_reassembles_in_order(api, hourly):
    import httpx

    from yhatlabs import Client

    many = pd.concat([hourly.assign(unique_id=hourly.unique_id + str(i)) for i in range(5)], ignore_index=True)
    c = Client(
        "test-key",
        base_url="https://fake.test",
        limits=Limits(max_series=3),
        max_concurrency=3,
        http_client=httpx.Client(transport=httpx.MockTransport(api)),
    )
    fc = c.forecast(many, h=2)
    assert len(api.requests) == 4
    assert all(len(r[2]["series"]) <= 3 for r in api.requests)
    for uid in many.unique_id.unique():
        got = fc.df[fc.df.unique_id == uid]["median"].iloc[0]
        assert np.isclose(got, many[many.unique_id == uid].y.iloc[-1])
    assert fc.usage["requests"] == 4 and len(fc.request_ids) == 4


def test_truncates_long_series(api):
    import httpx

    from yhatlabs import Client

    c = Client(
        "test-key",
        base_url="https://fake.test",
        limits=Limits(max_points_per_series=50),
        http_client=httpx.Client(transport=httpx.MockTransport(api)),
    )
    df = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=80, freq="D"), "y": np.arange(80.0)})
    with pytest.warns(UserWarning, match="most recent 50"):
        c.forecast(df, h=1)
    assert len(api.requests[-1][2]["series"][0]["values"]) == 50


def test_quantiles(client, hourly):
    fc = client.forecast(hourly, h=2, quantiles=[0.25, 0.5, 0.8])
    assert list(fc.df.columns)[2:] == ["q25", "median", "q80"]
    assert fc.derived == ["q25"]
    last = hourly[hourly.unique_id == "a"].y.iloc[-1]
    assert np.isclose(fc.df["q25"].iloc[0], last - 2.5)
    assert len(client.forecast(hourly, h=1, quantiles="all").quantile_levels) == 9
    with pytest.raises(UnsupportedError, match="extrapolation"):
        client.forecast(hourly, h=2, quantiles=[0.05, 0.5])
    with pytest.raises(DataError):
        client.forecast(hourly, h=2, quantiles=[1.5])


def test_multi_target(client, api, hourly):
    df = hourly.assign(price=hourly.y * 2)
    fc = client.forecast(df, h=2, target_col=["y", "price"])
    assert "targets" in api.requests[-1][2]["series"][0]
    assert list(fc.df.columns)[:3] == ["unique_id", "ds", "target"] and len(fc.df) == 8


def test_to_long(client, hourly):
    fc = client.forecast(hourly, h=2)
    long = fc.to_long()
    assert len(long) == 2 * 2 * 3
    assert set(long.output_type_id) == {0.1, 0.5, 0.9}
    assert not long["derived"].any()


def test_bad_horizon(client, hourly):
    with pytest.raises(DataError):
        client.forecast(hourly, h=0)
    with pytest.raises(DataError, match="maximum horizon"):
        client.forecast(hourly, h=5000)


async def test_async_forecast(aclient, hourly):
    async with aclient as c:
        fc = await c.forecast(hourly, h=3)
        assert len(fc.df) == 6
        assert (await c.models())[0]["id"] == "chakra-ts"
