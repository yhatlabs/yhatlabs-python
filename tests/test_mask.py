import json

import numpy as np
import pandas as pd
import pytest

from yhatlabs import APIError, load

REAL = ("store_sales", "promo", "temperature", "region", "machine", "churned", "spend")


def _wire(api) -> str:
    return " ".join(json.dumps(r[2], default=str) for r in api.requests if r[2])


def _table(n=60):
    rng = np.random.default_rng(0)
    spend = rng.normal(size=n)
    return pd.DataFrame(
        {
            "spend": spend,
            "region": ["n", "s"] * (n // 2),
            "machine": [f"m{i % 6}" for i in range(n)],
            "churned": (spend > 0).astype(int),
        }
    )


def test_off_by_default_sends_real_names(client, api):
    d = _table()
    client.predict(d[:40], "churned", d[40:].drop(columns="churned"))
    assert '"spend"' in _wire(api)


def test_forecast_sends_no_target_or_covariate_names(client, api, hourly):
    df = hourly.rename(columns={"y": "store_sales"}).assign(
        promo=lambda d: (d.ds.dt.hour == 9).astype(float), temperature=1.0
    )
    fut = df.groupby("unique_id").tail(6)[["unique_id", "ds"]].copy()
    fut["ds"] = fut["ds"] + pd.Timedelta(hours=6)
    fut["promo"] = 0.0
    fc = client.forecast(
        df, h=6, target_col=["store_sales", "temperature"], known=["promo"], future=fut, mask_columns=True
    )
    wire = _wire(api)
    assert not any(n in wire for n in REAL), wire[:300]
    assert set(fc.df["target"]) == {"store_sales", "temperature"}  # names restored in the result


def test_predict_fit_importance_and_errors_keep_real_names(client, api, tmp_path):
    d = _table()
    res = client.predict(
        d[:40], "churned", d[40:].drop(columns=["churned", "machine"]), groups="machine", mask_columns=True
    )
    assert len(res.df) == 20 and not any(n in _wire(api) for n in REAL)

    client.mask_columns = True  # the client's default applies when a call says nothing
    m = client.fit(d[:40].drop(columns="machine"), "churned")
    assert m.target == "churned" and m.mask
    imp = m.importance()
    assert set(imp["feature"]) == {"spend", "region"}
    m.predict(d[40:].drop(columns=["churned", "machine"]))
    assert not any(n in _wire(api) for n in REAL)

    back = load(m.save(tmp_path / "fit.json"), client)
    assert back.mask == m.mask
    back.predict(d[40:].drop(columns=["churned", "machine"]))
    assert not any(n in _wire(api) for n in REAL)

    wire_spend = m.mask["spend"]
    api.script = [(422, {"error": {"type": "invalid_request_error", "message": f"column '{wire_spend}' is constant"}})]
    with pytest.raises(APIError, match="'spend' is constant"):
        m.predict(d[40:].drop(columns=["churned", "machine"]))


def test_evaluate_table_masks_every_fold(client, api):
    d = _table(90)
    client.evaluate_table(d, "churned", cv=3, groups="machine", mask_columns=True)
    assert not any(n in _wire(api) for n in REAL)


def test_async_client_masks_too(aclient, api, hourly):
    import asyncio

    d = _table()

    async def go():
        await aclient.predict(d[:40], "churned", d[40:].drop(columns="churned"), mask_columns=True)
        m = await aclient.fit(d[:40], "churned", mask_columns=True)
        await m.predict(d[40:].drop(columns="churned"))
        df = hourly.rename(columns={"y": "store_sales"}).assign(temperature=1.0)
        await aclient.forecast(df, h=6, target_col=["store_sales", "temperature"], mask_columns=True)

    asyncio.run(go())
    assert not any(n in _wire(api) for n in REAL)
