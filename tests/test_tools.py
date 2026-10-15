import json
import logging

import numpy as np
import pandas as pd
import pytest

import yhatlabs
from yhatlabs import DataError, Limits


def test_check_series(hourly):
    df = pd.concat([hourly, hourly.iloc[[3]]])  # an exact duplicate row
    df = df.drop(index=[10, 11])  # missing timestamps
    df.loc[df.index[50], "y"] = np.nan
    rep = yhatlabs.check(df, h=200)
    codes = set(rep.issues.code)
    assert {"duplicate_rows", "missing_timestamps", "missing_values", "short_history"} <= codes
    assert rep.ok and len(rep.profile) == 2 and "DataCheck(timeseries: 2 series" in repr(rep)
    fixed = rep.repaired()
    assert len(fixed[fixed.unique_id == "a"]) == 96 and not fixed.duplicated(["unique_id", "ds"]).any()
    bad = hourly.copy()
    bad.loc[5, "y"] = 999.0
    bad = pd.concat([bad, bad.iloc[[5]].assign(y=1.0)])
    assert "duplicate_timestamps" in set(yhatlabs.check(bad).issues.code) and not yhatlabs.check(bad).ok


def test_check_table():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(300)],
            "a": rng.normal(size=300),
            "const": 1,
            "churn": (rng.random(300) < 0.03).astype(int),
        }
    )
    df["leak"] = df["churn"]
    rep = yhatlabs.check(df, target="churn")
    codes = set(rep.issues.code)
    assert {"id_like", "constant_column", "target_copy", "imbalanced"} <= codes and not rep.ok
    assert rep.kind == "tabular" and set(rep.profile.column) == set(df.columns)


def test_save_and_load_fit(client, api, tmp_path):
    train = pd.DataFrame({"a": range(30), "y": [1.0, 0.0] * 15})
    m = client.fit(train, "y", task="classify")
    rec = m.save(tmp_path / "m.json")
    assert rec["format"] == "yhatlabs.fit/1" and json.loads((tmp_path / "m.json").read_text())["id"] == m.id
    m2 = yhatlabs.load(tmp_path / "m.json", client=client)
    assert m2.id == m.id and m2.labels == [1.0, 0.0] and m2.features == ["a"]
    assert isinstance(m2.predict(train[["a"]].head(3)).predictions.iloc[0], float)
    with pytest.raises(DataError, match="Not a yhatlabs fit record"):
        yhatlabs.load({"format": "other"}, client=client)
    with pytest.warns(UserWarning, match="expired"):
        yhatlabs.load({**rec, "expires_at": 1}, client=client)


def test_with_options_and_whoami(client):
    slow = client.with_options(timeout=1800, max_retries=0)
    assert slow._t.timeout == 1800 and slow._t.max_retries == 0 and client._t.timeout == 300
    slow.close()
    assert client.models()  # the shared connection is still open
    me = client.whoami()
    assert me["key"] == "***" and me["models"] == ["chakra-ts"] and "used" in me["usage"]


def test_limits_fetched_lazily(api):
    import httpx

    from yhatlabs import Client

    c = Client("test-key", base_url="https://fake.test", http_client=httpx.Client(transport=httpx.MockTransport(api)))
    assert not api.requests
    assert c.limits.max_series == 2 and [r[1] for r in api.requests] == ["/v1/limits"]
    assert c.limits.max_series == 2 and len(api.requests) == 1  # cached


def test_logging(client, caplog):
    with caplog.at_level(logging.INFO, logger="yhatlabs"):
        client.models()
    assert any("GET /v1/models -> 200" in r.message for r in caplog.records)
    assert not any("test-key" in r.message for r in caplog.records)


def test_calendar_forecast(client, api, hourly):
    fc = client.forecast(hourly, h=6, calendar=True)
    s = api.requests[-1][2]["series"][0]
    assert set(s["known_covariates"]) == {"hour", "dayofweek", "month", "is_weekend"}
    assert len(s["known_covariates"]["hour"]) == 96 + 6 and s["known_covariates"]["hour"][96] == 0
    assert set(fc.provenance["settings"]["known"]) == {"hour", "dayofweek", "month", "is_weekend"}
    client.evaluate(hourly, h=6, windows=1, calendar={"features": ["hour"]})
    assert set(api.requests[-1][2]["series"][0]["known_covariates"]) == {"hour"}


def test_provenance_and_timing(client, hourly):
    fc = client.forecast(hourly, h=3)
    p = fc.provenance
    assert (
        p["model"] == "chakra-ts" and p["endpoint"] == "/v1/series/forecast" and p["sdk"].startswith("yhatlabs-python")
    )
    assert len(p["data_fingerprint"]) == 16 and p["settings"]["h"] == 3 and p["request_ids"]
    assert client.forecast(hourly, h=3).provenance["data_fingerprint"] == p["data_fingerprint"]
    assert fc.timing["total_s"] >= 0 and fc.timing["requests"] == 1
    res = client.predict(
        pd.DataFrame({"a": range(20), "y": np.arange(20.0) + 0.5}), "y", pd.DataFrame({"a": [1, 2]}), task="regress"
    )
    assert res.timing["fit_s"] == 0.1 and "total_s" in res.timing and res.provenance["settings"]["target"] == "y"


def test_calibration(client, hourly):
    ev = client.evaluate(hourly, h=12, windows=2)
    cal = ev.calibration
    assert set(cal.kind) == {"below", "interval"} and set(cal.model) == {"chakra-ts", "seasonal_naive"}
    row = cal[(cal.model == "chakra-ts") & (cal.kind == "interval") & np.isclose(cal.nominal, 0.8)]
    assert len(row) == 1 and 0 <= row.observed.iloc[0] <= 1 and np.isclose(row.gap.iloc[0], row.observed.iloc[0] - 0.8)
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    assert ev.plot_calibration() is not None


def test_polars_in_and_out(client, hourly):
    pl = pytest.importorskip("polars")
    fc = client.forecast(pl.from_pandas(hourly), h=2)
    assert len(fc.df) == 4 and fc.to_polars().shape == (4, 5)
    t = pl.DataFrame({"a": list(range(20)), "y": [0, 1] * 10})
    assert len(client.predict(t, "y", pl.DataFrame({"a": [1, 2]}), task="classify").predictions) == 2
    assert yhatlabs.check(pl.from_pandas(hourly)).ok


def test_progress_silent_without_terminal(capsys, client):
    from yhatlabs import _progress

    assert not _progress.enabled()  # pytest's stderr is not a terminal
    client.forecast(pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=20, freq="D"), "y": 1.0}), h=2)
    assert capsys.readouterr().err == ""


def test_cli_check_json_skills(tmp_path, capsys, hourly):
    from yhatlabs._cli import main

    src = tmp_path / "d.csv"
    hourly.drop(index=[3]).to_csv(src, index=False)
    assert main(["--json", "check", str(src), "--h", "12"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] and "missing_timestamps" in {i["code"] for i in out["issues"]}
    assert main(["skills", "install", "--dir", str(tmp_path)]) == 0
    skill = (tmp_path / ".claude/skills/yhatlabs/SKILL.md").read_text()
    assert skill.startswith("---\nname: yhatlabs") and (tmp_path / ".agents/skills/yhatlabs/SKILL.md").exists()
    assert "chakra-ts-" not in skill  # only public model names


def test_limits_class_from_api():
    lim = Limits.from_api({"forecast": {"max_series": 9}, "tabular": {"max_rows": 7}})
    assert lim.max_series == 9 and lim.max_tabular_rows == 7 and lim.max_horizon == Limits().max_horizon
