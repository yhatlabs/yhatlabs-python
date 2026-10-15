import numpy as np
import pandas as pd
import pytest

from yhatlabs import DataError
from yhatlabs._evaluate import baseline, score, season_for


def test_season_for():
    assert season_for("h") == 24 and season_for("15min") == 96 and season_for("MS") == 12 and season_for("D") == 1


def test_score_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0])
    q = np.repeat(y[:, None], 9, axis=1)
    s = score(y, q, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 1.0, ["mae", "mase", "wql", "sql", "coverage@0.8"])
    assert s["mae"] == 0 and s["wql"] == 0 and s["coverage@0.8"] == 1.0


def test_baseline_shape():
    q = baseline([1, 2, 3, 1, 2, 3, None], 4, 3, [0.1, 0.5, 0.9])
    assert q.shape == (4, 3) and np.all(q[:, 0] <= q[:, 2])


def test_evaluate(client, api, hourly):
    ev = client.evaluate(hourly, h=12, windows=3, baselines=("seasonal_naive", "naive"))
    assert set(ev.scores.model) == {"chakra-ts", "seasonal_naive", "naive"}
    assert len(ev.scores[ev.scores.model == "chakra-ts"]) == 2 * 3
    assert "skill_mase" in ev.summary.columns and ev.summary.loc["seasonal_naive", "skill_mase"] == pytest.approx(0)
    lengths = sorted({len(s["values"]) for _, _, b, _ in api.requests for s in b["series"]})
    assert lengths == [60, 72, 84]  # no window sees data past its cutoff
    assert ev.forecasts is not None and {"actual", "median", "cutoff"} <= set(ev.forecasts.columns)


def test_evaluate_known_covariates_cut(client, api, hourly):
    df = hourly.assign(temp=np.arange(len(hourly), dtype=float))
    client.evaluate(df, h=6, windows=2, known=["temp"], baselines=())
    for _, _, b, _ in api.requests:
        for s in b["series"]:
            assert len(s["known_covariates"]["temp"]) == len(s["values"]) + 6


def test_evaluate_too_short(client):
    df = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=10, freq="D"), "y": 1.0})
    with pytest.raises(DataError, match="long enough"):
        client.evaluate(df, h=5, windows=3)


def test_scale_undefined_for_exactly_seasonal_history():
    from yhatlabs._evaluate import _scale

    hist = list(np.tile(400 + 3 * np.sin(np.arange(24) * 2 * np.pi / 24), 5))
    assert np.isnan(_scale(hist, 24))
    assert _scale([1.0, 2.0, 4.0], 1) == 1.5
