"""Hourly load for two zones, end to end: calendar + temperature covariates, a backtest against seasonal-naive and against the
forecast you use today, a heatwave scenario, and an anomaly check on recent data."""

import numpy as np
import pandas as pd

from yhatlabs import Client
from yhatlabs.features import CALENDAR, add_calendar, make_future

rng = np.random.default_rng(0)
hours = pd.date_range("2026-06-01", periods=24 * 60, freq="h")
day = np.sin((hours.hour.to_numpy() - 6) * 2 * np.pi / 24)
temp = 25 + 7 * day + rng.normal(0, 2, len(hours)).cumsum() * 0.05
zones = []
for zone, base in (("north", 900.0), ("south", 1400.0)):
    load = (
        base + 120 * day + 18 * np.maximum(temp - 24, 0) + (hours.dayofweek >= 5) * -80 + rng.normal(0, 15, len(hours))
    )
    zones.append(pd.DataFrame({"unique_id": zone, "ds": hours, "y": load, "temp": temp, "zone_size": base}))
df = add_calendar(pd.concat(zones, ignore_index=True))
known = list(CALENDAR) + ["temp"]

# today's forecast, e.g. from your current vendor: same id/time columns plus a point 'forecast' (or quantile columns)
current = (
    df.groupby("unique_id")
    .tail(24 * 7)[["unique_id", "ds"]]
    .assign(forecast=df.groupby("unique_id").tail(24 * 7).y * 1.03)
)

with Client() as yh:
    ev = yh.evaluate(df, h=24, windows=7, known=known, static=["zone_size"], compare={"current": current})
    print(ev.summary[["mase", "wql", "coverage@0.8", "skill_mase"]].round(3))

    future = add_calendar(make_future(df, 24))
    future["temp"] = np.tile(temp[-24:], 2)  # your weather forecast for the next day
    out = yh.scenarios(
        df,
        24,
        {"forecast weather": future, "heatwave +6C": future.assign(temp=future.temp + 6)},
        known=known,
        static=["zone_size"],
        level=[80],
    )
    print(out.groupby(["scenario", "unique_id"])["median"].max().unstack().round(0))

    recent = yh.detect_anomalies(df, h=24, windows=3, known=known)
    print(recent[recent.anomaly][["unique_id", "ds", "y", "median", "score"]])
