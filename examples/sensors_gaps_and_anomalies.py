"""Sensor data with dropouts and a fault: fill the gaps from each series' own history, then flag readings far outside what the
model expected. Both use only data before the point in question."""

import numpy as np
import pandas as pd

from yhatlabs import Client

rng = np.random.default_rng(1)
t = pd.date_range("2026-10-01", periods=96 * 14, freq="15min")
frames = []
for s in ("pump_1", "pump_2", "pump_3"):
    v = 60 + 8 * np.sin(np.arange(len(t)) * 2 * np.pi / 96) + rng.normal(0, 1, len(t))
    frames.append(pd.DataFrame({"sensor": s, "time": t, "temp_c": v}))
df = pd.concat(frames, ignore_index=True)
df = df.drop(index=rng.choice(df.index[200:-200], 150, replace=False))  # dropouts: missing timestamps
fault = df.index[df.sensor == "pump_2"][-40]
df.loc[fault, "temp_c"] += 25  # a fault

with Client() as yh:
    filled = yh.fill_gaps(df, id_col="sensor", time_col="time", target_col="temp_c")
    print(f"filled {int(filled.temp_c_filled.sum())} readings")
    a = yh.detect_anomalies(
        filled[["sensor", "time", "temp_c"]], id_col="sensor", time_col="time", target_col="temp_c", h=96, windows=2
    )
    print(a[a.anomaly])
