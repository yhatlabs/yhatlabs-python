"""Forecast two hourly series a day ahead, with a known covariate. Run: YHAT_API_KEY=... python examples/forecast.py"""

import numpy as np
import pandas as pd

from yhatlabs import Client

hours = pd.date_range("2026-09-01", periods=24 * 28, freq="h")
rng = np.random.default_rng(0)
temp = 20 + 8 * np.sin(np.arange(len(hours) + 24) * 2 * np.pi / 24)
df = pd.concat(
    [
        pd.DataFrame(
            {
                "unique_id": zone,
                "ds": hours,
                "y": base + 3 * temp[: len(hours)] + rng.normal(0, 2, len(hours)),
                "temp": temp[: len(hours)],
            }
        )
        for zone, base in (("north", 400), ("south", 650))
    ]
)
future = pd.concat(
    [
        pd.DataFrame({"unique_id": z, "ds": pd.date_range(hours[-1], periods=25, freq="h")[1:], "temp": temp[-24:]})
        for z in ("north", "south")
    ]
)

with Client() as yh:
    fc = yh.forecast(df, h=24, known=["temp"], future=future, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    print(fc)
    print(fc.df.head(10))
