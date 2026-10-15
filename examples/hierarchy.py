"""Store forecasts that must add up to the region total. Forecast every store, then sum the medians bottom-up for the region and
compare with forecasting the region directly. (Medians add only approximately; for intervals on the total, forecast it directly.)"""

import numpy as np
import pandas as pd

from yhatlabs import Client

rng = np.random.default_rng(2)
days = pd.date_range("2026-01-01", periods=280, freq="D")
stores = []
for i in range(6):
    weekly = 1 + 0.3 * (days.dayofweek >= 5)
    sales = (50 + 10 * i) * weekly + rng.normal(0, 5, len(days))
    stores.append(pd.DataFrame({"unique_id": f"store_{i}", "ds": days, "y": sales}))
stores = pd.concat(stores, ignore_index=True)
region = stores.groupby("ds", as_index=False).y.sum().assign(unique_id="region")

with Client() as yh:
    fc = yh.forecast(pd.concat([stores, region], ignore_index=True), h=14)
    bottom_up = fc.df[fc.df.unique_id != "region"].groupby("ds")["median"].sum()
    direct = fc.df[fc.df.unique_id == "region"].set_index("ds")["median"]
    print(
        pd.DataFrame({"bottom_up": bottom_up, "direct": direct, "gap_%": (bottom_up / direct - 1) * 100})
        .round(1)
        .head(7)
    )
