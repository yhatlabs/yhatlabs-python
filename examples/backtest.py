"""Backtest ChakraTS against a seasonal-naive baseline on the last 3 days of each series."""

import numpy as np
import pandas as pd

from yhatlabs import Client

hours = pd.date_range("2026-09-01", periods=24 * 42, freq="h")
rng = np.random.default_rng(1)
df = pd.concat(
    [
        pd.DataFrame(
            {
                "unique_id": f"meter_{i}",
                "ds": hours,
                "y": 50 + 10 * np.sin(np.arange(len(hours)) * 2 * np.pi / 24 + i) + rng.normal(0, 3, len(hours)),
            }
        )
        for i in range(5)
    ]
)

with Client() as yh:
    ev = yh.evaluate(df, h=24, windows=3)
    print(ev.summary.round(3))
