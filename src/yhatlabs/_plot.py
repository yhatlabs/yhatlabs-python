"""Matplotlib plots (optional dependency)."""

from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd

from ._errors import ConfigurationError


def plot_forecast(fc: Any, history: Optional[pd.DataFrame], ids: Optional[List[Any]], max_series: int, ax: Any) -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ConfigurationError("plot() needs matplotlib: pip install matplotlib") from e
    df = fc.df
    id_col, time_col = df.columns[0], df.columns[1]
    qcols = [c for c in df.columns if c.startswith("q") or c == "median"]
    lo = next((c for c in qcols if c != "median"), None)
    hi = next((c for c in reversed(qcols) if c != "median"), None)
    ids = list(ids) if ids is not None else list(pd.unique(df[id_col]))[:max_series]
    if ax is None:
        _, axes = plt.subplots(len(ids), 1, figsize=(10, 2.6 * len(ids)), squeeze=False)
        axes = [a[0] for a in axes]
    else:
        axes = [ax] * len(ids)
    for a, uid in zip(axes, ids):
        f = df[df[id_col] == uid]
        if "target" in f.columns:
            f = f[f["target"] == f["target"].iloc[0]]
        if history is not None and id_col in history.columns:
            hcol = next((c for c in history.columns if c not in (id_col, time_col)), None)
            h = history[history[id_col] == uid].tail(max(3 * len(f), 50))
            if hcol is not None:
                a.plot(pd.to_datetime(h[time_col]), h[hcol], color="0.3", lw=1, label="history")
        if lo and hi:
            a.fill_between(f[time_col], f[lo], f[hi], alpha=0.25, label=f"{lo}-{hi}")
        if "median" in f.columns:
            a.plot(f[time_col], f["median"], lw=1.5, label="median")
        a.set_title(str(uid), fontsize=10)
        a.legend(loc="upper left", fontsize=8)
    return axes[0].figure


def plot_calibration(cal: pd.DataFrame, ax: Any) -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ConfigurationError("plot_calibration() needs matplotlib: pip install matplotlib") from e
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="calibrated")
    for model, c in cal[cal["kind"] == "below"].groupby("model", sort=False):
        ax.plot(c["nominal"], c["observed"], marker="o", ms=3, label=str(model))
    ax.set_xlabel("predicted quantile level")
    ax.set_ylabel("share of actuals below it")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    return ax.figure
