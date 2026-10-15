"""Matplotlib plots (optional dependency: pip install "yhatlabs[plot]"). Every function returns the matplotlib Figure."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._errors import ConfigurationError


def _plt() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ConfigurationError('Plots need matplotlib: pip install "yhatlabs[plot]"') from e
    return plt


def _level(c: str) -> Optional[float]:
    if c == "median":
        return 0.5
    if c.startswith("q"):
        try:
            return float(c[1:]) / 100
        except ValueError:
            return None
    return None


def _bands(cols: Sequence[str]) -> List[Tuple[str, str, float]]:
    """Symmetric quantile pairs present in the frame, widest first: [(q10, q90, 80.0), (q25, q75, 50.0)]."""
    lv = {c: _level(c) for c in cols}
    lv = {c: v for c, v in lv.items() if v is not None}
    out = []
    for c, v in lv.items():
        if v < 0.5:
            hi = next((d for d, w in lv.items() if abs(w - (1 - v)) < 1e-9), None)
            if hi:
                out.append((c, hi, round((1 - 2 * v) * 100, 6)))
    return sorted(out, key=lambda t: -t[2])


def _axes(n: int, ax: Any, height: float = 2.6) -> List[Any]:
    plt = _plt()
    if ax is not None:
        return [ax] * n
    _, axes = plt.subplots(n, 1, figsize=(10, height * n), squeeze=False, layout="constrained")
    return [a[0] for a in axes]


def _series_bands(a: Any, f: pd.DataFrame, x: Any, color: Any = None, label: str = "median") -> None:
    bands = _bands(f.columns)
    for i, (lo, hi, w) in enumerate(bands):
        a.fill_between(x, f[lo], f[hi], alpha=0.15 + 0.12 * i, color=color, lw=0, label=f"{w:g}% interval")
    if "median" in f.columns:
        a.plot(x, f["median"], lw=1.5, color=color, label=label)
    if "mean" in f.columns:
        a.plot(x, f["mean"], lw=1, ls="--", color=color or "C1", label="mean")


def _history(a: Any, history: Optional[pd.DataFrame], id_col: str, time_col: str, uid: Any, n: int) -> None:
    if history is None or id_col not in history.columns or time_col not in history.columns:
        return
    vcol = next(
        (c for c in history.columns if c not in (id_col, time_col) and pd.api.types.is_numeric_dtype(history[c])), None
    )
    if vcol is None:
        return
    h = history[history[id_col] == uid].sort_values(time_col).tail(n)
    a.plot(h[time_col], h[vcol], color="0.3", lw=1, label="history")


def _finish(axes: List[Any]) -> Any:
    """Compact date labels on time axes; returns the figure."""
    import matplotlib.dates as mdates

    for a in dict.fromkeys(axes):
        if isinstance(a.xaxis.get_major_formatter(), (mdates.AutoDateFormatter, mdates.ConciseDateFormatter)):
            loc = mdates.AutoDateLocator()
            a.xaxis.set_major_locator(loc)
            a.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    return axes[0].figure


def plot_forecast(fc: Any, history: Optional[pd.DataFrame], ids: Optional[List[Any]], max_series: int, ax: Any) -> Any:
    """Median with every central interval present as nested bands, after the end of `history` (same id / time columns)."""
    df = fc.df
    id_col, time_col = df.columns[0], df.columns[1]
    if history is not None and id_col not in history.columns:
        history = history.assign(**{id_col: df[id_col].iloc[0]})
    ids = list(ids) if ids is not None else list(pd.unique(df[id_col]))[:max_series]
    axes = _axes(len(ids), ax)
    for a, uid in zip(axes, ids):
        f = df[df[id_col] == uid]
        if "target" in f.columns:
            f = f[f["target"] == f["target"].iloc[0]]
        _history(a, history, id_col, time_col, uid, max(3 * len(f), 50))
        _series_bands(a, f, f[time_col])
        a.set_title(str(uid), fontsize=10)
        a.legend(loc="upper left", fontsize=8)
    return _finish(axes)


def plot_backtest(ev: Any, ids: Optional[List[Any]], max_series: int, model: Optional[str], ax: Any) -> Any:
    """Actuals across the scored windows, each window's forecast (median and intervals) and a line at every cutoff."""
    f = ev.forecasts
    if f is None or not len(f):
        raise ConfigurationError("This evaluation kept no forecasts to plot.")
    model = model or f["model"].iloc[0]
    f = f[f["model"] == model]
    meta = ("model", "window", "cutoff", "actual", "target")
    id_col, time_col = [c for c in f.columns if c not in meta][:2]
    ids = list(ids) if ids is not None else list(pd.unique(f[id_col]))[:max_series]
    axes = _axes(len(ids), ax)
    for a, uid in zip(axes, ids):
        g = f[f[id_col] == uid]
        if "target" in g.columns:
            g = g[g["target"] == g["target"].iloc[0]]
        actual = g.drop_duplicates(time_col).sort_values(time_col)
        a.plot(actual[time_col], actual["actual"], color="0.2", lw=1, marker=".", ms=3, label="actual")
        for i, (_, w) in enumerate(g.groupby("window", sort=True)):
            _series_bands(a, w, w[time_col], color="C0", label="median" if i == 0 else "_median")
            a.axvline(w["cutoff"].iloc[0], color="0.7", lw=0.8, ls=":")
        handles, labels = a.get_legend_handles_labels()
        seen = dict(zip(labels, handles))
        a.legend(seen.values(), seen.keys(), loc="upper left", fontsize=8)
        a.set_title(f"{uid} ({model})", fontsize=10)
    return _finish(axes)


def plot_anomalies(flags: pd.DataFrame, ids: Optional[List[Any]], max_series: int, ax: Any) -> Any:
    """The series, its expected range (q10 to q90 of each step's forecast) and the flagged points."""
    id_col, time_col = flags.columns[0], flags.columns[1]
    vcol = flags.columns[2]
    ids = list(ids) if ids is not None else list(pd.unique(flags[id_col]))[:max_series]
    axes = _axes(len(ids), ax)
    for a, uid in zip(axes, ids):
        f = flags[flags[id_col] == uid].sort_values(time_col)
        if "q10" in f.columns and "q90" in f.columns:
            a.fill_between(f[time_col], f["q10"], f["q90"], alpha=0.2, lw=0, label="expected (q10-q90)")
        a.plot(f[time_col], f[vcol], color="0.25", lw=1, label=str(vcol))
        bad = f[f["anomaly"].astype(bool)]
        a.scatter(bad[time_col], bad[vcol], color="C3", s=18, zorder=3, label=f"anomaly ({len(bad)})")
        a.set_title(str(uid), fontsize=10)
        a.legend(loc="upper left", fontsize=8)
    return _finish(axes)


def plot_scenarios(
    frame: pd.DataFrame, history: Optional[pd.DataFrame], ids: Optional[List[Any]], max_series: int, ax: Any
) -> Any:
    """One median line and interval band per scenario, per series."""
    id_col, time_col = frame.columns[1], frame.columns[2]
    ids = list(ids) if ids is not None else list(pd.unique(frame[id_col]))[:max_series]
    axes = _axes(len(ids), ax)
    for a, uid in zip(axes, ids):
        f = frame[frame[id_col] == uid]
        _history(a, history, id_col, time_col, uid, max(3 * f[time_col].nunique(), 50))
        for i, (name, g) in enumerate(f.groupby("scenario", sort=False)):
            bands = _bands(g.columns)
            if bands:
                lo, hi, _ = bands[0]
                a.fill_between(g[time_col], g[lo], g[hi], alpha=0.15, color=f"C{i}", lw=0)
            a.plot(g[time_col], g["median"], color=f"C{i}", lw=1.5, label=str(name))
        a.set_title(str(uid), fontsize=10)
        a.legend(loc="upper left", fontsize=8)
    return _finish(axes)


def plot_calibration(cal: pd.DataFrame, ax: Any) -> Any:
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5), layout="constrained")
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="calibrated")
    for model, c in cal[cal["kind"] == "below"].groupby("model", sort=False):
        ax.plot(c["nominal"], c["observed"], marker="o", ms=3, label=str(model))
    ax.set_xlabel("predicted quantile level")
    ax.set_ylabel("share of actuals below it")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    return ax.figure


def plot_reliability(table: pd.DataFrame, ax: Any) -> Any:
    """Classification calibration: mean predicted probability against the observed frequency, per probability bin."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5), layout="constrained")
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="calibrated")
    for cls, t in table.groupby("class", sort=False) if "class" in table.columns else [(None, table)]:
        ax.plot(t["predicted"], t["observed"], marker="o", ms=4, label="model" if cls is None else str(cls))
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    return ax.figure


def plot_predicted_vs_actual(pred: pd.DataFrame, actual: Any, ax: Any) -> Any:
    """Regression: predictions (with the widest interval as error bars, when present) against the true values."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5), layout="constrained")
    y = np.asarray(actual, dtype=float)
    p = pred["prediction"].to_numpy(dtype=float)
    bands = _bands(pred.columns)
    if bands:
        lo, hi, w = bands[0]
        err = np.vstack([p - pred[lo].to_numpy(float), pred[hi].to_numpy(float) - p]).clip(min=0)
        ax.errorbar(y, p, yerr=err, fmt="o", ms=3, alpha=0.6, elinewidth=0.6, label=f"prediction ({w:g}% interval)")
    else:
        ax.scatter(y, p, s=8, alpha=0.6, label="prediction")
    lim = [np.nanmin([y.min(), p.min()]), np.nanmax([y.max(), p.max()])]
    ax.plot(lim, lim, color="0.6", lw=1, ls="--")
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    ax.legend(fontsize=8)
    return ax.figure


def plot_importance(imp: pd.DataFrame, top: int, ax: Any) -> Any:
    plt = _plt()
    imp = imp.sort_values("importance", ascending=False).head(top).iloc[::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 0.3 * len(imp) + 1))
    ax.barh(
        imp["feature"].astype(str),
        imp["importance"],
        xerr=imp["stddev"] if "stddev" in imp.columns else None,
        color="C0",
    )
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel(f"score drop when shuffled ({imp.attrs.get('metric', 'metric')})")
    return ax.figure


def plot_effect(eff: pd.DataFrame, ax: Any) -> Any:
    """Partial dependence: the average prediction as one column varies, with the spread across rows."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
    col = eff.columns[0]
    numeric = pd.api.types.is_numeric_dtype(eff[col])
    x = eff[col] if numeric else np.arange(len(eff))
    for target, g in eff.groupby("output", sort=False) if "output" in eff.columns else [(None, eff)]:
        xs = g[col] if numeric else np.arange(len(g))
        ax.fill_between(xs, g["low"], g["high"], alpha=0.15, lw=0)
        ax.plot(xs, g["average"], marker="o", ms=3, label="average" if target is None else str(target))
    if not numeric:
        ax.set_xticks(x, eff[col].astype(str), rotation=30, ha="right")
    ax.set_xlabel(str(col))
    ax.set_ylabel(eff.attrs.get("ylabel", "prediction"))
    ax.legend(fontsize=8)
    return ax.figure
