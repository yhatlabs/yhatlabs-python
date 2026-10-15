"""Rolling-origin backtests on the client: cut each series at several points in its own history, forecast from each cut,
score against what happened, and compare with simple baselines. No extra endpoint is needed, and nothing that happens after
a cutoff is ever sent with that window's request (known covariates excepted: they are, by definition, known in advance).
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._errors import DataError
from ._forecast import Item, Prepared, qname

DEFAULT_METRICS = ("mase", "sql", "wql", "mae", "rmse", "coverage@0.8")
ERROR_METRICS = ("mase", "sql", "wql", "mae", "rmse")
BASELINES = ("seasonal_naive", "naive")
# Seasonal period by base frequency, as GluonTS / fev define it (used for MASE scaling and the seasonal-naive baseline).
_SEASONS = {
    "S": 3600,
    "s": 3600,
    "T": 1440,
    "min": 1440,
    "H": 24,
    "h": 24,
    "D": 1,
    "B": 5,
    "W": 1,
    "M": 12,
    "MS": 12,
    "ME": 12,
    "Q": 4,
    "QS": 4,
    "QE": 4,
    "A": 1,
    "Y": 1,
    "YS": 1,
    "YE": 1,
    "AS": 1,
}


def season_for(freq: str) -> int:
    off = pd.tseries.frequencies.to_offset(freq)
    name = off.name.split("-")[0]
    base = _SEASONS.get(name, 1)
    return max(1, base // off.n) if base > 1 else 1


Runner = Callable[[List[Item]], Tuple[List[dict], List[float]]]


def windows_for(T: int, h: int, windows: int, step: int) -> List[int]:
    return [T - h - (windows - 1 - w) * step for w in range(windows)]


def split(
    prep: Prepared, known: Sequence[str], h: int, windows: int, step: int, min_context: int
) -> Tuple[List[Item], List[dict], List[str]]:
    """Window items (history up to the cutoff) and, for each, the actuals and context needed to score it."""
    items, meta, warns = [], [], []
    skipped = 0
    for it in prep.items:
        T = it.length
        for w, c in enumerate(windows_for(T, h, windows, step)):
            if c < min_context:
                skipped += 1
                continue
            targets = {k: v[:c] for k, v in it.targets.items()}
            actual = {k: v[c : c + h] for k, v in it.targets.items()}
            if not any(x is not None for vs in targets.values() for x in vs) or not any(
                x is not None for vs in actual.values() for x in vs
            ):
                skipped += 1
                continue
            new = Item(
                uid=(it.uid, w),
                targets=targets,
                multi=it.multi,
                times=it.times[:c] if it.times is not None else None,
                future_times=it.times[c : c + h] if it.times is not None else np.arange(c, c + h),
            )
            for k, v in it.past.items():
                if k in known:
                    new.known[k] = v[: c + h]
                else:
                    new.past[k] = v[:c]
            for k, v in it.known.items():  # static columns
                new.known[k] = v[: c + h]
            items.append(new)
            meta.append(
                {
                    "uid": it.uid,
                    "window": w,
                    "cutoff": it.times[c - 1] if it.times is not None else c - 1,
                    "actual": actual,
                }
            )
    if skipped:
        warns.append(f"Skipped {skipped} (series, window) pairs with too little history or no observed values.")
    if not items:
        raise DataError(
            f"No series is long enough for {windows} windows of h={h}; use fewer windows, a smaller h or step."
        )
    return items, meta, warns


def _ffill(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    last = np.nan
    for i, v in enumerate(out):
        if np.isnan(v):
            out[i] = last
        else:
            last = v
    return out


def baseline(history: Sequence[Optional[float]], h: int, m: int, levels: Sequence[float]) -> np.ndarray:
    """[h, len(levels)] quantiles from a (seasonal) naive forecast with normal errors that widen with each season ahead."""
    y = _ffill(np.array([np.nan if v is None else v for v in history], dtype=float))
    finite = y[np.isfinite(y)]
    if len(finite) == 0:
        return np.full((h, len(levels)), np.nan)
    m = m if len(y) > m else 1
    last = y[-m:]
    point = np.array([last[t % m] for t in range(h)])
    point = np.where(np.isfinite(point), point, finite[-1])
    res = y[m:] - y[:-m]
    res = res[np.isfinite(res)]
    sigma = float(np.std(res)) if len(res) > 1 else 0.0
    sd = sigma * np.sqrt(np.ceil(np.arange(1, h + 1) / m))
    z = np.array([NormalDist().inv_cdf(q) for q in levels])
    return point[:, None] + sd[:, None] * z[None, :]


def _scale(history: Sequence[Optional[float]], m: int) -> float:
    y = np.array([np.nan if v is None else v for v in history], dtype=float)
    if len(y) <= m:
        m = 1
    d = np.abs(y[m:] - y[:-m])
    d = d[np.isfinite(d)]
    if not len(d):
        return float("nan")
    # A history that repeats exactly has a zero (or float-noise) scale; scaled metrics are undefined there, not huge.
    level = np.nanmean(np.abs(y)) if np.isfinite(y).any() else 0.0
    s = float(d.mean())
    return s if s > 1e-9 * max(1.0, level) else float("nan")


def score(
    y: np.ndarray, q: np.ndarray, levels: Sequence[float], scale: float, metrics: Sequence[str], strict: bool = True
) -> Dict[str, float]:
    """Y [h] (may hold NaN), q [h, n] quantiles at `levels`. fev-style definitions."""
    ok = np.isfinite(y)
    y, q = y[ok], q[ok]
    out: Dict[str, float] = {}
    if len(y) == 0:
        return {m: float("nan") for m in metrics}
    lv = np.asarray(levels, dtype=float)
    med = q[:, int(np.argmin(np.abs(lv - 0.5)))]
    err = y - med
    pin = np.maximum(lv * (y[:, None] - q), (lv - 1) * (y[:, None] - q))  # [h, n]
    for m in metrics:
        if m == "mae":
            out[m] = float(np.mean(np.abs(err)))
        elif m == "rmse":
            out[m] = float(np.sqrt(np.mean(err**2)))
        elif m == "mase":
            out[m] = float(np.mean(np.abs(err)) / scale) if scale and math.isfinite(scale) else float("nan")
        elif m == "wql":
            denom = np.sum(np.abs(y))
            out[m] = float(np.mean(2 * pin.sum(axis=0)) / denom) if denom > 0 else float("nan")
        elif m == "sql":
            out[m] = float(np.mean(2 * pin) / scale) if scale and math.isfinite(scale) else float("nan")
        elif m.startswith("coverage@"):
            width = float(m.split("@", 1)[1])
            lo, hi = (1 - width) / 2, 1 - (1 - width) / 2
            il, ih = np.argmin(np.abs(lv - lo)), np.argmin(np.abs(lv - hi))
            if abs(lv[il] - lo) > 1e-9 or abs(lv[ih] - hi) > 1e-9:
                if strict:
                    raise DataError(f"{m} needs the {lo:g} and {hi:g} quantiles.")
                out[m] = float("nan")
                continue
            out[m] = float(np.mean((y >= q[:, il]) & (y <= q[:, ih])))
        else:
            raise DataError(f"Unknown metric {m!r}; choose from {DEFAULT_METRICS} (coverage@<width> for any width).")
    return out


def summarize(scores: pd.DataFrame, metrics: Sequence[str], base: Optional[str]) -> pd.DataFrame:
    keys = [c for c in ("id", "target", "window") if c in scores.columns]
    summary = scores.groupby("model", sort=False)[list(metrics)].mean()
    if base is not None and base in summary.index:
        ref = scores[scores["model"] == base].set_index(keys)
        for m in metrics:
            if m not in ERROR_METRICS:
                continue
            col = []
            floor = 1e-12 * max(
                1.0, float(scores[m].abs().max())
            )  # a baseline error of float noise makes the ratio meaningless
            for model in summary.index:
                cur = scores[scores["model"] == model].set_index(keys)[m]
                den = ref[m].reindex(cur.index)
                r = (cur / den.where(den > floor)).replace([np.inf, -np.inf], np.nan).dropna()
                r = r[r > 0]
                col.append(float(1 - np.exp(np.mean(np.log(r)))) if len(r) else float("nan"))
            summary[f"skill_{m}"] = col
    return summary


def _compare_quantiles(
    f: pd.DataFrame, id_col: str, time_col: str
) -> Tuple[Dict[Tuple[Any, Any], np.ndarray], List[float]]:
    """A user's own forecasts: quantile columns (median, q10, q90, ...) or one point column ('forecast', 'yhat', 'y_hat', 'prediction')."""
    from ._forecast import qname

    levels = [lv / 100 for lv in range(1, 100)]
    qcols = {qname(lv): lv for lv in levels if qname(lv) in f.columns}
    if not qcols:
        point = next((c for c in ("forecast", "yhat", "y_hat", "prediction", "value") if c in f.columns), None)
        if point is None:
            raise DataError(
                f"compare forecasts need quantile columns (median, q10, ...) or a point column 'forecast'. Columns: {list(f.columns)}"
            )
        qcols, f = {"median": 0.5}, f.assign(median=f[point])
    names = sorted(qcols, key=lambda c: qcols[c])
    t = pd.to_datetime(f[time_col])
    table = {(u, ts): row for u, ts, row in zip(f[id_col], t, f[names].to_numpy(dtype=float))}
    return table, [qcols[c] for c in names]


def run(
    prep: Prepared,
    h: int,
    *,
    windows: int,
    step: Optional[int],
    models: Sequence[str],
    baselines: Sequence[str],
    metrics: Sequence[str],
    season: Optional[int],
    known: Sequence[str],
    compare: Optional[Dict[str, pd.DataFrame]] = None,
    forecaster: Callable[[str, List[Item]], Tuple[List[dict], List[float]]],
):
    """forecaster(model, items) -> (one reply per item, native quantile levels)."""
    if windows < 1:
        raise DataError("windows must be at least 1.")
    step = step or h
    unknown = [b for b in baselines if b not in BASELINES]
    if unknown:
        raise DataError(f"Unknown baselines {unknown}; available: {BASELINES}.")
    m = season or season_for(prep.freq)
    items, meta, warns = split(prep, known, h, windows, step, min_context=max(2 * m, 8) if m > 1 else 8)
    rows, fc_frames = [], []
    levels: Optional[List[float]] = None

    def add(
        model: str, mi: dict, it: Item, per_target: Dict[str, np.ndarray], lv: Sequence[float], strict: bool = True
    ) -> None:
        for col, q in per_target.items():
            y = np.array([np.nan if v is None else v for v in mi["actual"][col]], dtype=float)
            q = np.sort(np.asarray(q, dtype=float)[: len(y)], axis=1)
            sc = score(y, q, lv, _scale(it.targets[col], m), metrics, strict)
            row = {"model": model, "id": mi["uid"], "window": mi["window"], "cutoff": mi["cutoff"], **sc}
            if it.multi:
                row["target"] = col
            rows.append(row)
            f = pd.DataFrame(q, columns=[qname(x) for x in lv])
            f.insert(0, "actual", y)
            f.insert(0, prep.time_col, it.future_times[: len(y)])
            if it.multi:
                f.insert(0, "target", col)
            f.insert(0, "cutoff", mi["cutoff"])
            f.insert(0, "window", mi["window"])
            f.insert(0, prep.id_col, mi["uid"])
            f.insert(0, "model", model)
            fc_frames.append(f)

    for model in models:
        replies, native = forecaster(model, items)
        levels = list(native)
        for it, mi, r in zip(items, meta, replies):
            per = r["targets"] if "targets" in r else {prep.target_cols[0]: r["quantiles"]}
            add(model, mi, it, per, native)
    lv = levels or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for b in baselines:
        bm = m if b == "seasonal_naive" else 1
        for it, mi in zip(items, meta):
            add(b, mi, it, {c: baseline(v, h, bm, lv) for c, v in it.targets.items()}, lv)
    for name, f in (compare or {}).items():
        if prep.target_cols[1:]:
            raise DataError("compare= supports single-target series.")
        if prep.id_col not in f.columns:
            if len(prep.items) != 1:
                raise DataError(f"compare[{name!r}] needs the id column {prep.id_col!r} to match forecasts to series.")
            f = f.assign(**{prep.id_col: prep.items[0].uid})  # one series: the id is implied
        table, clv = _compare_quantiles(f, prep.id_col, prep.time_col)
        used = 0
        for it, mi in zip(items, meta):
            got = [table.get((mi["uid"], t)) for t in it.future_times]
            if all(g is not None for g in got):
                add(name, mi, it, {prep.target_cols[0]: np.vstack(got)}, clv, strict=False)
                used += 1
        if used < len(items):
            warns.append(
                f"compare[{name!r}] covered {used} of {len(items)} (series, window) pairs; only those were scored."
            )
    scores = pd.DataFrame(rows)
    base = baselines[0] if baselines else None
    summary = summarize(scores, metrics, base)
    forecasts = pd.concat(fc_frames, ignore_index=True) if fc_frames else None
    return scores, summary, forecasts, base, warns


def calibration_table(forecasts: Optional[pd.DataFrame]) -> pd.DataFrame:
    from ._forecast import qname

    cols = ["model", "kind", "nominal", "observed", "gap", "n"]
    if forecasts is None or not len(forecasts):
        return pd.DataFrame(columns=cols)
    levels = [lv / 100 for lv in range(1, 100)]
    present = {qname(lv): lv for lv in levels if qname(lv) in forecasts.columns}
    rows = []
    for model, f in forecasts.groupby("model", sort=False):
        y = f["actual"].to_numpy(dtype=float)
        have = {c: lv for c, lv in present.items() if f[c].notna().any()}
        ok = np.isfinite(y)
        for c, lv in sorted(have.items(), key=lambda kv: kv[1]):
            q = f[c].to_numpy(dtype=float)
            m = ok & np.isfinite(q)
            if m.any():
                obs = float(np.mean(y[m] < q[m]))
                rows.append(
                    {
                        "model": model,
                        "kind": "below",
                        "nominal": lv,
                        "observed": obs,
                        "gap": obs - lv,
                        "n": int(m.sum()),
                    }
                )
        by_level = {lv: c for c, lv in have.items()}
        for lo in sorted(lv for lv in by_level if lv < 0.5):
            hi = round(1 - lo, 6)
            if hi in by_level:
                a, b = f[by_level[lo]].to_numpy(dtype=float), f[by_level[hi]].to_numpy(dtype=float)
                m = ok & np.isfinite(a) & np.isfinite(b)
                if m.any():
                    obs, nom = float(np.mean((y[m] >= a[m]) & (y[m] <= b[m]))), round(hi - lo, 6)
                    rows.append(
                        {
                            "model": model,
                            "kind": "interval",
                            "nominal": nom,
                            "observed": obs,
                            "gap": obs - nom,
                            "n": int(m.sum()),
                        }
                    )
    return pd.DataFrame(rows, columns=cols)
