"""Time-series anomaly scores and gap filling, built on forecasts: a point is unusual when it falls far outside what the model
expected from the history before it; a gap is filled with what the model expects from the history before the gap. Nothing after
the scored window or the gap is ever used.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._errors import DataError
from ._evaluate import season_for, split
from ._forecast import ID_NAMES, TIME_NAMES, Item, Prepared, _freq_str, _infer_freq, _pandas, _pick

Z80 = 1.2815515655446004  # standard-normal 0.9 quantile: q10..q90 spans 2 * Z80 standard deviations
MIN_CONTEXT = 8

Forecaster = Callable[[List[Item], int], Tuple[List[dict], List[float]]]


def _cols(native: List[float]) -> Tuple[int, int, int]:
    lv = np.asarray(native)
    return int(np.argmin(np.abs(lv - 0.1))), int(np.argmin(np.abs(lv - 0.5))), int(np.argmin(np.abs(lv - 0.9)))


def anomalies(
    prep: Prepared, h: Optional[int], windows: int, threshold: float, known: List[str], forecaster: Forecaster
) -> Tuple[pd.DataFrame, List[str]]:
    """Scores the last `windows * h` points of every series. score = (actual - median) / sigma, where sigma is the forecast's own
    spread (q10..q90 read as a normal 80% band); |score| > threshold flags an anomaly. So a point the model was unsure about needs
    a larger miss to count.
    """
    h = h or max(season_for(prep.freq), 8)
    m = season_for(prep.freq)
    items, meta, warns = split(
        prep, known, h, windows, h, min_context=max(2 * m, MIN_CONTEXT) if m > 1 else MIN_CONTEXT
    )
    replies, native = forecaster(items, h)
    lo, mid, hi = _cols(native)
    rows = []
    for it, mi, r in zip(items, meta, replies):
        per = r["targets"] if "targets" in r else {prep.target_cols[0]: r["quantiles"]}
        for col, q in per.items():
            q = np.sort(np.asarray(q, dtype=float), axis=1)
            y = np.array([np.nan if v is None else v for v in mi["actual"][col]], dtype=float)
            n = len(y)
            sigma = (q[:n, hi] - q[:n, lo]) / (2 * Z80)
            floor = 1e-9 * max(1.0, float(np.nanmean(np.abs(q[:n, mid]))) if n else 1.0)
            score = (y - q[:n, mid]) / np.maximum(sigma, floor)
            f = pd.DataFrame(
                {
                    prep.id_col: mi["uid"],
                    prep.time_col: it.future_times[:n],
                    "y": y,
                    "median": q[:n, mid],
                    "q10": q[:n, lo],
                    "q90": q[:n, hi],
                    "score": score,
                }
            )
            if it.multi:
                f.insert(2, "target", col)
            rows.append(f)
    out = pd.concat(rows, ignore_index=True)
    out["anomaly"] = out["score"].abs() > threshold
    out.loc[out["y"].isna(), "anomaly"] = False
    sort = [prep.id_col] + (["target"] if "target" in out.columns else []) + [prep.time_col]
    return out.sort_values(sort, kind="stable").reset_index(drop=True), warns


def gaps(prep: Prepared, max_gap: int, forecaster: Forecaster) -> Tuple[pd.DataFrame, List[str]]:
    """Fills each run of missing target values with the median forecast from the history before it (covariates cut the same way).
    Runs at the very start of a series, or longer than `max_gap`, stay missing. Returns the regular grid with `<target>_filled` flags.
    """
    warns: List[str] = []
    jobs: Dict[int, List[Tuple[Item, Any, str, int]]] = {}  # gap length -> (sub-item, uid, column, start)
    skipped_start = skipped_long = 0
    frames = {}
    for it in prep.items:
        if it.times is None:
            raise DataError("fill_gaps needs timestamps (a DataFrame with a time column).")
        base = pd.DataFrame({prep.id_col: it.uid, prep.time_col: it.times})
        for col, vals in it.targets.items():
            base[col] = [np.nan if v is None else v for v in vals]
            base[f"{col}_filled"] = False
            missing = np.array([v is None for v in vals])
            i, T = 0, len(vals)
            while i < T:
                if not missing[i]:
                    i += 1
                    continue
                j = i
                while j < T and missing[j]:
                    j += 1
                if j == T:  # trailing gap = the future; that is a forecast, not a fill
                    break
                n = j - i
                if (~missing[:i]).sum() < MIN_CONTEXT:
                    skipped_start += 1
                elif n > max_gap:
                    skipped_long += 1
                else:
                    sub = Item(
                        uid=(it.uid, col, i),
                        targets={col: vals[:i]},
                        multi=False,
                        times=it.times[:i],
                        future_times=it.times[i:j],
                    )
                    sub.past = {k: v[:i] for k, v in it.past.items()}
                    sub.known = {k: v[:j] for k, v in it.known.items()}
                    jobs.setdefault(n, []).append((sub, it.uid, col, i))
                i = j
        frames[it.uid] = base
    if skipped_start:
        warns.append(f"Left {skipped_start} gap(s) unfilled: fewer than {MIN_CONTEXT} observed points before them.")
    if skipped_long:
        warns.append(f"Left {skipped_long} gap(s) longer than max_gap={max_gap} unfilled.")
    for n, group in jobs.items():
        replies, native = forecaster([g[0] for g in group], n)
        mid = _cols(native)[1]
        for (_sub, uid, col, start), r in zip(group, replies):
            med = np.asarray(r["quantiles"], dtype=float)[:n, mid]
            f = frames[uid]
            f.loc[start : start + n - 1, col] = med
            f.loc[start : start + n - 1, f"{col}_filled"] = True
    return pd.concat(frames.values(), ignore_index=True), warns


class AnomalyStream:
    """Anomaly scores for points as they arrive. Keeps the last `window` points of every series and scores only each update's
    new points, with one detect_anomalies call per update on the kept history plus the new points. Made by
    Client.anomaly_stream().
    """

    def __init__(self, client: Any, window: int, id_col: Optional[str], time_col: Optional[str], opts: Dict[str, Any]):
        if not isinstance(window, (int, np.integer)) or window < MIN_CONTEXT:
            raise DataError(f"window must be an integer of at least {MIN_CONTEXT} points per series.")
        self._client, self.window, self._opts = client, int(window), opts
        self.id_col, self.time_col = id_col, time_col
        self.history: Optional[pd.DataFrame] = None  # the kept points: at most `window` per series
        self._warmup: Optional[int] = None

    def __repr__(self) -> str:
        n = 0 if self.history is None else self.history[self.id_col].nunique()
        return f"AnomalyStream(window={self.window}, series={n})"

    def _ready_after(self, hist: pd.DataFrame) -> int:
        """Points a series needs before the scored window: what detect_anomalies needs as context (two seasons, at least 8)."""
        if self._warmup is None:
            freq = self._opts.get("freq")
            if freq is None:
                g = hist[hist[self.id_col] == hist[self.id_col].iloc[0]][self.time_col]
                off = _infer_freq(pd.DatetimeIndex(g)) if not pd.api.types.is_integer_dtype(g) else None
                freq = _freq_str(off) if off is not None else None
            m = season_for(freq) if freq else 1
            self._warmup = max(2 * m, MIN_CONTEXT) if m > 1 else MIN_CONTEXT
        return self._warmup

    def update(self, new: Any) -> pd.DataFrame:
        """Scores the new points (one detect_anomalies call) and adds them to the kept history. A series is scored once it has
        enough history before the new points (two seasons, at least 8 points); until then its points are only kept. A point
        sent again (same id and time) replaces the kept one.

        Args:
            new: The new points: a long table with the id, time and target columns (and the covariates the stream uses).

        Returns:
            The detect_anomalies columns (id, time, y, median, q10, q90, score, anomaly) for the new points only.
        """
        new = _pandas(new)
        if not isinstance(new, pd.DataFrame) or new.empty:
            raise DataError("update() takes a DataFrame of new points with the id, time and target columns.")
        new = new.rename(columns=str)
        cols = list(new.columns)
        self.time_col = self.time_col or _pick(cols, TIME_NAMES)
        if self.time_col not in cols:
            raise DataError(
                f"Could not find a time column; name it one of {TIME_NAMES} or pass time_col=. Columns: {cols}"
            )
        if self.id_col is None:
            self.id_col = _pick(cols, ID_NAMES) or "unique_id"
        idc, tc = self.id_col, self.time_col
        if idc not in cols:
            new = new.assign(**{idc: "series_0"})
        if not pd.api.types.is_integer_dtype(new[tc]):
            new[tc] = pd.to_datetime(new[tc])
        hist = new if self.history is None else pd.concat([self.history, new], ignore_index=True)
        hist = hist.drop_duplicates([idc, tc], keep="last").sort_values([idc, tc], kind="stable")
        keys = new[[idc, tc]].drop_duplicates()
        n_new = keys.groupby(idc).size()
        h = min(int(n_new.max()), self._client.limits.max_horizon)
        windows = -(-int(n_new.max()) // h)
        need = self._ready_after(hist)
        total = hist[idc].value_counts()
        ready = [u for u in n_new.index if total[u] - windows * h >= need]
        out = pd.DataFrame(columns=[idc, tc, "y", "median", "q10", "q90", "score", "anomaly"])
        if ready:
            scores = self._client.detect_anomalies(
                hist[hist[idc].isin(ready)], h=h, windows=windows, id_col=idc, time_col=tc, **self._opts
            )
            out = scores.merge(keys, on=[idc, tc]).reset_index(drop=True)
        self.history = hist.groupby(idc, sort=False).tail(self.window).reset_index(drop=True)
        return out
