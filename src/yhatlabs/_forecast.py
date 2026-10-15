"""Turning user data into forecast requests and API replies into DataFrames.

The public interface follows the YHat prediction spec (one table + column roles): a long DataFrame with an id, a timestamp and
one or more targets, plus optional covariate columns. The hosted API today takes plain arrays per series, so everything that
depends on time is handled here, on the client: sorting, frequency inference, filling gaps, aligning known future covariates,
building the future timestamps, splitting batches to stay inside the API's limits.
"""

from __future__ import annotations

import math
import warnings as _warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ._errors import DataError, UnsupportedError

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)
ID_NAMES = ("unique_id", "id", "item_id", "series_id")
TIME_NAMES = ("ds", "timestamp", "time", "date", "datetime")
TARGET_NAMES = ("y", "target", "value")


@dataclass
class Limits:
    """Size limits of one API request. The defaults are the hosted API's published caps; the client splits batches to fit."""

    max_series: int = 256
    max_points_per_series: int = 32_768
    max_total_points: int = 2_000_000
    max_horizon: int = 2048
    max_tabular_rows: int = 200_000
    max_tabular_bytes: int = 64 * 1024 * 1024
    max_columns: int = 256  # targets per series, and past / known covariates per series (each counted separately)
    capabilities: Dict[str, Any] = field(
        default_factory=dict
    )  # features the API has switched on; a missing flag means off

    @classmethod
    def from_api(cls, lim: dict) -> Limits:
        f, t = lim.get("forecast", {}), lim.get("tabular", {})
        d = cls()
        return cls(
            max_series=f.get("max_series", d.max_series),
            max_points_per_series=f.get("max_points_per_series", d.max_points_per_series),
            max_total_points=f.get("max_total_points", d.max_total_points),
            max_horizon=f.get("max_horizon", d.max_horizon),
            max_tabular_rows=t.get("max_rows", d.max_tabular_rows),
            max_tabular_bytes=t.get("max_bytes", d.max_tabular_bytes),
            max_columns=f.get("max_columns", d.max_columns),
            capabilities=dict(lim.get("capabilities") or {}),
        )

    def supports(self, feature: str) -> bool:
        return bool(self.capabilities.get(feature))


@dataclass
class Item:
    """One forecastable unit: a series id with one or more target columns of equal length T, plus covariates."""

    uid: Any
    targets: Dict[str, List[Optional[float]]]
    multi: bool
    times: Optional[pd.Index]  # history timestamps or integer steps (None for array input)
    future_times: Union[pd.Index, np.ndarray]
    past: Dict[str, list] = field(default_factory=dict)
    known: Dict[str, list] = field(default_factory=dict)  # length T + h

    @property
    def length(self) -> int:
        return len(next(iter(self.targets.values())))

    @property
    def points(self) -> int:
        T = self.length
        return T * len(self.targets) + T * len(self.past) + (T + len(self.future_times)) * len(self.known)

    def payload(self, wire_id: str) -> dict:
        p: dict = {"id": wire_id}
        if self.multi:
            p["targets"] = self.targets
        else:
            p["values"] = next(iter(self.targets.values()))
        if self.past:
            p["past_covariates"] = self.past
        if self.known:
            p["known_covariates"] = self.known
        return p


@dataclass
class Prepared:
    items: List[Item]
    freq: str
    id_col: str
    time_col: str
    target_cols: List[str]
    warnings: List[str]


def _pandas(x: Any) -> Any:
    """Polars (or anything else with .to_pandas()) in, pandas out; everything else unchanged."""
    if x is not None and not isinstance(x, (pd.DataFrame, pd.Series)) and hasattr(x, "to_pandas"):
        return x.to_pandas()
    return x


def _clean(v: Any) -> Optional[Union[float, str]]:
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, (bool, np.bool_)):
        return float(v)
    if v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, str):
        return v
    try:
        f = float(v)
        return None if not math.isfinite(f) else f
    except (TypeError, ValueError):
        return str(v)


def _numeric_list(
    s: Union[pd.Series, np.ndarray, list], what: str, warns: Optional[List[str]] = None
) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    n_inf = 0
    for j, v in enumerate(list(s)):
        c = _clean(v)
        if isinstance(c, str):
            raise DataError(f"{what} must be numeric; found {v!r} at position {j}.")
        if c is None and isinstance(v, (float, np.floating)) and math.isinf(float(v)):
            n_inf += 1
        out.append(c)
    if n_inf and warns is not None:
        warns.append(f"{what} has {n_inf} infinite values; they were sent as missing.")
    return out


def _rows(data: Any) -> Optional[list]:
    """A 2-D array or a list of sequences holds one series per row; returns the rows, or None for a single 1-D series."""
    if isinstance(data, np.ndarray):
        if data.ndim > 2:
            raise DataError(f"Arrays must be 1-D (one series) or 2-D (one series per row); got shape {data.shape}.")
        if data.ndim == 2 and data.shape[1] == 1:
            return None  # a column vector is one series
        return list(data) if data.ndim == 2 else None
    if len(data) and all(isinstance(r, (list, tuple, np.ndarray, pd.Series)) for r in data):
        return list(data)
    if any(isinstance(r, (list, tuple, np.ndarray, pd.Series)) for r in data):
        raise DataError("A list mixes numbers and sequences; pass one series (a flat list) or one sequence per series.")
    return None


def _cov_list(s: Union[pd.Series, list]) -> list:
    return [_clean(v) for v in list(s)]


def _pick(cols: Sequence[str], names: Sequence[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for n in names:
        if n in low:
            return low[n]
    return None


def _infer_freq(times: pd.DatetimeIndex) -> Optional[pd.DateOffset]:
    if len(times) < 3:
        return None
    try:
        f = pd.infer_freq(times)
    except (ValueError, TypeError):
        f = None
    if f:
        return pd.tseries.frequencies.to_offset(f)
    diffs = pd.Series(times[1:] - times[:-1])
    mode = diffs.mode()
    if len(mode) == 0 or mode.iloc[0] <= pd.Timedelta(0):
        return None
    return pd.tseries.frequencies.to_offset(mode.iloc[0])


def _freq_str(off: pd.DateOffset) -> str:
    return off.freqstr


def _regularize(df: pd.DataFrame, time_col: str, off: pd.DateOffset) -> Tuple[pd.DataFrame, int]:
    """Reindex one series onto its regular time grid; returns the frame and how many timestamps were missing."""
    full = pd.date_range(df[time_col].iloc[0], df[time_col].iloc[-1], freq=off)
    if len(full) == len(df) and (full == pd.DatetimeIndex(df[time_col])).all():
        return df, 0
    out = df.set_index(time_col).reindex(full)
    out.index.name = time_col
    out = out.reset_index()
    return out, len(full) - len(df)


def _regularize_steps(df: pd.DataFrame, time_col: str, step: int) -> Tuple[pd.DataFrame, int]:
    t = df[time_col].to_numpy()
    full = np.arange(t[0], t[-1] + step, step)
    if len(full) == len(df):
        return df, 0
    out = df.set_index(time_col).reindex(full)
    out.index.name = time_col
    return out.reset_index(), len(full) - len(df)


def prepare(
    data: Any,
    h: int,
    *,
    id_col: Optional[str] = None,
    time_col: Optional[str] = None,
    target_col: Optional[Union[str, Sequence[str]]] = None,
    freq: Optional[str] = None,
    known: Optional[Sequence[str]] = None,
    past: Optional[Sequence[str]] = None,
    future: Optional[pd.DataFrame] = None,
    limits: Optional[Limits] = None,
    static: Optional[Sequence[str]] = None,
) -> Prepared:
    """Validate and convert user input into API items. Raises DataError with a fix-it message on bad input."""
    limits = limits or Limits()
    data, future = _pandas(data), _pandas(future)
    if not isinstance(h, (int, np.integer)) or h < 1:
        raise DataError("h (the horizon) must be a positive integer.")
    if h > limits.max_horizon:
        raise DataError(f"h={h} is above the API's maximum horizon of {limits.max_horizon}.")
    warns: List[str] = []

    # ---- plain arrays / dicts / Series without timestamps ----
    if isinstance(data, (list, tuple, np.ndarray)) or (isinstance(data, dict) and not isinstance(data, pd.DataFrame)):
        if known or past or static or future is not None:
            raise DataError("Covariates need a DataFrame input with id and time columns.")
        if freq is None:
            raise DataError(
                "Without timestamps, pass freq= (e.g. 'h', 'D', 'W', 'MS') so the model knows the seasonality."
            )
        if isinstance(data, dict):
            series = data
        else:
            rows = _rows(data)
            series = {f"series_{i}": r for i, r in enumerate(rows)} if rows is not None else {"series_0": data}
        if not series:
            raise DataError("No series to forecast.")
        items = []
        for uid, arr in series.items():
            vals = _numeric_list(np.asarray(arr, dtype=object).ravel(), f"series {uid!r}", warns)
            if not any(v is not None for v in vals):
                raise DataError(f"series {uid!r} has no observed values.")
            items.append(
                Item(
                    uid=uid,
                    targets={"y": vals},
                    multi=False,
                    times=None,
                    future_times=np.arange(len(vals), len(vals) + h),
                )
            )
        return Prepared(items, str(freq), "unique_id", "ds", ["y"], warns)

    if isinstance(data, pd.Series):
        name = data.name if data.name is not None else "y"
        data = data.to_frame(name=name)
        if target_col is None:
            target_col = name
    if not isinstance(data, pd.DataFrame):
        raise DataError(
            f"Unsupported input type {type(data).__name__}; pass a pandas DataFrame, Series, numpy array or dict of arrays."
        )

    if data.empty:
        raise DataError("The table has no rows.")
    df = data.copy()
    if (
        time_col is None
        and isinstance(df.index, pd.DatetimeIndex)
        and _pick(df.columns.astype(str), TIME_NAMES) is None
    ):
        df = df.reset_index()
        time_col = df.columns[0] if df.columns[0] != "index" else "ds"
        df = df.rename(columns={df.columns[0]: time_col})
    cols = [str(c) for c in df.columns]
    df.columns = cols
    id_col = id_col or _pick(cols, ID_NAMES)
    time_col = time_col or _pick(cols, TIME_NAMES)
    if time_col is None:
        raise DataError(f"Could not find a time column; name it one of {TIME_NAMES} or pass time_col=. Columns: {cols}")
    if time_col not in cols:
        raise DataError(f"time_col {time_col!r} is not a column. Columns: {cols}")
    known, past, static = list(known or []), list(past or []), list(static or [])
    if target_col is None:
        target_col = _pick(cols, TARGET_NAMES)
        if target_col is None:
            rest = [
                c
                for c in cols
                if c not in {id_col, time_col, *known, *past, *static} and pd.api.types.is_numeric_dtype(df[c])
            ]
            if len(rest) != 1:
                raise DataError(
                    f"Could not tell which column to forecast; pass target_col=. Numeric candidates: {rest}"
                )
            target_col = rest[0]
    targets = [target_col] if isinstance(target_col, str) else list(target_col)
    for c in targets + known + past + static + ([id_col] if id_col else []):
        if c not in cols:
            raise DataError(f"Column {c!r} not found. Columns: {cols}")
    for what, n in (
        ("targets", len(targets)),
        ("past covariates", len(past)),
        ("known and static covariates", len(known) + len(static)),
    ):
        if n > limits.max_columns:
            raise DataError(
                f"{n} {what}; the API takes at most {limits.max_columns} per series. Select the most useful ones."
            )
    overlap = set(targets) & set(known + past + static)
    if overlap:
        raise DataError(f"Columns {sorted(overlap)} are both targets and covariates.")
    ignored = [c for c in cols if c not in {id_col, time_col, *targets, *known, *past, *static}]
    if ignored:
        warns.append(
            f"Ignored columns {ignored}; pass them as known=[...] (values known for the future) or past=[...] (history only) to use them."
        )
    if id_col is None:
        id_col = "unique_id"
        df[id_col] = "series_0"
    if df[id_col].isna().any():
        bad = df.index[df[id_col].isna()]
        raise DataError(
            f"{len(bad)} rows have no series id in {id_col!r} (e.g. index {list(bad[:3])}); fill or drop them."
        )

    # Integers in the time column are steps (1, 2, 3, ...), not nanoseconds since 1970: they keep their values and the
    # future continues them. The model still needs to know what one step is, for the seasonality.
    int_time = pd.api.types.is_integer_dtype(df[time_col]) and not pd.api.types.is_bool_dtype(df[time_col])
    if int_time:
        if freq is None:
            raise DataError(
                f"Column {time_col!r} holds integer steps; pass freq= to say what one step is (e.g. 'h', 'D', 'W', 'MS')."
            )
        if (
            known
            and future is not None
            and time_col in future.columns
            and not pd.api.types.is_integer_dtype(future[time_col])
        ):
            raise DataError(f"The history uses integer steps in {time_col!r}, so future= must too.")
    else:
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except (ValueError, TypeError) as e:
            raise DataError(f"Column {time_col!r} could not be parsed as timestamps: {e}") from e
    if df[time_col].isna().any():
        raise DataError(f"Column {time_col!r} has missing timestamps.")
    if df.duplicated([id_col, time_col]).any():
        dup = df[df.duplicated([id_col, time_col], keep=False)][[id_col, time_col]].head(3).values.tolist()
        raise DataError(f"Duplicate (id, time) rows, e.g. {dup}. Each series needs one row per timestamp.")
    df = df.sort_values([id_col, time_col], kind="stable")

    fut = None
    if known:
        if future is None:
            raise DataError(
                f"known={known} needs future= : a DataFrame with {id_col!r}, {time_col!r} and those columns over the next {h} steps."
            )
        fut = future.copy()
        fut.columns = [str(c) for c in fut.columns]
        if id_col not in fut.columns:
            if df[id_col].nunique() == 1:
                fut[id_col] = df[id_col].iloc[0]
            else:
                raise DataError(f"future= needs the id column {id_col!r}.")
        if time_col not in fut.columns:
            raise DataError(f"future= needs the time column {time_col!r}.")
        missing = [c for c in known if c not in fut.columns]
        if missing:
            raise DataError(f"future= is missing known covariates {missing}.")
        if not int_time:
            fut[time_col] = pd.to_datetime(fut[time_col])
    elif future is not None:
        warns.append("future= was given but no known=[...] columns; it was ignored.")

    offsets = {}
    groups = list(df.groupby(id_col, sort=False))
    req_off = pd.tseries.frequencies.to_offset(freq) if freq else None
    filled_total = 0
    items: List[Item] = []
    for uid, g in groups:
        g = g.reset_index(drop=True)
        n_obs = len(g)
        if int_time:
            steps = np.diff(g[time_col].to_numpy())
            off = int(np.gcd.reduce(steps)) if len(steps) else 1
            offsets[uid] = req_off
            g, n_filled = _regularize_steps(g, time_col, off)
        else:
            off = req_off or _infer_freq(pd.DatetimeIndex(g[time_col]))
            if off is None:
                raise DataError(
                    f"Could not infer the frequency of series {uid!r} (fewer than 3 points or irregular); pass freq=."
                )
            offsets[uid] = off
            g, n_filled = _regularize(g, time_col, off)
        if n_filled > n_obs:
            step = f"step {off}" if int_time else f"frequency {_freq_str(off)!r}"
            raise DataError(
                f"Series {uid!r} has {n_obs} rows, but on a regular grid at {step} {n_filled} of {n_obs + n_filled} "
                f"timestamps would be missing. The timestamps look irregular: pass freq= for the spacing you want, "
                f"or resample first (e.g. df.set_index('{time_col}').resample('D').mean())."
            )
        filled_total += n_filled
        T = len(g)
        if T > limits.max_points_per_series:
            g = g.iloc[T - limits.max_points_per_series :].reset_index(drop=True)
            warns.append(
                f"Series {uid!r} has {T} points; only the most recent {limits.max_points_per_series} were sent."
            )
            T = len(g)
        tvals = {c: _numeric_list(g[c], f"series {uid!r} column {c!r}", warns) for c in targets}
        if not any(v is not None for vs in tvals.values() for v in vs):
            raise DataError(f"Series {uid!r} has no observed values.")
        if int_time:
            times = pd.Index(g[time_col].to_numpy())
            future_times = pd.Index(np.arange(1, h + 1) * off + times[-1])
        else:
            times = pd.DatetimeIndex(g[time_col])
            future_times = pd.date_range(times[-1], periods=h + 1, freq=off)[1:]
        item = Item(uid=uid, targets=tvals, multi=len(targets) > 1, times=times, future_times=future_times)
        for c in past:
            item.past[c] = _cov_list(g[c])
        for c in (
            static
        ):  # constant per series: sent as a known covariate over context and horizon (the API has no static slot)
            vals = pd.unique(g[c].dropna())
            if len(vals) > 1:
                raise DataError(
                    f"static column {c!r} changes within series {uid!r} ({list(vals[:3])}...); pass it as known= or past=."
                )
            item.known[c] = [_clean(vals[0]) if len(vals) else None] * (T + h)
        if known:
            f = fut[fut[id_col] == uid].set_index(time_col)
            fv = f.reindex(future_times)
            if fv[known].isna().all(axis=1).any():
                gap = [str(t) for t in fv.index[fv[known].isna().all(axis=1)][:3]]
                raise DataError(
                    f"future= has no values for series {uid!r} at {gap}...; it must cover the {h} steps after {times[-1]}."
                )
            for c in known:
                item.known[c] = _cov_list(g[c]) + _cov_list(fv[c])
        items.append(item)
    if filled_total:
        warns.append(f"Filled {filled_total} missing timestamps with gaps (sent as missing values).")
    distinct = {_freq_str(o) for o in offsets.values()}
    if len(distinct) > 1:
        raise DataError(
            f"Series have different frequencies {sorted(distinct)}; forecast each frequency separately or pass freq=."
        )
    return Prepared(items, distinct.pop(), id_col, time_col, targets, warns)


def batches(items: List[Item], limits: Limits) -> List[List[Item]]:
    """Greedy packing so that every request respects the series and total-points caps."""
    out: List[List[Item]] = []
    cur: List[Item] = []
    pts = 0
    for it in items:
        p = it.points
        if p > limits.max_total_points:
            raise DataError(
                f"Series {it.uid!r} alone has {p} points (with covariates), above the per-request limit {limits.max_total_points}."
            )
        if cur and (len(cur) >= limits.max_series or pts + p > limits.max_total_points):
            out.append(cur)
            cur, pts = [], 0
        cur.append(it)
        pts += p
    if cur:
        out.append(cur)
    return out


def resolve_quantiles(quantiles: Any, native: Sequence[float]) -> Tuple[List[float], List[str]]:
    """Returns the requested levels and which of them are derived (interpolated) rather than native model outputs."""
    if quantiles is None:
        levels = list(DEFAULT_QUANTILES)
    elif isinstance(quantiles, str) and quantiles == "all":
        levels = list(native)
    else:
        levels = sorted({float(q) for q in quantiles})
    lo, hi = min(native), max(native)
    bad = [q for q in levels if not (0 < q < 1)]
    if bad:
        raise DataError(f"Quantile levels must be between 0 and 1; got {bad}.")
    outside = [q for q in levels if q < lo - 1e-9 or q > hi + 1e-9]
    if outside:
        raise UnsupportedError(
            f"The model returns quantiles from {lo} to {hi}; {outside} would need extrapolation, which this client does not do."
        )
    derived = [qname(q) for q in levels if not any(abs(q - n) < 1e-9 for n in native)]
    return levels, derived


def qname(q: float) -> str:
    """Column name for a quantile level: 0.5 -> 'median', 0.1 -> 'q10', 0.025 -> 'q2.5'."""
    if abs(q - 0.5) < 1e-12:
        return "median"
    return f"q{round(q * 100, 6):g}"


def _interp(qs: np.ndarray, native: np.ndarray, levels: Sequence[float]) -> np.ndarray:
    """Qs [h, n_native] -> [h, len(levels)], linear in the level between neighbouring native quantiles."""
    qs = np.sort(qs, axis=1)
    return np.stack([np.array([np.interp(lv, native, row) for lv in levels]) for row in qs])


def build_frame(
    prep: Prepared, replies: Dict[Any, Any], native: Sequence[float], levels: Sequence[float]
) -> pd.DataFrame:
    """replies: uid -> {'quantiles': [h][n]} or {'targets': {col: [h][n]}}."""
    nat = np.asarray(native, dtype=float)
    rows = []
    for it in prep.items:
        r = replies[it.uid]
        per_target = r["targets"] if "targets" in r else {prep.target_cols[0]: r["quantiles"]}
        for col, q in per_target.items():
            arr = _interp(np.asarray(q, dtype=float), nat, levels)
            frame = pd.DataFrame(arr, columns=[qname(lv) for lv in levels])
            frame.insert(0, prep.time_col, it.future_times[: len(frame)])
            frame.insert(0, prep.id_col, it.uid)
            if it.multi:
                frame.insert(2, "target", col)
            rows.append(frame)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out


def warn_all(msgs: List[str]) -> None:
    for m in msgs:
        _warnings.warn(m, stacklevel=3)
