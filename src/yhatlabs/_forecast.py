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
    pricing: Dict[str, Any] = field(
        default_factory=dict
    )  # what calls cost, as the API publishes it (estimate_table reads it)
    tabular_direct: Dict[str, Dict[str, int]] = field(
        default_factory=dict
    )  # per preset, the largest table a direct call takes ({"rows", "columns"}); larger ones run as jobs
    max_tabular_columns: Optional[int] = None
    max_cells_async: Optional[int] = None

    def direct_fits(self, preset: Optional[str], rows: int, columns: int) -> bool:
        """Whether a table call of this size gets a direct answer (True when the API publishes no direct size)."""
        d = self.tabular_direct.get({"full": "pro"}.get(preset or "fast", preset or "fast"))
        return not d or (rows <= d.get("rows", rows) and columns <= d.get("columns", columns))

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
            pricing=dict(lim.get("pricing") or {}),
            tabular_direct={k: dict(v) for k, v in (t.get("direct") or {}).items() if isinstance(v, dict)},
            max_tabular_columns=t.get("max_columns"),
            max_cells_async=t.get("max_cells_async"),
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


LAZY_NAMES = {"spark": "Spark DataFrame", "dask": "Dask DataFrame", "ray": "Ray Dataset", "polars": "polars LazyFrame"}


def _lazy(x: Any) -> Optional[str]:
    """Which lazy or distributed frame `x` is (spark, dask, ray, polars LazyFrame), by its type's module; None otherwise.
    Nothing is imported: only objects of those libraries match.
    """
    root = type(x).__module__.split(".")[0]
    if root == "pyspark" and hasattr(x, "toPandas"):
        return "spark"
    if root in ("dask", "dask_expr") and hasattr(x, "compute"):
        return "dask"
    if root == "ray" and hasattr(x, "to_pandas"):
        return "ray"
    if root == "polars" and type(x).__name__ == "LazyFrame":
        return "polars"
    return None


def _columns(x: Any, kind: str) -> int:
    if kind == "ray":
        return len(x.schema().names)
    if kind == "polars":
        return len(x.collect_schema() if hasattr(x, "collect_schema") else x.schema)
    return len(x.columns)


def _collect(x: Any, kind: str, n: Optional[int]) -> pd.DataFrame:
    """The frame in pandas: all of it, or its first n rows (so a frame past the limit is never pulled in whole)."""
    if kind == "spark":
        return (x if n is None else x.limit(n)).toPandas()
    if kind == "dask":
        return x.compute() if n is None else x.head(n, npartitions=-1)
    if kind == "ray":
        return (x if n is None else x.limit(n)).to_pandas()
    return (x if n is None else x.head(n)).collect().to_pandas()


def _pandas(x: Any, limits: Optional[Limits] = None) -> Any:
    """Polars, Spark, Dask and Ray frames (or anything else with .to_pandas()) in, pandas out; everything else unchanged.
    With limits (a table for the API), a lazy or distributed frame larger than the API takes raises DataError before it is
    collected in full.
    """
    if x is None or isinstance(x, (pd.DataFrame, pd.Series)):
        return x
    kind = _lazy(x)
    if kind is None:
        return x.to_pandas() if hasattr(x, "to_pandas") else x
    if limits is None:
        return _collect(x, kind, None)
    cap = limits.max_tabular_rows
    if limits.max_cells_async:
        cap = min(cap, limits.max_cells_async // max(1, _columns(x, kind)))
    df = _collect(x, kind, cap + 1)
    if len(df) > cap:
        raise DataError(
            f"The {LAZY_NAMES[kind]} has more than {cap} rows, more than the API takes in one call. Sample or aggregate it "
            "first (e.g. .sample() or a group-by), then pass the result."
        )
    return df


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


FREQ_ALIASES = (
    "D (daily), W (weekly), MS or M (monthly), QS (quarterly), YS (yearly), h (hourly), 15min, 30min, B (business days)"
)


def to_offset(freq: Any) -> pd.DateOffset:
    """freq= as a pandas offset, or a DataError naming the aliases people use."""
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", FutureWarning)
            return pd.tseries.frequencies.to_offset(freq)
    except (ValueError, TypeError):
        raise DataError(f"freq={freq!r} is not a pandas frequency. Common ones: {FREQ_ALIASES}.") from None


def _infer_freq(times: pd.DatetimeIndex) -> Optional[pd.DateOffset]:
    if len(times) < 3:
        return None
    try:
        with _warnings.catch_warnings():  # pandas warns about its own alias renames here; not the caller's concern
            _warnings.simplefilter("ignore", FutureWarning)
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
    if known and future is None:
        # one table: rows after a series' last value, with the target blank and the known columns filled, are its future
        last = df[targets].notna().any(axis=1).groupby(df[id_col]).transform(lambda m: m[::-1].cummax()[::-1])
        if (~last).any():
            future, df = df.loc[~last, [id_col, time_col, *known]], df.loc[last]
            warns.append(f"Used the {int((~last).sum())} rows after the last value as future= (their {known} values).")

    fut = None
    if known:
        if future is None:
            raise DataError(
                f"known={known} needs future= : a DataFrame with {id_col!r}, {time_col!r} and those columns over the next {h} steps"
                " (or rows after the last value in the same table, with the target blank)."
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
    req_off = to_offset(freq) if freq else None
    filled_total = 0
    short = []
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
        if T < h:
            short.append(uid)
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
    if short:
        warns.append(
            f"The horizon ({h} steps) is longer than the history of {len(short)} series (e.g. {short[0]!r}); forecasts that"
            " far ahead of so little data are rough."
        )
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


def mean_from_quantiles(qs: np.ndarray, native: np.ndarray) -> np.ndarray:
    """The mean is the integral of the quantile function over (0, 1): linear between the native levels, and an exponential tail
    beyond the outermost ones, its scale read from the outermost pair (a skewed forecast's mean moves toward its long tail).
    Checked on deciles: exact on symmetric shapes, within 1% on gamma and exponential, 6% low on a lognormal with sigma 1.
    qs: [h, n_native] sorted per row -> [h].
    """
    u = np.asarray(native, dtype=float)
    mid = ((qs[:, 1:] + qs[:, :-1]) / 2 * np.diff(u)).sum(axis=1)
    b_lo = (qs[:, 1] - qs[:, 0]) / np.log(u[1] / u[0])
    b_hi = (qs[:, -1] - qs[:, -2]) / np.log((1 - u[-2]) / (1 - u[-1]))
    return mid + u[0] * (qs[:, 0] - b_lo) + (1 - u[-1]) * (qs[:, -1] + b_hi)


def cdf_at(qs: np.ndarray, levels: Sequence[float], x: np.ndarray) -> np.ndarray:
    """P(Y <= x) per row from that row's quantiles: linear between the levels, the same exponential tails as
    mean_from_quantiles beyond the outermost ones. qs: [n, k] (sorted per row), x: [n] -> [n].
    """
    u = np.asarray(levels, dtype=float)
    qs = np.sort(np.asarray(qs, dtype=float), axis=1)
    x = np.asarray(x, dtype=float)
    out = np.empty(len(qs))
    with np.errstate(divide="ignore", invalid="ignore"):
        b_lo = (qs[:, 1] - qs[:, 0]) / np.log(u[1] / u[0])
        b_hi = (qs[:, -1] - qs[:, -2]) / np.log((1 - u[-2]) / (1 - u[-1]))
        for i in range(len(qs)):
            q, xi = qs[i], x[i]
            if xi < q[0]:
                out[i] = u[0] * np.exp((xi - q[0]) / b_lo[i]) if b_lo[i] > 0 else 0.0
            elif xi > q[-1]:
                out[i] = 1 - (1 - u[-1]) * np.exp((q[-1] - xi) / b_hi[i]) if b_hi[i] > 0 else 1.0
            else:  # q[j-1] <= x < q[j]; on a repeated quantile (a point mass, e.g. zeros) x gets the highest level it reaches
                j = int(np.searchsorted(q, xi, side="right"))
                if j == len(q) or xi == q[j - 1]:
                    out[i] = u[j - 1]
                else:
                    out[i] = u[j - 1] + (u[j] - u[j - 1]) * (xi - q[j - 1]) / (q[j] - q[j - 1])
    return out


def build_frame(
    prep: Prepared, replies: Dict[Any, Any], native: Sequence[float], levels: Sequence[float], mean: bool = False
) -> pd.DataFrame:
    """replies: uid -> {'quantiles': [h][n]} or {'targets': {col: [h][n]}}. mean=True adds a `mean` column (mean_from_quantiles)."""
    nat = np.asarray(native, dtype=float)
    rows = []
    for it in prep.items:
        r = replies[it.uid]
        per_target = r["targets"] if "targets" in r else {prep.target_cols[0]: r["quantiles"]}
        for col, q in per_target.items():
            raw = np.sort(np.asarray(q, dtype=float), axis=1)
            frame = pd.DataFrame(_interp(raw, nat, levels), columns=[qname(lv) for lv in levels])
            if mean:
                frame["mean"] = mean_from_quantiles(raw, nat)
            frame.insert(0, prep.time_col, it.future_times[: len(frame)])
            frame.insert(0, prep.id_col, it.uid)
            if it.multi:
                frame.insert(2, "target", col)
            rows.append(frame)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out


def _steps(it: Item, start: int, n: int) -> Any:
    """Timestamps (or integer steps) of positions start .. start + n - 1 of an item's sent history, continuing past its end."""
    T = it.length
    hist = it.times if it.times is not None else pd.Index(np.arange(T))
    pos = np.arange(start, start + n)
    inside = pos[pos < T]
    vals = list(hist[inside])
    if len(inside) < n:
        vals += list(it.future_times[: n - len(inside)])
    return vals


def build_samples(prep: Prepared, kept: List[Tuple[Item, dict]]) -> Optional[pd.DataFrame]:
    """Sample paths as a long table: id, time, (target), sample, value. None when none were asked for."""
    frames = []
    for it, r in kept:
        per = r.get("target_samples") or ({prep.target_cols[0]: r["samples"]} if "samples" in r else None)
        if not per:
            continue
        for col, S in per.items():
            S = np.asarray(S, dtype=float)
            n, h = S.shape
            f = pd.DataFrame(
                {
                    prep.id_col: it.uid,
                    prep.time_col: np.tile(np.asarray(it.future_times[:h], dtype=object), n),
                    "sample": np.repeat(np.arange(n), h),
                    "value": S.reshape(-1),
                }
            )
            if it.multi:
                f.insert(2, "target", col)
            frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else None


def build_history(prep: Prepared, kept: List[Tuple[Item, dict]], levels: Sequence[float]) -> Optional[pd.DataFrame]:
    """The past windows the model scored itself on (its backtest), as a table: id, cutoff, time, (target), quantile columns."""
    frames = []
    for it, r in kept:
        per = r.get("target_history") or ({prep.target_cols[0]: r["history"]} if "history" in r else None)
        if not per:
            continue
        for col, windows in per.items():
            for w in windows:
                q = np.asarray(w["quantiles"], dtype=float)
                cut = int(w["cutoff"])
                f = pd.DataFrame(q, columns=[qname(lv) for lv in levels])
                f.insert(0, prep.time_col, _steps(it, cut, len(q)))
                f.insert(0, "cutoff", _steps(it, cut - 1, 1)[0])
                f.insert(0, prep.id_col, it.uid)
                if it.multi:
                    f.insert(3, "target", col)
                frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else None


def _caller_level() -> int:
    """The stacklevel of the first frame outside this package, so a warning points at the user's line."""
    import sys

    pkg = __name__.rsplit(".", 1)[0]
    f, level = sys._getframe(1), 1
    while f is not None and (f.f_globals.get("__name__") or "").split(".")[0] == pkg:
        f, level = f.f_back, level + 1
    return level


def warn_all(msgs: List[str]) -> None:
    for m in msgs:
        _warnings.warn(m, stacklevel=_caller_level())
