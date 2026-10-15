"""Local data checks before anything is sent (no API key, no call): what the data looks like to the model, what will be
changed on the way (sorting, gap filling, truncation), and what is likely to hurt accuracy. Works for series and for tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._forecast import ID_NAMES, TARGET_NAMES, TIME_NAMES, Limits, _infer_freq, _pick

SEVERITIES = ("error", "warning", "note")


@dataclass
class DataCheck:
    """`issues`: one row per finding (severity error / warning / note, code, where, detail, suggestion).
    `profile`: one row per series (time series) or per column (tables). `ok` is False when any error was found.
    """

    kind: str  # "timeseries" or "tabular"
    issues: pd.DataFrame
    profile: pd.DataFrame
    _data: Any = field(default=None, repr=False)
    _cols: dict = field(default_factory=dict, repr=False)

    @property
    def ok(self) -> bool:
        return not (self.issues["severity"] == "error").any() if len(self.issues) else True

    def __repr__(self) -> str:
        counts = self.issues["severity"].value_counts().to_dict() if len(self.issues) else {}
        parts = (
            ", ".join(f"{counts[s]} {s}{'s' if counts[s] != 1 else ''}" for s in SEVERITIES if s in counts)
            or "no issues"
        )
        return f"DataCheck({self.kind}: {len(self.profile)} {'series' if self.kind == 'timeseries' else 'columns'}, {parts})"

    def _repr_html_(self) -> str:
        head = f"<p><b>{self!r}</b></p>"
        return head + (self.issues._repr_html_() if len(self.issues) else "<p>Nothing to fix.</p>")

    def repaired(self) -> pd.DataFrame:
        """Time series only: the data sorted, exact duplicate rows dropped, duplicate timestamps averaged, and each series put on
        its regular grid (missing timestamps added as empty rows). The same steps forecast() applies, made visible.
        """
        if self.kind != "timeseries":
            raise ValueError("repaired() applies to time-series checks.")
        df, c = self._data, self._cols
        id_col, time_col, target = c["id"], c["time"], c["target"]
        df = df.drop_duplicates()
        num = [k for k in df.columns if k not in (id_col, time_col) and pd.api.types.is_numeric_dtype(df[k])]
        other = [k for k in df.columns if k not in (id_col, time_col, *num)]
        df = df.groupby([id_col, time_col], as_index=False, sort=True).agg(
            {**{k: "mean" for k in num}, **{k: "first" for k in other}}
        )
        out = []
        for uid, g in df.groupby(id_col, sort=False):
            off = _infer_freq(pd.DatetimeIndex(g[time_col]))
            if off is None:
                out.append(g)
                continue
            grid = pd.date_range(g[time_col].iloc[0], g[time_col].iloc[-1], freq=off)
            g = g.set_index(time_col).reindex(grid).rename_axis(time_col).reset_index()
            g[id_col] = uid
            out.append(g)
        res = pd.concat(out, ignore_index=True)
        return res[[id_col, time_col, target] + [k for k in res.columns if k not in (id_col, time_col, target)]]


def _issue(rows: list, severity: str, code: str, where: Any, detail: str, suggestion: str = "") -> None:
    rows.append({"severity": severity, "code": code, "where": where, "detail": detail, "suggestion": suggestion})


def _frame(data: Any) -> pd.DataFrame:
    if hasattr(data, "to_pandas") and not isinstance(data, pd.DataFrame):  # polars
        data = data.to_pandas()
    if isinstance(data, pd.Series):
        data = data.to_frame(name=data.name or "y")
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.rename_axis("ds").reset_index()
    if not isinstance(data, pd.DataFrame):
        raise TypeError("check() takes a pandas or polars DataFrame (or a Series with a DatetimeIndex).")
    return data.rename(columns=str)


def check(
    data: Any,
    *,
    h: Optional[int] = None,
    target: Optional[Union[str, Sequence[str]]] = None,
    id_col: Optional[str] = None,
    time_col: Optional[str] = None,
    freq: Optional[str] = None,
    limits: Optional[Limits] = None,
) -> DataCheck:
    """Checks data locally. With a time column it is read as time series (pass h= to also check the horizon); without one, as a
    table to predict `target` from. Nothing leaves your machine.

    >>> import yhatlabs
    >>> yhatlabs.check(df, h=24)            # DataCheck(timeseries: 12 series, 1 warning, 2 notes)

    Args:
        data: A time-series table, or a tabular table with target=.
        h: The horizon you plan to forecast (checks the history is long enough).
        target: For tables: the target column(s); turns on the tabular checks.
        limits: Request caps to check against. Default: the published ones.
        freq: The series' frequency as a pandas alias (h, D, W-SUN, MS, 15min, ...). Inferred from the timestamps when
            left out; required for arrays and integer time steps.
        id_col: The column naming each series. Default: unique_id, id, item_id or series_id if present, else one series.
        time_col: The timestamp column. Default: ds, timestamp, time, date or datetime if present.
    """
    df = _frame(data)
    cols = list(df.columns)
    tcol = time_col or _pick(cols, TIME_NAMES)
    if tcol is None or (target is not None and time_col is None and _pick(cols, TIME_NAMES) is None):
        if target is None:
            raise ValueError("No time column found; for a table pass target= (the column to predict).")
        return _check_table(df, target if isinstance(target, str) else list(target)[0], limits or Limits())
    return _check_series(df, h, target, id_col, tcol, freq, limits or Limits())


def _check_series(df: pd.DataFrame, h, target, id_col, time_col, freq, limits: Limits) -> DataCheck:
    rows: list = []
    cols = list(df.columns)
    id_col = id_col or _pick(cols, ID_NAMES)
    if target is None:
        target = _pick(cols, TARGET_NAMES)
        if target is None:
            numeric = [c for c in cols if c not in (id_col, time_col) and pd.api.types.is_numeric_dtype(df[c])]
            target = numeric[0] if len(numeric) == 1 else None
    target = target if isinstance(target, str) or target is None else list(target)[0]
    if target is None or target not in cols:
        _issue(
            rows,
            "error",
            "no_target",
            "data",
            "Could not tell which column to forecast.",
            "Name it y/target/value or pass target=.",
        )
        return DataCheck("timeseries", pd.DataFrame(rows), pd.DataFrame())
    df = df.copy()
    if id_col is None:
        id_col = "unique_id"
        df[id_col] = "series_0"
    try:
        df[time_col] = pd.to_datetime(df[time_col])
    except (ValueError, TypeError) as e:
        _issue(
            rows,
            "error",
            "bad_timestamps",
            time_col,
            f"Timestamps could not be parsed: {e}",
            "Use ISO dates or pandas datetimes.",
        )
        return DataCheck("timeseries", pd.DataFrame(rows), pd.DataFrame())
    if df[time_col].isna().any():
        _issue(
            rows,
            "error",
            "missing_timestamps",
            time_col,
            f"{int(df[time_col].isna().sum())} rows have no timestamp.",
            "Drop or fill them.",
        )
    if not pd.api.types.is_numeric_dtype(df[target]):
        _issue(
            rows,
            "error",
            "non_numeric_target",
            target,
            f"{target!r} is not numeric ({df[target].dtype}).",
            "Convert it to numbers.",
        )
    exact = int(df.duplicated().sum())
    if exact:
        _issue(rows, "warning", "duplicate_rows", "data", f"{exact} exact duplicate rows.", "repaired() drops them.")
    dup = int(df.duplicated([id_col, time_col]).sum()) - exact
    if dup > 0:
        _issue(
            rows,
            "error",
            "duplicate_timestamps",
            "data",
            f"{dup} timestamps appear more than once within a series with different values.",
            "Aggregate them (repaired() averages them) or add the column that tells them apart to the id.",
        )
    profile, freqs = [], {}
    other = [c for c in cols if c not in (id_col, time_col, target)]
    for uid, g in df.dropna(subset=[time_col]).sort_values(time_col).groupby(id_col, sort=False):
        t = pd.DatetimeIndex(g[time_col].drop_duplicates())
        off = pd.tseries.frequencies.to_offset(freq) if freq else _infer_freq(t)
        y = pd.to_numeric(g[target], errors="coerce")
        n, n_missing = len(g), int(y.isna().sum())
        gaps = 0
        if off is not None and len(t) > 1:
            gaps = max(0, len(pd.date_range(t[0], t[-1], freq=off)) - len(t))
            freqs[uid] = off.freqstr
        else:
            _issue(
                rows,
                "error",
                "unknown_frequency",
                uid,
                "Fewer than 3 points or irregular timestamps; no frequency.",
                "Pass freq= or add history.",
            )
        obs = y.dropna()
        if len(obs) and obs.nunique() == 1:
            _issue(
                rows,
                "note",
                "constant",
                uid,
                f"The target is constant ({obs.iloc[0]}).",
                "A flat forecast is all the data supports.",
            )
        if len(obs) >= 20:
            med = obs.median()
            mad = (obs - med).abs().median() or (obs.std() or 0)
            out = int(((obs - med).abs() > 8 * mad * 1.4826).sum()) if mad else 0
            if out:
                _issue(
                    rows,
                    "note",
                    "extreme_values",
                    uid,
                    f"{out} values are far from the rest (over 8 robust standard deviations).",
                    "Check they are real; yh.detect_anomalies() scores them against the model's expectation.",
                )
        if (obs < 0).any() and (obs >= 0).mean() > 0.98:
            _issue(
                rows,
                "note",
                "rare_negatives",
                uid,
                f"{int((obs < 0).sum())} negative values in an otherwise non-negative series.",
                "Check the sign.",
            )
        if h is not None:
            need = max(2 * h, 16)
            if len(obs) < need:
                _issue(
                    rows,
                    "warning",
                    "short_history",
                    uid,
                    f"{len(obs)} observed points for h={h}.",
                    f"About {need}+ points give steadier forecasts.",
                )
        if n + gaps > limits.max_points_per_series:
            _issue(
                rows,
                "note",
                "truncated",
                uid,
                f"{n + gaps} points; only the most recent {limits.max_points_per_series} are sent.",
                "",
            )
        if gaps:
            share = gaps / (n + gaps)
            _issue(
                rows,
                "warning" if share > 0.2 else "note",
                "missing_timestamps",
                uid,
                f"{gaps} missing timestamps ({share:.0%} of the grid).",
                "They are sent as gaps; yh.fill_gaps() fills them from the history before each gap.",
            )
        if n_missing:
            share = n_missing / n
            _issue(
                rows,
                "warning" if share > 0.2 else "note",
                "missing_values",
                uid,
                f"{n_missing} missing target values ({share:.0%}).",
                "They are sent as gaps; the model handles them.",
            )
        profile.append(
            {
                "series": uid,
                "start": t[0] if len(t) else None,
                "end": t[-1] if len(t) else None,
                "points": n,
                "freq": off.freqstr if off is not None else None,
                "missing_timestamps": gaps,
                "missing_values": n_missing,
                "mean": float(obs.mean()) if len(obs) else np.nan,
                "std": float(obs.std()) if len(obs) > 1 else np.nan,
            }
        )
    if len(set(freqs.values())) > 1:
        counts = pd.Series(freqs).value_counts().to_dict()
        _issue(
            rows,
            "error",
            "mixed_frequencies",
            "data",
            f"Series have different frequencies: {counts}.",
            "Forecast each frequency separately or pass freq=.",
        )
    ends = pd.Series({p["series"]: p["end"] for p in profile if p["end"] is not None})
    if len(ends) > 1 and ends.nunique() > 1:
        late = (ends < ends.max()).sum()
        _issue(
            rows,
            "note",
            "ragged_ends",
            "data",
            f"{late} of {len(ends)} series end before the latest timestamp {ends.max()}.",
            "Each series is forecast from its own last point, so their horizons cover different dates.",
        )
    if other:
        _issue(
            rows,
            "note",
            "extra_columns",
            "data",
            f"Columns {other} are not used unless passed as known=, past= or static=.",
            "",
        )
    issues = pd.DataFrame(rows, columns=["severity", "code", "where", "detail", "suggestion"])
    issues["_o"] = issues["severity"].map({s: i for i, s in enumerate(SEVERITIES)})
    issues = issues.sort_values("_o", kind="stable").drop(columns="_o").reset_index(drop=True)
    return DataCheck(
        "timeseries", issues, pd.DataFrame(profile), df, {"id": id_col, "time": time_col, "target": target}
    )


def _check_table(df: pd.DataFrame, target: str, limits: Limits) -> DataCheck:
    rows: list = []
    if target not in df.columns:
        _issue(rows, "error", "no_target", target, f"{target!r} is not a column.", f"Columns: {list(df.columns)}")
        return DataCheck("tabular", pd.DataFrame(rows), pd.DataFrame())
    dupe_cols = sorted({c for c in df.columns if list(df.columns).count(c) > 1})
    if dupe_cols:
        _issue(rows, "error", "duplicate_columns", "data", f"Duplicate column names {dupe_cols}.", "Rename them.")
    y = df[target]
    labeled = int(y.notna().sum())
    if labeled < len(df):
        _issue(
            rows,
            "note",
            "unlabeled_rows",
            target,
            f"{len(df) - labeled} rows have no target.",
            "predict(train) without test= predicts exactly those.",
        )
    if y.dropna().nunique() < 2:
        _issue(
            rows,
            "error",
            "single_class",
            target,
            "The target has fewer than 2 distinct values.",
            "There is nothing to learn.",
        )
    if len(df) > limits.max_tabular_rows:
        _issue(
            rows,
            "error",
            "too_many_rows",
            "data",
            f"{len(df)} rows; the API takes {limits.max_tabular_rows} per call.",
            "Subsample the training rows.",
        )
    is_class = not pd.api.types.is_float_dtype(y) or y.dropna().nunique() <= 20
    if is_class and labeled:
        share = y.value_counts(normalize=True)
        if share.min() < 0.05:
            _issue(
                rows,
                "note",
                "imbalanced",
                target,
                f"The rarest class is {share.min():.1%} of rows ({share.idxmin()!r}).",
                "Judge with probabilities and a cost-based threshold rather than accuracy (examples/decision_threshold.py).",
            )
        if pd.api.types.is_integer_dtype(y) and y.nunique() > 2:
            _issue(
                rows,
                "note",
                "integer_target",
                target,
                f"Integer target with {y.nunique()} values: could be classes or a count.",
                "Pass task='classify' or task='regress' so it is not guessed.",
            )
    profile = []
    for c in df.columns:
        s = df[c]
        miss = float(s.isna().mean())
        uniq = int(s.nunique(dropna=True))
        profile.append({"column": c, "dtype": str(s.dtype), "missing": miss, "distinct": uniq})
        if c == target:
            continue
        if uniq <= 1:
            _issue(
                rows,
                "note",
                "constant_column",
                c,
                "The column has a single value.",
                "It carries no information; drop it.",
            )
        if miss > 0.5:
            _issue(
                rows, "note", "mostly_missing", c, f"{miss:.0%} missing.", "Fine to keep; missing values are handled."
            )
        text = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
        if uniq > 0.95 * len(df) and (text or pd.api.types.is_integer_dtype(s)) and len(df) > 50:
            _issue(
                rows,
                "warning",
                "id_like",
                c,
                f"{uniq} distinct values in {len(df)} rows: looks like an id.",
                "Drop identifier columns; they don't generalise.",
            )
        elif text and uniq > 200:
            _issue(
                rows,
                "note",
                "high_cardinality",
                c,
                f"{uniq} distinct text values.",
                "Free text is not modelled; group rare values or drop it.",
            )
        both = df[[c, target]].dropna()
        if len(both) > 20 and both[c].astype(str).equals(both[target].astype(str)):
            _issue(rows, "error", "target_copy", c, "The column equals the target.", "Remove it: it leaks the answer.")
        elif (
            len(both) > 50
            and pd.api.types.is_numeric_dtype(both[c])
            and pd.api.types.is_numeric_dtype(both[target])
            and both[c].std() > 0
            and both[target].std() > 0
        ):
            r = np.corrcoef(both[c].astype(float), both[target].astype(float))[0, 1]
            if np.isfinite(r) and abs(r) > 0.995:
                _issue(
                    rows,
                    "warning",
                    "near_copy",
                    c,
                    f"Correlation with the target is {r:.3f}.",
                    "Check it is known at prediction time; it may leak the answer.",
                )
    issues = pd.DataFrame(rows, columns=["severity", "code", "where", "detail", "suggestion"])
    issues["_o"] = issues["severity"].map({s: i for i, s in enumerate(SEVERITIES)})
    issues = issues.sort_values("_o", kind="stable").drop(columns="_o").reset_index(drop=True)
    return DataCheck("tabular", issues, pd.DataFrame(profile), df, {"target": target})
