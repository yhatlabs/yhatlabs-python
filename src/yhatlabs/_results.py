"""Result objects returned by the client."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd


def _prob(values: pd.DataFrame, levels: List[float], above: Any, below: Any) -> np.ndarray:
    """P(Y > above) or P(Y < below) per row, from the row's quantile columns (cdf_at)."""
    from ._errors import DataError
    from ._forecast import cdf_at

    if (above is None) == (below is None):
        raise DataError("Pass exactly one of above= or below=.")
    if len(levels) < 2:
        raise DataError(
            "prob() needs at least two quantile levels; request more (quantiles='all' gives every native level)."
        )
    x = np.broadcast_to(np.asarray(above if above is not None else below, dtype=float), (len(values),))
    p = cdf_at(values.to_numpy(dtype=float), levels, x)
    return 1 - p if above is not None else p


def _levels_of(columns: Any) -> List[float]:
    """Quantile column names back to levels: median -> 0.5, q10 -> 0.1, q2.5 -> 0.025."""
    out = []
    for c in columns:
        if c == "median":
            out.append(0.5)
        elif isinstance(c, str) and c.startswith("q"):
            try:
                out.append(float(c[1:]) / 100)
            except ValueError:
                pass
    return out


@dataclass
class ForecastResult:
    """Forecasts as a DataFrame: id, timestamp, (target if several), then one column per quantile (`median`, `q10`, `q90`...).

    `derived` lists quantile columns that were interpolated between the model's native quantiles rather than produced directly.
    """

    df: pd.DataFrame
    model: str
    freq: str
    quantile_levels: List[float]
    native_quantile_levels: List[float]
    derived: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    request_ids: List[str] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(
        default_factory=list
    )  # on_error="warn": the series whose batch failed, and why
    provenance: Dict[str, Any] = field(
        default_factory=dict
    )  # what produced this: model, settings, request ids, data fingerprint
    timing: Dict[str, Any] = field(default_factory=dict)  # seconds: total (client side, wall clock) and per request
    samples: Optional[pd.DataFrame] = None  # samples=N: id, time, sample, value (one path per sample number)
    history: Optional[pd.DataFrame] = None  # history=True: the model's own past-window forecasts, by cutoff

    def sample_paths(self, uid: Any = None) -> Any:
        """Sample paths of one series as an array [samples, h] (the first series by default). Sums, maxima or counts over the
        horizon get honest uncertainty from these: e.g. np.quantile(fc.sample_paths("a").sum(axis=1), [0.05, 0.95]).
        """
        from ._errors import DataError

        if self.samples is None:
            raise DataError("No sample paths: forecast with samples=N.")
        idc = self.samples.columns[0]
        uid = self.samples[idc].iloc[0] if uid is None else uid
        s = self.samples[self.samples[idc] == uid]
        return s.pivot(index="sample", columns=s.columns[1], values="value").to_numpy()

    def to_polars(self) -> Any:
        import polars as pl

        return pl.from_pandas(self.df)

    def _repr_html_(self) -> str:
        return (
            f"<p><b>ForecastResult</b> {self.model}, {len(self.df)} rows, quantiles {self.quantile_levels}</p>"
            + self.df.head(10)._repr_html_()
        )

    def plot(
        self,
        history: Optional[pd.DataFrame] = None,
        ids: Optional[List[Any]] = None,
        max_series: int = 4,
        ax: Any = None,
    ) -> Any:
        """Median and every central interval present (nested bands) per series, after the end of `history` if given (same
        id / time columns). Needs matplotlib.
        """
        from ._plot import plot_forecast

        return plot_forecast(self, history, ids, max_series, ax)

    def __repr__(self) -> str:
        return f"ForecastResult(model={self.model!r}, rows={len(self.df)}, quantiles={self.quantile_levels}, derived={self.derived})"

    def to_long(self) -> pd.DataFrame:
        """Hubverse-style long format: one row per (id, time, output): output_type 'quantile', output_type_id the level."""
        from ._forecast import qname

        idc = self._keys()
        frames = []
        for q in self.quantile_levels:
            f = self.df[idc].copy()
            f["output_type"] = "quantile"
            f["output_type_id"] = q
            f["value"] = self.df[qname(q)].values
            f["derived"] = qname(q) in self.derived
            frames.append(f)
        if "mean" in self.df.columns:
            f = self.df[idc].copy()
            f["output_type"] = "mean"
            f["output_type_id"] = None
            f["value"] = self.df["mean"].values
            f["derived"] = "mean" in self.derived
            frames.append(f)
        return pd.concat(frames, ignore_index=True)

    def _keys(self) -> List[str]:
        """The id, time and (for several targets) target columns: everything that is not a forecast value."""
        from ._forecast import qname

        values = {qname(q) for q in self.quantile_levels} | {"mean"}
        return [c for c in self.df.columns if c not in values]

    def intervals(self, levels: Union[float, Sequence[float]] = 80) -> pd.DataFrame:
        """Central prediction intervals by width in percent: `lower_80` / `upper_80` columns (from q10 / q90), next to the
        median. The quantiles must be in the forecast: request them with level=[80] (or quantiles=) when forecasting.
        """
        from ._errors import DataError
        from ._forecast import qname

        widths = [levels] if isinstance(levels, (int, float)) else list(levels)
        out = self.df[self._keys()].copy()
        if "median" in self.df.columns:
            out["median"] = self.df["median"].values
        for w in widths:
            lo, hi = qname(round(0.5 - w / 200, 6)), qname(round(0.5 + w / 200, 6))
            if lo not in self.df.columns or hi not in self.df.columns:
                raise DataError(f"A {w:g}% interval needs {lo} and {hi}; forecast with level=[{w:g}] to get them.")
            out[f"lower_{w:g}"] = self.df[lo].values
            out[f"upper_{w:g}"] = self.df[hi].values
        return out

    def prob(self, *, above: Any = None, below: Any = None) -> pd.DataFrame:
        """Probability of exceeding (above=) or staying under (below=) a limit at every forecast step, read from the forecast's
        quantiles. The limit is one number or one per row. More levels give a finer answer: forecast with quantiles="all".
        Beyond the outermost quantile the tail is extrapolated, so probabilities under 10% or over 90% are estimates.
        """
        from ._forecast import qname

        out = self.df[self._keys()].copy()
        out["prob"] = _prob(self.df[[qname(q) for q in self.quantile_levels]], self.quantile_levels, above, below)
        return out

    def to_wide(self, value: str = "median") -> pd.DataFrame:
        """One row per timestamp, one column per series (per series and target when there are several targets): the shape a
        spreadsheet or a plotting tool expects. value: which column to spread (median, mean, q10, ...).
        """
        from ._errors import DataError

        if value not in self.df.columns or value in self._keys():
            raise DataError(
                f"value must be one of the forecast columns; got {value!r}. Columns: {list(self.df.columns)}"
            )
        keys = self._keys()
        time_col, rest = keys[1], [keys[0]] + keys[2:]
        wide = self.df.pivot_table(index=time_col, columns=rest, values=value, aggfunc="first", sort=False)
        if len(rest) == 1:
            wide.columns = list(wide.columns)
        wide.columns.name = None
        return wide


@dataclass
class PredictionResult:
    """Tabular predictions aligned to the rows that were scored (`index` matches the input test rows)."""

    predictions: pd.Series
    model: str
    probabilities: Optional[pd.DataFrame] = None
    classes: Optional[List[Any]] = None
    fit: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    request_id: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    quantiles: Optional[pd.DataFrame] = None  # regression with quantiles=: one column per level (q5, median, q95, ...)
    fit_id: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, Any] = field(
        default_factory=dict
    )  # seconds: total (client side) plus the server's fit / predict split
    checks: Any = field(default=None, repr=False, compare=False)  # _profile.Checks against the training table
    prediction_sets: Optional[pd.Series] = None  # classification with set_coverage=: the classes per row (a list)
    set_coverage: Optional[float] = None
    interval_method: Optional[str] = None  # how regression quantiles were made: "distribution" (per row) or "conformal"
    mean: Optional[pd.Series] = (
        None  # regression with a per-row distribution: its mean (the prediction column is the fit's point)
    )

    @property
    def out_of_range(self) -> Optional[pd.Series]:
        """Per row, the columns whose value is outside the training range, or a category training never had ([] = none).
        Predictions on such rows are extrapolations; treat their intervals with care. None when no training profile exists.
        """
        return self.checks.out_of_range() if self.checks is not None else None

    @property
    def novelty(self) -> Optional[pd.Series]:
        """Per row, how unlike the training data it is: the distance to its nearest training rows as a percentile of how far
        training rows are from each other (0.5 typical, 0.99 farther than 99% of them). Computed on first access.
        """
        return self.checks.novelty() if self.checks is not None else None

    def to_polars(self) -> Any:
        import polars as pl

        return pl.from_pandas(self.df.reset_index())

    def __repr__(self) -> str:
        kind = (
            "classification"
            if self.probabilities is not None
            else ("intervals" if self.quantiles is not None else "prediction")
        )
        return f"PredictionResult(model={self.model!r}, rows={len(self.predictions)}, {kind})"

    def _repr_html_(self) -> str:
        return (
            f"<p><b>PredictionResult</b> {self.model}, {len(self.predictions)} rows</p>"
            + self.df.head(10)._repr_html_()
        )

    def prob(self, *, above: Any = None, below: Any = None) -> pd.Series:
        """Regression: probability that each row's value exceeds (above=) or stays under (below=) a limit (one number or one
        per row), read from the prediction intervals. Predict with several quantiles, e.g. quantiles=[0.05, 0.25, 0.5, 0.75, 0.95].
        Classification already returns probabilities (result.probabilities).
        """
        from ._errors import DataError
        from ._forecast import qname

        if self.quantiles is None:
            raise DataError("prob() reads the prediction intervals: predict with quantiles=[...] (regression).")
        levels = sorted(lv for lv in _levels_of(self.quantiles.columns))
        return pd.Series(
            _prob(self.quantiles[[qname(lv) for lv in levels]], levels, above, below),
            index=self.quantiles.index,
            name="prob",
        )

    @property
    def df(self) -> pd.DataFrame:
        out = pd.DataFrame({"prediction": self.predictions})
        if self.probabilities is not None:
            out = out.join(self.probabilities.add_prefix("p_"))
        if self.quantiles is not None:
            out = out.join(self.quantiles)
        if self.mean is not None:
            out = out.join(self.mean)
        if self.prediction_sets is not None:
            out = out.join(self.prediction_sets)
        return out


@dataclass
class EvaluationResult:
    """Backtest output. `scores`: one row per (model, series, [target], window) with each metric. `summary`: per model, the mean
    of each metric, plus `skill_<metric>` = 1 - geometric mean of the model's error relative to the baseline (higher is better).
    """

    scores: pd.DataFrame
    summary: pd.DataFrame
    forecasts: Optional[pd.DataFrame]
    baseline: Optional[str]
    windows: int
    h: int
    warnings: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"EvaluationResult(models={list(self.summary.index)}, windows={self.windows}, h={self.h})"

    @property
    def calibration(self) -> pd.DataFrame:
        """How honest the uncertainty is, per model, pooled over every scored point:
        kind "below": share of actuals below each predicted quantile (a calibrated q10 has 10% below it);
        kind "interval": share inside each central interval (a calibrated 80% interval holds 80%).
        gap = observed - nominal: near 0 is calibrated; a negative interval gap means intervals too narrow.
        """
        from ._evaluate import calibration_table

        return calibration_table(self.forecasts)

    def plot(
        self, ids: Optional[List[Any]] = None, max_series: int = 4, model: Optional[str] = None, ax: Any = None
    ) -> Any:
        """Actuals across the windows, each window's forecast and its cutoff. Needs matplotlib."""
        from ._plot import plot_backtest

        return plot_backtest(self, ids, max_series, model, ax)

    def plot_calibration(self, ax: Any = None) -> Any:
        """Reliability plot: nominal vs observed share below each quantile, one line per model. Needs matplotlib."""
        from ._plot import plot_calibration

        return plot_calibration(self.calibration, ax)
