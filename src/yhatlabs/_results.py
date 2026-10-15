"""Result objects returned by the client."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


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
        """Median and the widest interval per series, after the last `history` points if given (same id/time/target columns).
        Needs matplotlib.
        """
        from ._plot import plot_forecast

        return plot_forecast(self, history, ids, max_series, ax)

    def __repr__(self) -> str:
        return f"ForecastResult(model={self.model!r}, rows={len(self.df)}, quantiles={self.quantile_levels}, derived={self.derived})"

    def to_long(self) -> pd.DataFrame:
        """Hubverse-style long format: one row per (id, time, output): output_type 'quantile', output_type_id the level."""
        from ._forecast import qname

        idc = [c for c in self.df.columns if c not in {qname(q) for q in self.quantile_levels}]
        frames = []
        for q in self.quantile_levels:
            f = self.df[idc].copy()
            f["output_type"] = "quantile"
            f["output_type_id"] = q
            f["value"] = self.df[qname(q)].values
            f["derived"] = qname(q) in self.derived
            frames.append(f)
        return pd.concat(frames, ignore_index=True)


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

    @property
    def df(self) -> pd.DataFrame:
        out = pd.DataFrame({"prediction": self.predictions})
        if self.probabilities is not None:
            out = out.join(self.probabilities.add_prefix("p_"))
        if self.quantiles is not None:
            out = out.join(self.quantiles)
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

    def plot_calibration(self, ax: Any = None) -> Any:
        """Reliability plot: nominal vs observed share below each quantile, one line per model. Needs matplotlib."""
        from ._plot import plot_calibration

        return plot_calibration(self.calibration, ax)
