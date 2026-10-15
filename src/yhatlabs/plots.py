"""Plots for every result (needs matplotlib: pip install "yhatlabs[plot]"). Each returns the matplotlib Figure.

from yhatlabs import plots
plots.forecast(fc, history=df)          # median and nested interval bands after the history
plots.backtest(ev)                      # actuals, each window's forecast, the cutoffs
plots.anomalies(flags)                  # series, expected range, flagged points
plots.scenarios(sc, history=df)         # one line and band per scenario
plots.calibration(ev)                   # forecast quantiles: nominal against observed
plots.reliability(ev_table)             # classification: predicted probability against observed frequency
plots.predicted_vs_actual(res, y)       # regression, with intervals as error bars
plots.importance(imp)                   # permutation importance bars
plots.effect(eff)                       # partial dependence curve
"""

from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd

from . import _plot


def forecast(
    fc: Any,
    history: Optional[pd.DataFrame] = None,
    ids: Optional[List[Any]] = None,
    max_series: int = 4,
    ax: Any = None,
) -> Any:
    """Median and nested interval bands per series, after the end of `history`."""
    return _plot.plot_forecast(fc, history, ids, max_series, ax)


def backtest(
    ev: Any, ids: Optional[List[Any]] = None, max_series: int = 4, model: Optional[str] = None, ax: Any = None
) -> Any:
    """Actuals across the backtest windows, each window's forecast and its cutoff."""
    return _plot.plot_backtest(ev, ids, max_series, model, ax)


def anomalies(flags: pd.DataFrame, ids: Optional[List[Any]] = None, max_series: int = 4, ax: Any = None) -> Any:
    """The series, its expected range and the flagged points (from detect_anomalies)."""
    return _plot.plot_anomalies(flags, ids, max_series, ax)


def scenarios(
    frame: pd.DataFrame,
    history: Optional[pd.DataFrame] = None,
    ids: Optional[List[Any]] = None,
    max_series: int = 4,
    ax: Any = None,
) -> Any:
    """One median line and interval band per scenario, per series."""
    return _plot.plot_scenarios(frame, history, ids, max_series, ax)


def calibration(ev: Any, ax: Any = None) -> Any:
    """Forecast calibration: nominal quantile level against the share of actuals below it."""
    return _plot.plot_calibration(ev.calibration if hasattr(ev, "calibration") else ev, ax)


def reliability(ev: Any, ax: Any = None) -> Any:
    """Classification calibration: predicted probability against observed frequency (from evaluate_table)."""
    return _plot.plot_reliability(ev.reliability if hasattr(ev, "reliability") else ev, ax)


def predicted_vs_actual(res: Any, actual: Any, ax: Any = None) -> Any:
    """Regression: predictions against true values, with the widest interval as error bars."""
    return _plot.plot_predicted_vs_actual(res.df if hasattr(res, "df") else res, actual, ax)


def importance(imp: pd.DataFrame, top: int = 20, ax: Any = None) -> Any:
    """Permutation importance bars with their spread (from FittedModel.importance)."""
    return _plot.plot_importance(imp, top, ax)


def effect(eff: pd.DataFrame, ax: Any = None) -> Any:
    """An effect curve: the average prediction as one column varies, with the spread across rows."""
    return _plot.plot_effect(eff, ax)
