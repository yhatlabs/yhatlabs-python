"""Helpers for building forecast inputs: calendar covariates and the future frame for known covariates.

from yhatlabs.features import add_calendar, make_future
future = add_calendar(make_future(df, h=24))          # id, time + hour, dayofweek, ... for the next 24 steps
fc = yh.forecast(add_calendar(df), h=24, known=CALENDAR, future=future)
"""

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from ._errors import ConfigurationError, DataError
from ._forecast import prepare

CALENDAR = ("hour", "dayofweek", "month", "is_weekend")
_PARTS = {
    "minute": lambda t: t.dt.minute,
    "hour": lambda t: t.dt.hour,
    "dayofweek": lambda t: t.dt.dayofweek,
    "day": lambda t: t.dt.day,
    "dayofyear": lambda t: t.dt.dayofyear,
    "weekofyear": lambda t: t.dt.isocalendar().week.astype(int),
    "month": lambda t: t.dt.month,
    "quarter": lambda t: t.dt.quarter,
    "year": lambda t: t.dt.year,
    "is_weekend": lambda t: (t.dt.dayofweek >= 5).astype(int),
    "is_month_start": lambda t: t.dt.is_month_start.astype(int),
    "is_month_end": lambda t: t.dt.is_month_end.astype(int),
}


def add_calendar(
    df: pd.DataFrame,
    features: Sequence[str] = CALENDAR,
    *,
    time_col: Optional[str] = None,
    holidays: Optional[str] = None,
    subdiv: Optional[str] = None,
) -> pd.DataFrame:
    """Adds calendar columns (known for any future date, so pass them as known=). holidays: a country code such as "US" or "DE"
    adds is_holiday (needs `pip install holidays`); subdiv a state or province, e.g. "TX".
    """
    from ._forecast import TIME_NAMES, _pick

    out = df.copy()
    time_col = time_col or _pick([str(c) for c in out.columns], TIME_NAMES)
    if time_col is None:
        raise DataError(f"No time column found; pass time_col=. Columns: {list(out.columns)}")
    t = pd.to_datetime(out[time_col])
    unknown = [f for f in features if f not in _PARTS]
    if unknown:
        raise DataError(f"Unknown calendar features {unknown}; choose from {sorted(_PARTS)}.")
    for f in features:
        out[f] = _PARTS[f](t).to_numpy()
    if holidays:
        try:
            import holidays as hol
        except ImportError as e:
            raise ConfigurationError("holidays= needs the holidays package: pip install holidays") from e
        cal = hol.country_holidays(holidays, subdiv=subdiv, years=range(t.dt.year.min(), t.dt.year.max() + 1))
        out["is_holiday"] = t.dt.date.map(lambda d: int(d in cal)).to_numpy()
    return out


def make_future(
    df: pd.DataFrame,
    h: int,
    *,
    id_col: Optional[str] = None,
    time_col: Optional[str] = None,
    freq: Optional[str] = None,
) -> pd.DataFrame:
    """The id and timestamps of the next `h` steps of every series, at the series' own frequency: fill in your known covariates
    (prices, planned promotions, weather forecasts) and pass it as future=.
    """
    prep = prepare(df, h, id_col=id_col, time_col=time_col, freq=freq)
    return pd.concat(
        [pd.DataFrame({prep.id_col: it.uid, prep.time_col: it.future_times}) for it in prep.items], ignore_index=True
    )
