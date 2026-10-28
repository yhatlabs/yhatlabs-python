"""How much each covariate moves a forecast: the forecast with every covariate, against the forecast with one covariate set to
a neutral value (numbers: the series' mean over the history and the future; categories and flags: their most common value).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._errors import DataError
from ._forecast import ID_NAMES, _pick
from ._results import ForecastResult


def covariates(data: Any, known: Optional[List[str]], past: Optional[List[str]]) -> List[str]:
    if not isinstance(data, pd.DataFrame):
        raise DataError("explain_forecast takes a DataFrame with id, time, target and covariate columns.")
    names = list(known or []) + list(past or [])
    if not names:
        raise DataError("explain_forecast explains covariates: pass known=[...] and / or past=[...].")
    missing = [c for c in names if c not in data.columns]
    if missing:
        raise DataError(f"Covariates {missing} are not columns of data. Columns: {list(data.columns)}")
    return names


def _neutral(s: pd.Series) -> Any:
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s.mean()
    mode = s.mode(dropna=True)
    return mode.iloc[0] if len(mode) else np.nan


def neutralise(
    data: pd.DataFrame, future: Optional[pd.DataFrame], col: str, id_col: Optional[str]
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """`col` set to its neutral value per series, in the history and the future alike."""
    idc = id_col or _pick([str(c) for c in data.columns], ID_NAMES)
    fut = future if future is not None and col in future.columns else None
    keep = [idc, col] if idc else [col]
    both = pd.concat([data[keep], fut[keep]] if fut is not None else [data[keep]], ignore_index=True)
    data = data.copy()
    if fut is not None:
        future = fut = fut.copy()
    frames = [data] + ([fut] if fut is not None else [])
    if idc:
        value = both.groupby(idc, sort=False)[col].agg(_neutral)
        for f in frames:
            f[col] = f[idc].map(value).to_numpy()
    else:
        v = _neutral(both[col])
        for f in frames:
            f[col] = v
    return data, future


def effects(base: ForecastResult, without: Dict[str, ForecastResult]) -> pd.DataFrame:
    keys = base._keys()
    b = base.df[keys + ["median"]]
    parts = []
    for name, res in without.items():
        m = b.merge(res.df[keys + ["median"]].rename(columns={"median": "median_without"}), on=keys, how="inner")
        m.insert(len(keys), "covariate", name)
        m["effect"] = m["median"] - m["median_without"]
        parts.append(m)
    out = pd.concat(parts, ignore_index=True)
    summary = (
        out.groupby("covariate", sort=False)["effect"]
        .agg(lambda e: float(e.abs().mean()))
        .rename("mean_abs_effect")
        .sort_values(ascending=False)
        .reset_index()
    )
    out.attrs["summary"] = summary
    return out
