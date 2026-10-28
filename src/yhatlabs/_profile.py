"""What the training table looked like, kept on the client so every prediction can say where it is outside that experience.

A profile holds per-column ranges (numbers) and categories (text, up to MAX_CATEGORIES), plus a sample of at most SAMPLE_ROWS
training rows. `out_of_range` names, per predicted row, the columns outside the training range or with a category training
never had. `novelty` is how far the row is from its nearest training rows, as a percentile of how far training rows are from
each other: 0.5 is typical, 0.99 farther than 99% of training rows (the model is extrapolating there). Nothing is sent anywhere.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ._errors import DataError

SAMPLE_ROWS = 2000
MAX_CATEGORIES = 500
CHUNK = 1024


def build(train: pd.DataFrame, features: List[str], seed: int = 0) -> Dict[str, Any]:
    cols: Dict[str, Any] = {}
    for c in features:
        s = train[c]
        if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
            vals = s.dropna().astype(str)
            vc = vals.value_counts()
            cols[c] = {
                "kind": "category",
                "categories": vc.index[:MAX_CATEGORIES].tolist(),
                "complete": len(vc) <= MAX_CATEGORIES,
            }
        else:
            x = s.to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if not len(x):
                continue
            sd = float(np.std(x))
            cols[c] = {
                "kind": "number",
                "min": float(x.min()),
                "max": float(x.max()),
                "mean": float(x.mean()),
                "std": sd if sd > 0 else 1.0,
            }
    sample = train[list(cols)].iloc[_sample_positions(len(train), seed)]
    prof = {"columns": cols, "sample": _json_rows(sample), "rows": int(len(train))}
    enc = _encode(prof, sample)
    prof["reference"] = _reference(enc).tolist()
    return prof


def _sample_positions(n: int, seed: int) -> np.ndarray:
    return pd.RangeIndex(n).to_series().sample(min(n, SAMPLE_ROWS), random_state=seed).to_numpy()


def _json_rows(df: pd.DataFrame) -> Dict[str, list]:
    out = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
            out[c] = [None if pd.isna(v) else str(v) for v in s]
        else:
            out[c] = [None if not np.isfinite(v) else float(v) for v in s.to_numpy(dtype=float)]
    return out


def _encode(prof: Dict[str, Any], df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Numbers standardised (missing -> the mean), categories as codes (-1 = unseen or missing)."""
    num, cat = [], []
    for c, info in prof["columns"].items():
        s = df[c] if c in df.columns else pd.Series([None] * len(df))
        if info["kind"] == "number":
            x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            x = np.where(np.isfinite(x), (x - info["mean"]) / info["std"], 0.0)
            num.append(x)
        else:
            lookup = {v: i for i, v in enumerate(info["categories"])}
            cat.append(np.array([lookup.get(None if pd.isna(v) else str(v), -1) for v in s]))
    n = len(df)
    return {
        "num": np.column_stack(num) if num else np.zeros((n, 0)),
        "cat": np.column_stack(cat) if cat else np.zeros((n, 0), int),
    }


def _nn(
    a: Dict[str, np.ndarray], b: Dict[str, np.ndarray], skip_self: bool = False, skip: Optional[np.ndarray] = None
) -> np.ndarray:
    """Distance from every row of a to its nearest row of b: Euclidean on standardised numbers, plus 1 per differing category.
    Chunked so memory stays at CHUNK x len(b) floats whatever the column count. skip: per row of a, a row of b it is not
    compared with (itself; -1 for none); skip_self is skip = 0, 1, 2, ... (a and b the same rows).
    """
    out = np.empty(len(a["num"]))
    if skip_self:
        skip = np.arange(len(out))
    bn, bc = b["num"], b["cat"]
    bsq = (bn**2).sum(axis=1)
    for i in range(0, len(out), CHUNK):
        an, ac = a["num"][i : i + CHUNK], a["cat"][i : i + CHUNK]
        with np.errstate(all="ignore"):  # numpy 2.0 on macOS (Accelerate) warns on finite matmuls; the values are right
            d2 = (an**2).sum(axis=1)[:, None] + bsq[None, :] - 2 * an @ bn.T
        for j in range(ac.shape[1]):
            d2 += (ac[:, j : j + 1] != bc[None, :, j]) | (ac[:, j : j + 1] < 0)
        if skip is not None:
            sk = skip[i : i + CHUNK]
            r = np.flatnonzero(sk >= 0)
            d2[r, sk[r]] = np.inf
        out[i : i + CHUNK] = np.sqrt(np.maximum(d2.min(axis=1), 0))
    return out


def _reference(enc: Dict[str, np.ndarray]) -> np.ndarray:
    if len(enc["num"]) < 2:
        return np.zeros(1)
    return np.sort(_nn(enc, enc, skip_self=True))


def out_of_range(prof: Dict[str, Any], test: pd.DataFrame) -> pd.Series:
    flags: List[List[str]] = [[] for _ in range(len(test))]
    for c, info in prof["columns"].items():
        if c not in test.columns:
            continue
        s = test[c]
        if info["kind"] == "number":
            x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            bad = np.isfinite(x) & ((x < info["min"]) | (x > info["max"]))
        elif info["complete"]:
            known = set(info["categories"])
            bad = np.array([not pd.isna(v) and str(v) not in known for v in s])
        else:
            continue
        for i in np.flatnonzero(bad):
            flags[i].append(c)
    return pd.Series(flags, index=test.index, name="out_of_range")


def novelty(prof: Dict[str, Any], test: pd.DataFrame) -> pd.Series:
    sample = pd.DataFrame(prof["sample"])
    d = _nn(_encode(prof, test), _encode(prof, sample))
    ref = np.asarray(prof["reference"])
    return pd.Series(np.searchsorted(ref, d, side="right") / len(ref), index=test.index, name="novelty")


class Checks:
    """Lazily computed out-of-range flags and novelty for one prediction (novelty costs a nearest-row search)."""

    def __init__(self, prof: Optional[Dict[str, Any]], test: pd.DataFrame):
        self.prof, self.test = prof, test
        self._oor: Optional[pd.Series] = None
        self._nov: Optional[pd.Series] = None

    def out_of_range(self) -> Optional[pd.Series]:
        if self.prof is not None and self._oor is None:
            self._oor = out_of_range(self.prof, self.test)
        return self._oor

    def novelty(self) -> Optional[pd.Series]:
        if self.prof is not None and self.prof.get("sample") is not None and self._nov is None:
            self._nov = novelty(self.prof, self.test)
        return self._nov


def _loo_out_of_range(prof: Dict[str, Any], df: pd.DataFrame) -> pd.Series:
    """Per row, the columns where it lies outside the range of the other rows: the only row at the minimum or maximum,
    or the only row with its category.
    """
    flags: List[List[str]] = [[] for _ in range(len(df))]
    for c, info in prof["columns"].items():
        s = df[c]
        if info["kind"] == "number":
            x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            fin = np.sort(x[np.isfinite(x)])
            if len(fin) < 2:
                continue
            bad = np.isfinite(x) & ((x < fin[1]) | (x > fin[-2]))
        elif info["complete"]:
            v = s.astype(str).where(s.notna())
            bad = (v.map(v.value_counts()) == 1).to_numpy()
        else:
            continue
        for i in np.flatnonzero(bad):
            flags[i].append(c)
    return pd.Series(flags, index=df.index, name="out_of_range")


def outliers(train: Any, data: Any = None, *, threshold: float = 0.99) -> pd.DataFrame:
    """Scores rows by how unlike the training rows they are, locally: nothing is sent and no key is needed. novelty is the
    distance from a row to its nearest training rows, as a percentile of how far training rows are from each other (0.5 is
    typical, 0.99 farther than 99% of them); the same measure as PredictionResult.novelty.

    >>> import yhatlabs
    >>> yhatlabs.outliers(train, new_rows)    # novelty, outlier, out_of_range per new row
    >>> yhatlabs.outliers(train)              # each training row against the others

    Args:
        train: The reference table (pandas, polars, Spark, Dask or Ray).
        data: The rows to score, against the columns they share with train. Default: the training rows themselves, each
            against the other rows (leave-one-out).
        threshold: Novelty above which a row is flagged as an outlier.

    Returns:
        One row per scored row (same index): novelty, outlier (novelty > threshold) and out_of_range (the columns outside
        the training range, or a category training never had; without data, outside the range of the other rows).
    """
    from ._tabular import _frame, _str_columns

    if not 0 < threshold < 1:
        raise DataError("threshold is a novelty percentile, strictly between 0 and 1 (e.g. 0.99).")
    tr = _str_columns(_frame(train, "train"), "train")
    te = None if data is None else _str_columns(_frame(data, "data"), "data")
    features = [c for c in tr.columns if te is None or c in te.columns]
    if len(tr) < 2 or not features:
        raise DataError("outliers needs at least 2 training rows and columns that train and data share.")
    prof = build(tr.reset_index(drop=True), features)
    if not prof["columns"]:
        raise DataError("No column of train has values to compare rows on.")
    sample = _encode(prof, pd.DataFrame(prof["sample"]))
    ref = np.asarray(prof["reference"])
    if te is None:
        pos = np.full(len(tr), -1)
        pos[_sample_positions(len(tr), 0)] = np.arange(len(sample["num"]))
        d = _nn(_encode(prof, tr), sample, skip=pos)
        oor = _loo_out_of_range(prof, tr)
    else:
        d = _nn(_encode(prof, te), sample)
        oor = out_of_range(prof, te)
    rows = tr if te is None else te
    nov = np.searchsorted(ref, d, side="right") / len(ref)
    return pd.DataFrame({"novelty": nov, "outlier": nov > threshold, "out_of_range": oor.to_numpy()}, index=rows.index)
