"""Cross-validated evaluation of tabular predictions on the caller's own data: every row is predicted once by a fit that never
saw it, then scored for accuracy and for how honest its uncertainty is (calibration of probabilities, coverage of intervals).

Splits: shuffled k-fold; grouped (rows sharing a group, e.g. one machine or one batch, stay in the same fold, so the score is
for new groups); or time-ordered (expanding window: fold k trains on everything before its block, never on the future).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._errors import DataError

DEFAULT_QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def infer_task(y: pd.Series, task: Optional[str]) -> str:
    """The task the folds are run as, fixed up front so every fold agrees: numbers with fractions, or more than 20 distinct
    integers, are regression; anything else is classification. Pass task= to override.
    """
    if task is not None:
        return task
    obs = y.dropna()
    if pd.api.types.is_numeric_dtype(obs) and not pd.api.types.is_bool_dtype(obs):
        frac = not np.allclose(obs.to_numpy(dtype=float) % 1, 0)
        if frac or obs.nunique() > 20:
            return "regress"
    return "classify"


def folds(
    df: pd.DataFrame, cv: int, groups: Any, time_col: Optional[str], seed: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Positional (train, test) index pairs."""
    n = len(df)
    if not isinstance(cv, (int, np.integer)) or cv < 2:
        raise DataError("cv must be an integer of at least 2.")
    if groups is not None and time_col is not None:
        raise DataError("Pass groups= or time_col=, not both.")
    rng = np.random.default_rng(seed)
    if time_col is not None:
        if time_col not in df.columns:
            raise DataError(f"time_col {time_col!r} is not a column.")
        order = np.argsort(
            pd.to_datetime(df[time_col]).to_numpy()
            if not pd.api.types.is_numeric_dtype(df[time_col])
            else df[time_col].to_numpy(),
            kind="stable",
        )
        blocks = np.array_split(order, cv + 1)
        return [(np.concatenate(blocks[: k + 1]), blocks[k + 1]) for k in range(cv)]
    if groups is not None:
        g = df[groups].to_numpy() if isinstance(groups, str) else np.asarray(groups)
        if len(g) != n:
            raise DataError(f"groups has {len(g)} values for {n} rows.")
        uniq = pd.unique(g)
        if len(uniq) < cv:
            raise DataError(f"groups has {len(uniq)} distinct values; grouped cv={cv} needs at least {cv}.")
        sizes = pd.Series(g).value_counts()
        load = np.zeros(cv)
        where: Dict[Any, int] = {}
        for gid in sizes.sample(frac=1, random_state=seed).sort_values(ascending=False, kind="stable").index:
            k = int(np.argmin(load))  # biggest groups first, each to the emptiest fold: folds of similar size
            where[gid] = k
            load[k] += sizes[gid]
        fold_of = np.array([where[v] for v in g])
    else:
        fold_of = rng.permutation(np.arange(n) % cv)
    return [(np.flatnonzero(fold_of != k), np.flatnonzero(fold_of == k)) for k in range(cv)]


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the ROC curve via ranks (Mann-Whitney), ties averaged."""
    pos = y.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def classification_scores(y: pd.Series, proba: pd.DataFrame) -> Dict[str, float]:
    classes = list(proba.columns)
    P = proba.to_numpy(dtype=float)
    keys = [str(c) for c in classes]
    idx = pd.Series(range(len(classes)), index=keys)
    yi = idx.reindex([str(v) for v in y]).to_numpy()
    ok = ~np.isnan(yi)
    yi = yi[ok].astype(int)
    P = P[ok]
    onehot = np.eye(len(classes))[yi]
    pt = np.clip(P[np.arange(len(yi)), yi], 1e-15, 1)
    out = {
        "accuracy": float((P.argmax(axis=1) == yi).mean()),
        "log_loss": float(-np.log(pt).mean()),
        "brier": float(((P - onehot) ** 2).sum(axis=1).mean()),
    }
    if len(classes) == 2:
        out["auc"] = _auc(yi == 1, P[:, 1])
    else:
        out["auc"] = float(np.nanmean([_auc(yi == k, P[:, k]) for k in range(len(classes))]))  # one-vs-rest, macro
    conf = P.max(axis=1)
    hit = P.argmax(axis=1) == yi
    bins = np.minimum((conf * 10).astype(int), 9)
    out["ece"] = float(
        sum(abs(conf[bins == b].mean() - hit[bins == b].mean()) * (bins == b).mean() for b in np.unique(bins))
    )
    return out


def reliability(y: pd.Series, proba: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Binary: the positive class's predicted probability against how often it happened. Multiclass: the top class's
    probability (confidence) against how often the top class was right. Bins with no rows are left out.
    """
    classes = list(proba.columns)
    P = proba.to_numpy(dtype=float)
    ys = np.array([str(v) for v in y])
    if len(classes) == 2:
        p, hit, label = P[:, 1], ys == str(classes[1]), str(classes[1])
    else:
        p, hit, label = P.max(axis=1), ys == np.array([str(classes[i]) for i in P.argmax(axis=1)]), "top class"
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = [
        {
            "class": label,
            "bin": f"{k / n_bins:.1f}-{(k + 1) / n_bins:.1f}",
            "predicted": p[b == k].mean(),
            "observed": hit[b == k].mean(),
            "count": int((b == k).sum()),
        }
        for k in range(n_bins)
        if (b == k).any()
    ]
    return pd.DataFrame(rows)


def regression_scores(
    y: np.ndarray, pred: np.ndarray, q: Optional[pd.DataFrame], levels: Sequence[float]
) -> Dict[str, float]:
    err = y - pred
    out = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1 - np.sum(err**2) / np.sum((y - y.mean()) ** 2)) if len(y) > 1 and np.var(y) > 0 else float("nan"),
    }
    if q is not None:
        Q = q.to_numpy(dtype=float)
        lv = np.asarray(levels)
        pin = np.maximum(lv * (y[:, None] - Q), (lv - 1) * (y[:, None] - Q))
        out["crps"] = float(
            2 * pin.mean()
        )  # the mean pinball loss over the levels, x2: a quantile approximation of CRPS
        for lo_i, lo in enumerate(lv):
            hi_i = int(np.argmin(np.abs(lv - (1 - lo))))
            if lo < 0.5 and abs(lv[hi_i] - (1 - lo)) < 1e-9:
                w = round((1 - 2 * lo) * 100, 6)
                inside = (y >= Q[:, lo_i]) & (y <= Q[:, hi_i])
                out[f"coverage_{w:g}"] = float(inside.mean())
                out[f"width_{w:g}"] = float(np.mean(Q[:, hi_i] - Q[:, lo_i]))
    return out


def coverage_table(scores: Dict[str, float]) -> pd.DataFrame:
    rows = [
        {
            "interval": float(k.split("_")[1]),
            "nominal": float(k.split("_")[1]) / 100,
            "observed": v,
            "width": scores.get(f"width_{k.split('_')[1]}"),
        }
        for k, v in scores.items()
        if k.startswith("coverage_")
    ]
    t = pd.DataFrame(rows)
    if len(t):
        t["gap"] = t["observed"] - t["nominal"]
    return t


def pit(y: np.ndarray, q: pd.DataFrame, levels: Sequence[float], n_bins: int = 10) -> pd.DataFrame:
    """Probability integral transform: where each actual falls in its own predicted distribution. Calibrated predictions give
    a flat histogram; a U shape means intervals too narrow, a hump too wide, a slope a bias.
    """
    from ._forecast import cdf_at

    u = cdf_at(q.to_numpy(dtype=float), levels, y)
    counts, edges = np.histogram(u, bins=n_bins, range=(0, 1))
    return pd.DataFrame(
        {"bin": [f"{a:.1f}-{b:.1f}" for a, b in zip(edges[:-1], edges[1:])], "share": counts / max(len(u), 1)}
    )


@dataclass
class TableEvaluation:
    """Cross-validated scores of a tabular model on your data. `predictions`: every row's out-of-fold prediction (and fold);
    `scores`: one row per fold; `summary`: the scores over all rows pooled. Classification adds `reliability`; regression with
    intervals adds `coverage` and `pit`.
    """

    task: str
    split: str
    predictions: pd.DataFrame
    scores: pd.DataFrame
    summary: Dict[str, float]
    reliability: Optional[pd.DataFrame] = None
    coverage: Optional[pd.DataFrame] = None
    pit: Optional[pd.DataFrame] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    request_ids: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        main = {
            k: round(v, 4)
            for k, v in self.summary.items()
            if k in ("accuracy", "log_loss", "auc", "mae", "rmse", "r2", "crps")
        }
        return f"TableEvaluation({self.task}, {self.split}, {len(self.scores)} folds, {main})"

    def plot(self, ax: Any = None) -> Any:
        """Classification: reliability diagram. Regression: predicted against actual, with the widest interval. Needs matplotlib."""
        from . import _plot

        if self.reliability is not None:
            return _plot.plot_reliability(self.reliability, ax)
        return _plot.plot_predicted_vs_actual(self.predictions, self.predictions["actual"], ax)


def assemble(
    task: str,
    split: str,
    data: pd.DataFrame,
    target: str,
    parts: List[Tuple[int, np.ndarray, Any]],
    levels: Optional[List[float]],
) -> TableEvaluation:
    """parts: (fold, test positions, PredictionResult) -> the evaluation."""
    frames, rows, rids, used = [], [], [], 0
    for k, pos, res in parts:
        y = data[target].iloc[pos]
        f = res.df.reset_index(drop=True)
        f.insert(0, "actual", y.values)
        f.insert(0, "fold", k)
        f["_pos"] = pos
        frames.append(f)
        sc = (
            classification_scores(y, res.probabilities)
            if task == "classify"
            else regression_scores(y.to_numpy(float), res.predictions.to_numpy(float), res.quantiles, levels or [])
        )
        rows.append({"fold": k, "rows": len(pos), **sc})
        rids.append(res.request_id or "")
        used += int((res.usage or {}).get("rows", len(pos)))
    pred = pd.concat(frames, ignore_index=True).sort_values("_pos", kind="stable")
    pred.index = data.index[pred.pop("_pos").to_numpy()]
    out = TableEvaluation(
        task, split, pred, pd.DataFrame(rows), {}, usage={"rows": used, "requests": len(parts)}, request_ids=rids
    )
    if task == "classify":
        proba = pred[[c for c in pred.columns if c.startswith("p_")]]
        classes = parts[0][2].classes or []
        proba.columns = classes
        out.summary = classification_scores(pred["actual"], proba)
        out.reliability = reliability(pred["actual"], proba)
    else:
        qcols = parts[0][2].quantiles.columns if parts[0][2].quantiles is not None else None
        q = pred[list(qcols)] if qcols is not None else None
        out.summary = regression_scores(
            pred["actual"].to_numpy(float), pred["prediction"].to_numpy(float), q, levels or []
        )
        if q is not None:
            out.coverage = coverage_table(out.summary)
            out.pit = pit(pred["actual"].to_numpy(float), q, levels or [])
    out.summary = {k: (round(v, 6) if isinstance(v, float) and math.isfinite(v) else v) for k, v in out.summary.items()}
    return out
