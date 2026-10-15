"""Hierarchies: series that add up (stores into regions into a total), forecast so the forecasts add up too.

    from yhatlabs import hierarchy
    data, tree = hierarchy.aggregate(sales, levels=[["region"], ["region", "store"]], time_col="date", target_col="units")
    fc = yh.forecast(data, h=28, time_col="date", target_col="units")
    fc = hierarchy.reconcile(fc, tree)

`aggregate` builds every aggregate series from the bottom ones (and the tree saying who sums into whom); `reconcile` adjusts the
forecasts of all levels so each parent equals the sum of its children, using the structure alone (WLS with structural weights:
a node's forecast counts in proportion to how few bottom series it sums, so a store's own forecast is trusted for the store and the
total's for the total). It needs no backtest residuals.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._errors import DataError
from ._forecast import _pandas

SEP = "/"


def aggregate(
    data: Any,
    levels: Sequence[Sequence[str]],
    *,
    time_col: str = "ds",
    target_col: str = "y",
    total: Optional[str] = "total",
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Every level's series from the bottom rows. levels: from the top down, each a list of the columns that define it; the last
    level is the bottom (one series per distinct combination). -> (long table with a unique_id column, {parent id: [child ids]}).
    """
    df = _pandas(data).copy()
    levels = [list(lv) for lv in levels]
    for lv in levels:
        missing = [c for c in lv if c not in df.columns]
        if missing:
            raise DataError(f"levels name columns {missing} that the table does not have.")
    for upper, lower in zip(levels, levels[1:]):
        if upper != lower[: len(upper)]:
            raise DataError(f"Each level must extend the one above: {lower} does not start with {upper}.")
    ident = lambda row, cols: SEP.join(f"{c}={row[c]}" for c in cols)  # noqa: E731
    frames, tree = [], {}
    chain = ([[]] if total else []) + levels
    for depth, cols in enumerate(chain):
        g = (
            df.groupby(cols + [time_col], dropna=False)[target_col].sum().reset_index()
            if cols
            else df.groupby(time_col)[target_col].sum().reset_index()
        )
        g["unique_id"] = g.apply(lambda r, cols=cols: ident(r, cols), axis=1) if cols else total
        frames.append(g[["unique_id", time_col, target_col]])
        if depth + 1 < len(chain):
            nxt = chain[depth + 1]
            pairs = df[nxt].drop_duplicates()
            for _, r in pairs.iterrows():
                parent = ident(r, cols) if cols else total
                tree.setdefault(parent, []).append(ident(r, nxt))
    tree = {k: sorted(set(v)) for k, v in tree.items()}
    return pd.concat(frames, ignore_index=True), tree


def _bottom(tree: Dict[str, List[str]], node: str) -> List[str]:
    kids = tree.get(node)
    if not kids:
        return [node]
    return [b for k in kids for b in _bottom(tree, k)]


def summing_matrix(tree: Dict[str, List[str]], ids: Sequence[Any]) -> Tuple[np.ndarray, List[Any]]:
    """S [nodes, bottom]: each node's row sums its bottom descendants. -> (S, the bottom ids in column order)."""
    ids = list(ids)
    bottom = [i for i in ids if not tree.get(i)]
    col = {b: j for j, b in enumerate(bottom)}
    S = np.zeros((len(ids), len(bottom)))
    for r, node in enumerate(ids):
        for b in _bottom(tree, node):
            if b not in col:
                raise DataError(f"{node!r} sums {b!r}, which has no forecast; forecast every series of the hierarchy.")
            S[r, col[b]] = 1.0
    return S, bottom


def reconcile(fc: Any, tree: Dict[str, List[str]]) -> Any:
    """The forecast with every value column (median, mean, each quantile) adjusted so parents equal the sum of their children.
    Quantiles are reconciled column by column: exact for the median and mean, an approximation for the interval edges (sums of
    quantiles are not quantiles of sums); use sample paths when the uncertainty of an aggregate matters.
    """
    df = fc.df
    id_col, time_col = df.columns[0], df.columns[1]
    keys = fc._keys()
    if len(keys) > 2:
        raise DataError("reconcile() takes one target per series.")
    values = [c for c in df.columns if c not in keys]
    ids = list(pd.unique(df[id_col]))
    unknown = [p for p in tree if p not in ids]
    if unknown:
        raise DataError(f"The tree names series without a forecast: {unknown[:5]}.")
    S, _ = summing_matrix(tree, ids)
    w = 1.0 / S.sum(axis=1)  # structural weights: a node summing many bottom series counts for less
    W = np.diag(w)
    P = np.linalg.solve(S.T @ W @ S, S.T @ W)  # bottom = P y_hat; reconciled = S P y_hat
    G = S @ P
    out = df.copy()
    for col in values:
        wide = df.pivot(index=time_col, columns=id_col, values=col)[ids]
        rec = wide.to_numpy() @ G.T
        out[col] = (
            pd.DataFrame(rec, index=wide.index, columns=ids)
            .stack()
            .reindex(pd.MultiIndex.from_arrays([df[time_col], df[id_col]]))
            .to_numpy()
        )
    res = copy.copy(fc)
    res.df = out
    res.derived = sorted(set(fc.derived) | set(values))
    res.provenance = {
        **fc.provenance,
        "reconciled": {"method": "wls_structural", "nodes": len(ids), "bottom": int(S.shape[1])},
    }
    return res
