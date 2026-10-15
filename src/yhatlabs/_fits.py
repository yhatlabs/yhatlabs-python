"""Fitted tabular models: fit once on the server, predict and explain many times.

A fit lives on our side until it expires (`ttl_hours`, default 24 h, at most 7 days) or you delete it; each prediction sends only
the rows to score. The handle is picklable: it keeps the fit id and what it needs to check inputs and restore label types, and
reattaches to a client with `bind()` (the scikit-learn estimators do this for you).
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import _profile, _tabular
from ._errors import ConfigurationError, DataError, NotFoundError, UnsupportedError
from ._forecast import Limits, warn_all
from ._results import PredictionResult
from ._version import __version__

RECORD_FORMAT = "yhatlabs.fit/1"


def _not_deployed(e: NotFoundError) -> UnsupportedError:
    return UnsupportedError(
        f"This API does not offer fitted models yet ({e}); use Client.predict for one-shot fit-and-predict."
    )


@dataclass
class FittedModel:
    id: str
    model: str
    target: str
    problem_type: Optional[str]
    classes: Optional[List[Any]]
    fit: Dict[str, Any]
    created_at: int
    expires_at: int
    intervals: bool  # regression fits: whether quantiles= is available
    features: Optional[List[str]] = None
    labels: List[Any] = field(default_factory=list)
    _client: Any = field(default=None, repr=False, compare=False)
    profile: Optional[Dict[str, Any]] = field(
        default=None, repr=False
    )  # the training table's ranges and a row sample (_profile)

    @classmethod
    def from_reply(cls, reply: dict, features: Optional[List[str]], labels: List[Any], client: Any) -> FittedModel:
        classes = reply.get("classes")
        if classes is not None and labels:
            classes = _tabular._restore_labels(classes, labels)
        fit = reply.get("fit") or {}
        return cls(
            reply["id"],
            reply.get("model", ""),
            reply.get("target", ""),
            fit.get("problem_type"),
            classes,
            fit,
            int(reply.get("created_at", 0)),
            int(reply.get("expires_at", 0)),
            bool(reply.get("intervals")),
            features,
            labels or list(classes or []),
            client,
        )

    def save(self, path: Any = None) -> Dict[str, Any]:
        """Writes the fit's record to `path` as JSON, and returns it. Load it later, in another process or on another machine,
        with yhatlabs.load(path). The fit itself lives on our side; the record holds the training profile (column ranges,
        categories and a sample of at most 2,000 training rows, for out_of_range and novelty) unless the fit was made with
        profile=False.
        """
        rec = {
            "format": RECORD_FORMAT,
            "sdk": f"yhatlabs-python {__version__}",
            "base_url": getattr(self._client, "base_url", None),
            "id": self.id,
            "model": self.model,
            "target": self.target,
            "problem_type": self.problem_type,
            "classes": self.classes,
            "labels": self.labels,
            "features": self.features,
            "fit": self.fit,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "intervals": self.intervals,
            "profile": self.profile,
        }
        if path is not None:
            Path(path).write_text(json.dumps(rec, indent=2, default=str))
        return rec

    @property
    def expired(self) -> bool:
        """By the local clock; the API is the authority (a deleted fit is gone before it expires)."""
        return bool(self.expires_at) and self.expires_at <= time.time()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def bind(self, client: Any) -> FittedModel:
        self._client = client
        return self

    def _c(self) -> Any:
        if self._client is None:
            raise ConfigurationError("This fitted model is not attached to a client; call .bind(Client()).")
        return self._client

    def __enter__(self) -> FittedModel:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.delete()

    def _test(self, test: Any, quantiles: Any, limits: Limits):
        if self.features is not None:
            te, warns = _tabular.prepare_test(test, self.features, self.target, limits)
        else:
            te = _tabular._str_columns(_tabular._frame(test, "test"), "test")
            te, warns = te.drop(columns=[self.target], errors="ignore"), []
        q = _tabular.check_quantiles(quantiles, self.problem_type)
        if q and not self.intervals:
            raise UnsupportedError(
                "This fit has no prediction intervals (too few rows to estimate out-of-sample errors)."
            )
        return te, q, warns

    def _result(
        self, reply: dict, te: pd.DataFrame, warns: List[str], want_q: bool = False, elapsed: float = 0.0
    ) -> PredictionResult:
        preds, proba, classes, q = _tabular.parse(reply, te, self.labels, want_q)
        res = PredictionResult(
            preds, self.model, proba, classes, self.fit, warns, reply.get("id"), reply.get("usage") or {}, q, self.id
        )
        warn_all(res.warnings)
        res.timing = {"total_s": round(elapsed, 3), "restored": bool(reply.get("restored"))}
        res.checks = _profile.Checks(self.profile, te) if self.profile else None
        return res

    def predict(self, test: Any, *, quantiles: Any = None) -> PredictionResult:
        """Score rows with this fit. quantiles: regression only, any levels in (0, 1), e.g. [0.05, 0.5, 0.95]."""
        c = self._c()
        te, q, warns = self._test(test, quantiles, c.limits)
        t0 = time.monotonic()
        reply = c._t.request(
            "POST", f"/v1/fits/{self.id}/predict", _tabular.test_payload(te, q, c.limits.max_tabular_bytes)
        )
        return self._result(reply, te, warns, bool(q), time.monotonic() - t0)

    def predict_proba(self, test: Any) -> pd.DataFrame:
        res = self.predict(test)
        if res.probabilities is None:
            raise DataError("predict_proba applies to classification fits.")
        return res.probabilities

    def importance(
        self,
        data: Any = None,
        *,
        subsample_size: int = 1000,
        num_shuffle_sets: int = 3,
        time_limit: Optional[float] = None,
    ) -> pd.DataFrame:
        """Permutation importance per input column: how much the score drops when the column is shuffled (higher = more important).
        data: labeled rows the model has not seen (with the target column); best for medium/full fits. Without it, a fast fit uses its
        own validation split and a bagged fit falls back to training rows (in-sample, so importance is optimistic; see .attrs['source']).
        """
        c = self._c()
        return _importance(
            c._t.request(
                "POST",
                f"/v1/fits/{self.id}/importance",
                _importance_body(data, subsample_size, num_shuffle_sets, time_limit),
            )
        )

    def effect(self, col: str, *, grid: Any = 20, data: Any = None, rows: int = 200, seed: int = 0) -> pd.DataFrame:
        """How the prediction moves as one column varies, everything else as in real rows (partial dependence).

        Takes `rows` rows (from data=, else the training sample kept on this handle), sets `col` to each grid value in turn
        and predicts them all in one call: `average` is the mean prediction (class probability for classification, one curve
        per class in `output`), `low` / `high` the 10th and 90th percentile across rows, which shows whether the effect is the
        same everywhere. grid: a number of points (quantiles of the column; categories by frequency) or the values themselves.
        Cost: rows x grid predicted rows (200 x 20 = 4,000 by default). Plot it with yhatlabs.plots.effect().
        """
        stacked, values = _effect_rows(self, col, grid, data, rows, seed)
        return _effect_frame(self.predict(stacked), col, values, len(stacked) // len(values))

    def delete(self) -> None:
        self._c()._t.request("DELETE", f"/v1/fits/{self.id}")

    def info(self) -> Dict[str, Any]:
        return self._c()._t.request("GET", f"/v1/fits/{self.id}")


class AsyncFittedModel(FittedModel):
    async def predict(self, test: Any, *, quantiles: Any = None) -> PredictionResult:  # type: ignore[override]
        c = self._c()
        te, q, warns = self._test(test, quantiles, c.limits)
        t0 = time.monotonic()
        reply = await c._t.request(
            "POST", f"/v1/fits/{self.id}/predict", _tabular.test_payload(te, q, c.limits.max_tabular_bytes)
        )
        return self._result(reply, te, warns, bool(q), time.monotonic() - t0)

    async def predict_proba(self, test: Any) -> pd.DataFrame:  # type: ignore[override]
        res = await self.predict(test)
        if res.probabilities is None:
            raise DataError("predict_proba applies to classification fits.")
        return res.probabilities

    async def importance(
        self,
        data: Any = None,
        *,
        subsample_size: int = 1000,
        num_shuffle_sets: int = 3,
        time_limit: Optional[float] = None,
    ) -> pd.DataFrame:  # type: ignore[override]
        c = self._c()
        return _importance(
            await c._t.request(
                "POST",
                f"/v1/fits/{self.id}/importance",
                _importance_body(data, subsample_size, num_shuffle_sets, time_limit),
            )
        )

    async def effect(
        self, col: str, *, grid: Any = 20, data: Any = None, rows: int = 200, seed: int = 0
    ) -> pd.DataFrame:  # type: ignore[override]
        stacked, values = _effect_rows(self, col, grid, data, rows, seed)
        return _effect_frame(await self.predict(stacked), col, values, len(stacked) // len(values))

    async def delete(self) -> None:  # type: ignore[override]
        await self._c()._t.request("DELETE", f"/v1/fits/{self.id}")

    async def info(self) -> Dict[str, Any]:  # type: ignore[override]
        return await self._c()._t.request("GET", f"/v1/fits/{self.id}")

    async def __aenter__(self) -> AsyncFittedModel:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.delete()


def _effect_rows(fm: FittedModel, col: str, grid: Any, data: Any, rows: int, seed: int) -> Tuple[pd.DataFrame, list]:
    """The stacked copies to predict for an effect curve, and the grid values."""
    if data is not None:
        base = _tabular._str_columns(_tabular._frame(data, "data"), "data").drop(columns=[fm.target], errors="ignore")
    elif fm.profile:
        base = pd.DataFrame(fm.profile["sample"])
        for c in fm.features or []:
            if c not in base.columns:
                base[c] = None
    else:
        raise DataError("effect() needs data= (rows to vary) when the fit has no training profile.")
    if col not in base.columns:
        raise DataError(f"{col!r} is not a feature of this fit.")
    base = base.sample(min(rows, len(base)), random_state=seed)
    if isinstance(grid, (list, tuple, np.ndarray, pd.Series)):
        values = list(grid)
    else:
        info = (fm.profile or {}).get("columns", {}).get(col)
        s = base[col].dropna()
        if (info or {}).get("kind") == "category" or not pd.api.types.is_numeric_dtype(s):
            values = (info["categories"] if info else list(s.value_counts().index))[: int(grid)]
        else:
            values = list(np.unique(np.quantile(s.to_numpy(dtype=float), np.linspace(0.02, 0.98, int(grid)))))
    stacked = pd.concat([base.assign(**{col: v}) for v in values], ignore_index=True)
    return stacked, values


def _effect_frame(res: PredictionResult, col: str, values: list, per: int) -> pd.DataFrame:
    """Average prediction per grid value, with the 10th to 90th percentile across the varied rows."""
    out = []
    if res.probabilities is not None:
        P = res.probabilities.to_numpy(dtype=float)
        for i, v in enumerate(values):
            block = P[i * per : (i + 1) * per]
            for j, cls in enumerate(res.probabilities.columns):
                out.append(
                    {
                        col: v,
                        "output": cls,
                        "average": block[:, j].mean(),
                        "low": np.quantile(block[:, j], 0.1),
                        "high": np.quantile(block[:, j], 0.9),
                    }
                )
        frame = pd.DataFrame(out)
        frame.attrs["ylabel"] = "probability"
    else:
        y = res.predictions.to_numpy(dtype=float)
        for i, v in enumerate(values):
            block = y[i * per : (i + 1) * per]
            out.append(
                {col: v, "average": block.mean(), "low": np.quantile(block, 0.1), "high": np.quantile(block, 0.9)}
            )
        frame = pd.DataFrame(out)
    frame.attrs.update(rows=per, predicted_rows=per * len(values))
    return frame


GROUP_FORMAT = "yhatlabs.fitgroup/1"


@dataclass
class GroupPrediction:
    """Predictions of a FitGroup: `results[target]` is each target's PredictionResult; `df` puts them side by side, with
    two-level columns (target, field), so df["strength"] is that target's table.
    """

    results: Dict[str, PredictionResult]

    def __getitem__(self, target: str) -> PredictionResult:
        return self.results[target]

    @property
    def df(self) -> pd.DataFrame:
        return pd.concat({t: r.df for t, r in self.results.items()}, axis=1)

    @property
    def out_of_range(self) -> Optional[pd.Series]:
        return next(iter(self.results.values())).out_of_range

    @property
    def novelty(self) -> Optional[pd.Series]:
        return next(iter(self.results.values())).novelty

    def __repr__(self) -> str:
        n = len(next(iter(self.results.values())).predictions)
        return f"GroupPrediction(targets={list(self.results)}, rows={n})"


@dataclass
class FitGroup:
    """One fit per target, from Client.fit(train, target=[...]): each target is fitted on the features only (the other targets
    are left out), and predicted, explained and deleted together. `fits[target]` is that target's FittedModel.
    """

    fits: Dict[str, FittedModel]

    @property
    def targets(self) -> List[str]:
        return list(self.fits)

    def __getitem__(self, target: str) -> FittedModel:
        return self.fits[target]

    def __iter__(self) -> Any:
        return iter(self.fits.items())

    def __len__(self) -> int:
        return len(self.fits)

    def predict(self, test: Any, *, quantiles: Any = None) -> GroupPrediction:
        """Every target for the same rows (one call per target); quantiles= applies to the regression targets."""
        return GroupPrediction(
            {
                t: m.predict(test, quantiles=quantiles if m.problem_type == "regression" else None)
                for t, m in self.fits.items()
            }
        )

    def importance(self, data: Any = None, **kw: Any) -> pd.DataFrame:
        """Permutation importance per target, stacked with a `target` column."""
        frames = []
        for t, m in self.fits.items():
            d = (
                None
                if data is None
                else _tabular._frame(data, "data").drop(columns=[o for o in self.fits if o != t], errors="ignore")
            )
            frames.append(m.importance(d, **kw).assign(target=t))
        out = pd.concat(frames, ignore_index=True)
        return out[["target"] + [c for c in out.columns if c != "target"]]

    def effect(self, col: str, **kw: Any) -> pd.DataFrame:
        frames = [m.effect(col, **kw).assign(target=t) for t, m in self.fits.items()]
        out = pd.concat(frames, ignore_index=True)
        return out[["target"] + [c for c in out.columns if c != "target"]]

    def delete(self) -> None:
        for m in self.fits.values():
            m.delete()

    def save(self, path: Any = None) -> Dict[str, Any]:
        rec = {"format": GROUP_FORMAT, "fits": {t: m.save() for t, m in self.fits.items()}}
        if path is not None:
            Path(path).write_text(json.dumps(rec, indent=2, default=str))
        return rec

    def bind(self, client: Any) -> FitGroup:
        for m in self.fits.values():
            m.bind(client)
        return self

    def __enter__(self) -> FitGroup:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.delete()

    def __repr__(self) -> str:
        return f"FitGroup(targets={self.targets})"


class AsyncFitGroup(FitGroup):
    async def predict(self, test: Any, *, quantiles: Any = None) -> GroupPrediction:  # type: ignore[override]
        import asyncio

        outs = await asyncio.gather(
            *[
                m.predict(test, quantiles=quantiles if m.problem_type == "regression" else None)
                for m in self.fits.values()
            ]
        )
        return GroupPrediction(dict(zip(self.fits, outs)))

    async def importance(self, data: Any = None, **kw: Any) -> pd.DataFrame:  # type: ignore[override]
        frames = []
        for t, m in self.fits.items():
            d = (
                None
                if data is None
                else _tabular._frame(data, "data").drop(columns=[o for o in self.fits if o != t], errors="ignore")
            )
            frames.append((await m.importance(d, **kw)).assign(target=t))
        out = pd.concat(frames, ignore_index=True)
        return out[["target"] + [c for c in out.columns if c != "target"]]

    async def effect(self, col: str, **kw: Any) -> pd.DataFrame:  # type: ignore[override]
        frames = [(await m.effect(col, **kw)).assign(target=t) for t, m in self.fits.items()]
        out = pd.concat(frames, ignore_index=True)
        return out[["target"] + [c for c in out.columns if c != "target"]]

    async def delete(self) -> None:  # type: ignore[override]
        for m in self.fits.values():
            await m.delete()

    async def __aenter__(self) -> AsyncFitGroup:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.delete()


IMPORTANCE_MAX_BYTES = 16 * 1024 * 1024


def _importance_body(data: Any, subsample_size: int, num_shuffle_sets: int, time_limit: Optional[float]) -> dict:
    body: dict = {"subsample_size": int(subsample_size), "num_shuffle_sets": int(num_shuffle_sets)}
    if time_limit is not None:
        body["time_limit"] = float(time_limit)
    if data is not None:
        enc, size = _tabular.encode_table(_tabular._str_columns(_tabular._frame(data, "data"), "data"))
        if size > IMPORTANCE_MAX_BYTES:
            raise DataError("data for importance may be at most 16 MB; pass a sample (a few thousand rows is plenty).")
        body["data"] = enc
    return body


def _importance(reply: dict) -> pd.DataFrame:
    out = pd.DataFrame(reply["features"], columns=["feature", "importance", "stddev"])
    out.attrs.update(source=reply.get("source"), metric=reply.get("metric"), fit_id=reply.get("fit_id"))
    return out


def load(source: Union[str, Path, Dict[str, Any]], client: Any = None) -> FittedModel:
    """A fitted model from a record written by FittedModel.save() (or a FitGroup from FitGroup.save()), attached to `client`
    (default: a new Client()). Makes no request; if the fit has expired or been deleted, its first predict raises NotFoundError.
    """
    rec = source if isinstance(source, dict) else json.loads(Path(source).read_text())
    if rec.get("format") == GROUP_FORMAT:
        if client is None:
            from ._client import Client

            client = Client(base_url=next(iter(rec["fits"].values())).get("base_url"))
        return FitGroup({t: load(r, client) for t, r in rec["fits"].items()})  # type: ignore[return-value]
    if rec.get("format") != RECORD_FORMAT:
        raise DataError(f"Not a yhatlabs fit record (format {rec.get('format')!r}, expected {RECORD_FORMAT!r}).")
    if client is None:
        from ._client import Client

        client = Client(base_url=rec.get("base_url"))
    if rec.get("base_url") and getattr(client, "base_url", None) not in (None, rec["base_url"]):
        warnings.warn(f"This fit was made on {rec['base_url']}; the client points at {client.base_url}.", stacklevel=2)
    fm = FittedModel(
        rec["id"],
        rec["model"],
        rec["target"],
        rec.get("problem_type"),
        rec.get("classes"),
        rec.get("fit") or {},
        int(rec.get("created_at") or 0),
        int(rec.get("expires_at") or 0),
        bool(rec.get("intervals")),
        rec.get("features"),
        rec.get("labels") or [],
        client,
        rec.get("profile"),
    )
    if fm.expired:
        warnings.warn(
            f"Fit {fm.id} expired at {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(fm.expires_at))}; fit again.",
            stacklevel=2,
        )
    return fm
