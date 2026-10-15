"""Client and AsyncClient: the public entry points."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import httpx
import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

from . import _anomaly, _evaluate, _profile, _table_eval, _tabular
from ._errors import APIError, DataError, NotFoundError, UnsupportedError, YHatError
from ._fits import AsyncFitGroup, AsyncFittedModel, FitGroup, FittedModel, _not_deployed
from ._forecast import Item, Limits, Prepared, _pandas, batches, build_frame, prepare, resolve_quantiles, warn_all
from ._results import EvaluationResult, ForecastResult, PredictionResult
from ._table_eval import TableEvaluation
from ._transport import AsyncTransport, SyncTransport
from ._version import __version__

DEFAULT_TS_MODEL = "chakra-ts"
DEFAULT_TAB_MODEL = "chakra-tab"
ASSUMED_NATIVE = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
)  # checked against the reply; rejects bad levels before sending

Target = Optional[Union[str, Sequence[str]]]
Cols = Optional[Sequence[str]]


def _body(
    batch: List[Item], h: int, freq: str, model: str, season: Optional[int], extra: Optional[dict] = None
) -> dict:
    body: dict = {
        "model": model,
        "horizon": h,
        "freq": freq,
        "series": [it.payload(f"s{j}") for j, it in enumerate(batch)],
    }
    if season is not None:
        body["season"] = int(season)
    body.update(extra or {})
    return body


def _forecast_extra(limits: Limits, quantiles: Any, samples: Optional[int], seed: Optional[int], history: bool) -> dict:
    """Request fields for levels beyond the model's deciles, sample paths and past windows, each used once the API lists it."""
    extra: dict = {}
    try:
        resolve_quantiles(quantiles, ASSUMED_NATIVE)
    except UnsupportedError:
        if not limits.supports("forecast_quantile_levels"):
            raise
        extra["quantile_levels"] = resolve_quantiles(quantiles, [1e-6, 1 - 1e-6])[0]
    if samples is not None:
        if not isinstance(samples, (int, np.integer)) or not 1 <= samples <= 1000:
            raise DataError("samples must be an integer from 1 to 1000.")
        if not limits.supports("forecast_samples"):
            raise UnsupportedError("This API does not return sample paths yet.")
        extra["samples"] = int(samples)
        if seed is not None:
            extra["seed"] = int(seed)
    if history:
        if not limits.supports("forecast_history"):
            raise UnsupportedError("This API does not return past-window forecasts yet.")
        extra["history"] = True
    return extra


def _unpack(batch: List[Item], reply: dict) -> List[dict]:
    by_id = {f["id"]: f for f in reply["forecasts"]}
    return [by_id[f"s{j}"] for j in range(len(batch))]


def _native(replies: List[dict]) -> List[float]:
    levels = {tuple(r["quantile_levels"]) for r in replies}
    if len(levels) > 1:
        raise DataError("The API returned different quantile levels across batches; retry the request.")
    return list(next(iter(levels)))


def _levels(quantiles: Any, level: Optional[Sequence[float]]) -> Any:
    """level=[80, 90] (central interval widths in percent) -> the matching quantiles plus the median."""
    if level is None:
        return quantiles
    if quantiles is not None:
        raise DataError("Pass quantiles= or level=, not both.")
    out = {0.5}
    for lv in level:
        if not 0 < float(lv) < 100:
            raise DataError(f"level values are interval widths in percent, between 0 and 100; got {lv}.")
        out |= {round(0.5 - float(lv) / 200, 6), round(0.5 + float(lv) / 200, 6)}
    return sorted(out)


def _targets(target: Sequence[str], train: Any) -> List[str]:
    targets = list(target)
    if not targets or len(set(targets)) != len(targets) or not all(isinstance(t, str) for t in targets):
        raise DataError("target must be a column name or a list of distinct column names.")
    cols = set(map(str, _tabular._frame(_pandas(train), "train").columns))
    missing = [t for t in targets if t not in cols]
    if missing:
        raise DataError(f"targets {missing} are not columns of train.")
    return targets


def _extras(prep: Prepared, outs: List[Optional[dict]], res: ForecastResult):
    """Sample paths and past-window forecasts, when the request asked for them, as long tables."""
    from ._forecast import build_history, build_samples

    kept = [(it, o) for it, o in zip(prep.items, outs) if o is not None]
    return build_samples(prep, kept), build_history(prep, kept, res.native_quantile_levels)


def _fold_refs(base: Optional[dict], tr: Any, te: Any, target: str, drop: List[str]) -> Optional[Dict[str, dict]]:
    """A fold of an uploaded table: its training and test rows, by position, with the test target (and group column) left out."""
    if base is None:
        return None
    train = {**base, "rows": [int(i) for i in tr], **({"drop": list(drop)} if drop else {})}
    return {"train": train, "test": {**base, "rows": [int(i) for i in te], "drop": [target, *drop]}}


def _forecast_result(
    prep: Prepared,
    model: str,
    quantiles: Any,
    items: List[Item],
    outs: List[Optional[dict]],
    replies: List[dict],
    errors: List[Dict[str, Any]],
    mean: bool = False,
) -> ForecastResult:
    if not replies:
        raise errors[0]["exception"]
    native = _native(replies)
    levels, derived = resolve_quantiles(quantiles, native)
    if mean:
        derived = derived + ["mean"]
    kept = Prepared(
        [it for it, o in zip(items, outs) if o is not None],
        prep.freq,
        prep.id_col,
        prep.time_col,
        prep.target_cols,
        prep.warnings,
    )
    frame = build_frame(kept, {it.uid: o for it, o in zip(items, outs) if o is not None}, native, levels, mean)
    usage = {"series": sum(int((r.get("usage") or {}).get("series", 0)) for r in replies), "requests": len(replies)}
    warns = list(prep.warnings) + [f"{len(e['ids'])} series failed ({e['error']}); see result.errors." for e in errors]
    return ForecastResult(
        frame,
        model,
        prep.freq,
        levels,
        native,
        derived,
        warns,
        [r.get("id", "") for r in replies],
        usage,
        [{k: v for k, v in e.items() if k != "exception"} for e in errors],
    )


def _predict_result(
    model: str, reply: dict, test: pd.DataFrame, labels: List[Any], warns: List[str], want_q: bool = False
) -> PredictionResult:
    preds, proba, classes, q = _tabular.parse(reply, test, labels, want_q)
    fit = reply.get("fit") or {}
    res = PredictionResult(preds, model, proba, classes, fit, warns, reply.get("id"), reply.get("usage") or {}, q)
    res.prediction_sets, res.set_coverage = _tabular.parse_sets(reply, test, labels), reply.get("set_coverage")
    res.interval_method, res.mean = reply.get("interval_method"), _tabular.parse_mean(reply, test)
    res.timing = {k: fit[k] for k in ("fit_s", "predict_s") if fit.get(k) is not None}
    return res


def _prep(
    data: Any, h: int, limits: Limits, *, freq, id_col, time_col, target_col, known, past, static, future
) -> Prepared:
    return prepare(
        data,
        h,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        freq=freq,
        known=known,
        past=past,
        future=future,
        limits=limits,
        static=static,
    )


def _impute_task(s: pd.Series) -> Optional[str]:
    if pd.api.types.is_float_dtype(s) or (
        pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and s.nunique() > 20
    ):
        return "regress"
    return "classify"


Calendar = Union[bool, Sequence[str], Dict[str, Any], None]


def _with_calendar(
    data: Any,
    future: Any,
    h: Optional[int],
    calendar: Calendar,
    known: Cols,
    id_col: Optional[str],
    time_col: Optional[str],
    freq: Optional[str],
    target_col: Target = None,
):
    """calendar=True (or a list of features, or {"features": [...], "holidays": "US", "subdiv": "TX"}): adds calendar columns as
    known covariates, to the history and, when forecasting, to the future (built here when not given; dates are always known).
    """
    if not calendar:
        return data, future, known
    from ._forecast import _pandas
    from .features import CALENDAR, add_calendar, make_future

    opts = calendar if isinstance(calendar, dict) else {}
    feats = list(opts.get("features") or (calendar if isinstance(calendar, (list, tuple)) else CALENDAR))
    hol = {"holidays": opts.get("holidays"), "subdiv": opts.get("subdiv")}
    data = _pandas(data)
    if h is not None:  # the future frame first, from the data as given (before calendar columns are added to it)
        if future is None:
            tc = target_col if isinstance(target_col, str) else (list(target_col)[0] if target_col else None)
            future = make_future(data, h, id_col=id_col, time_col=time_col, target_col=tc, freq=freq)
        future = add_calendar(_pandas(future), feats, time_col=time_col, **hol)
    data = add_calendar(data, feats, time_col=time_col, **hol)
    names = feats + (["is_holiday"] if hol["holidays"] else [])
    return data, future, list(known or []) + [n for n in names if n not in (known or [])]


def _fingerprint(data: Any) -> Optional[str]:
    try:
        from ._forecast import _pandas

        d = _pandas(data)
        if isinstance(d, (pd.DataFrame, pd.Series)):
            return hashlib.sha256(hash_pandas_object(d, index=False).to_numpy().tobytes()).hexdigest()[:16]
    except Exception:
        return None
    return None


def _provenance(
    base_url: str, endpoint: str, model: str, request_ids: List[str], data: Any, settings: Dict[str, Any]
) -> Dict[str, Any]:
    """Enough to say later exactly what produced a result: re-run it, cite it, or quote the request ids to us."""
    return {
        "sdk": f"yhatlabs-python {__version__}",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_url": base_url,
        "endpoint": endpoint,
        "model": model,
        "request_ids": [r for r in request_ids if r],
        "data_fingerprint": _fingerprint(data),
        "settings": {k: v for k, v in settings.items() if v is not None},
    }


class _Base:
    def __init__(self, transport: Any, limits: Optional[Limits], max_concurrency: int):
        self._t = transport
        self._limits = limits
        self.max_concurrency = max(1, max_concurrency)
        self._tables: Dict[str, str] = {}  # sha256 of an uploaded parquet file -> its table id, so a table is sent once

    @property
    def limits(self) -> Limits:
        if self._limits is None:
            self.limits = self._fetch_limits()
        return self._limits  # type: ignore[return-value]

    @limits.setter
    def limits(self, value: Limits) -> None:
        self._limits = value
        self._t.gzip = value.supports(
            "gzip_requests"
        )  # bodies over GZIP_MIN_BYTES go compressed once the API takes them

    def _fetch_limits(self) -> Limits:
        return Limits()

    @property
    def base_url(self) -> str:
        return self._t.base_url

    def with_options(
        self,
        *,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        max_concurrency: Optional[int] = None,
    ) -> Any:
        """A copy of this client with other settings, sharing its connection: yh.with_options(timeout=1800).fit(...)."""
        new = copy.copy(self)
        new._t = copy.copy(self._t)
        new._t._own = False  # closing the copy leaves the shared connection open
        if timeout is not None:
            new._t.timeout = float(timeout)
        if max_retries is not None:
            new._t.max_retries = int(max_retries)
        if max_concurrency is not None:
            new.max_concurrency = max(1, int(max_concurrency))
        return new

    def _key_hint(self) -> str:
        k = self._t.api_key
        return f"{k[:3]}...{k[-4:]}" if len(k) > 10 else "***"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(base_url={self.base_url!r})"


class Client(_Base):
    """Synchronous client for the YHat Labs prediction API.

    >>> from yhatlabs import Client
    >>> yh = Client()                      # reads YHAT_API_KEY (and YHAT_BASE_URL, if set)
    >>> fc = yh.forecast(df, h=24)         # df: unique_id, ds, y (+ covariates)
    >>> m = yh.fit(train, "churn")         # tabular: fit once, then m.predict(rows) as often as you like
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        timeout: float = 300.0,
        max_retries: int = 3,
        max_concurrency: int = 4,
        limits: Optional[Limits] = None,
        http_client: Optional[httpx.Client] = None,
    ):
        super().__init__(SyncTransport(api_key, base_url, timeout, max_retries, http_client), limits, max_concurrency)

    def __enter__(self) -> Client:
        return self

    def _fetch_limits(self) -> Limits:
        """The API's own caps, fetched once on first need (an API without /v1/limits falls back to the published defaults)."""
        try:
            return Limits.from_api(self._t.request("GET", "/v1/limits"))
        except YHatError:
            return Limits()

    def whoami(self) -> Dict[str, Any]:
        """Checks the key and shows what it can do: the models it may use and its quota."""
        return {
            "base_url": self.base_url,
            "key": self._key_hint(),
            "models": [m["id"] for m in self.models()],
            "usage": self.usage(),
        }

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self._t.close()

    # ---- time series ----

    def _send(
        self,
        items: List[Item],
        h: int,
        freq: str,
        model: str,
        season: Optional[int],
        on_error: str = "raise",
        extra: Optional[dict] = None,
    ):
        """-> (one reply per item, None where its batch failed), the raw replies, and the failures."""
        bs = batches(items, self.limits)

        def one(b: List[Item]):
            try:
                reply = self._t.request("POST", "/v1/forecast", _body(b, h, freq, model, season, extra))
                return _unpack(b, reply), reply, None
            except APIError as e:
                if on_error == "raise":
                    raise
                return [None] * len(b), None, {"ids": [it.uid for it in b], "error": str(e), "exception": e}

        if len(bs) == 1 or self.max_concurrency == 1:
            done = [one(b) for b in bs]
        else:
            with ThreadPoolExecutor(min(self.max_concurrency, len(bs))) as ex:
                done = list(ex.map(one, bs))
        return (
            [f for out, _, _ in done for f in out],
            [r for _, r, _ in done if r is not None],
            [e for _, _, e in done if e],
        )

    def _forecaster(self, h_fixed: Optional[int], freq: str, model: str, season: Optional[int]):
        def run(items: List[Item], h: Optional[int] = None) -> Tuple[List[dict], List[float]]:
            out, replies, _ = self._send(items, h or h_fixed, freq, model, season)
            return out, _native(replies)

        return run

    def forecast(
        self,
        data: Any,
        h: int,
        *,
        model: str = DEFAULT_TS_MODEL,
        quantiles: Any = None,
        level: Optional[Sequence[float]] = None,
        mean: bool = False,
        freq: Optional[str] = None,
        id_col: Optional[str] = None,
        time_col: Optional[str] = None,
        target_col: Target = None,
        known: Cols = None,
        past: Cols = None,
        static: Cols = None,
        future: Optional[pd.DataFrame] = None,
        season: Optional[int] = None,
        on_error: str = "raise",
        calendar: Calendar = None,
        samples: Optional[int] = None,
        seed: Optional[int] = None,
        history: bool = False,
    ) -> ForecastResult:
        """Forecast the next `h` steps of every series in `data`.

        data: long DataFrame (id, timestamp, target[s], covariates), a Series with a DatetimeIndex, an array, or a dict of arrays
        (arrays need freq=). quantiles: levels in [0.1, 0.9] (default 0.1, 0.5, 0.9) or "all"; or level=[80] for central intervals.
        mean=True adds a `mean` column next to the median (the point forecast), computed from the native quantiles and listed in
        result.derived; use it when the quantity is summed later (totals over time or series), where the median is biased.
        known: columns whose future values you know (pass them in `future`, see yhatlabs.features.make_future); past: history-only
        columns; static: per-series constants (region, store size). on_error="warn" returns the series that succeeded when some
        batches fail (failures in result.errors). calendar=True adds hour / weekday / month / weekend as known covariates, with
        no future= needed for them ({"holidays": "US"} adds a holiday flag).
        """
        t0 = time.monotonic()
        data, future, known = _with_calendar(data, future, h, calendar, known, id_col, time_col, freq, target_col)
        quantiles = _levels(quantiles, level)
        extra = _forecast_extra(self.limits, quantiles, samples, seed, history)
        prep = _prep(
            data,
            h,
            self.limits,
            freq=freq,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            known=known,
            past=past,
            static=static,
            future=future,
        )
        out, replies, errors = self._send(prep.items, h, prep.freq, model, season, on_error, extra)
        res = _forecast_result(prep, model, quantiles, prep.items, out, replies, errors, mean)
        res.samples, res.history = _extras(prep, out, res)
        res.timing = {"total_s": round(time.monotonic() - t0, 3), "requests": len(replies)}
        res.provenance = _provenance(
            self.base_url,
            "/v1/forecast",
            model,
            res.request_ids,
            data,
            {
                "h": h,
                "freq": prep.freq,
                "quantiles": res.quantile_levels,
                "known": known,
                "past": past,
                "static": static,
                "season": season,
                "series": len(prep.items),
            },
        )
        warn_all(res.warnings)
        return res

    def scenarios(
        self,
        data: Any,
        h: int,
        futures: Dict[str, pd.DataFrame],
        *,
        known: Sequence[str],
        model: str = DEFAULT_TS_MODEL,
        quantiles: Any = None,
        level: Optional[Sequence[float]] = None,
        **kw: Any,
    ) -> pd.DataFrame:
        """What-if: one forecast per scenario of the known covariates (e.g. {"base": future, "price +10%": future_up}), stacked
        with a `scenario` column so the outcomes compare side by side.
        """
        if not futures:
            raise DataError("futures must map scenario names to future DataFrames.")
        frames = []
        for name, fut in futures.items():
            f = self.forecast(data, h, model=model, quantiles=quantiles, level=level, known=known, future=fut, **kw).df
            f.insert(0, "scenario", name)
            frames.append(f)
        return pd.concat(frames, ignore_index=True)

    def evaluate(
        self,
        data: Any,
        h: int,
        *,
        windows: int = 3,
        step: Optional[int] = None,
        models: Sequence[str] = (DEFAULT_TS_MODEL,),
        baselines: Sequence[str] = ("seasonal_naive",),
        metrics: Sequence[str] = _evaluate.DEFAULT_METRICS,
        compare: Optional[Dict[str, pd.DataFrame]] = None,
        season: Optional[int] = None,
        freq: Optional[str] = None,
        id_col: Optional[str] = None,
        time_col: Optional[str] = None,
        target_col: Target = None,
        known: Cols = None,
        past: Cols = None,
        static: Cols = None,
        calendar: Calendar = None,
    ) -> EvaluationResult:
        """Rolling-origin backtest: `windows` cutoffs `step` apart (default h) at the end of each series, each forecast `h` ahead and
        scored against the actuals. Known covariates are read from `data` itself. compare: your own forecasts to score alongside,
        {"current": df} with id, time and quantile columns (median, q10, ...) or a point column 'forecast'. The summary gives
        skill against the first baseline (1 - geometric-mean error ratio; > 0 beats it).
        """
        data, _, known = _with_calendar(data, None, None, calendar, known, id_col, time_col, freq, target_col)
        known = list(known or [])
        prep = _prep(
            data,
            1,
            self.limits,
            freq=freq,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            known=None,
            past=known + list(past or []),
            static=static,
            future=None,
        )
        out = _evaluate.run(
            prep,
            h,
            windows=windows,
            step=step,
            models=models,
            baselines=baselines,
            metrics=metrics,
            season=season,
            known=known,
            compare=compare,
            forecaster=lambda m, items: self._forecaster(h, prep.freq, m, season)(items),
        )
        res = EvaluationResult(out[0], out[1], out[2], out[3], windows, h, list(prep.warnings) + out[4])
        warn_all(res.warnings)
        return res

    def detect_anomalies(
        self,
        data: Any,
        *,
        h: Optional[int] = None,
        windows: int = 5,
        threshold: float = 3.0,
        model: str = DEFAULT_TS_MODEL,
        freq: Optional[str] = None,
        id_col: Optional[str] = None,
        time_col: Optional[str] = None,
        target_col: Target = None,
        known: Cols = None,
        past: Cols = None,
        static: Cols = None,
        season: Optional[int] = None,
        calendar: Calendar = None,
    ) -> pd.DataFrame:
        """Scores the last windows x h points of each series against forecasts made from the history before each window.
        score = (actual - median) / the forecast's own spread; |score| > threshold marks `anomaly`. h defaults to one season.
        """
        data, _, known = _with_calendar(data, None, None, calendar, known, id_col, time_col, freq, target_col)
        known = list(known or [])
        prep = _prep(
            data,
            1,
            self.limits,
            freq=freq,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            known=None,
            past=known + list(past or []),
            static=static,
            future=None,
        )
        out, warns = _anomaly.anomalies(
            prep, h, windows, threshold, known, self._forecaster(None, prep.freq, model, season)
        )
        warn_all(list(prep.warnings) + warns)
        return out

    def fill_gaps(
        self,
        data: Any,
        *,
        max_gap: Optional[int] = None,
        model: str = DEFAULT_TS_MODEL,
        freq: Optional[str] = None,
        id_col: Optional[str] = None,
        time_col: Optional[str] = None,
        target_col: Target = None,
        known: Cols = None,
        past: Cols = None,
        static: Cols = None,
        season: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fills missing values (and missing timestamps) inside each series with the median forecast from the history before the
        gap. Returns the regular grid: id, time, targets, and a `<target>_filled` flag per target.
        """
        known = list(known or [])
        prep = _prep(
            data,
            1,
            self.limits,
            freq=freq,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            known=None,
            past=known + list(past or []),
            static=static,
            future=None,
        )
        for it in prep.items:  # known columns run over the whole history here, so a gap's horizon reads them directly
            for k in known:
                it.known[k] = it.past.pop(k)
        out, warns = _anomaly.gaps(
            prep, max_gap or self.limits.max_horizon, self._forecaster(None, prep.freq, model, season)
        )
        warn_all([w for w in prep.warnings if not w.startswith("Filled ")] + warns)  # filling them is the point here
        return out

    # ---- tabular ----

    def predict(
        self,
        train: Any,
        target: str,
        test: Any = None,
        *,
        model: str = DEFAULT_TAB_MODEL,
        task: Optional[str] = None,
        preset: Optional[str] = None,
        eval_metric: Optional[str] = None,
        time_limit: Optional[float] = None,
        quantiles: Any = None,
        seed: Optional[int] = None,
        groups: Optional[str] = None,
        time_col: Optional[str] = None,
        set_coverage: Optional[float] = None,
        _refs: Optional[Dict[str, dict]] = None,
    ) -> PredictionResult:
        """Fit on `train` and predict `test` in one call (nothing is kept; to predict again from the same fit, use fit()).

        Without test=, the rows of `train` whose target is missing are predicted.
        task: "classify", "regress" or None to let the server infer it. Pass it for integer targets: a rating from 1 to 5 or a
        small count may otherwise be inferred as classification.
        preset: "fast" (the default: one holdout split), "medium" (3-fold bagging, about 3x slower and more accurate; what our
        TabArena numbers use) or "full". eval_metric: what validation optimises, e.g. log_loss (default), roc_auc, f1; rmse
        (default), mae, r2 for regression. It does not change the decision rule. quantiles: regression prediction intervals,
        e.g. [0.05, 0.5, 0.95] (split-conformal on out-of-sample errors). time_limit: seconds for the whole call. seed: the
        server uses 0 when omitted, so results are already reproducible.
        """
        tr, te, pt, q, warns, extra = _tabular.prepare(
            train,
            target,
            test,
            task=task,
            preset=preset,
            quantiles=quantiles,
            limits=self.limits,
            eval_metric=eval_metric,
            groups=groups,
            time_col=time_col,
            set_coverage=set_coverage,
        )
        refs = _refs if _refs is not None else self._refs(train=tr, test=te)
        body = _tabular.payload(
            model,
            tr,
            target,
            te,
            pt,
            preset,
            eval_metric,
            time_limit,
            seed,
            self.limits.max_tabular_bytes,
            q,
            extra,
            refs,
        )
        t0 = time.monotonic()
        res = _predict_result(
            model,
            self._t.request("POST", "/v1/tabular/predict", body),
            te,
            _tabular.label_values(tr[target]),
            warns,
            bool(q),
        )
        res.timing["total_s"] = round(time.monotonic() - t0, 3)
        res.checks = _profile.Checks(_profile.build(tr, _tabular.features_of(tr, target, groups)), te)
        res.provenance = _provenance(
            self.base_url,
            "/v1/tabular/predict",
            model,
            [res.request_id or ""],
            train,
            {
                "target": target,
                "task": task,
                "preset": preset,
                "eval_metric": eval_metric,
                "seed": seed,
                "quantiles": q,
                "rows": len(te),
            },
        )
        warn_all(res.warnings)
        return res

    def estimate_table(
        self,
        train: Any = None,
        *,
        target: Optional[str] = None,
        test: Any = None,
        preset: Optional[str] = None,
        train_rows: Optional[int] = None,
        features: Optional[int] = None,
        predict_rows: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Credits a tabular call will cost, before making it: a fit on `train` (rows x feature columns, times the preset's
        folds) plus predicting `test`. Pass tables, or the counts (train_rows, features, predict_rows). Uses the rates the API
        publishes, so it stays right when they change. -> {"fit", "predict", "total", "unit"}
        """
        if train is not None:
            df = _tabular._frame(_pandas(train), "train")
            train_rows = len(df) if train_rows is None else train_rows
            features = (df.shape[1] - (1 if target in df.columns else 0)) if features is None else features
        if test is not None:
            predict_rows = len(_tabular._frame(_pandas(test), "test")) if predict_rows is None else predict_rows
        rates = (self.limits.pricing or {}).get("tabular") or {}
        return _tabular.estimate(
            rates, int(train_rows or 0), int(features or 1), preset or "fast", int(predict_rows or 0)
        )

    def _upload(self, raw: bytes) -> dict:
        """Sends a parquet file once (POST /v1/uploads, then the upload it names) and returns the table reference."""
        sha = hashlib.sha256(raw).hexdigest()
        if sha not in self._tables:
            slot = self._t.request("POST", "/v1/uploads", {"sha256": sha, "size_bytes": len(raw), "format": "parquet"})
            if not slot.get("exists"):
                self._t.put_bytes(slot["upload"]["url"], raw, bool(slot["upload"].get("auth")))
            self._tables[sha] = slot["id"]
        return {"format": "table", "id": self._tables[sha]}

    def _refs(self, **tables: Optional[pd.DataFrame]) -> Dict[str, dict]:
        """Large tables as uploaded-table references (when the API offers uploads); small ones stay inline."""
        if not self.limits.supports("table_uploads"):
            return {}
        out = {}
        for name, df in tables.items():
            raw = _tabular.upload_candidate(df) if df is not None else None
            if raw is not None:
                out[name] = self._upload(raw)
        return out

    def evaluate_table(
        self,
        data: Any,
        target: str,
        *,
        cv: int = 5,
        groups: Any = None,
        time_col: Optional[str] = None,
        model: str = DEFAULT_TAB_MODEL,
        task: Optional[str] = None,
        preset: Optional[str] = None,
        eval_metric: Optional[str] = None,
        quantiles: Any = None,
        seed: int = 0,
    ) -> TableEvaluation:
        """Cross-validate on your own table: each row is predicted by a fit that never saw it, then scored for accuracy and for
        how honest the uncertainty is.

        groups: a column name (or one value per row) whose rows must stay together, e.g. a machine or a batch, so the score is
        for groups the model has not seen. time_col: time-ordered folds instead (each fold trains only on earlier rows).
        Classification: accuracy, log loss, Brier score, AUC, calibration error, and a reliability table. Regression: MAE,
        RMSE, R2 and, with intervals (default quantiles 0.05 to 0.95), CRPS, coverage per interval and a PIT histogram.
        Cost: one fit-and-predict call per fold; the rows predicted add up to the table's size.
        """
        df = _tabular._frame(_pandas(data), "data")
        if target not in df.columns:
            raise DataError(f"target {target!r} is not a column. Columns: {list(df.columns)}")
        df = df[df[target].notna()]
        tk = _table_eval.infer_task(df[target], task)
        q = (list(_table_eval.DEFAULT_QUANTILES) if quantiles is None else quantiles) if tk == "regress" else None
        split = "time-ordered" if time_col else ("grouped" if groups is not None else "shuffled")
        drop = [groups] if isinstance(groups, str) else []
        parts, warns = [], []
        full = _tabular.upload_candidate(df) if self.limits.supports("table_uploads") else None
        base = (
            self._upload(full) if full is not None else None
        )  # a large table is uploaded once; each fold names its rows
        for k, (tr, te) in enumerate(_table_eval.folds(df, cv, groups, time_col, seed)):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                res = self.predict(
                    df.iloc[tr].drop(columns=drop),
                    target,
                    test=df.iloc[te].drop(columns=[target, *drop]),
                    model=model,
                    task=tk,
                    preset=preset,
                    eval_metric=eval_metric,
                    quantiles=q,
                    seed=seed,
                    _refs=_fold_refs(base, tr, te, target, drop),
                )
            warns += [str(x.message) for x in w]
            parts.append((k, te, res))
        out = _table_eval.assemble(tk, split, df, target, parts, sorted(q) if q else None)
        out.warnings = list(dict.fromkeys(warns))
        warn_all(out.warnings)
        return out

    def fit(
        self,
        train: Any,
        target: Union[str, Sequence[str]],
        *,
        model: str = DEFAULT_TAB_MODEL,
        task: Optional[str] = None,
        preset: Optional[str] = None,
        eval_metric: Optional[str] = None,
        time_limit: Optional[float] = None,
        seed: Optional[int] = None,
        ttl_hours: Optional[float] = None,
        profile: bool = True,
        groups: Optional[str] = None,
        time_col: Optional[str] = None,
    ) -> Union[FittedModel, FitGroup]:
        """Fit once on our side and get a handle to predict (with intervals for regression) and explain as often as you like;
        each prediction sends only the rows to score. The fit is kept for ttl_hours (default 24, at most 168) or until
        .delete(); `with yh.fit(...) as m:` deletes it on exit. Options as in predict().
        profile=True keeps a summary of the training table on the handle (ranges, categories, up to 2,000 sample rows), so
        predictions report out_of_range and novelty; it stays on your machine (and in .save() records). False drops it.
        target=[...] fits each target on the features alone (the other targets left out) and returns a FitGroup that predicts,
        explains and deletes them together; task= may then be a dict per target. One fit's quota per target.
        """
        if not isinstance(target, str):
            targets = _targets(target, train)
            opts = dict(
                model=model,
                preset=preset,
                eval_metric=eval_metric,
                time_limit=time_limit,
                seed=seed,
                ttl_hours=ttl_hours,
                profile=profile,
                groups=groups,
                time_col=time_col,
            )
            frame = _tabular._frame(_pandas(train), "train")
            made: Dict[str, FittedModel] = {}

            def one(t: str) -> FittedModel:
                m = self.fit(
                    frame.drop(columns=[o for o in targets if o != t]),
                    t,
                    task=task.get(t) if isinstance(task, dict) else task,
                    **opts,
                )  # type: ignore[arg-type]
                made[t] = m  # type: ignore[assignment]
                return m  # type: ignore[return-value]

            try:
                with ThreadPoolExecutor(min(4, len(targets))) as ex:
                    fits = list(ex.map(one, targets))
            except BaseException:
                for m in made.values():
                    m.delete()
                raise
            return FitGroup(dict(zip(targets, fits)))
        _tabular.check_options(preset, eval_metric)
        tr, pt, warns = _tabular.prepare_train(train, target, task=task, limits=self.limits)
        extra = _tabular.split_options(tr, target, groups, time_col, None, pt)
        body = _tabular.fit_payload(
            model,
            tr,
            target,
            pt,
            preset,
            eval_metric,
            time_limit,
            seed,
            ttl_hours,
            self.limits.max_tabular_bytes,
            extra,
            self._refs(train=tr),
        )
        try:
            reply = self._t.request("POST", "/v1/fits", body)
        except NotFoundError as e:
            if e.type == "not_found":
                raise _not_deployed(e) from e
            raise
        warn_all(warns)
        fm = FittedModel.from_reply(
            reply, _tabular.features_of(tr, target, groups), _tabular.label_values(tr[target]), self
        )
        fm.profile = _profile.build(tr, fm.features or []) if profile else None
        return fm

    def get_fit(self, fit_id: str) -> FittedModel:
        """Reattach to a fit by id (labels come back as text unless the handle was pickled)."""
        return FittedModel.from_reply(self._t.request("GET", f"/v1/fits/{fit_id}"), None, [], self)

    def fits(self) -> List[Dict[str, Any]]:
        """This key's live fits: [{id, model, target, created_at, expires_at}]."""
        return self._t.request("GET", "/v1/fits")["data"]

    def impute(
        self, df: pd.DataFrame, columns: Cols = None, *, model: str = DEFAULT_TAB_MODEL, preset: Optional[str] = None
    ) -> pd.DataFrame:
        """Fills missing cells of a table: for each column with gaps, the model learns it from the other columns on the rows
        where it is present and predicts the rest (one call per column). The result's .attrs['filled'] marks what was filled.
        """
        out = df.copy()
        filled = pd.DataFrame(False, index=df.index, columns=df.columns)
        cols = [c for c in (columns or df.columns) if df[c].isna().any()]
        for c in cols:
            known_rows = df[c].notna()
            if known_rows.sum() < 10 or df.loc[known_rows, c].nunique() < 2:
                warn_all([f"Column {c!r}: too few observed values to learn it; left as is."])
                continue
            res = self.predict(
                df.loc[known_rows],
                c,
                df.loc[~known_rows].drop(columns=[c]),
                model=model,
                task=_impute_task(df[c]),
                preset=preset,
            )
            out.loc[~known_rows, c] = res.predictions.to_numpy()
            filled.loc[~known_rows, c] = True
        out.attrs["filled"] = filled
        return out

    # ---- account ----

    def models(self) -> List[Dict[str, Any]]:
        """Models this key can use: [{id, kind, description}]."""
        return self._t.request("GET", "/v1/models")["models"]

    def usage(self) -> Dict[str, Any]:
        return self._t.request("GET", "/v1/usage")

    def server_limits(self, apply: bool = True) -> Dict[str, Any]:
        """The API's request caps (GET /v1/limits); with apply=True the client splits batches by them from now on."""
        lim = self._t.request("GET", "/v1/limits")
        if apply:
            self.limits = Limits.from_api(lim)
        return lim


class AsyncClient(_Base):
    """asyncio version of Client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        timeout: float = 300.0,
        max_retries: int = 3,
        max_concurrency: int = 4,
        limits: Optional[Limits] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        super().__init__(AsyncTransport(api_key, base_url, timeout, max_retries, http_client), limits, max_concurrency)

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._t.aclose()

    async def _send(
        self,
        items: List[Item],
        h: int,
        freq: str,
        model: str,
        season: Optional[int],
        on_error: str = "raise",
        extra: Optional[dict] = None,
    ):
        sem = asyncio.Semaphore(self.max_concurrency)

        async def one(b: List[Item]):
            async with sem:
                try:
                    reply = await self._t.request("POST", "/v1/forecast", _body(b, h, freq, model, season, extra))
                    return _unpack(b, reply), reply, None
                except APIError as e:
                    if on_error == "raise":
                        raise
                    return [None] * len(b), None, {"ids": [it.uid for it in b], "error": str(e), "exception": e}

        done = await asyncio.gather(*(one(b) for b in batches(items, self.limits)))
        return (
            [f for out, _, _ in done for f in out],
            [r for _, r, _ in done if r is not None],
            [e for _, _, e in done if e],
        )

    async def forecast(
        self,
        data: Any,
        h: int,
        *,
        model: str = DEFAULT_TS_MODEL,
        quantiles: Any = None,
        level: Optional[Sequence[float]] = None,
        mean: bool = False,
        freq: Optional[str] = None,
        id_col: Optional[str] = None,
        time_col: Optional[str] = None,
        target_col: Target = None,
        known: Cols = None,
        past: Cols = None,
        static: Cols = None,
        future: Optional[pd.DataFrame] = None,
        season: Optional[int] = None,
        on_error: str = "raise",
        calendar: Calendar = None,
        samples: Optional[int] = None,
        seed: Optional[int] = None,
        history: bool = False,
    ) -> ForecastResult:
        t0 = time.monotonic()
        data, future, known = _with_calendar(data, future, h, calendar, known, id_col, time_col, freq, target_col)
        quantiles = _levels(quantiles, level)
        extra = _forecast_extra(self.limits, quantiles, samples, seed, history)
        prep = _prep(
            data,
            h,
            self.limits,
            freq=freq,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            known=known,
            past=past,
            static=static,
            future=future,
        )
        out, replies, errors = await self._send(prep.items, h, prep.freq, model, season, on_error, extra)
        res = _forecast_result(prep, model, quantiles, prep.items, out, replies, errors, mean)
        res.samples, res.history = _extras(prep, out, res)
        res.timing = {"total_s": round(time.monotonic() - t0, 3), "requests": len(replies)}
        res.provenance = _provenance(
            self.base_url,
            "/v1/forecast",
            model,
            res.request_ids,
            data,
            {
                "h": h,
                "freq": prep.freq,
                "quantiles": res.quantile_levels,
                "known": known,
                "past": past,
                "static": static,
                "season": season,
                "series": len(prep.items),
            },
        )
        warn_all(res.warnings)
        return res

    async def evaluate(
        self,
        data: Any,
        h: int,
        *,
        windows: int = 3,
        step: Optional[int] = None,
        models: Sequence[str] = (DEFAULT_TS_MODEL,),
        baselines: Sequence[str] = ("seasonal_naive",),
        metrics: Sequence[str] = _evaluate.DEFAULT_METRICS,
        compare: Optional[Dict[str, pd.DataFrame]] = None,
        season: Optional[int] = None,
        freq: Optional[str] = None,
        id_col: Optional[str] = None,
        time_col: Optional[str] = None,
        target_col: Target = None,
        known: Cols = None,
        past: Cols = None,
        static: Cols = None,
        calendar: Calendar = None,
    ) -> EvaluationResult:
        data, _, known = _with_calendar(data, None, None, calendar, known, id_col, time_col, freq, target_col)
        known = list(known or [])
        prep = _prep(
            data,
            1,
            self.limits,
            freq=freq,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            known=None,
            past=known + list(past or []),
            static=static,
            future=None,
        )
        m = season or _evaluate.season_for(prep.freq)
        items, _, _ = (
            _evaluate.split(prep, known, h, windows, step or h, min_context=max(2 * m, 8) if m > 1 else 8)
            if windows >= 1
            else ([], [], [])
        )
        done = {}
        for model in models:  # the forecasts first, concurrently per model; scoring then runs on them
            out, replies, _ = await self._send(items, h, prep.freq, model, season)
            done[model] = (out, _native(replies))
        out = _evaluate.run(
            prep,
            h,
            windows=windows,
            step=step,
            models=models,
            baselines=baselines,
            metrics=metrics,
            season=season,
            known=known,
            compare=compare,
            forecaster=lambda model, _items: done[model],
        )
        res = EvaluationResult(out[0], out[1], out[2], out[3], windows, h, list(prep.warnings) + out[4])
        warn_all(res.warnings)
        return res

    async def predict(
        self,
        train: Any,
        target: str,
        test: Any = None,
        *,
        model: str = DEFAULT_TAB_MODEL,
        task: Optional[str] = None,
        preset: Optional[str] = None,
        eval_metric: Optional[str] = None,
        time_limit: Optional[float] = None,
        quantiles: Any = None,
        seed: Optional[int] = None,
        groups: Optional[str] = None,
        time_col: Optional[str] = None,
        set_coverage: Optional[float] = None,
        _refs: Optional[Dict[str, dict]] = None,
    ) -> PredictionResult:
        tr, te, pt, q, warns, extra = _tabular.prepare(
            train,
            target,
            test,
            task=task,
            preset=preset,
            quantiles=quantiles,
            limits=self.limits,
            eval_metric=eval_metric,
            groups=groups,
            time_col=time_col,
            set_coverage=set_coverage,
        )
        refs = _refs if _refs is not None else await self._refs(train=tr, test=te)
        body = _tabular.payload(
            model,
            tr,
            target,
            te,
            pt,
            preset,
            eval_metric,
            time_limit,
            seed,
            self.limits.max_tabular_bytes,
            q,
            extra,
            refs,
        )
        t0 = time.monotonic()
        res = _predict_result(
            model,
            await self._t.request("POST", "/v1/tabular/predict", body),
            te,
            _tabular.label_values(tr[target]),
            warns,
            bool(q),
        )
        res.timing["total_s"] = round(time.monotonic() - t0, 3)
        res.checks = _profile.Checks(_profile.build(tr, _tabular.features_of(tr, target, groups)), te)
        res.provenance = _provenance(
            self.base_url,
            "/v1/tabular/predict",
            model,
            [res.request_id or ""],
            train,
            {
                "target": target,
                "task": task,
                "preset": preset,
                "eval_metric": eval_metric,
                "seed": seed,
                "quantiles": q,
                "rows": len(te),
            },
        )
        warn_all(res.warnings)
        return res

    async def _upload(self, raw: bytes) -> dict:  # type: ignore[override]
        sha = hashlib.sha256(raw).hexdigest()
        if sha not in self._tables:
            slot = await self._t.request(
                "POST", "/v1/uploads", {"sha256": sha, "size_bytes": len(raw), "format": "parquet"}
            )
            if not slot.get("exists"):
                await self._t.put_bytes(slot["upload"]["url"], raw, bool(slot["upload"].get("auth")))
            self._tables[sha] = slot["id"]
        return {"format": "table", "id": self._tables[sha]}

    async def _refs(self, **tables: Optional[pd.DataFrame]) -> Dict[str, dict]:  # type: ignore[override]
        if not self.limits.supports("table_uploads"):
            return {}
        out = {}
        for name, df in tables.items():
            raw = _tabular.upload_candidate(df) if df is not None else None
            if raw is not None:
                out[name] = await self._upload(raw)
        return out

    async def evaluate_table(
        self,
        data: Any,
        target: str,
        *,
        cv: int = 5,
        groups: Any = None,
        time_col: Optional[str] = None,
        model: str = DEFAULT_TAB_MODEL,
        task: Optional[str] = None,
        preset: Optional[str] = None,
        eval_metric: Optional[str] = None,
        quantiles: Any = None,
        seed: int = 0,
    ) -> TableEvaluation:
        """Client.evaluate_table, with the folds run concurrently."""
        df = _tabular._frame(_pandas(data), "data")
        if target not in df.columns:
            raise DataError(f"target {target!r} is not a column. Columns: {list(df.columns)}")
        df = df[df[target].notna()]
        tk = _table_eval.infer_task(df[target], task)
        q = (list(_table_eval.DEFAULT_QUANTILES) if quantiles is None else quantiles) if tk == "regress" else None
        split = "time-ordered" if time_col else ("grouped" if groups is not None else "shuffled")
        drop = [groups] if isinstance(groups, str) else []
        splits = _table_eval.folds(df, cv, groups, time_col, seed)
        full = _tabular.upload_candidate(df) if self.limits.supports("table_uploads") else None
        base = await self._upload(full) if full is not None else None
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = await asyncio.gather(
                *[
                    self.predict(
                        df.iloc[tr].drop(columns=drop),
                        target,
                        test=df.iloc[te].drop(columns=[target, *drop]),
                        model=model,
                        task=tk,
                        preset=preset,
                        eval_metric=eval_metric,
                        quantiles=q,
                        seed=seed,
                        _refs=_fold_refs(base, tr, te, target, drop),
                    )
                    for tr, te in splits
                ]
            )
        out = _table_eval.assemble(
            tk,
            split,
            df,
            target,
            [(k, te, r) for k, ((_, te), r) in enumerate(zip(splits, results))],
            sorted(q) if q else None,
        )
        out.warnings = list(dict.fromkeys(str(x.message) for x in w))
        warn_all(out.warnings)
        return out

    async def fit(
        self,
        train: Any,
        target: Union[str, Sequence[str]],
        *,
        model: str = DEFAULT_TAB_MODEL,
        task: Optional[str] = None,
        preset: Optional[str] = None,
        eval_metric: Optional[str] = None,
        time_limit: Optional[float] = None,
        seed: Optional[int] = None,
        ttl_hours: Optional[float] = None,
        profile: bool = True,
        groups: Optional[str] = None,
        time_col: Optional[str] = None,
    ) -> Union[AsyncFittedModel, AsyncFitGroup]:
        if not isinstance(target, str):
            targets = _targets(target, train)
            opts = dict(
                model=model,
                preset=preset,
                eval_metric=eval_metric,
                time_limit=time_limit,
                seed=seed,
                ttl_hours=ttl_hours,
                profile=profile,
                groups=groups,
                time_col=time_col,
            )
            frame = _tabular._frame(_pandas(train), "train")
            outs = await asyncio.gather(
                *[
                    self.fit(
                        frame.drop(columns=[o for o in targets if o != t]),
                        t,
                        task=task.get(t) if isinstance(task, dict) else task,
                        **opts,
                    )
                    for t in targets
                ],  # type: ignore[arg-type]
                return_exceptions=True,
            )
            errs = [o for o in outs if isinstance(o, BaseException)]
            if errs:
                for o in outs:
                    if not isinstance(o, BaseException):
                        await o.delete()  # type: ignore[union-attr]
                raise errs[0]
            return AsyncFitGroup(dict(zip(targets, outs)))  # type: ignore[arg-type]
        _tabular.check_options(preset, eval_metric)
        tr, pt, warns = _tabular.prepare_train(train, target, task=task, limits=self.limits)
        extra = _tabular.split_options(tr, target, groups, time_col, None, pt)
        body = _tabular.fit_payload(
            model,
            tr,
            target,
            pt,
            preset,
            eval_metric,
            time_limit,
            seed,
            ttl_hours,
            self.limits.max_tabular_bytes,
            extra,
            await self._refs(train=tr),
        )
        try:
            reply = await self._t.request("POST", "/v1/fits", body)
        except NotFoundError as e:
            if e.type == "not_found":
                raise _not_deployed(e) from e
            raise
        warn_all(warns)
        fm = FittedModel.from_reply(
            reply, _tabular.features_of(tr, target, groups), _tabular.label_values(tr[target]), self
        )
        fm.profile = _profile.build(tr, fm.features or []) if profile else None
        return AsyncFittedModel(**{k: getattr(fm, k) for k in fm.__dataclass_fields__})

    async def models(self) -> List[Dict[str, Any]]:
        return (await self._t.request("GET", "/v1/models"))["models"]

    async def usage(self) -> Dict[str, Any]:
        return await self._t.request("GET", "/v1/usage")

    async def server_limits(self, apply: bool = True) -> Dict[str, Any]:
        lim = await self._t.request("GET", "/v1/limits")
        if apply:
            self.limits = Limits.from_api(lim)
        return lim
