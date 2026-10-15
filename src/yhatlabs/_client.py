"""Client and AsyncClient: the public entry points."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import httpx
import pandas as pd

from . import _anomaly, _evaluate, _tabular
from ._errors import APIError, DataError, NotFoundError, YHatError
from ._fits import AsyncFittedModel, FittedModel, _not_deployed
from ._forecast import Item, Limits, Prepared, batches, build_frame, prepare, resolve_quantiles, warn_all
from ._results import EvaluationResult, ForecastResult, PredictionResult
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


def _body(batch: List[Item], h: int, freq: str, model: str, season: Optional[int]) -> dict:
    body: dict = {
        "model": model,
        "horizon": h,
        "freq": freq,
        "series": [it.payload(f"s{j}") for j, it in enumerate(batch)],
    }
    if season is not None:
        body["season"] = int(season)
    return body


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


def _forecast_result(
    prep: Prepared,
    model: str,
    quantiles: Any,
    items: List[Item],
    outs: List[Optional[dict]],
    replies: List[dict],
    errors: List[Dict[str, Any]],
) -> ForecastResult:
    if not replies:
        raise errors[0]["exception"]
    native = _native(replies)
    levels, derived = resolve_quantiles(quantiles, native)
    kept = Prepared(
        [it for it, o in zip(items, outs) if o is not None],
        prep.freq,
        prep.id_col,
        prep.time_col,
        prep.target_cols,
        prep.warnings,
    )
    frame = build_frame(kept, {it.uid: o for it, o in zip(items, outs) if o is not None}, native, levels)
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
    data = add_calendar(_pandas(data), feats, time_col=time_col, **hol)
    names = feats + (["is_holiday"] if hol["holidays"] else [])
    if h is not None:
        future = (
            _pandas(future) if future is not None else make_future(data, h, id_col=id_col, time_col=time_col, freq=freq)
        )
        future = add_calendar(future, feats, time_col=time_col, **hol)
    return data, future, list(known or []) + [n for n in names if n not in (known or [])]


def _fingerprint(data: Any) -> Optional[str]:
    try:
        from ._forecast import _pandas

        d = _pandas(data)
        if isinstance(d, (pd.DataFrame, pd.Series)):
            return hashlib.sha256(pd.util.hash_pandas_object(d, index=False).to_numpy().tobytes()).hexdigest()[:16]
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

    @property
    def limits(self) -> Limits:
        if self._limits is None:
            self._limits = self._fetch_limits()
        return self._limits

    @limits.setter
    def limits(self, value: Limits) -> None:
        self._limits = value

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

    def _send(self, items: List[Item], h: int, freq: str, model: str, season: Optional[int], on_error: str = "raise"):
        """-> (one reply per item, None where its batch failed), the raw replies, and the failures."""
        bs = batches(items, self.limits)

        def one(b: List[Item]):
            try:
                reply = self._t.request("POST", "/v1/forecast", _body(b, h, freq, model, season))
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
    ) -> ForecastResult:
        """Forecast the next `h` steps of every series in `data`.

        data: long DataFrame (id, timestamp, target[s], covariates), a Series with a DatetimeIndex, an array, or a dict of arrays
        (arrays need freq=). quantiles: levels in [0.1, 0.9] (default 0.1, 0.5, 0.9) or "all"; or level=[80] for central intervals.
        known: columns whose future values you know (pass them in `future`, see yhatlabs.features.make_future); past: history-only
        columns; static: per-series constants (region, store size). on_error="warn" returns the series that succeeded when some
        batches fail (failures in result.errors). calendar=True adds hour / weekday / month / weekend as known covariates, with
        no future= needed for them ({"holidays": "US"} adds a holiday flag).
        """
        t0 = time.monotonic()
        data, future, known = _with_calendar(data, future, h, calendar, known, id_col, time_col, freq)
        quantiles = _levels(quantiles, level)
        resolve_quantiles(quantiles, ASSUMED_NATIVE)
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
        out, replies, errors = self._send(prep.items, h, prep.freq, model, season, on_error)
        res = _forecast_result(prep, model, quantiles, prep.items, out, replies, errors)
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
        data, _, known = _with_calendar(data, None, None, calendar, known, id_col, time_col, freq)
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
        data, _, known = _with_calendar(data, None, None, calendar, known, id_col, time_col, freq)
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
        tr, te, pt, q, warns = _tabular.prepare(
            train,
            target,
            test,
            task=task,
            preset=preset,
            quantiles=quantiles,
            limits=self.limits,
            eval_metric=eval_metric,
        )
        body = _tabular.payload(
            model, tr, target, te, pt, preset, eval_metric, time_limit, seed, self.limits.max_tabular_bytes, q
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

    def fit(
        self,
        train: Any,
        target: str,
        *,
        model: str = DEFAULT_TAB_MODEL,
        task: Optional[str] = None,
        preset: Optional[str] = None,
        eval_metric: Optional[str] = None,
        time_limit: Optional[float] = None,
        seed: Optional[int] = None,
        ttl_hours: Optional[float] = None,
    ) -> FittedModel:
        """Fit once on our side and get a handle to predict (with intervals for regression) and explain as often as you like;
        each prediction sends only the rows to score. The fit is kept for ttl_hours (default 24, at most 168) or until
        .delete(); `with yh.fit(...) as m:` deletes it on exit. Options as in predict().
        """
        _tabular.check_options(preset, eval_metric)
        tr, pt, warns = _tabular.prepare_train(train, target, task=task, limits=self.limits)
        body = _tabular.fit_payload(
            model, tr, target, pt, preset, eval_metric, time_limit, seed, ttl_hours, self.limits.max_tabular_bytes
        )
        try:
            reply = self._t.request("POST", "/v1/fits", body)
        except NotFoundError as e:
            if e.type == "not_found":
                raise _not_deployed(e) from e
            raise
        warn_all(warns)
        return FittedModel.from_reply(
            reply, [c for c in tr.columns if c != target], _tabular.label_values(tr[target]), self
        )

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
        self, items: List[Item], h: int, freq: str, model: str, season: Optional[int], on_error: str = "raise"
    ):
        sem = asyncio.Semaphore(self.max_concurrency)

        async def one(b: List[Item]):
            async with sem:
                try:
                    reply = await self._t.request("POST", "/v1/forecast", _body(b, h, freq, model, season))
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
    ) -> ForecastResult:
        t0 = time.monotonic()
        data, future, known = _with_calendar(data, future, h, calendar, known, id_col, time_col, freq)
        quantiles = _levels(quantiles, level)
        resolve_quantiles(quantiles, ASSUMED_NATIVE)
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
        out, replies, errors = await self._send(prep.items, h, prep.freq, model, season, on_error)
        res = _forecast_result(prep, model, quantiles, prep.items, out, replies, errors)
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
        data, _, known = _with_calendar(data, None, None, calendar, known, id_col, time_col, freq)
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
    ) -> PredictionResult:
        tr, te, pt, q, warns = _tabular.prepare(
            train,
            target,
            test,
            task=task,
            preset=preset,
            quantiles=quantiles,
            limits=self.limits,
            eval_metric=eval_metric,
        )
        body = _tabular.payload(
            model, tr, target, te, pt, preset, eval_metric, time_limit, seed, self.limits.max_tabular_bytes, q
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

    async def fit(
        self,
        train: Any,
        target: str,
        *,
        model: str = DEFAULT_TAB_MODEL,
        task: Optional[str] = None,
        preset: Optional[str] = None,
        eval_metric: Optional[str] = None,
        time_limit: Optional[float] = None,
        seed: Optional[int] = None,
        ttl_hours: Optional[float] = None,
    ) -> AsyncFittedModel:
        _tabular.check_options(preset, eval_metric)
        tr, pt, warns = _tabular.prepare_train(train, target, task=task, limits=self.limits)
        body = _tabular.fit_payload(
            model, tr, target, pt, preset, eval_metric, time_limit, seed, ttl_hours, self.limits.max_tabular_bytes
        )
        try:
            reply = await self._t.request("POST", "/v1/fits", body)
        except NotFoundError as e:
            if e.type == "not_found":
                raise _not_deployed(e) from e
            raise
        warn_all(warns)
        fm = FittedModel.from_reply(
            reply, [c for c in tr.columns if c != target], _tabular.label_values(tr[target]), self
        )
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
