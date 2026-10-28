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

from . import _anomaly, _compare, _evaluate, _explain, _jobs, _profile, _table_eval, _tabular
from ._compare import Comparison
from ._errors import APIError, DataError, NotFoundError, UnsupportedError, YHatError
from ._fits import AsyncFitGroup, AsyncFittedModel, FitGroup, FittedModel, _not_deployed
from ._forecast import (
    Item,
    Limits,
    Prepared,
    _lazy,
    _pandas,
    batches,
    build_frame,
    prepare,
    resolve_quantiles,
    warn_all,
)
from ._jobs import AsyncJob, Job
from ._mask import Mask, masked
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


def _fit_extras(body: dict, ttl_days: Optional[float], store: Optional[bool]) -> None:
    if ttl_days is not None:
        body.pop("ttl_hours", None)
        body["ttl_days"] = float(ttl_days)
    if store is False:
        body["store"] = False


def _estimate_counts(
    train: Any,
    target: Optional[str],
    test: Any,
    train_rows: Optional[int],
    features: Optional[int],
    predict_rows: Optional[int],
) -> Tuple[int, int, int]:
    if train is not None:
        df = _tabular._frame(_pandas(train), "train")
        train_rows = len(df) if train_rows is None else train_rows
        features = (df.shape[1] - (1 if target in df.columns else 0)) if features is None else features
    if test is not None:
        predict_rows = len(_tabular._frame(_pandas(test), "test")) if predict_rows is None else predict_rows
    return int(train_rows or 0), int(features or 1), int(predict_rows or 0)


def _estimate_body(call: str, counts: Tuple[int, int, int], preset: Optional[str], task: Optional[str]) -> dict:
    rows, features, test = counts
    body = {"call": call, "train_rows": rows, "features": features, "preset": preset or "fast"}
    if call == "predict":
        body["rows"] = test
    if _tabular.norm_task(task) == "regress":
        body["regression"] = True
    return body


def _estimate_result(fit: dict, pred: Optional[dict]) -> Dict[str, Any]:
    f = float(fit.get("credits") or 0)
    p = float((pred or {}).get("credits") or 0)
    tokens = int(fit.get("tokens") or 0) + int((pred or {}).get("tokens") or 0)
    return {"fit": f, "predict": p, "total": round(f + p, 6), "tokens": tokens, "unit": "credits"}


def _compare_body(body: dict, models: Optional[List[str]]) -> dict:
    """The /v1/table/compare request for a table predict: the same tables and target, without the fit's options."""
    out = {k: body[k] for k in ("model", "target", "train", "test", "problem_type", "store") if k in body}
    if models is not None:
        out["models"] = models
    return out


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


def _mask_series(batch: List[Item], body: dict) -> Mask:
    """mask_columns=True on a forecast: only target (with several targets) and covariate names are on the wire."""
    m = Mask.of(k for it in batch for d in ((it.targets if it.multi else {}), it.past, it.known) for k in d)
    for p in body["series"]:
        for k in ("targets", "past_covariates", "known_covariates"):
            if k in p:
                p[k] = m.keys(p[k])
    return m


def _unpack(batch: List[Item], reply: dict, mask: Optional[Mask] = None) -> List[dict]:
    if mask:
        for f in reply["forecasts"]:
            for k in ("targets", "target_samples", "target_history"):
                if k in f:
                    f[k] = mask.unkeys(f[k])
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
        if 0 < float(lv) <= 1:
            raise DataError(
                f"levels are percentages, e.g. level=[80, 95]; for quantiles use quantiles=[0.1, 0.9] (got {lv})."
            )
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
    if any("credits" in (r.get("usage") or {}) for r in replies):
        usage["credits"] = round(sum(float((r.get("usage") or {}).get("credits") or 0) for r in replies), 6)
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
    def __init__(self, transport: Any, limits: Optional[Limits], max_concurrency: int, mask_columns: bool = False):
        self._t = transport
        self._limits = limits
        self.max_concurrency = max(1, max_concurrency)
        self.mask_columns = bool(mask_columns)
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
        mask_columns: Optional[bool] = None,
    ) -> Any:
        """A copy of this client with other settings, sharing its connection: yh.with_options(timeout=1800).fit(...).

        Args:
            timeout: Seconds per request.
            max_retries: Retries of timeouts, connection errors and 408 / 409 / 429 / 5xx replies.
            max_concurrency: Requests sent in parallel when a large input is split.
            mask_columns: Send column names as c0, c1, ... (results keep your names).
        """
        new = copy.copy(self)
        new._t = copy.copy(self._t)
        new._t._own = False  # closing the copy leaves the shared connection open
        if timeout is not None:
            new._t.timeout = float(timeout)
        if max_retries is not None:
            new._t.max_retries = int(max_retries)
        if max_concurrency is not None:
            new.max_concurrency = max(1, int(max_concurrency))
        if mask_columns is not None:
            new.mask_columns = bool(mask_columns)
        return new

    def _masking(self, mask_columns: Optional[bool]) -> bool:
        return self.mask_columns if mask_columns is None else bool(mask_columns)

    def _table(self, x: Any) -> Any:
        """A Spark, Dask, Ray or polars lazy table in pandas, checked against the API's table size first; else unchanged."""
        return _pandas(x, self.limits) if _lazy(x) else x

    def _key_hint(self) -> str:
        k = self._t.api_key
        return f"{k[:3]}...{k[-4:]}" if len(k) > 10 else "***"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(base_url={self.base_url!r})"

    def _direct(self, preset: Optional[str], tr: pd.DataFrame, target: str, groups: Optional[str]) -> bool:
        """Whether the training table gets a direct answer, by the sizes /v1/limits publishes (else it runs as a job)."""
        return self.limits.direct_fits(preset, len(tr), len(_tabular.features_of(tr, target, groups)))

    def _compare_request(
        self,
        train: Any,
        target: str,
        test: Any,
        models: Any,
        model: str,
        task: Optional[str],
        store: Optional[bool],
        mask_columns: Optional[bool],
    ) -> Tuple[dict, pd.Index, Optional[Mask]]:
        train, test = self._table(train), self._table(test)
        picked = _compare.check_models(models)
        tr, te, pt, _, _, _ = _tabular.prepare(
            train, target, test, task=task, preset=None, quantiles=None, limits=self.limits
        )
        m = Mask.of([*tr.columns, *te.columns]) if self._masking(mask_columns) else None
        wtr, wte = (m.frame(tr), m.frame(te)) if m else (tr, te)
        body = _tabular.payload(
            model, wtr, m.name(target) if m else target, wte, pt, None, None, None, None, self.limits.max_tabular_bytes
        )
        if store is False:
            body["store"] = False
        return _compare_body(body, picked), te.index, m


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
        mask_columns: bool = False,
    ):
        """

        Args:
            api_key: Your key. Default: the YHAT_API_KEY environment variable.
            base_url: The API's address. Default: YHAT_BASE_URL, else https://api.yhatlabs.com.
            timeout: Seconds per request.
            max_retries: Retries of timeouts, connection errors and 408 / 409 / 429 / 5xx replies, with backoff.
            max_concurrency: Requests sent in parallel when a large input is split.
            limits: Request caps to split batches by; read from GET /v1/limits on first use when left out.
            http_client: An httpx.Client to send requests with (proxies, custom transports).
            mask_columns: Send every column name as c0, c1, ... instead of its own; results come back under your names.
                Values, category labels and series ids are sent as they are. Each call can override it.
        """
        super().__init__(
            SyncTransport(api_key, base_url, timeout, max_retries, http_client), limits, max_concurrency, mask_columns
        )

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
        mask: bool = False,
    ):
        """-> (one reply per item, None where its batch failed), the raw replies, and the failures."""
        bs = batches(items, self.limits)

        def one(b: List[Item]):
            try:
                body = _body(b, h, freq, model, season, extra)
                m = _mask_series(b, body) if mask else None
                with masked(m):
                    reply = self._t.request("POST", "/v1/series/forecast", body)
                return _unpack(b, reply, m), reply, None
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

    def _forecaster(self, h_fixed: Optional[int], freq: str, model: str, season: Optional[int], mask: bool = False):
        def run(items: List[Item], h: Optional[int] = None) -> Tuple[List[dict], List[float]]:
            out, replies, _ = self._send(items, h or h_fixed, freq, model, season, mask=mask)
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
        mask_columns: Optional[bool] = None,
        store: Optional[bool] = None,
    ) -> ForecastResult:
        """Forecast the next `h` steps of every series in `data`.

        data: long DataFrame (id, timestamp, target[s], covariates), a Series with a DatetimeIndex, an array, or a dict of arrays
        (arrays need freq=). quantiles: any levels in (0, 1) (default 0.1, 0.5, 0.9) or "all"; or level=[80] for central intervals.
        mean=True adds a `mean` column next to the median (the point forecast), computed from the native quantiles and listed in
        result.derived; use it when the quantity is summed later (totals over time or series), where the median is biased.
        known: columns whose future values you know (pass them in `future`, see yhatlabs.features.make_future); past: history-only
        columns; static: per-series constants (region, store size). on_error="warn" returns the series that succeeded when some
        batches fail (failures in result.errors). calendar=True adds hour / weekday / month / weekend as known covariates, with
        no future= needed for them ({"holidays": "US"} adds a holiday flag).

        Args:
            mean: Add a mean column next to the median, computed from the quantiles (for forecasts summed later).
            on_error: "raise" (default): the first failed batch raises. "warn": return the series that succeeded;
                failures in result.errors.
            samples: Return this many sample paths per series (1 to 1000) in result.samples.
            seed: Random seed for the sample paths.
            history: Also return the model's forecasts of the windows just before the forecast (result.history).
            store: False: the call isn't shown in your Logs (see the data-handling page in the docs).
            mask_columns: Send target and covariate names as c0, c1, ...; the result keeps your names. Default: the client's
                setting (False).
            calendar: True adds hour, weekday, month and weekend as known covariates (no future= needed for them);
                {"holidays": "US"} also adds a holiday flag.
            data: The history: a long table with an id column, a time column and the target (pandas, polars, Spark,
                Dask or Ray), a Series with a DatetimeIndex, a numpy array (one series per row), or a dict of arrays.
                Arrays need freq=.
            freq: The series' frequency as a pandas alias (h, D, W-SUN, MS, 15min, ...). Inferred from the timestamps
                when left out; required for arrays and integer time steps.
            future: The known covariates over the next h steps: id, time and the known columns.
            h: How many steps to forecast, at the series' own frequency.
            id_col: The column naming each series. Default: unique_id, id, item_id or series_id if present, else one
                series.
            known: Covariate columns whose future values you know (prices, weather forecasts, holidays); their future
                values go in future=.
            level: Central intervals by width in percent, e.g. [80, 95]; the alternative to quantiles=.
            model: The model name, as GET /v1/models lists it.
            past: Covariate columns known only up to now (observed loads, sensor readings).
            quantiles: Quantile levels in (0, 1), e.g. [0.1, 0.5, 0.9], or "all" for every level the model produces.
                Default [0.1, 0.5, 0.9].
            season: The seasonal period in steps (24 for daily cycles of hourly data), to fix it instead of inferring it
                from freq.
            static: Columns that are constant per series (region, store size).
            target_col: The column (or list of columns) to forecast. Default: y, target or value if present, else the
                one numeric column.
            time_col: The timestamp column. Default: ds, timestamp, time, date or datetime if present.
        """
        t0 = time.monotonic()
        data, future, known = _with_calendar(data, future, h, calendar, known, id_col, time_col, freq, target_col)
        quantiles = _levels(quantiles, level)
        extra = _forecast_extra(self.limits, quantiles, samples, seed, history)
        if store is False:
            extra["store"] = False
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
        out, replies, errors = self._send(
            prep.items, h, prep.freq, model, season, on_error, extra, self._masking(mask_columns)
        )
        res = _forecast_result(prep, model, quantiles, prep.items, out, replies, errors, mean)
        res.samples, res.history = _extras(prep, out, res)
        res.timing = {"total_s": round(time.monotonic() - t0, 3), "requests": len(replies)}
        res.provenance = _provenance(
            self.base_url,
            "/v1/series/forecast",
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

        Args:
            futures: {scenario name: future table of the known covariates}, one table per scenario.
            known: The covariate columns the scenarios set.
            kw: Any other forecast() option (id_col, time_col, freq, past, static, mask_columns, ...).
            data: The history: a long table with an id column, a time column and the target (pandas, polars, Spark,
                Dask or Ray), a Series with a DatetimeIndex, a numpy array (one series per row), or a dict of arrays.
                Arrays need freq=.
            h: How many steps to forecast, at the series' own frequency.
            level: Central intervals by width in percent, e.g. [80, 95]; the alternative to quantiles=.
            model: The model name, as GET /v1/models lists it.
            quantiles: Quantile levels in (0, 1), e.g. [0.1, 0.5, 0.9], or "all" for every level the model produces.
                Default [0.1, 0.5, 0.9].
        """
        data = _pandas(data)  # once, not per scenario
        if not futures:
            raise DataError("futures must map scenario names to future DataFrames.")
        frames = []
        for name, fut in futures.items():
            f = self.forecast(data, h, model=model, quantiles=quantiles, level=level, known=known, future=fut, **kw).df
            f.insert(0, "scenario", name)
            frames.append(f)
        return pd.concat(frames, ignore_index=True)

    def explain_forecast(
        self,
        data: Any,
        h: int,
        *,
        known: Cols = None,
        past: Cols = None,
        future: Optional[pd.DataFrame] = None,
        model: str = DEFAULT_TS_MODEL,
        freq: Optional[str] = None,
        id_col: Optional[str] = None,
        time_col: Optional[str] = None,
        target_col: Target = None,
        static: Cols = None,
        season: Optional[int] = None,
        calendar: Calendar = None,
        mask_columns: Optional[bool] = None,
    ) -> pd.DataFrame:
        """How much each covariate moves the forecast: forecasts once with every covariate, then once per covariate with it
        set to a neutral value (numbers: the series' mean over the history and the future; categories and flags: their most
        common value). Costs one forecast plus one more per covariate in known and past.

        Args:
            known: The known covariates to explain (their future values in future=, or in the same table).
            past: The past covariates to explain.
            data: The history: a long table with an id column, a time column, the target and the covariates.
            h: How many steps to forecast, at the series' own frequency.
            future: The known covariates over the next h steps: id, time and the known columns.
            calendar: True adds hour, weekday, month and weekend as known covariates (used in every forecast, not
                explained).
            static: Columns that are constant per series (used in every forecast, not explained).
            freq: The series' frequency as a pandas alias (h, D, W-SUN, MS, 15min, ...). Inferred from the timestamps
                when left out; required for arrays and integer time steps.
            id_col: The column naming each series. Default: unique_id, id, item_id or series_id if present, else one
                series.
            mask_columns: Send target and covariate names as c0, c1, ...; results keep your names. Default: the
                client's setting (False).
            model: The model name, as GET /v1/models lists it.
            season: The seasonal period in steps (24 for daily cycles of hourly data), to fix it instead of inferring it
                from freq.
            target_col: The column (or list of columns) to forecast. Default: y, target or value if present, else the
                one numeric column.
            time_col: The timestamp column. Default: ds, timestamp, time, date or datetime if present.

        Returns:
            One row per covariate and forecast step: id, time, (target), covariate, median (with every covariate),
            median_without (with this one neutral) and effect (median - median_without). .attrs["summary"] has the
            average absolute effect per covariate, largest first.
        """
        data, future = _pandas(data), _pandas(future)
        names = _explain.covariates(data, known, past)

        def run(d: Any, f: Any) -> ForecastResult:
            return self.forecast(
                d,
                h,
                known=known,
                past=past,
                future=f,
                model=model,
                freq=freq,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                static=static,
                season=season,
                calendar=calendar,
                mask_columns=mask_columns,
            )

        base = run(data, future)
        return _explain.effects(base, {c: run(*_explain.neutralise(data, future, c, id_col)) for c in names})

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
        mask_columns: Optional[bool] = None,
    ) -> EvaluationResult:
        """Rolling-origin backtest: `windows` cutoffs `step` apart (default h) at the end of each series, each forecast `h` ahead and
        scored against the actuals. Known covariates are read from `data` itself. compare: your own forecasts to score alongside,
        {"current": df} with id, time and quantile columns (median, q10, ...) or a point column 'forecast'. The summary gives
        skill against the first baseline (1 - geometric-mean error ratio; > 0 beats it).

        Args:
            mask_columns: Send target and covariate names as c0, c1, ...; results keep your names. Default: the
                client's setting (False).
            windows: How many backtest windows, ending at the end of each series.
            step: Steps between window cutoffs. Default: h (windows do not overlap).
            models: The models to score.
            baselines: Simple forecasts to score against: seasonal_naive (repeats the last season: a week for daily data,
                a day for hourly; the last value when the history is shorter than two seasons) and naive (repeats the last
                value).
            metrics: Scores to compute per window: mase, sql, wql, mae, rmse, coverage@<level>.
            compare: Your own forecasts to score on the same windows: {name: table with the time (and id) column and
                quantile columns, or a forecast column}.
            calendar: True adds hour, weekday, month and weekend as known covariates (no future= needed for them);
                {"holidays": "US"} also adds a holiday flag.
            data: The history: a long table with an id column, a time column and the target (pandas, polars, Spark,
                Dask or Ray), a Series with a DatetimeIndex, a numpy array (one series per row), or a dict of arrays.
                Arrays need freq=.
            freq: The series' frequency as a pandas alias (h, D, W-SUN, MS, 15min, ...). Inferred from the timestamps
                when left out; required for arrays and integer time steps.
            h: How many steps to forecast, at the series' own frequency.
            id_col: The column naming each series. Default: unique_id, id, item_id or series_id if present, else one
                series.
            known: Covariate columns whose future values you know (prices, weather forecasts, holidays); their future
                values go in future=.
            past: Covariate columns known only up to now (observed loads, sensor readings).
            season: The seasonal period in steps (24 for daily cycles of hourly data), to fix it instead of inferring it
                from freq.
            static: Columns that are constant per series (region, store size).
            target_col: The column (or list of columns) to forecast. Default: y, target or value if present, else the
                one numeric column.
            time_col: The timestamp column. Default: ds, timestamp, time, date or datetime if present.
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
            forecaster=lambda m, items: self._forecaster(h, prep.freq, m, season, self._masking(mask_columns))(items),
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
        mask_columns: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Scores the last windows x h points of each series against forecasts made from the history before each window.
        score = (actual - median) / the forecast's own spread; |score| > threshold marks `anomaly`. h defaults to one season.

        Args:
            mask_columns: Send target and covariate names as c0, c1, ...; results keep your names. Default: the
                client's setting (False).
            h: Points scored per window. Default: max(season, 8).
            windows: How many windows back from the end are scored.
            threshold: Score above which a point is flagged (its distance from the forecast, in units of the forecast's
                spread).
            calendar: True adds hour, weekday, month and weekend as known covariates (no future= needed for them);
                {"holidays": "US"} also adds a holiday flag.
            data: The history: a long table with an id column, a time column and the target (pandas, polars, Spark,
                Dask or Ray), a Series with a DatetimeIndex, a numpy array (one series per row), or a dict of arrays.
                Arrays need freq=.
            freq: The series' frequency as a pandas alias (h, D, W-SUN, MS, 15min, ...). Inferred from the timestamps
                when left out; required for arrays and integer time steps.
            id_col: The column naming each series. Default: unique_id, id, item_id or series_id if present, else one
                series.
            known: Covariate columns whose future values you know (prices, weather forecasts, holidays); their future
                values go in future=.
            model: The model name, as GET /v1/models lists it.
            past: Covariate columns known only up to now (observed loads, sensor readings).
            season: The seasonal period in steps (24 for daily cycles of hourly data), to fix it instead of inferring it
                from freq.
            static: Columns that are constant per series (region, store size).
            target_col: The column (or list of columns) to forecast. Default: y, target or value if present, else the
                one numeric column.
            time_col: The timestamp column. Default: ds, timestamp, time, date or datetime if present.
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
            prep,
            h,
            windows,
            threshold,
            known,
            self._forecaster(None, prep.freq, model, season, self._masking(mask_columns)),
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
        mask_columns: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Fills missing values (and missing timestamps) inside each series with the median forecast from the history before the
        gap. Returns the regular grid: id, time, targets, and a `<target>_filled` flag per target.

        Args:
            mask_columns: Send target and covariate names as c0, c1, ...; results keep your names. Default: the
                client's setting (False).
            max_gap: Longest run of missing points to fill. Default: the API's maximum horizon.
            data: The history: a long table with an id column, a time column and the target (pandas, polars, Spark,
                Dask or Ray), a Series with a DatetimeIndex, a numpy array (one series per row), or a dict of arrays.
                Arrays need freq=.
            freq: The series' frequency as a pandas alias (h, D, W-SUN, MS, 15min, ...). Inferred from the timestamps
                when left out; required for arrays and integer time steps.
            id_col: The column naming each series. Default: unique_id, id, item_id or series_id if present, else one
                series.
            known: Covariate columns whose future values you know (prices, weather forecasts, holidays); their future
                values go in future=.
            model: The model name, as GET /v1/models lists it.
            past: Covariate columns known only up to now (observed loads, sensor readings).
            season: The seasonal period in steps (24 for daily cycles of hourly data), to fix it instead of inferring it
                from freq.
            static: Columns that are constant per series (region, store size).
            target_col: The column (or list of columns) to forecast. Default: y, target or value if present, else the
                one numeric column.
            time_col: The timestamp column. Default: ds, timestamp, time, date or datetime if present.
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
            prep,
            max_gap or self.limits.max_horizon,
            self._forecaster(None, prep.freq, model, season, self._masking(mask_columns)),
        )
        warn_all([w for w in prep.warnings if not w.startswith("Filled ")] + warns)  # filling them is the point here
        return out

    def anomaly_stream(
        self,
        *,
        window: int = 512,
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
        mask_columns: Optional[bool] = None,
    ) -> _anomaly.AnomalyStream:
        """Anomaly detection on points as they arrive: stream.update(new_points) scores only the new points, against forecasts
        from the recent history before them, and keeps the last `window` points of every series for the next update. Each
        update is one detect_anomalies call (on the kept history plus the new points), whatever the number of series.

        Args:
            window: Points kept per series as history for the next update.
            threshold: Score above which a point is flagged (its distance from the forecast, in units of the forecast's
                spread).
            calendar: True adds hour, weekday, month and weekend as known covariates; {"holidays": "US"} also adds a
                holiday flag.
            known: Covariate columns whose future values you know, sent with each update's points.
            past: Covariate columns known only up to now, sent with each update's points.
            static: Columns that are constant per series (region, store size).
            freq: The series' frequency as a pandas alias (h, D, W-SUN, MS, 15min, ...). Inferred from the timestamps
                when left out; required for arrays and integer time steps.
            id_col: The column naming each series. Default: unique_id, id, item_id or series_id if present, else one
                series.
            mask_columns: Send target and covariate names as c0, c1, ...; results keep your names. Default: the
                client's setting (False).
            model: The model name, as GET /v1/models lists it.
            season: The seasonal period in steps (24 for daily cycles of hourly data), to fix it instead of inferring it
                from freq.
            target_col: The column (or list of columns) to forecast. Default: y, target or value if present, else the
                one numeric column.
            time_col: The timestamp column. Default: ds, timestamp, time, date or datetime if present.
        """
        opts = dict(
            threshold=threshold,
            model=model,
            freq=freq,
            target_col=target_col,
            known=known,
            past=past,
            static=static,
            season=season,
            calendar=calendar,
            mask_columns=mask_columns,
        )
        return _anomaly.AnomalyStream(self, window, id_col, time_col, opts)

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
        compare: Any = None,
        store: Optional[bool] = None,
        background: bool = False,
        _refs: Optional[Dict[str, dict]] = None,
        mask_columns: Optional[bool] = None,
        _mask: Optional[Mask] = None,
    ) -> Union[PredictionResult, Job]:
        """Fit on `train` and predict `test` in one call (to predict again from the same fit, use fit()).

        Without test=, the rows of `train` whose target is missing are predicted.
        task: "classify", "regress" or None to let the server infer it. Pass it for integer targets: a rating from 1 to 5 or a
        small count may otherwise be inferred as classification.
        preset: "fast" (the default: one holdout split), "medium" (3-fold bagging, about 3x slower and more accurate) or "pro"
        (8-fold bagging, the most accurate). eval_metric: what validation optimises, e.g. log_loss (default), roc_auc, f1; rmse
        (default), mae, r2 for regression. It does not change the decision rule. quantiles: regression prediction intervals,
        e.g. [0.05, 0.5, 0.95], from each row's own predictive distribution. time_limit: seconds for the whole call. seed: the
        server uses 0 when omitted, so results are already reproducible.

        Args:
            mask_columns: Send column names as c0, c1, ...; results keep your names. Default: the client's setting
                (False).
            quantiles: Regression: interval levels in (0, 1), e.g. [0.05, 0.5, 0.95], up to 32.
            time_col: A column of times: validation folds are contiguous periods of it, so intervals hold for later
                periods. It stays a feature.
            eval_metric: What validation optimises: log_loss (default), roc_auc, f1, ... for classification; rmse
                (default), mae, r2 for regression.
            groups: A column whose rows belong together (machine, batch, patient, store): validation folds hold out
                whole groups, so intervals hold for groups the model has not seen. The column is not used as a feature.
            model: The model name, as GET /v1/models lists it.
            preset: "fast" (default: one holdout split, the quickest), "medium" (3-fold bagging, about 3x the time) or "pro" (8-fold bagging, the most accurate; "full" is the same).
            seed: Random seed; the server uses 0 when it is left out, so results are reproducible.
            set_coverage: Classification: return prediction sets that hold the true class this often, e.g. 0.9.
            target: The column to predict.
            task: "classify", "regress", or None to let the server infer it. Pass it for integer targets (a 1-5 rating
                could be either).
            test: The rows to predict, with the same feature columns as train. Without it, the rows of train whose
                target is empty are predicted.
            time_limit: Seconds for the whole call.
            train: The training table: one row per example, the target column included.
            compare: Also train standard models on the same rows, for comparison: True for XGBoost, a random forest and a
                linear model, or a list of them ("xgboost", "random_forest", "linear", "lightgbm"). Their
                predictions, with a naive baseline, are in result.comparison; result.comparison.score(actual) scores them.
            store: False: the call isn't shown in your Logs (see the data-handling page in the docs).
            background: Return a Job at once instead of waiting: job.wait() gives the result. A table too large for a
                direct call runs as a job anyway, and is waited for unless background=True.
        """
        train, test = self._table(train), self._table(test)
        picked = _compare.check_models(compare) if compare else None
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
        m = _mask or (Mask.of([*tr.columns, *te.columns]) if self._masking(mask_columns) else None)
        wtr, wte = (m.frame(tr), m.frame(te)) if m else (tr, te)
        refs = _refs if _refs is not None else self._refs(train=wtr, test=wte)
        body = _tabular.payload(
            model,
            wtr,
            m.name(target) if m else target,
            wte,
            pt,
            preset,
            eval_metric,
            time_limit,
            seed,
            self.limits.max_tabular_bytes,
            q,
            m.options(extra) if m else extra,
            refs,
        )
        if store is False:
            body["store"] = False
        t0 = time.monotonic()

        def finish(reply: Any, comparison: Optional[Comparison]) -> PredictionResult:
            res = _predict_result(
                model,
                reply,
                te,
                _tabular.label_values(tr[target]),
                warns,
                bool(q),
            )
            res.comparison = comparison
            res.timing["total_s"] = round(time.monotonic() - t0, 3)
            res.checks = _profile.Checks(_profile.build(tr, _tabular.features_of(tr, target, groups)), te)
            res.provenance = _provenance(
                self.base_url,
                "/v1/table/predict",
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

        cmp_body = _compare_body(body, picked) if compare else None
        if background:
            with masked(m):
                job = self._t.request("POST", "/v1/table/predict", {**body, "async": True})
            comparison = self._comparison(cmp_body, te.index) if cmp_body else None
            return Job(self._t, job, lambda reply: finish(reply, comparison))
        with masked(m):
            reply = _jobs.call(self._t, "/v1/table/predict", body, not self._direct(preset, tr, target, groups))
        return finish(reply, self._comparison(cmp_body, te.index) if cmp_body else None)

    def _comparison(self, body: dict, index: pd.Index) -> Comparison:
        return _compare.from_reply(self._t.request("POST", "/v1/table/compare", body), index)

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
        task: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Credits a tabular call will cost, before making it: a fit on `train` plus predicting `test`, priced by the API at
        your organization's rates (POST /v1/estimate). Pass tables, or the counts (train_rows, features, predict_rows).
        -> {"fit", "predict", "total", "tokens", "unit"}

        Args:
            train: The training table (or pass train_rows and features).
            target: The target column, left out of the feature count.
            test: The rows to predict (or pass predict_rows).
            train_rows: Training rows, when no table is passed.
            features: Feature columns, when no table is passed.
            predict_rows: Rows to predict, when no table is passed.
            preset: "fast" (default: one holdout split, the quickest), "medium" (3-fold bagging, about 3x the time) or "pro" (8-fold bagging, the most accurate; "full" is the same).
            task: "regress" adds what regression's per-row intervals cost.
        """
        train, test = self._table(train), self._table(test)
        _tabular.check_options(preset, None)
        counts = _estimate_counts(train, target, test, train_rows, features, predict_rows)
        fit = self._t.request("POST", "/v1/estimate", _estimate_body("fit", counts, preset, task))
        pred = (
            self._t.request("POST", "/v1/estimate", _estimate_body("predict", counts, preset, task))
            if counts[2]
            else None
        )
        return _estimate_result(fit, pred)

    def compare(
        self,
        train: Any,
        target: str,
        test: Any,
        *,
        models: Any = None,
        model: str = DEFAULT_TAB_MODEL,
        task: Optional[str] = None,
        store: Optional[bool] = None,
        mask_columns: Optional[bool] = None,
    ) -> Comparison:
        """Train standard models on `train` and predict `test` with each, to compare with ChakraTab on the same rows:
        LightGBM, XGBoost, a random forest and a linear model, plus a naive baseline (the most common class, or the mean).
        result.score(actual, ChakraTab=prediction) scores them side by side. A direct call: tables up to the direct size.

        Args:
            train: The training table: one row per example, the target column included.
            target: The column to predict.
            test: The rows to predict, with the same feature columns as train.
            models: Which standard models: a list of "xgboost", "random_forest", "linear", "lightgbm". Default: the first three.
            model: The model name, as GET /v1/models lists it.
            task: "classify", "regress", or None to let the server infer it.
            store: False: the call isn't shown in your Logs (see the data-handling page in the docs).
            mask_columns: Send column names as c0, c1, ...; results keep your names. Default: the client's setting
                (False).
        """
        body, index, m = self._compare_request(train, target, test, models, model, task, store, mask_columns)
        with masked(m):
            return self._comparison(body, index)

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
        mask_columns: Optional[bool] = None,
    ) -> TableEvaluation:
        """Cross-validate on your own table: each row is predicted by a fit that never saw it, then scored for accuracy and for
        how honest the uncertainty is.

        groups: a column name (or one value per row) whose rows must stay together, e.g. a machine or a batch, so the score is
        for groups the model has not seen. time_col: time-ordered folds instead (each fold trains only on earlier rows).
        Classification: accuracy, log loss, Brier score, AUC, calibration error, and a reliability table. Regression: MAE,
        RMSE, R2 and, with intervals (default quantiles 0.05 to 0.95), CRPS, coverage per interval and a PIT histogram.
        Cost: one fit-and-predict call per fold; the rows predicted add up to the table's size.

        Args:
            mask_columns: Send column names as c0, c1, ...; results keep your names. Default: the client's setting
                (False).
            data: The whole table, target included; every row is predicted once by a fit that did not see it.
            cv: Number of folds, at least 2.
            groups: A column (or one value per row) whose rows stay in the same fold.
            time_col: Time-ordered folds instead: each fold trains only on rows before its block.
            quantiles: Regression interval levels to score. Default: 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95.
            eval_metric: What validation optimises: log_loss (default), roc_auc, f1, ... for classification; rmse
                (default), mae, r2 for regression.
            model: The model name, as GET /v1/models lists it.
            preset: "fast" (default: one holdout split, the quickest), "medium" (3-fold bagging, about 3x the time) or "pro" (8-fold bagging, the most accurate; "full" is the same).
            seed: Random seed; the server uses 0 when it is left out, so results are reproducible.
            target: The column to predict.
            task: "classify", "regress", or None to let the server infer it. Pass it for integer targets (a 1-5 rating
                could be either).
        """
        df = _tabular._frame(data, "data", self.limits)
        if target not in df.columns:
            raise DataError(f"target {target!r} is not a column. Columns: {list(df.columns)}")
        df = df[df[target].notna()]
        tk = _table_eval.infer_task(df[target], task)
        q = (list(_table_eval.DEFAULT_QUANTILES) if quantiles is None else quantiles) if tk == "regress" else None
        split = "time-ordered" if time_col else ("grouped" if groups is not None else "shuffled")
        drop = [groups] if isinstance(groups, str) else []
        parts, warns = [], []
        m = Mask.of(df.columns) if self._masking(mask_columns) else None
        full = _tabular.upload_candidate(m.frame(df) if m else df) if self.limits.supports("table_uploads") else None
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
                    _refs=_fold_refs(
                        base, tr, te, m.name(target) if m else target, [m.name(d) for d in drop] if m else drop
                    ),
                    _mask=m,
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
        mask_columns: Optional[bool] = None,
        ttl_days: Optional[float] = None,
        store: Optional[bool] = None,
        background: bool = False,
    ) -> Union[FittedModel, FitGroup, Job]:
        """Fit once on our side and get a handle to predict (with intervals for regression) and explain as often as you like;
        each prediction sends only the rows to score. The fit is kept 30 days after its last use (each prediction renews it;
        ttl_days= sets 1 to 365) or until .delete(); `with yh.fit(...) as m:` deletes it on exit. Options as in predict().
        profile=True keeps a summary of the training table on the handle (ranges, categories, up to 2,000 sample rows), so
        predictions report out_of_range and novelty; it stays on your machine unless saved to a file (.save(), whose
        sample=False leaves the rows out). False drops it.
        target=[...] fits each target on the features alone (the other targets left out) and returns a FitGroup that predicts,
        explains and deletes them together; task= may then be a dict per target. One fit's quota per target.

        Args:
            mask_columns: Send column names as c0, c1, ...; results keep your names. Default: the client's setting
                (False).
            target: The column to predict, or a list of columns: one fit per target, returned as a FitGroup.
            ttl_days: How many days the fit is kept after its last use, 1 to 365. Default: 30.
            ttl_hours: The same in hours (ttl_days takes precedence).
            store: False: the call isn't shown in your Logs (see the data-handling page in the docs).
            background: Return a Job at once instead of waiting: job.wait() gives the fitted model. One target only.
            profile: Keep a summary of the training table on the handle for out_of_range and novelty (stays on your
                machine).
            time_col: A column of times: validation folds are contiguous periods of it. It stays a feature.
            eval_metric: What validation optimises: log_loss (default), roc_auc, f1, ... for classification; rmse
                (default), mae, r2 for regression.
            groups: A column whose rows belong together (machine, batch, patient, store): validation folds hold out
                whole groups, so intervals hold for groups the model has not seen. The column is not used as a feature.
            model: The model name, as GET /v1/models lists it.
            preset: "fast" (default: one holdout split, the quickest), "medium" (3-fold bagging, about 3x the time) or "pro" (8-fold bagging, the most accurate; "full" is the same).
            seed: Random seed; the server uses 0 when it is left out, so results are reproducible.
            task: "classify", "regress", or None to let the server infer it. Pass it for integer targets (a 1-5 rating
                could be either).
            time_limit: Seconds for the whole call.
            train: The training table: one row per example, the target column included.
        """
        train = self._table(train)
        if not isinstance(target, str):
            targets = _targets(target, train)
            opts = dict(
                model=model,
                preset=preset,
                eval_metric=eval_metric,
                time_limit=time_limit,
                seed=seed,
                ttl_hours=ttl_hours,
                ttl_days=ttl_days,
                store=store,
                profile=profile,
                groups=groups,
                time_col=time_col,
                mask_columns=mask_columns,
            )
            if background:
                raise DataError("background=True fits one target at a time.")
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
        m = Mask.of(tr.columns) if self._masking(mask_columns) else None
        wtr = m.frame(tr) if m else tr
        body = _tabular.fit_payload(
            model,
            wtr,
            m.name(target) if m else target,
            pt,
            preset,
            eval_metric,
            time_limit,
            seed,
            ttl_hours,
            self.limits.max_tabular_bytes,
            m.options(extra) if m else extra,
            self._refs(train=wtr),
        )
        _fit_extras(body, ttl_days, store)

        def finish(reply: Any) -> FittedModel:
            fm = FittedModel.from_reply(
                reply, _tabular.features_of(tr, target, groups), _tabular.label_values(tr[target]), self
            )
            fm.profile = _profile.build(tr, fm.features or []) if profile else None
            if m:
                fm.target, fm.mask = target, m.wire
            return fm

        try:
            with masked(m):
                if background:
                    job = self._t.request("POST", "/v1/table/fits", {**body, "async": True})
                else:
                    reply = _jobs.call(self._t, "/v1/table/fits", body, not self._direct(preset, tr, target, groups))
        except NotFoundError as e:
            if e.type == "not_found":
                raise _not_deployed(e) from e
            raise
        warn_all(warns)
        return Job(self._t, job, finish) if background else finish(reply)

    def get_fit(self, fit_id: str) -> FittedModel:
        """Reattach to a fit by id (labels come back as text unless the handle was pickled)."""
        return FittedModel.from_reply(self._t.request("GET", f"/v1/table/fits/{fit_id}"), None, [], self)

    def fits(self) -> List[Dict[str, Any]]:
        """This key's live fits: [{id, model, target, created_at, expires_at}]."""
        return self._t.request("GET", "/v1/table/fits")["data"]

    def impute(
        self,
        df: pd.DataFrame,
        columns: Cols = None,
        *,
        model: str = DEFAULT_TAB_MODEL,
        preset: Optional[str] = None,
        mask_columns: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Fills missing cells of a table: for each column with gaps, the model learns it from the other columns on the rows
        where it is present and predicts the rest (one call per column). The result's .attrs['filled'] marks what was filled.

        Args:
            mask_columns: Send column names as c0, c1, ...; results keep your names. Default: the client's setting
                (False).
            df: The table with missing values.
            columns: Which columns to fill. Default: every column with missing values.
            model: The model name, as GET /v1/models lists it.
            preset: "fast" (default: one holdout split, the quickest), "medium" (3-fold bagging, about 3x the time) or "pro" (8-fold bagging, the most accurate; "full" is the same).
        """
        df = self._table(df)
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
                mask_columns=mask_columns,
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
        """Series (forecasts) and credits (tables) used and left, this month and in total."""
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
        mask_columns: bool = False,
    ):
        super().__init__(
            AsyncTransport(api_key, base_url, timeout, max_retries, http_client), limits, max_concurrency, mask_columns
        )

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
        mask: bool = False,
    ):
        sem = asyncio.Semaphore(self.max_concurrency)

        async def one(b: List[Item]):
            async with sem:
                try:
                    body = _body(b, h, freq, model, season, extra)
                    m = _mask_series(b, body) if mask else None
                    with masked(m):
                        reply = await self._t.request("POST", "/v1/series/forecast", body)
                    return _unpack(b, reply, m), reply, None
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
        mask_columns: Optional[bool] = None,
        store: Optional[bool] = None,
    ) -> ForecastResult:
        t0 = time.monotonic()
        data, future, known = _with_calendar(data, future, h, calendar, known, id_col, time_col, freq, target_col)
        quantiles = _levels(quantiles, level)
        extra = _forecast_extra(self.limits, quantiles, samples, seed, history)
        if store is False:
            extra["store"] = False
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
        out, replies, errors = await self._send(
            prep.items, h, prep.freq, model, season, on_error, extra, self._masking(mask_columns)
        )
        res = _forecast_result(prep, model, quantiles, prep.items, out, replies, errors, mean)
        res.samples, res.history = _extras(prep, out, res)
        res.timing = {"total_s": round(time.monotonic() - t0, 3), "requests": len(replies)}
        res.provenance = _provenance(
            self.base_url,
            "/v1/series/forecast",
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

    async def explain_forecast(
        self,
        data: Any,
        h: int,
        *,
        known: Cols = None,
        past: Cols = None,
        future: Optional[pd.DataFrame] = None,
        model: str = DEFAULT_TS_MODEL,
        freq: Optional[str] = None,
        id_col: Optional[str] = None,
        time_col: Optional[str] = None,
        target_col: Target = None,
        static: Cols = None,
        season: Optional[int] = None,
        calendar: Calendar = None,
        mask_columns: Optional[bool] = None,
    ) -> pd.DataFrame:
        data, future = _pandas(data), _pandas(future)
        names = _explain.covariates(data, known, past)

        def run(d: Any, f: Any) -> Any:
            return self.forecast(
                d,
                h,
                known=known,
                past=past,
                future=f,
                model=model,
                freq=freq,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                static=static,
                season=season,
                calendar=calendar,
                mask_columns=mask_columns,
            )

        base, *rest = await asyncio.gather(
            run(data, future), *(run(*_explain.neutralise(data, future, c, id_col)) for c in names)
        )
        return _explain.effects(base, dict(zip(names, rest)))

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
        mask_columns: Optional[bool] = None,
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
            out, replies, _ = await self._send(items, h, prep.freq, model, season, mask=self._masking(mask_columns))
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
        compare: Any = None,
        store: Optional[bool] = None,
        background: bool = False,
        _refs: Optional[Dict[str, dict]] = None,
        mask_columns: Optional[bool] = None,
        _mask: Optional[Mask] = None,
    ) -> Union[PredictionResult, AsyncJob]:
        train, test = self._table(train), self._table(test)
        picked = _compare.check_models(compare) if compare else None
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
        m = _mask or (Mask.of([*tr.columns, *te.columns]) if self._masking(mask_columns) else None)
        wtr, wte = (m.frame(tr), m.frame(te)) if m else (tr, te)
        refs = _refs if _refs is not None else await self._refs(train=wtr, test=wte)
        body = _tabular.payload(
            model,
            wtr,
            m.name(target) if m else target,
            wte,
            pt,
            preset,
            eval_metric,
            time_limit,
            seed,
            self.limits.max_tabular_bytes,
            q,
            m.options(extra) if m else extra,
            refs,
        )
        if store is False:
            body["store"] = False
        t0 = time.monotonic()

        def finish(reply: Any, comparison: Optional[Comparison]) -> PredictionResult:
            res = _predict_result(
                model,
                reply,
                te,
                _tabular.label_values(tr[target]),
                warns,
                bool(q),
            )
            res.comparison = comparison
            res.timing["total_s"] = round(time.monotonic() - t0, 3)
            res.checks = _profile.Checks(_profile.build(tr, _tabular.features_of(tr, target, groups)), te)
            res.provenance = _provenance(
                self.base_url,
                "/v1/table/predict",
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

        cmp_body = _compare_body(body, picked) if compare else None
        if background:
            with masked(m):
                job = await self._t.request("POST", "/v1/table/predict", {**body, "async": True})
            comparison = await self._comparison(cmp_body, te.index) if cmp_body else None
            return AsyncJob(self._t, job, lambda reply: finish(reply, comparison))
        with masked(m):
            reply = await _jobs.acall(self._t, "/v1/table/predict", body, not self._direct(preset, tr, target, groups))
        return finish(reply, await self._comparison(cmp_body, te.index) if cmp_body else None)

    async def _comparison(self, body: dict, index: pd.Index) -> Comparison:  # type: ignore[override]
        return _compare.from_reply(await self._t.request("POST", "/v1/table/compare", body), index)

    async def compare(  # type: ignore[override]
        self,
        train: Any,
        target: str,
        test: Any,
        *,
        models: Any = None,
        model: str = DEFAULT_TAB_MODEL,
        task: Optional[str] = None,
        store: Optional[bool] = None,
        mask_columns: Optional[bool] = None,
    ) -> Comparison:
        """Client.compare."""
        body, index, m = self._compare_request(train, target, test, models, model, task, store, mask_columns)
        with masked(m):
            return await self._comparison(body, index)

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
        mask_columns: Optional[bool] = None,
    ) -> TableEvaluation:
        """Client.evaluate_table, with the folds run concurrently."""
        df = _tabular._frame(data, "data", self.limits)
        if target not in df.columns:
            raise DataError(f"target {target!r} is not a column. Columns: {list(df.columns)}")
        df = df[df[target].notna()]
        tk = _table_eval.infer_task(df[target], task)
        q = (list(_table_eval.DEFAULT_QUANTILES) if quantiles is None else quantiles) if tk == "regress" else None
        split = "time-ordered" if time_col else ("grouped" if groups is not None else "shuffled")
        drop = [groups] if isinstance(groups, str) else []
        splits = _table_eval.folds(df, cv, groups, time_col, seed)
        m = Mask.of(df.columns) if self._masking(mask_columns) else None
        full = _tabular.upload_candidate(m.frame(df) if m else df) if self.limits.supports("table_uploads") else None
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
                        _refs=_fold_refs(
                            base, tr, te, m.name(target) if m else target, [m.name(d) for d in drop] if m else drop
                        ),
                        _mask=m,
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
        mask_columns: Optional[bool] = None,
        ttl_days: Optional[float] = None,
        store: Optional[bool] = None,
        background: bool = False,
    ) -> Union[AsyncFittedModel, AsyncFitGroup, AsyncJob]:
        train = self._table(train)
        if not isinstance(target, str):
            targets = _targets(target, train)
            opts = dict(
                model=model,
                preset=preset,
                eval_metric=eval_metric,
                time_limit=time_limit,
                seed=seed,
                ttl_hours=ttl_hours,
                ttl_days=ttl_days,
                store=store,
                profile=profile,
                groups=groups,
                time_col=time_col,
                mask_columns=mask_columns,
            )
            if background:
                raise DataError("background=True fits one target at a time.")
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
        m = Mask.of(tr.columns) if self._masking(mask_columns) else None
        wtr = m.frame(tr) if m else tr
        body = _tabular.fit_payload(
            model,
            wtr,
            m.name(target) if m else target,
            pt,
            preset,
            eval_metric,
            time_limit,
            seed,
            ttl_hours,
            self.limits.max_tabular_bytes,
            m.options(extra) if m else extra,
            await self._refs(train=wtr),
        )
        _fit_extras(body, ttl_days, store)

        def finish(reply: Any) -> AsyncFittedModel:
            fm = FittedModel.from_reply(
                reply, _tabular.features_of(tr, target, groups), _tabular.label_values(tr[target]), self
            )
            fm.profile = _profile.build(tr, fm.features or []) if profile else None
            if m:
                fm.target, fm.mask = target, m.wire
            return AsyncFittedModel(**{k: getattr(fm, k) for k in fm.__dataclass_fields__})

        try:
            with masked(m):
                if background:
                    job = await self._t.request("POST", "/v1/table/fits", {**body, "async": True})
                else:
                    reply = await _jobs.acall(
                        self._t, "/v1/table/fits", body, not self._direct(preset, tr, target, groups)
                    )
        except NotFoundError as e:
            if e.type == "not_found":
                raise _not_deployed(e) from e
            raise
        warn_all(warns)
        return AsyncJob(self._t, job, finish) if background else finish(reply)

    async def models(self) -> List[Dict[str, Any]]:
        return (await self._t.request("GET", "/v1/models"))["models"]

    async def usage(self) -> Dict[str, Any]:
        return await self._t.request("GET", "/v1/usage")

    async def server_limits(self, apply: bool = True) -> Dict[str, Any]:
        lim = await self._t.request("GET", "/v1/limits")
        if apply:
            self.limits = Limits.from_api(lim)
        return lim
