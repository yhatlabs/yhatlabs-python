"""Turning tables into /v1/tabular/predict requests and replies into a PredictionResult.

The hosted endpoint is stateless fit-and-predict: every call carries the training rows and the rows to score. Two calling
styles map onto it: train + test (the usual), or one table where the rows with a missing target are the ones to predict.
"""

from __future__ import annotations

import base64
import io
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._errors import DataError, UnsupportedError
from ._forecast import Limits, qname

# Metric names as the API checks them: lowercase, digits allowed (f1, r2, f1_macro).
EVAL_METRIC = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
EXACT_INT = 2**53  # larger ints lose precision as JSON numbers on the way through the gateway

TASKS = ("classify", "regress")
PRESETS = ("fast", "medium", "full")
PARQUET_ABOVE_BYTES = 4 * 1024 * 1024  # JSON rows below this; parquet (when pyarrow is installed) above


def _cell(v: Any) -> Any:
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v) if abs(int(v)) < EXACT_INT else str(v)
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        ts = pd.Timestamp(v)
        return (ts.tz_convert("UTC") if ts.tzinfo is not None else ts).isoformat()
    if isinstance(v, pd.Timedelta):
        return v.total_seconds()
    if isinstance(v, str):
        return v
    return str(v)


def to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    cols = [str(c) for c in df.columns]
    return [{c: _cell(v) for c, v in zip(cols, row)} for row in df.itertuples(index=False, name=None)]


UPLOAD_ABOVE_BYTES = (
    8 * 1024 * 1024
)  # parquet files above this go to POST /v1/uploads (when the API offers it) instead of the body


def parquet_bytes(df: pd.DataFrame) -> Optional[bytes]:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return None
    buf = io.BytesIO()
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    out.reset_index(drop=True).to_parquet(buf, index=False)
    return buf.getvalue()


def _parquet_b64(df: pd.DataFrame) -> Optional[dict]:
    raw = parquet_bytes(df)
    return None if raw is None else {"format": "parquet_b64", "bytes": base64.b64encode(raw).decode("ascii")}


def upload_candidate(df: pd.DataFrame) -> Optional[bytes]:
    """The table's parquet bytes when it is large enough to upload rather than send inline, else None."""
    sample = to_records(df.head(200))
    if len(json.dumps(sample, default=str)) * max(1.0, len(df) / max(1, len(sample))) < UPLOAD_ABOVE_BYTES:
        return None
    raw = parquet_bytes(df)
    return raw if raw is not None and len(raw) > UPLOAD_ABOVE_BYTES else None


def encode_table(df: pd.DataFrame) -> Tuple[Any, float]:
    """The table as the API takes it, and its (estimated) encoded size in bytes. The size is estimated from a sample first, so a
    large table goes straight to parquet without building every JSON row.
    """
    sample = to_records(df.head(200))
    size = len(json.dumps(sample, default=str)) * max(1.0, len(df) / max(1, len(sample)))
    if size > PARQUET_ABOVE_BYTES:
        pq = _parquet_b64(df)
        if pq is not None:
            return pq, len(pq["bytes"])
    return to_records(df), size


def _frame(x: Any, what: str) -> pd.DataFrame:
    from ._forecast import _pandas

    x = _pandas(x)
    if isinstance(x, pd.DataFrame):
        return x
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], dict):
        return pd.DataFrame(list(x))
    if isinstance(x, np.ndarray) and x.ndim == 2:
        return pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
    raise DataError(f"{what} must be a pandas DataFrame (or a list of row dicts / a 2-D numpy array).")


def _str_columns(df: pd.DataFrame, what: str) -> pd.DataFrame:
    out = df.rename(columns=str)
    dup = sorted({c for c in out.columns if list(out.columns).count(c) > 1})
    if dup:
        raise DataError(f"{what} has duplicate column names {dup} once names are converted to text; rename them.")
    return out


def problem_type(y: pd.Series, task: Optional[str]) -> Optional[str]:
    if task is not None and task not in TASKS:
        raise DataError(f"task must be one of {TASKS} or None (infer); got {task!r}.")
    if (
        pd.api.types.is_datetime64_any_dtype(y)
        or pd.api.types.is_timedelta64_dtype(y)
        or pd.api.types.is_complex_dtype(y)
    ):
        raise DataError(
            f"The target has dtype {y.dtype}, which is neither numbers nor classes; convert it (e.g. to days or a number) first."
        )
    obs = y.dropna()
    if len(obs) and obs.map(lambda v: isinstance(v, (list, dict, set, tuple))).any():
        raise DataError("The target holds lists or dicts; each row needs one number or one class label.")
    n = obs.nunique()
    if n < 2:
        raise DataError(
            f"The target has {n} distinct value{'s' if n != 1 else ''} in train; there is nothing to learn."
        )
    if task is None:
        return None  # the server infers it
    if task == "regress":
        return "regression"
    return "binary" if n == 2 else "multiclass"


def check_options(preset: Optional[str], eval_metric: Optional[str]) -> None:
    if preset is not None and preset not in PRESETS:
        raise DataError(f"preset must be one of {PRESETS}; got {preset!r}.")
    if eval_metric is not None and not EVAL_METRIC.match(eval_metric):
        raise DataError(
            f"eval_metric {eval_metric!r} is not a metric name. Use e.g. log_loss, roc_auc, accuracy, f1, balanced_accuracy "
            "for classification; rmse, mae, r2 for regression."
        )


def check_quantiles(quantiles: Any, pt: Optional[str]) -> Optional[List[float]]:
    if quantiles is None:
        return None
    if pt in ("binary", "multiclass"):
        raise DataError("quantiles= applies to regression only; classification returns class probabilities.")
    levels = sorted({float(q) for q in quantiles})
    if not levels or len(levels) > 32 or not all(0 < q < 1 for q in levels):
        raise DataError("quantiles must be 1 to 32 levels strictly between 0 and 1, e.g. [0.05, 0.5, 0.95].")
    return levels


def prepare_train(
    train: Any, target: str, *, task: Optional[str], limits: Limits
) -> Tuple[pd.DataFrame, Optional[str], List[str]]:
    """Validate the training table; returns (train with an observed target, problem_type, warnings)."""
    warns: List[str] = []
    train = _str_columns(_frame(train, "train"), "train")
    if target not in train.columns:
        raise DataError(f"target {target!r} is not a column of train. Columns: {list(train.columns)}")
    missing = train[target].isna()
    if missing.any():
        warns.append(f"Dropped {int(missing.sum())} train rows with a missing target.")
        train = train[~missing]
    if len(train) == 0:
        raise DataError("train has no rows with an observed target.")
    if len(train) > limits.max_tabular_rows:
        raise DataError(
            f"train has {len(train)} rows; the API takes at most {limits.max_tabular_rows} per call. Subsample it."
        )
    return train, problem_type(train[target], task), warns


def prepare_test(test: Any, features: List[str], target: str, limits: Limits) -> Tuple[pd.DataFrame, List[str]]:
    """Validate the rows to score against the training columns; keeps the caller's index."""
    warns: List[str] = []
    test = _str_columns(_frame(test, "test"), "test")
    if target in test.columns:
        if test[target].notna().any():
            warns.append(f"test has the target column {target!r}; it was dropped before sending.")
        test = test.drop(columns=[target])
    if len(test) == 0:
        raise DataError("test has no rows.")
    lacking = [c for c in features if c not in test.columns]
    if lacking:
        raise DataError(f"test is missing feature columns {lacking} that train has.")
    extra = [c for c in test.columns if c not in features]
    if extra:
        warns.append(f"Ignored test columns {extra} that train does not have.")
    if len(test) > limits.max_tabular_rows:
        raise DataError(
            f"test has {len(test)} rows; the API takes at most {limits.max_tabular_rows} per call. Split it."
        )
    return test[features], warns


def split_options(
    train: pd.DataFrame,
    target: str,
    groups: Optional[str],
    time_col: Optional[str],
    set_coverage: Any,
    pt: Optional[str],
) -> Dict[str, Any]:
    """groups= / time_col= (validation folds that hold out whole groups or periods) and set_coverage= (classification
    prediction sets), checked against the training table; -> the request fields.
    """
    out: Dict[str, Any] = {}
    if groups is not None and time_col is not None:
        raise DataError("Pass groups= or time_col=, not both.")
    for name, col in (("groups", groups), ("time_col", time_col)):
        if col is None:
            continue
        if not isinstance(col, str) or col not in train.columns:
            raise DataError(f"{name}={col!r} must name a column of train. Columns: {list(train.columns)}")
        if col == target:
            raise DataError(f"{name} cannot be the target.")
        out[name] = col
    if groups is not None:
        if train[groups].isna().any():
            raise DataError(f"groups column {groups!r} has missing values; every row needs its group.")
        if train[groups].nunique() < 2:
            raise DataError(f"groups column {groups!r} needs at least 2 groups.")
    cov = check_coverage(set_coverage, pt)
    if cov is not None:
        out["set_coverage"] = cov
    return out


def check_coverage(set_coverage: Any, pt: Optional[str]) -> Optional[float]:
    if set_coverage is None:
        return None
    if isinstance(set_coverage, bool) or not isinstance(set_coverage, (int, float)) or not 0 < float(set_coverage) < 1:
        raise DataError("set_coverage must be a number strictly between 0 and 1, e.g. 0.9.")
    if pt == "regression":
        raise DataError("set_coverage= applies to classification; for regression use quantiles=.")
    return float(set_coverage)


def features_of(train: pd.DataFrame, target: str, groups: Optional[str] = None) -> List[str]:
    """The feature columns: everything but the target, and but the groups column (a group id is not used as a feature)."""
    return [c for c in train.columns if c != target and c != groups]


def prepare(
    train: Any,
    target: str,
    test: Any,
    *,
    task: Optional[str],
    preset: Optional[str],
    quantiles: Any,
    limits: Limits,
    eval_metric: Optional[str] = None,
    groups: Optional[str] = None,
    time_col: Optional[str] = None,
    set_coverage: Any = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str], Optional[List[float]], List[str], Dict[str, Any]]:
    """One-shot fit-and-predict inputs; returns (train, test, problem_type, quantile levels, warnings, extra request fields)."""
    check_options(preset, eval_metric)
    if test is None:
        full = _str_columns(_frame(train, "train"), "train")
        if target not in full.columns:
            raise DataError(f"target {target!r} is not a column of train. Columns: {list(full.columns)}")
        missing = full[target].isna()
        if not missing.any():
            raise DataError(
                "Pass test= (the rows to predict), or leave the target empty in the rows you want predicted."
            )
        train, test = full[~missing], full[missing].drop(columns=[target])
    tr, pt, warns = prepare_train(train, target, task=task, limits=limits)
    extra = split_options(tr, target, groups, time_col, set_coverage, pt)
    te, w2 = prepare_test(test, features_of(tr, target, groups), target, limits)
    return tr, te, pt, check_quantiles(quantiles, pt), warns + w2, extra


def _check_size(sizes: List[float], parquet: bool, max_bytes: Optional[int]) -> None:
    total = sum(sizes)
    if max_bytes is not None and total > max_bytes:
        hint = "subsample train" if parquet else "install pyarrow (sent as parquet, much smaller) or subsample train"
        raise DataError(
            f"The request is about {total / 2**20:.0f} MB; the API takes at most {max_bytes / 2**20:.0f} MB. {hint.capitalize()}."
        )


def _options(
    body: dict,
    pt: Optional[str],
    preset: Optional[str],
    eval_metric: Optional[str],
    time_limit: Optional[float],
    seed: Optional[int],
) -> dict:
    if preset is not None:
        body["preset"] = preset
    if pt is not None:
        body["problem_type"] = pt
    if eval_metric is not None:
        body["eval_metric"] = eval_metric
    if time_limit is not None:
        body["time_limit"] = float(time_limit)
    if seed is not None:
        body["seed"] = int(seed)
    return body


def payload(
    model: str,
    train: pd.DataFrame,
    target: str,
    test: pd.DataFrame,
    pt: Optional[str],
    preset: Optional[str],
    eval_metric: Optional[str],
    time_limit: Optional[float],
    seed: Optional[int] = None,
    max_bytes: Optional[int] = None,
    quantiles: Optional[List[float]] = None,
    extra: Optional[Dict[str, Any]] = None,
    refs: Optional[Dict[str, dict]] = None,
) -> dict:
    refs = refs or {}
    tr, tr_size = (refs["train"], 0.0) if "train" in refs else encode_table(train)
    te, te_size = (refs["test"], 0.0) if "test" in refs else encode_table(test)
    _check_size([tr_size, te_size], isinstance(tr, dict), max_bytes)
    body = _options(
        {"model": model, "target": target, "train": tr, "test": te}, pt, preset, eval_metric, time_limit, seed
    )
    if quantiles:
        body["quantiles"] = quantiles
    body.update(extra or {})
    return body


def fit_payload(
    model: str,
    train: pd.DataFrame,
    target: str,
    pt: Optional[str],
    preset: Optional[str],
    eval_metric: Optional[str],
    time_limit: Optional[float],
    seed: Optional[int],
    ttl_hours: Optional[float],
    max_bytes: Optional[int],
    extra: Optional[Dict[str, Any]] = None,
    refs: Optional[Dict[str, dict]] = None,
) -> dict:
    tr, size = (refs["train"], 0.0) if refs and "train" in refs else encode_table(train)
    _check_size([size], isinstance(tr, dict), max_bytes)
    body = _options({"model": model, "target": target, "train": tr}, pt, preset, eval_metric, time_limit, seed)
    if ttl_hours is not None:
        body["ttl_hours"] = float(ttl_hours)
    body.update(extra or {})
    return body


def test_payload(
    test: pd.DataFrame,
    quantiles: Optional[List[float]],
    max_bytes: Optional[int],
    set_coverage: Optional[float] = None,
    ref: Optional[dict] = None,
) -> dict:
    te, size = (ref, 0.0) if ref is not None else encode_table(test)
    _check_size([size], isinstance(te, dict), max_bytes)
    body: dict = {"test": te}
    if quantiles:
        body["quantiles"] = quantiles
    if set_coverage is not None:
        body["set_coverage"] = float(set_coverage)
    return body


def parse_mean(reply: dict, test: pd.DataFrame) -> Optional[pd.Series]:
    m = reply.get("mean")
    return None if m is None else pd.Series(np.asarray(m, dtype=float), index=test.index, name="mean")


def parse_sets(reply: dict, test: pd.DataFrame, labels: List[Any]) -> Optional[pd.Series]:
    sets = reply.get("prediction_sets")
    if sets is None:
        return None
    return pd.Series([_restore_labels(list(s), labels) for s in sets], index=test.index, name="prediction_set")


def _key(v: Any) -> str:
    """One spelling per label across the JSON round trip: 1, 1.0, '1' and np.int64(1) all match; True stays apart from 1."""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, str):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(f)) if f.is_integer() else repr(f)


def label_values(train_y: pd.Series) -> List[Any]:
    """The distinct training labels as plain Python values: what predictions are mapped back to."""
    return [v.item() if isinstance(v, np.generic) else v for v in pd.unique(train_y.dropna())]


def _restore_labels(values: List[Any], labels: List[Any]) -> List[Any]:
    """The server returns labels as JSON; map them back to the training labels' values and type (1.0 rather than '1')."""
    lookup = {_key(v): v for v in labels}
    return [lookup.get(_key(v), v) for v in values]


def parse(
    reply: dict, test: pd.DataFrame, labels: List[Any], want_quantiles: bool = False
) -> Tuple[pd.Series, Optional[pd.DataFrame], Optional[List[Any]], Optional[pd.DataFrame]]:
    """-> (predictions, probabilities, classes, quantiles), all indexed like `test`."""
    preds = reply["predictions"]
    if len(preds) != len(test):
        raise DataError(f"The API returned {len(preds)} predictions for {len(test)} test rows.")
    classes = reply.get("classes")
    probs = reply.get("probabilities")
    if classes is not None and probs is not None:
        classes = _restore_labels(classes, labels)
        preds = _restore_labels(preds, labels)
        proba = pd.DataFrame(np.asarray(probs, dtype=float), index=test.index, columns=classes)
        return pd.Series(preds, index=test.index, name="prediction"), proba, classes, None
    try:
        out = pd.Series(np.asarray(preds, dtype=float), index=test.index, name="prediction")
    except (TypeError, ValueError):  # labels without probabilities (the server inferred a classification task)
        return pd.Series(_restore_labels(preds, labels), index=test.index, name="prediction"), None, None, None
    q = None
    if want_quantiles and reply.get("quantiles") is None:
        raise UnsupportedError("This API does not return prediction intervals yet; call without quantiles=.")
    if reply.get("quantiles") is not None:
        levels = reply["quantile_levels"]
        q = pd.DataFrame(
            np.asarray(reply["quantiles"], dtype=float), index=test.index, columns=[qname(lv) for lv in levels]
        )
    return out, None, None, q


def estimate(rates: Dict[str, Any], train_rows: int, features: int, preset: str, predict_rows: int) -> Dict[str, Any]:
    """Credits of a fit and a prediction under the API's published rates (GET /v1/limits, pricing.tabular)."""
    if not rates:
        raise UnsupportedError(
            "This API publishes no tabular pricing; the estimate needs GET /v1/limits to include it."
        )
    lo = int(rates.get("min_credits", 1))
    mult = (rates.get("preset_multiplier") or {}).get(preset, 1)
    fit = max(lo, math.ceil(train_rows * max(1, features) / rates["fit_cells_per_credit"])) * mult if train_rows else 0
    pred = max(lo, math.ceil(predict_rows / rates["predict_rows_per_credit"])) if predict_rows else 0
    return {"fit": fit, "predict": pred, "total": fit + pred, "unit": rates.get("unit", "credits")}
