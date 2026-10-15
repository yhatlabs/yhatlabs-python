# Changelog

## Unreleased

- `orjson` is now a default dependency, for faster encoding of large requests (the `[fast]` extra now adds pyarrow only).

## 0.1.1 (2026-10-15)

- Fixed: requests failed without the optional `orjson` package (a plain `pip install yhatlabs`). The standard-library JSON
  fallback is now used correctly.
- CI runs the full test suite on an install without any extras; tests that need an extra skip themselves.
- `pandas.util.hash_pandas_object` is imported explicitly rather than through the `pd.util` attribute.

## 0.1.0 (2026-10-15)

- **Data checks:** `yhatlabs.check()` runs locally for series and tables, gives findings with suggested fixes, and
  `.repaired()` returns the cleaned series.
- **Fit records:** `FittedModel.save()` and `yhatlabs.load()` reuse a fit in another process or on another machine.
- **Client settings:** `Client.with_options()`, `Client.whoami()`, and limits read lazily from the API.
- **Visibility:**
  - `YHAT_LOG` request logging;
  - a terminal activity indicator (`YHAT_PROGRESS=0` turns it off);
  - `.provenance` and `.timing` on results.
- **Forecasting:** `calendar=` for forecast, evaluate and detect_anomalies; `EvaluationResult.calibration` and
  `plot_calibration()`.
- **Polars:** polars input, and `.to_polars()` on results.
- **CLI:** `yhatlabs check`, `whoami`, `skills install` (agent skill), and `--json` output.

- **Fit once, predict many:** `Client.fit` returns a `FittedModel` with `predict`, `predict_proba`, `importance`, `delete`,
  and a context manager. The scikit-learn estimators use it, and fall back to one-shot calls on an API without `/v1/fits`.
- **Regression prediction intervals** (`quantiles=`), from split-conformal out-of-sample errors.
- **Permutation importance** per input column, and `Client.impute` for filling missing cells in tables.
- **Time series:**
  - `static=` covariates and the `level=` alias;
  - `scenarios` (what-if) and `detect_anomalies`;
  - `fill_gaps`;
  - `evaluate(compare=...)`, to score your own forecasts alongside;
  - `on_error="warn"` for partial results;
  - `AsyncClient.evaluate`;
  - `ForecastResult.plot()`.
- `yhatlabs.features`: `add_calendar` (with holidays) and `make_future`.
- The `yhat` command line.
- **Plumbing:**
  - orjson encoding when installed;
  - large tables go to parquet without building JSON rows first;
  - `server_limits()` reads `/v1/limits`;
  - asking for intervals from an API that does not return them raises an error.

- `Client` / `AsyncClient` with `forecast`, `predict`, `models`, `usage`; `Client.evaluate` for rolling-origin backtests.
- Client-side time handling: frequency inference, gap filling, future timestamps, known/past covariate alignment, batching.
- Quantile selection with interpolation between native levels (reported in `derived`).
- scikit-learn estimators `ChakraTabClassifier` and `ChakraTabRegressor`.
- Typed errors with request ids; retries with backoff on transient failures.
- Tabular: float class labels survive the JSON round trip; local checks for one-class targets, duplicate column names,
  eval_metric names and the 64 MB body cap; huge ints sent as text; tz-aware timestamps sent as UTC.
- Errors streamed in a 200 body are raised, and retried once at most (a long fit is not re-run several times).
