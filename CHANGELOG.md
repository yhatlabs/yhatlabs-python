# Changelog

## 0.2.0 (2026-10-15)

- `to_polars()` without polars installed says what to install instead of raising a bare `ModuleNotFoundError`.
- A model that is starting up (its first call after a quiet period) is waited for, up to 5 minutes, instead of failing
  after the usual retries.
- **Breaking:** the API paths name the input, then the task: `POST /v1/series/forecast`, `POST /v1/table/predict` and
  `/v1/table/fits/...`. Releases up to 0.1.1 call the old paths, which the API no longer serves.
- `mask_columns=True` (on `Client`, `with_options` or any call that sends a table or series) sends column names as `c0`, `c1`,
  ...; results, stored fits, saved fit records and error messages keep your names. Values, category labels and series ids
  are sent as they are. Off by default.
- Fixed: `calendar=` failed when the time or target column had a custom name (for example `Date` and `Temp`).
- Fixed: `evaluate(compare=...)` on a single series failed when the compared forecasts had no id column.
- The CLI prints warnings as one clean line (`yhatlabs: warning: ...`) instead of Python's warning format.
- Bad input now fails before anything is sent, with a message that says what to change:
  - an empty table;
  - timestamps so irregular that more than half of the regular grid would be missing (0.1.1 filled the gaps and warned);
  - rows without a series id (0.1.1 dropped them silently);
  - more targets or covariates than the API takes (`max_columns` from `/v1/limits`);
  - a tabular target holding dates, durations, complex numbers, lists or dicts.
- Integer time columns are steps (1, 2, 3, ...): the forecast continues them, and `freq=` says what one step is. 0.1.1 read
  them as nanoseconds since 1970.
- A 2-D array or a list of lists is one series per row (`series_0`, `series_1`, ...). 0.1.1 flattened it into one series.
- Infinite values in a target are reported in a warning; they are still sent as missing.
- Every POST carries an `Idempotency-Key` header that stays the same across its retries, so the API can recognise a retry.
- `Limits` has `max_columns` and `capabilities`, read from `/v1/limits`. `limits.supports(name)` tells whether the API has
  switched a feature on.
- Request bodies over 1 MB are sent gzip-compressed once the API lists the `gzip_requests` capability.
- Forecasts:
  - `forecast(..., mean=True)` adds a `mean` column, computed from the quantiles (listed in `derived`). Use it when forecasts
    are summed later; the median stays the point forecast.
  - `fc.intervals(80)` returns `lower_80` / `upper_80` next to the median; `fc.to_wide()` gives one column per series.
  - `fc.prob(above=limit)` / `fc.prob(below=limit)`: the chance of crossing a limit at each step.
- Forecasts, uncertainty (each switched on when the API lists it in its capabilities):
  - any quantile levels, including 95% and 99% intervals (`level=[95]`, `quantiles=[0.01, ...]`);
  - `samples=N` (and `seed=`): sample paths over the horizon in `fc.samples`, `fc.sample_paths(id)`; sums, peaks and counts
    over the horizon get honest intervals from them;
  - `history=True`: the model's own forecasts of the last windows before the forecast, with their cutoffs (`fc.history`).
- Tables:
  - `yh.evaluate_table(df, target, cv=5, groups=..., time_col=...)` cross-validates on your data: shuffled, grouped (a group
    never sits on both sides) or time-ordered folds. It reports accuracy, log loss, AUC, Brier and calibration error with a
    reliability table, or MAE, RMSE, R2, CRPS, coverage per interval and a PIT histogram.
  - Predictions carry `out_of_range` (columns outside the training range, or unseen categories) and `novelty` (how far a row
    is from the training rows, as a percentile). Fits keep a training profile for this (`fit(..., profile=False)` drops it);
    it is saved with `.save()`.
  - `res.prob(above=limit)` on regression predictions with intervals.
  - `model.effect(col)`: how the prediction moves as one column varies (partial dependence, with the spread across rows).
  - `yh.fit(train, target=["a", "b"])` fits each target on the features alone and returns a `FitGroup`; its `predict`,
    `importance`, `effect`, `save` and `delete` cover every target.
- Tables, uncertainty:
  - Regression intervals come from each row's own predictive distribution where the API offers it (`res.interval_method`
    is `"distribution"`), so an easy row gets a narrow interval and a hard or unusual one a wide one; `res.mean` is that
    distribution's mean.
  - `groups="machine"` or `time_col="date"` on `fit` and `predict`: the model's own validation holds out whole groups or
    periods, so its intervals hold for groups it has not seen (random folds overstate them).
  - `set_coverage=0.9` on classification: `res.prediction_sets` lists, per row, the classes that together hold the true one
    90% of the time.
  - `yh.estimate_table(...)`: the credits a tabular call will cost, from the rates the API publishes.
  - Tables over 8 MB (as parquet) are uploaded once, by content hash, and referred to by id; `evaluate_table` uploads a large
    table once for all its folds. Used when the API lists the `table_uploads` capability.
- `yhatlabs.hierarchy`: `aggregate()` builds every level's series (stores into regions into a total) and `reconcile()` makes the
  forecasts add up across the levels.
- Plots: `yhatlabs.plots` has forecast (nested interval bands), backtest, anomalies, scenarios, calibration, reliability,
  predicted against actual, importance and effect. `EvaluationResult.plot()` and `TableEvaluation.plot()` use them.

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
