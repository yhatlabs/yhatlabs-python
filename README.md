# YHat Labs Python Library

The official Python library for the YHat Labs time-series and tabular foundation models. Frontier AI models for
forecasting, anomaly detection, classification, regression and feature importance, built for production
workloads: no feature engineering or model training, with uncertainty and explanations built in.

- **Time-series forecasting** with **ChakraTS**: a drop-in alternative to ARIMA, Prophet or a trained deep-learning model.
  It forecasts one series or thousands, with a range of likely outcomes around every point.
- **Tabular prediction** with **ChakraTab**: classification and regression on tables, typically more accurate than tuned
  gradient boosting such as XGBoost or LightGBM on public benchmarks.

Both are pretrained foundation models: they work on your data as it is, with nothing to train, tune or maintain.

## Installation

```sh
pip install yhatlabs
```

Optional extras: `[sklearn]` for the scikit-learn estimators, `[parquet]` for faster large tables, `[plot]` for charts,
`[fast]` for faster large uploads, or `[all]`. Requires Python 3.9 or later.

## Usage

The full documentation of the models and the API is in the [Docs](https://yhatlabs.com/models/docs).

Every call needs an API key. [Get an API key here](https://yhatlabs.com/models/access), then set it in your environment:

```sh
export YHAT_API_KEY="<your-key>"
```

The client reads it from there, so it never ends up in your code. You can also pass `Client(api_key=...)`, or keep it in a
`.env` file loaded with [python-dotenv](https://pypi.org/project/python-dotenv/).

To use your own deployment instead of the hosted API, set its address:

```sh
export YHAT_BASE_URL="https://your-deployment.example.com"
```

Pick what you want to do:

- **[Time-series forecasting](#time-series-forecasting)**: forecasts with uncertainty, backtests, what-if scenarios, anomalies
  and gaps.
- **[Tabular prediction](#tabular-prediction)**: classification and regression, prediction intervals, column importance,
  scikit-learn estimators.
- **[Check your data first](#check-your-data-first)**: local checks before anything is sent.
- **[Advanced usage](#advanced-usage)**: the command line, coding agents, async, configuration and errors.

## Time-series forecasting

ChakraTS forecasts one series or many at once. Every forecast comes with an uncertainty range, so you see both the most
likely path and how far reality could stray from it.

```python
import pandas as pd
from yhatlabs import Client

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
df = pd.read_csv(url + "airline-passengers.csv")

yh = Client()
fc = yh.forecast(
    df, h=24, time_col="Month", target_col="Passengers", level=[80]
)
print(fc.df.head(3))
```

```
  unique_id      Month    q10  median    q90
0  series_0 1961-01-01  430.5   448.5  468.0
1  series_0 1961-02-01  408.9   430.9  452.0
2  series_0 1961-03-01  449.0   486.7  508.2
```

![Airline passengers forecast with an 80% interval](https://raw.githubusercontent.com/yhatlabs/yhatlabs-python/main/docs/images/forecast.png)

For many series, add an id column (`unique_id`, `id` or `item_id`). The time column (`ds`, `timestamp`, `date`) and the target
(`y`, `target`, `value`) are found by name as well. The client parses timestamps, infers the frequency, fills missing
timestamps and splits large inputs into requests for you.

### What you can do

| Goal | Call |
|---|---|
| Forecast with a range of outcomes | `yh.forecast(df, h)` |
| Wide intervals for rare outcomes (95%, 99%) | `yh.forecast(df, h, level=[95, 99])` |
| Totals, peaks and runs over the horizon | `yh.forecast(df, h, samples=1000)`, then `fc.sample_paths()` |
| The chance of crossing a limit | `fc.prob(above=limit)` |
| Use inputs you know ahead (prices, weather, promotions) | `yh.forecast(df, h, known=[...], future=...)` |
| Add calendar effects (hour, weekday, holidays) | `yh.forecast(df, h, calendar=True)` |
| Measure accuracy on your own history | `yh.evaluate(df, h, windows=3)` |
| Compare with the forecast you use today | `yh.evaluate(df, h, compare={"current": mine})` |
| Ask "what if" (a heatwave, a price change) | `yh.scenarios(df, h, {"base": f0, "heatwave": f1}, known=[...])` |
| Flag unusual points | `yh.detect_anomalies(df)` |
| Flag unusual points as they arrive | `stream = yh.anomaly_stream()`, then `stream.update(new_points)` |
| See how much each input moves the forecast | `yh.explain_forecast(df, h, known=[...], future=...)` |
| Fill gaps inside series | `yh.fill_gaps(df)` |
| Forecasts that add up (stores, regions, total) | `yhatlabs.hierarchy.aggregate(...)`, then `hierarchy.reconcile(fc, tree)` |

What a forecast used is in `fc.usage`: credits for keys of an organization, series for keys without one.

### Inputs and uncertainty

Forecasts get better when you tell the model what you already know about the future, and more useful when the uncertainty
matches your decision.

```python
from yhatlabs.features import make_future

# the next 24 timestamps of every series, to fill in with what you know ahead
future = make_future(df, h=24)
future["temperature"] = weather_forecast

fc = yh.forecast(
    df,
    h=24,
    known=["temperature"],
    future=future,
    calendar=True,
    quantiles=[0.05, 0.5, 0.95],
)
fc.plot(history=df)
```

One table works too: rows after a series' last value, with the target blank and the `known=` columns filled in, are taken
as its future, so `yh.forecast(df, h=24, known=["temperature"])` needs no `future=`.

- **Uncertainty:** the default is the median plus the 0.1 and 0.9 quantiles.
  - `quantiles=` takes any levels strictly between 0 and 1, and `level=[80, 95, 99]` gives central intervals by width.
  - Intervals are calibrated on each series' own past windows: an 80% interval aims to hold 80% of the actual values.
    Check it on your data with `yh.evaluate` (its calibration table).
  - `quantiles="all"` returns every level the model produces; levels between those are interpolated and listed in
    `fc.derived`.
  - `mean=True` adds the mean, for forecasts you will add up; `samples=N` returns sample paths; `history=True` returns
    the model's forecasts of earlier windows.
- **Covariates:**
  - `known=[...]` are columns you know ahead, with their future values in `future=`;
  - `past=[...]` are observed only in the history;
  - `static=[...]` are constants per series, such as region or store size;
  - `calendar=True` adds hour, weekday, month and weekend, with no future frame needed; `calendar={"holidays": "US"}` adds
    holidays.
- **Several targets per series:** `target_col=["load", "price"]` forecasts them jointly.

### Backtesting

Before relying on a forecast, measure it on your own history: what would it have predicted at past points, and how close
was it?

```python
ev = yh.evaluate(
    df, h=24, windows=4, time_col="Month", target_col="Passengers"
)
ev.summary      # accuracy per model, and skill vs seasonal naive
ev.calibration  # does the 80% interval really hold 80%?
```

- **How it works:** `evaluate` cuts each series at several points near its end, forecasts from each cut using only the data
  before it, and scores the forecasts against what happened.
- **Skill:** `skill_<metric>` compares with the first baseline. Above 0 means the model beats it.
- **Your own forecast:** `compare={"current": df}` scores it alongside. It takes id and time plus quantile columns
  (`median`, `q10`, ...) or a single `forecast` column.
- **Calibration:** `ev.calibration` reports the observed share below each quantile and inside each interval, next to the
  nominal level. `ev.plot_calibration()` draws it.

### Anomalies and gaps

The same model that forecasts can tell you which recent points it did not expect, and what a missing value most likely was.

```python
# a score and an anomaly flag for each recent point
flags = yh.detect_anomalies(df, h=24, windows=5)

# missing values filled from the history before each gap
filled = yh.fill_gaps(df)
```

- **Scores:** each point is compared with a forecast made from the data before it. The score is the distance from the median
  in units of that forecast's own spread, so a point the model was unsure about needs a larger miss to count. `threshold`
  (default 3) sets what gets flagged.
- **Gaps:** each gap gets the median forecast from the data before it, and `<target>_filled` marks what was filled.

## Tabular prediction

ChakraTab learns the relationship between the columns of a table from your labeled rows, then predicts the answer for new
rows. It handles numbers, categories, dates and missing values as they are.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from yhatlabs import Client

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

yh = Client()
model = yh.fit(X_train.assign(diagnosis=y_train), target="diagnosis")
res = model.predict(X_test)
print(res.df.head(3))
```

```
     prediction    p_0    p_1
512           0  1.000  0.000
457           1  0.013  0.987
439           1  0.000  1.000
```

The fit happens once, on our side; every `predict` after it sends only the new rows. Each prediction from a saved model
is charged on the training table's size (0.5 x its cells) plus the predicted cells. For a number instead of a class, ask
for an interval as well:

```python
model = yh.fit(train, target="progression", task="regress")
# prediction, q5, median and q95 for every test row
res = model.predict(test, quantiles=[0.05, 0.5, 0.95])

# which columns drive the predictions
model.importance(holdout)
```

![Predictions with 90% intervals on held-out rows, and column importance](https://raw.githubusercontent.com/yhatlabs/yhatlabs-python/main/docs/images/tabular.png)

### What you can do

| Goal | Call |
|---|---|
| Classify (with probabilities) or regress | `yh.fit(train, target).predict(test)` |
| Get prediction intervals for numbers | `model.predict(test, quantiles=[0.05, 0.95])` |
| Get sets of likely classes | `model.predict(test, set_coverage=0.9).prediction_sets` |
| Validate on new machines, patients or periods | `yh.fit(train, target, groups="machine")` or `time_col="date"` |
| Measure accuracy and calibration on your table | `yh.evaluate_table(df, target, cv=5)` |
| See how one column moves the prediction | `model.effect("temperature")` |
| See which columns matter | `model.importance(holdout)` |
| Find rows unlike the training rows | `yhatlabs.outliers(train, test)` |
| Reuse a fit later, anywhere | `model.save("model.json")`, then `yhatlabs.load("model.json")` |
| Predict once, without saving a fit | `yh.predict(train, target, test)` |
| Fill missing cells | `yh.impute(df)` |
| Use it in scikit-learn | `ChakraTabClassifier()`, `ChakraTabRegressor()` |
| Compare with standard models on the same rows | `yh.predict(train, target, test, compare=True)`, then `res.comparison.score(actual, ChakraTab=res)` |
| Standard models alone, on the same split | `yh.compare(train, target, test)` |
| Run a large table in the background | `job = yh.fit(train, target, background=True)`, then `job.wait()` |
| Know the cost before you run | `yh.estimate_table(train, target=..., test=...)` |

### Fitting, intervals and importance

A fit stays available so you can score new rows as they arrive, explain the model, and share it with a teammate, without
sending the training data again.

- **Lifetime:** a fit is kept 30 days after its last use (each prediction renews it; `ttl_days=` sets 1 to 365), or
  until `model.delete()`. A `with yh.fit(...) as model:` block deletes it on exit. `model.save(path)` writes a record that `yhatlabs.load()` reattaches anywhere.
  The file includes a sample of up to 2,000 training rows for the out-of-range and novelty checks; pass `sample=False`
  to leave it out (`profile=False` on `fit` skips the profile entirely).
- **Classes and numbers:** classification returns `predictions` and `probabilities`, and with `set_coverage=` a set of
  classes per row that holds the true one that often. Regression returns `predictions` and, with `quantiles=`, an interval
  from a calibrated distribution for every row: wider where the row is harder to call or lies outside the training data.
  Pass `task="classify"` or `task="regress"` for integer targets, which could be either.
- **Rows that belong together:** with `groups="machine"` (or `time_col=` for rows over time), validation holds out whole
  groups or later periods, so scores and intervals hold for groups the model has not seen.
- **Presets:** `fast` is the default (seconds on a few thousand rows); `medium` is more accurate and about 2-3x slower;
  `pro` is the most accurate, about 4-8x slower than `fast` on the same table. `eval_metric=`, `time_limit=` and `seed=`
  are also accepted.
- **Regression columns:** `prediction` is the point forecast to use; with `quantiles=`, `median` and `mean` describe each
  row's predictive distribution. Ranges add a fixed cost per call, and about 10-20 s the first time on a saved model.
- **Large tables:** a table larger than a direct call takes runs as a background job, and the call waits for it, so the
  result is the same. `background=True` returns the job at once instead (`job.status()`, `job.wait()`, `job.cancel()`).
- **Importance:** how much the score drops when a column is shuffled. Labeled rows the model has not seen give the most honest
  numbers.

### scikit-learn

ChakraTab drops into existing scikit-learn code as an estimator, so pipelines, grid searches and cross-validation work
unchanged.

```python
from yhatlabs.integrations import ChakraTabClassifier
from sklearn.model_selection import cross_val_score

cross_val_score(ChakraTabClassifier(preset="medium"), X, y, cv=5)
```

The estimators fit once and reuse the fit for every prediction, and they pickle. `ChakraTabRegressor.predict_quantiles(X)`
gives intervals.

## Check your data first

Most surprises in a forecast or a prediction start in the data. `check` looks at it locally, before anything is sent, and
tells you what it found and how to fix it.

```python
import yhatlabs

# series: duplicates, gaps, frequency, short history, extreme values
yhatlabs.check(df, h=24)

# tables: id-like columns, columns that leak the target, imbalance
yhatlabs.check(train, target="churn")
```

- **Local and free:** it needs no key, and nothing leaves your machine.
- **What you get:** each finding has a severity, a location and a suggested fix. `.ok` is False when something would fail.
- **Seeing the clean-up:** for series, `.repaired()` returns the data after the same clean-up `forecast` applies.

## Advanced usage

<details>
<summary><b>Command line and coding agents</b></summary>

Everything above also runs from the terminal, on CSV or parquet files, which suits scripts, schedulers and AI coding
agents.

```sh
yhatlabs check sales.csv --h 28
yhatlabs forecast sales.csv --h 28 --level 80 -o forecast.csv
yhatlabs evaluate sales.csv --h 28 --windows 3
yhatlabs anomalies meters.parquet --only
yhatlabs predict train.csv --target churn --test new.csv -o scored.csv
yhatlabs whoami
```

- **Machine-readable output:** `--json` on any command, before or after it. `check` exits with code 2 on errors, so it
  can gate a pipeline.
- **Account:** `yhatlabs models`, `usage`, `limits` and `fits` print the models your key can call, what it has used, the
  API's limits and your saved models. `predict` takes `--preset pro`, `--compare` and `--no-store`; `forecast` takes
  `--no-store`.
- **Coding agents:** `yhatlabs skills install` writes a skill into the current project for Claude Code, Cursor, Codex and
  similar agents, so they know the calls and the rules.
- **Short alias:** `yhat` works too.

</details>

<details>
<summary><b>Async</b></summary>

For services and notebooks that already run an event loop, `AsyncClient` offers the same calls as coroutines.

```python
from yhatlabs import AsyncClient

async with AsyncClient() as yh:
    fc = await yh.forecast(df, h=24)
    model = await yh.fit(train, target="churn")
```

</details>

<details>
<summary><b>Configuration and visibility</b></summary>

The defaults suit most work. These settings are there for long jobs, self-hosted endpoints and debugging.

| Setting | How |
|---|---|
| API key | `Client(api_key=...)` or `YHAT_API_KEY` |
| Server | `base_url=` or `YHAT_BASE_URL` (default `https://api.yhatlabs.com`) |
| Timeout and retries | `timeout=` (default 300 s), `max_retries=` (default 3); for a single call, `yh.with_options(timeout=1800)` |
| Parallel requests | `max_concurrency=` (default 4) |
| Column names | `mask_columns=True` sends them as `c0`, `c1`, ...; results keep your names (default off) |
| Request logging | `YHAT_LOG=info`: method, path, status, time and request id (never data or keys) |
| Activity line | shown in terminals for calls over 2 s; `YHAT_PROGRESS=0` turns it off |

- **Request caps:** read from the API on first use, and applied when splitting inputs.
- **Your key:** `yh.whoami()` checks it and shows the models it can use and its quota.
- **Provenance:** every result carries `.provenance`: model, settings, request ids, SDK version and a fingerprint of the
  input, enough to rerun or cite it. `.timing` gives the client-side total and, for tables, the server's fit/predict split.
- **Polars, Spark, Dask and Ray:** input works wherever pandas does (polars LazyFrames too), and results have
  `.to_polars()`. A distributed table larger than the API takes raises `DataError` before it is collected: sample or
  aggregate it first.

</details>

<details>
<summary><b>Errors and retries</b></summary>

Errors tell you whose side the problem is on and what to do next. Temporary problems are retried for you.

- **Before anything is sent:** `DataError` says how to fix the input, and `UnsupportedError` means the model can't do what
  was asked.
- **From the API:** `APIError` and its subclasses (`AuthenticationError`, `RateLimitError`, `ServiceUnavailableError`, ...)
  carry `.status`, `.type` and `.request_id`. Quote the request id to us and we can trace the call.
- **Retries:** timeouts, connection errors, 408, 409, 429 and 5xx are retried with backoff, honouring `Retry-After`. A
  used-up quota is not retried. Every POST carries an `Idempotency-Key`, so a retried call never runs twice.
- **Partial results:** for large forecasts, `on_error="warn"` returns the series from batches that succeeded and lists the
  rest in `fc.errors`.

</details>

<details>
<summary><b>Your data</b></summary>

Your inputs are sent to the YHat Labs API to be processed.
- **One-shot calls:** kept as described in our [privacy policy](https://yhatlabs.com/privacy).
- **Fits:** a saved model is kept until it expires or you delete it.
- **Large tables:** with the `[parquet]` extra, tables over 8 MB are uploaded once and reused by reference.
- **Column names:** `mask_columns=True` keeps them on your machine; values are sent as they are.
- **Logs:** `store=False` on a call keeps it out of your Logs.
- **More:** see [SECURITY.md](SECURITY.md) for how keys and data are handled, and how to report a security issue.

</details>

<details>
<summary><b>Examples</b></summary>

Runnable scripts in [`examples/`](examples):
- **Time series:** energy load with weather and a heatwave scenario; sensors with gaps and a fault; stores that add up to a
  region.
- **Tables:** a fit with intervals and importance; a decision threshold set from business costs; validation for rows that
  arrive over time; a comparison with gradient boosting.

</details>

<details>
<summary><b>Development</b></summary>

```sh
git clone https://github.com/yhatlabs/yhatlabs-python && cd yhatlabs-python
uv venv && uv pip install -e ".[dev]"
.venv/bin/pytest -m "not live"      # offline tests against a fake API
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the checks, the changelog rule and how releases are made.
`python docs/readme_images.py` regenerates the images in this README from live calls.

</details>

## Support

Open an issue (include the `request_id` from any error, never your key), or write to founders@yhatlabs.com.

## License

Apache-2.0
