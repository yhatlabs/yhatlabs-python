---
name: yhatlabs
description: Forecast time series and predict tabular targets with the YHat Labs API (ChakraTS, ChakraTab) through the yhatlabs Python package or CLI. Use when asked to forecast, backtest, detect anomalies, fill gaps, classify or regress on a table, evaluate a model on the user's own data, estimate uncertainty or risk (intervals, probabilities of crossing a limit, prediction sets), or explain a tabular model.
---

# yhatlabs

Hosted models: `chakra-ts` (time series) and `chakra-tab` (tables). Key in `YHAT_API_KEY` (never print or log it).

## Check the data first (local, no key, no request)

```python
import yhatlabs

yhatlabs.check(df, h=24)  # time series: duplicates, gaps, frequency, short history
yhatlabs.check(df, target="churn")  # tables: id-like columns, target leaks, imbalance
```

## Time series (long format: id, timestamp, target, covariates)

```python
from yhatlabs import Client

yh = Client()
fc = yh.forecast(df, h=24, level=[80, 95], calendar=True)  # fc.df: id, time, q2.5, q10, median, q90, q97.5
fc.intervals(95)  # lower_95 / upper_95 next to the median
fc.prob(above=limit)  # chance of crossing a limit at each step
fc = yh.forecast(df, h=24, mean=True)  # add the mean when forecasts are summed later
fc = yh.forecast(df, h=24, samples=500)  # sample paths: totals or peaks over the horizon
np.quantile(fc.sample_paths("store_1").sum(axis=1), [0.05, 0.95])
fc = yh.forecast(df, h=24, known=["temp"], future=future)  # future: id, time, temp for the next 24 steps
fc.to_wide()  # one column per series
ev = yh.evaluate(df, h=24, windows=3)  # ev.summary (skill vs seasonal naive), ev.calibration, ev.plot()
yh.detect_anomalies(df)  # score, anomaly per recent point
yh.fill_gaps(df)  # fills missing values inside series

from yhatlabs import hierarchy  # series that add up (stores -> regions -> total)

data, tree = hierarchy.aggregate(sales, levels=[["region"], ["region", "store"]], time_col="date", target_col="units")
fc = hierarchy.reconcile(yh.forecast(data, h=28, time_col="date", target_col="units"), tree)
```

## Tables

```python
yh.evaluate_table(df, "target", cv=5, groups="machine")  # accuracy AND calibration on the user's own data
m = yh.fit(
    train, "target", groups="machine"
)  # fit once; groups / time_col keep validation honest for new groups / periods
res = m.predict(
    test, quantiles=[0.05, 0.5, 0.95]
)  # regression: per-row intervals (res.interval_method == "distribution")
res.prob(above=200)  # chance each row exceeds a limit
res.out_of_range, res.novelty  # where a row lies outside the training data (novelty > 0.95: extrapolating)
m.predict(test, set_coverage=0.9).prediction_sets  # classification: the classes that hold the true one 90% of the time
m.effect("temperature")  # how the prediction moves with one column
m.importance(holdout)  # permutation importance per column
yh.fit(train, ["strength", "pass"])  # several targets, one FitGroup
yh.estimate_table(train, target="target", test=test, preset="medium")  # credits before running (decimals, e.g. 0.02)
m.save("model.json")
yhatlabs.load("model.json")
yh.predict(train, "target", test)  # one-shot, no fit saved
res = yh.predict(train, "target", test, compare=True)  # also LightGBM, XGBoost, random forest, linear, a baseline
res.comparison.score(actual, ChakraTab=res)  # every model scored the same way on the test rows
job = yh.fit(train, "target", preset="pro", background=True)  # returns at once; job.wait() gives the model
```

Plots: `from yhatlabs import plots` then `plots.forecast(fc, history=df)`, `plots.backtest(ev)`, `plots.reliability(ev_table)`,
`plots.effect(m.effect("x"))`, `plots.importance(imp)` (needs `pip install "yhatlabs[plot]"`).

## CLI (add --json for machine-readable output)

```
yhatlabs check data.csv --h 24
yhatlabs forecast data.csv --h 24 --level 80 -o out.csv
yhatlabs evaluate data.csv --h 24 --windows 3
yhatlabs anomalies data.csv --only
yhatlabs predict train.csv --target y --test test.csv -o scored.csv
yhatlabs evaluate-table data.csv --target y --cv 5
yhatlabs whoami
```

## Rules
- Errors raised before sending (`DataError`) say how to fix the input; API errors carry `.request_id`.
- Pass `task="classify"` or `task="regress"` for integer targets.
- Rows grouped by machine, batch, patient or store: pass `groups=`; rows over time: `time_col=` (or validate on later rows).
- Report uncertainty with the numbers the result gives (coverage, intervals, novelty), not a single point.
- Large tables are uploaded once automatically, and a table too large for a direct call runs as a background job that the
  call waits for; nothing else to do.
- Presets: "fast" (default), "medium", "pro" (most accurate, slowest).
- `store=False` on a call: it isn't shown in the user's Logs.
