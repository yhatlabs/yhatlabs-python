---
name: yhatlabs
description: Forecast time series and predict tabular targets with the YHat Labs API (ChakraTS, ChakraTab) through the yhatlabs Python package or CLI. Use when asked to forecast, backtest, detect anomalies, fill gaps, classify or regress on a table, or explain a tabular model.
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
fc = yh.forecast(df, h=24, level=[80], calendar=True)  # fc.df: id, time, q10, median, q90
fc = yh.forecast(df, h=24, known=["temp"], future=future)  # future: id, time, temp for the next 24 steps
ev = yh.evaluate(df, h=24, windows=3)  # ev.summary (skill vs seasonal naive), ev.calibration
yh.detect_anomalies(df)  # score, anomaly per recent point
yh.fill_gaps(df)  # fills missing values inside series
```

## Tables

```python
m = yh.fit(train, "target")  # fit once
m.predict(test, quantiles=[0.05, 0.95])  # regression intervals; classification gives .probabilities
m.importance(holdout)  # permutation importance per column
m.save("model.json")
yhatlabs.load("model.json")
yh.predict(train, "target", test)  # one-shot, nothing kept
```

## CLI (add --json for machine-readable output)

```
yhatlabs check data.csv --h 24
yhatlabs forecast data.csv --h 24 --level 80 -o out.csv
yhatlabs evaluate data.csv --h 24 --windows 3
yhatlabs anomalies data.csv --only
yhatlabs predict train.csv --target y --test test.csv -o scored.csv
yhatlabs whoami
```

## Rules
- Errors raised before sending (`DataError`) say how to fix the input; API errors carry `.request_id`.
- Forecast quantiles span 0.1 to 0.9; ask for others between those levels only.
- Pass `task="classify"` or `task="regress"` for integer targets.
- Validate on rows later in time than the training rows when rows arrive over time.
