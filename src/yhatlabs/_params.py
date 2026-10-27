"""Descriptions of the parameters many public functions share, for the generated Python reference (each function's docstring
`Args:` adds or overrides its own). Required or not, and the default, come from the signatures themselves.
"""

SHARED = {
    "data": "The history: a long table with an id column, a time column and the target (pandas or polars), a Series with a "
    "DatetimeIndex, a numpy array (one series per row), or a dict of arrays. Arrays need freq=.",
    "h": "How many steps to forecast, at the series' own frequency.",
    "model": "The model name, as GET /v1/models lists it.",
    "quantiles": 'Quantile levels in (0, 1), e.g. [0.1, 0.5, 0.9], or "all" for every level the model produces. Default '
    "[0.1, 0.5, 0.9].",
    "level": "Central intervals by width in percent, e.g. [80, 95]; the alternative to quantiles=.",
    "freq": "The series' frequency as a pandas alias (h, D, W-SUN, MS, 15min, ...). Inferred from the timestamps when left out; "
    "required for arrays and integer time steps.",
    "id_col": "The column naming each series. Default: unique_id, id, item_id or series_id if present, else one series.",
    "time_col": "The timestamp column. Default: ds, timestamp, time, date or datetime if present.",
    "target_col": "The column (or list of columns) to forecast. Default: y, target or value if present, else the one numeric column.",
    "known": "Covariate columns whose future values you know (prices, weather forecasts, holidays); their future values go in "
    "future=.",
    "past": "Covariate columns known only up to now (observed loads, sensor readings).",
    "static": "Columns that are constant per series (region, store size).",
    "future": "The known covariates over the next h steps: id, time and the known columns.",
    "season": "The seasonal period in steps (24 for daily cycles of hourly data), to fix it instead of inferring it from freq.",
    "calendar": 'True adds hour, weekday, month and weekend as known covariates (no future= needed for them); {"holidays": "US"} '
    "also adds a holiday flag.",
    "train": "The training table: one row per example, the target column included.",
    "target": "The column to predict.",
    "test": "The rows to predict, with the same feature columns as train. Without it, the rows of train whose target is empty "
    "are predicted.",
    "task": '"classify", "regress", or None to let the server infer it. Pass it for integer targets (a 1-5 rating could be either).',
    "preset": '"fast" (default: one holdout split, the quickest), "medium" (3-fold bagging, about 3x the time) or "pro" (8-fold bagging, the most accurate; "full" is the same).',
    "eval_metric": "What validation optimises: log_loss (default), roc_auc, f1, ... for classification; rmse (default), mae, r2 "
    "for regression.",
    "time_limit": "Seconds for the whole call.",
    "seed": "Random seed; the server uses 0 when it is left out, so results are reproducible.",
    "groups": "A column whose rows belong together (machine, batch, patient, store): validation folds hold out whole groups, so "
    "intervals hold for groups the model has not seen. The column is not used as a feature.",
    "mask_columns": "Send column names as c0, c1, ... instead of your own; every result comes back under your names. Values, "
    "category labels and series ids are sent as they are. Default: the client's setting (False).",
    "set_coverage": "Classification: return prediction sets that hold the true class this often, e.g. 0.9.",
}
