"""scikit-learn estimators over ChakraTab. fit() fits once on our side (Client.fit) and keeps the handle, so each predict() sends
only the rows to score. Against an API without fitted models, fit() just stores the data and each predict() fits and predicts in
one call. Works in Pipelines, cross_val_score and GridSearchCV; a fitted estimator pickles (it keeps the fit id).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

from ._client import DEFAULT_TAB_MODEL, Client
from ._errors import DataError, UnsupportedError
from ._fits import FittedModel

try:
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
except ImportError:  # usable without scikit-learn (clone and pipelines need it)

    class BaseEstimator:  # type: ignore[no-redef]
        def get_params(self, deep: bool = True) -> dict:
            import inspect

            return {k: getattr(self, k) for k in inspect.signature(type(self).__init__).parameters if k != "self"}

        def set_params(self, **params: object) -> BaseEstimator:
            for k, v in params.items():
                setattr(self, k, v)
            return self

    class ClassifierMixin:  # type: ignore[no-redef]
        _estimator_type = "classifier"

    class RegressorMixin:  # type: ignore[no-redef]
        _estimator_type = "regressor"


_TARGET = "__yhat_target__"


def _as_frame(X: Any, columns: Optional[list]) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.rename(columns=str)
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise DataError(f"X must be 2-D; got shape {arr.shape}.")
    return pd.DataFrame(arr, columns=columns or [f"x{i}" for i in range(arr.shape[1])])


class _ChakraTab(BaseEstimator):
    _task = ""

    def __init__(
        self,
        model: str = DEFAULT_TAB_MODEL,
        preset: Optional[str] = None,
        eval_metric: Optional[str] = None,
        time_limit: Optional[float] = None,
        seed: Optional[int] = None,
        ttl_hours: Optional[float] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.preset = preset
        self.eval_metric = eval_metric
        self.time_limit = time_limit
        self.seed = seed
        self.ttl_hours = ttl_hours
        self.api_key = api_key
        self.base_url = base_url

    def _client(self) -> Client:
        c = getattr(self, "_yhat_client", None)
        if c is None:
            c = Client(self.api_key, base_url=self.base_url)
            self._yhat_client = c
        return c

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_yhat_client", None)
        state.pop("_last", None)
        return state

    def fit(self, X: Any, y: Any) -> Any:
        X = _as_frame(X, None)
        if _TARGET in X.columns:
            raise DataError(f"X may not have a column named {_TARGET!r}.")
        y = np.asarray(y).ravel()
        if len(y) != len(X):
            raise DataError(f"X has {len(X)} rows but y has {len(y)}.")
        self.n_features_in_ = X.shape[1]
        self._columns = list(X.columns)
        if not all(c.startswith("x") and c[1:].isdigit() for c in self._columns):
            self.feature_names_in_ = np.asarray(self._columns, dtype=object)
        self._last = None
        train = X.reset_index(drop=True).assign(**{_TARGET: y})
        try:
            self.fit_ = self._client().fit(
                train,
                _TARGET,
                model=self.model,
                task=self._task,
                preset=self.preset,
                eval_metric=self.eval_metric,
                time_limit=self.time_limit,
                seed=self.seed,
                ttl_hours=self.ttl_hours,
            )
            self._train = None
        except UnsupportedError:  # an API without /v1/table/fits: one fit-and-predict call per predict
            self.fit_ = None
            self._train = train
        return self

    def _call(self, X: Any, quantiles: Any = None):
        if not hasattr(self, "fit_"):
            raise DataError(f"This {type(self).__name__} is not fitted yet; call fit(X, y) first.")
        X = _as_frame(X, self._columns).reset_index(drop=True)
        key = (X.shape, int(hash_pandas_object(X, index=False).sum()), tuple(quantiles or ()))
        last = getattr(self, "_last", None)
        if last is not None and last[0] == key:
            return last[1]
        if self.fit_ is not None:
            fm: FittedModel = self.fit_ if self.fit_._client is not None else self.fit_.bind(self._client())
            res = fm.predict(X, quantiles=quantiles)
        else:
            res = self._client().predict(
                self._train,
                _TARGET,
                X,
                model=self.model,
                task=self._task,
                preset=self.preset,
                eval_metric=self.eval_metric,
                time_limit=self.time_limit,
                seed=self.seed,
                quantiles=quantiles,
            )
        self._last = (key, res)
        return res

    def importance(self, X: Any = None, y: Any = None, **kw: Any) -> pd.DataFrame:
        """Permutation importance per column (see FittedModel.importance); X, y: held-out labeled rows, recommended."""
        if getattr(self, "fit_", None) is None:
            raise UnsupportedError("importance needs an API with fitted models.")
        data = None
        if X is not None:
            data = _as_frame(X, self._columns).reset_index(drop=True).assign(**{_TARGET: np.asarray(y).ravel()})
        fm = self.fit_ if self.fit_._client is not None else self.fit_.bind(self._client())
        return fm.importance(data, **kw)


class ChakraTabClassifier(ClassifierMixin, _ChakraTab):
    _task = "classify"

    def fit(self, X: Any, y: Any) -> ChakraTabClassifier:
        super().fit(X, y)
        self.classes_ = np.unique(np.asarray(y).ravel())
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        res = self._call(X)
        missing = [c for c in self.classes_ if c not in set(res.probabilities.columns)]
        if missing:
            raise DataError(
                f"The API returned classes {list(res.probabilities.columns)}, expected {list(self.classes_)}."
            )
        return res.probabilities[list(self.classes_)].to_numpy()

    def predict(self, X: Any) -> np.ndarray:
        # the server's predictions are the most probable class (no threshold tuning), the same as argmax of predict_proba
        return np.asarray(self._call(X).predictions.tolist(), dtype=self.classes_.dtype)


class ChakraTabRegressor(RegressorMixin, _ChakraTab):
    _task = "regress"

    def predict(self, X: Any) -> np.ndarray:
        return self._call(X).predictions.to_numpy(dtype=float)

    def predict_quantiles(self, X: Any, quantiles: Any = (0.05, 0.5, 0.95)) -> pd.DataFrame:
        """Prediction intervals: one column per level (split-conformal on out-of-sample errors)."""
        q = self._call(X, list(quantiles)).quantiles.copy()
        if isinstance(X, pd.DataFrame):
            q.index = X.index  # rows line up with the caller's X, like the rest of their DataFrame
        return q
