"""Standard models on the same rows (POST /v1/table/compare), to put a ChakraTab prediction in context: LightGBM, XGBoost, a
random forest and a linear model, each trained once on `train`, and a naive baseline (the most common class, or the mean).
The API returns their predictions; scoring happens here, the same way for every model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ._errors import DataError
from ._table_eval import classification_scores, regression_scores

MODELS = ("lightgbm", "xgboost", "random_forest", "linear")


def check_models(models: Any) -> Optional[List[str]]:
    if models is None or models is True:
        return None
    picked = [models] if isinstance(models, str) else list(models)
    bad = [m for m in picked if m not in MODELS]
    if bad or not picked:
        raise DataError(f"compare models must be among {list(MODELS)}; got {models!r}.")
    return picked


@dataclass
class Comparison:
    """Each standard model's predictions for the test rows (index = the test rows), and the naive baseline's."""

    problem_type: Optional[str]
    predictions: Dict[str, pd.Series]
    probabilities: Dict[str, pd.DataFrame] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)  # a model that didn't train, and why
    fit_s: Dict[str, Optional[float]] = field(default_factory=dict)
    baseline: Optional[str] = None  # the baseline's label, a key of `predictions`
    usage: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None

    def score(self, actual: Any, **others: Any) -> pd.DataFrame:
        """Scores every model against the true values of the test rows, one row per model, best first.

        Classification: accuracy, balanced accuracy and, where probabilities exist, log loss (and ROC AUC for two classes). Regression: MAE, RMSE, R2.

        Args:
            actual: The true target values of the test rows, in the same order.
            **others: More predictions to score alongside, by name: a PredictionResult (e.g. ChakraTab=res) or a sequence.
        """
        y = pd.Series(np.asarray(actual))
        named: Dict[str, Any] = {**others, **self.predictions}
        probs = {
            **{k: v.probabilities for k, v in others.items() if getattr(v, "probabilities", None) is not None},
            **self.probabilities,
        }
        rows = []
        for name, p in named.items():
            pred = getattr(p, "predictions", p)
            pred = np.asarray(pred)
            if len(pred) != len(y):
                raise DataError(f"{name}: {len(pred)} predictions for {len(y)} true values.")
            rows.append({"model": name, **self._metrics(y, pred, probs.get(name))})
        for name, why in self.errors.items():
            rows.append({"model": name, "error": why})
        out = pd.DataFrame(rows)
        key = "mae" if self.problem_type == "regression" else "accuracy"
        if key in out.columns:
            out = out.sort_values(key, ascending=key == "mae", na_position="last")
        return out.reset_index(drop=True)

    def _metrics(self, y: pd.Series, pred: np.ndarray, proba: Optional[pd.DataFrame]) -> Dict[str, float]:
        if self.problem_type == "regression":
            return regression_scores(y.to_numpy(dtype=float), pred.astype(float), None, [])
        ys, ps = y.astype(str).to_numpy(), pd.Series(pred).astype(str).to_numpy()
        recall = [float((ps[ys == c] == c).mean()) for c in np.unique(ys)]
        out = {"accuracy": float((ys == ps).mean()), "balanced_accuracy": float(np.mean(recall))}
        if proba is not None:
            sc = classification_scores(y.reset_index(drop=True), proba.reset_index(drop=True))
            out["log_loss"] = sc["log_loss"]
            if proba.shape[1] == 2:
                out["roc_auc"] = sc["auc"]
        return out


def from_reply(reply: dict, index: pd.Index) -> Comparison:
    classes = reply.get("classes")
    preds: Dict[str, pd.Series] = {}
    probs: Dict[str, pd.DataFrame] = {}
    errors: Dict[str, str] = {}
    fit_s: Dict[str, Optional[float]] = {}
    for m in reply.get("models") or []:
        label = str(m.get("label") or m.get("model"))
        if "error" in m:
            errors[label] = str(m["error"])
            continue
        preds[label] = pd.Series(m.get("predictions"), index=index, name=label)
        fit_s[label] = m.get("fit_s")
        if m.get("probabilities") is not None and classes:
            probs[label] = pd.DataFrame(np.asarray(m["probabilities"], dtype=float), index=index, columns=classes)
    base = reply.get("baseline") or {}
    label = base.get("label")
    if label and base.get("predictions") is not None:
        preds[str(label)] = pd.Series(base["predictions"], index=index, name=str(label))
    return Comparison(
        reply.get("problem_type"), preds, probs, errors, fit_s, label, reply.get("usage") or {}, reply.get("id")
    )
