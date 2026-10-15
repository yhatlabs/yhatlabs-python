"""Pick the decision threshold from what mistakes cost, not 0.5. A missed fraud costs far more than a false alarm, so the best
cut-off for flagging is usually much lower than 0.5. The model gives calibrated probabilities; the threshold is your call."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from yhatlabs import Client

X, y = make_classification(n_samples=6000, n_features=12, weights=[0.97], random_state=0)  # 3% positives, like fraud
X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=0)

COST_MISSED, COST_FALSE_ALARM = 500.0, 10.0  # what each kind of mistake costs you

with Client() as yh:
    m = yh.fit(pd.DataFrame(X_train).assign(fraud=y_train), "fraud", task="classify", eval_metric="log_loss")
    p_val = m.predict_proba(pd.DataFrame(X_val))[1].to_numpy()

    def cost(th, p, y):
        flag = p >= th
        return COST_MISSED * ((~flag) & (y == 1)).sum() + COST_FALSE_ALARM * (flag & (y == 0)).sum()

    grid = np.linspace(0.01, 0.9, 90)
    best = grid[np.argmin([cost(t, p_val, y_val) for t in grid])]  # chosen on validation rows...
    p_test = m.predict_proba(pd.DataFrame(X_test))[1].to_numpy()
    print(
        f"best threshold {best:.2f}: test cost {cost(best, p_test, y_test):,.0f} vs {cost(0.5, p_test, y_test):,.0f} at 0.5"
    )  # ...scored on test
    m.delete()
