"""ChakraTab against a tuned-by-default gradient boosting model on your own split: the comparison a team usually asks for first."""

import time

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from yhatlabs.integrations import ChakraTabRegressor

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X, y = X.iloc[:8000], y.iloc[:8000]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

for name, est in [
    ("HistGradientBoosting", HistGradientBoostingRegressor(random_state=0)),
    ("ChakraTab", ChakraTabRegressor(preset="medium")),
]:
    t = time.time()
    pred = est.fit(X_train, y_train).predict(X_test)
    print(f"{name:<22} MAE {mean_absolute_error(y_test, pred):.4f}  ({time.time() - t:.0f}s)")
