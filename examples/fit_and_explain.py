"""Fit once, then predict with intervals and see which columns matter. Run: YHAT_API_KEY=... python examples/fit_and_explain.py"""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from yhatlabs import Client

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

with (
    Client() as yh,
    yh.fit(X_train.assign(progression=y_train), "progression", task="regress", preset="medium") as model,
):
    res = model.predict(X_test, quantiles=[0.05, 0.5, 0.95])  # each call sends only the rows to score
    print(res.df.head())
    inside = ((y_test >= res.quantiles["q5"]) & (y_test <= res.quantiles["q95"])).mean()
    print(f"90% intervals cover {inside:.0%} of held-out rows")

    # held-out labeled rows give the most honest importance; without them a medium fit falls back to training rows
    print(model.importance(X_test.assign(progression=y_test)).head(5))
# leaving the `with` block deletes the fit; without it, it expires after ttl_hours (default 24)
