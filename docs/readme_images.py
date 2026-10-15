"""Regenerates the README images and output tables from live API calls. Run: YHAT_API_KEY=... python docs/readme_images.py"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

from yhatlabs import Client

OUT = "docs/images"
INK, BAND, HIST = "#1f3b73", "#9db7e8", "#555555"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

yh = Client()

# --- time series: monthly airline passengers, forecast two years ahead
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv")
fc = yh.forecast(df, h=24, time_col="Month", target_col="Passengers", level=[80])  # exactly as in the README
print(fc.df.head(3).round(1))
f = fc.df
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(pd.to_datetime(df.Month), df.Passengers, color=HIST, lw=1.2, label="history")
ax.fill_between(f.Month, f.q10, f.q90, color=BAND, alpha=0.6, lw=0, label="80% interval")
ax.plot(f.Month, f["median"], color=INK, lw=1.8, label="ChakraTS median")
ax.set_title("Airline passengers: 24-month forecast", loc="left", fontsize=11)
ax.set_ylabel("passengers (thousands)")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}/forecast.png", dpi=160)

# --- tables: classification
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = yh.fit(X_train.assign(diagnosis=y_train), target="diagnosis")
res = model.predict(X_test)
print(res.df.head(3).round(3).to_string())
print("accuracy", round((res.predictions == y_test).mean(), 3))
model.delete()

# --- tables: regression with intervals
X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
reg = yh.fit(X_train.assign(progression=y_train), target="progression", task="regress")
r = reg.predict(X_test, quantiles=[0.05, 0.5, 0.95])
print(r.df.head(3).round(1).to_string())
cover = ((y_test >= r.quantiles.q5) & (y_test <= r.quantiles.q95)).mean()
print("90% interval coverage", round(cover, 3))
imp = reg.importance(X_test.assign(progression=y_test))
reg.delete()
order = np.argsort(r.predictions.to_numpy())
xs = np.arange(len(order))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [2.2, 1]})
a1.fill_between(
    xs,
    r.quantiles.q5.to_numpy()[order],
    r.quantiles.q95.to_numpy()[order],
    color=BAND,
    alpha=0.6,
    lw=0,
    label="90% interval",
)
a1.plot(xs, r.predictions.to_numpy()[order], color=INK, lw=1.6, label="prediction")
a1.scatter(xs, y_test.to_numpy()[order], s=9, color=HIST, zorder=3, label="actual")
a1.set_title(f"Held-out patients: {cover:.0%} inside the 90% interval", loc="left", fontsize=11)
a1.set_xlabel("patients, sorted by prediction")
a1.set_ylabel("disease progression")
a1.legend(frameon=False, loc="upper left")
top = imp.head(6).iloc[::-1]
a2.barh(top.feature, top.importance, color=INK)
a2.set_title("What drives it", loc="left", fontsize=11)
a2.set_xlabel("importance")
fig.tight_layout()
fig.savefig(f"{OUT}/tabular.png", dpi=160)
print("saved", OUT)
