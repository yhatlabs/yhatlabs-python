"""Rows that arrive over time (churn by month, loan applications) must be validated by time, not at random: a random split lets
the model learn from the future it is tested on, and the score looks better than it will be in production."""

import numpy as np
import pandas as pd

from yhatlabs import Client

rng = np.random.default_rng(0)
n = 3000
df = pd.DataFrame(
    {
        "signup": pd.date_range("2025-01-01", periods=n, freq="3h"),
        "plan": rng.choice(["free", "pro"], n),
        "logins_30d": rng.poisson(8, n),
        "tickets": rng.poisson(1, n),
    }
)
drift = np.linspace(-1, 1, n)  # behaviour changes over the year
df["churned"] = (rng.random(n) < 1 / (1 + np.exp(-(drift - 0.2 * df.logins_30d + 0.8 * df.tickets)))).astype(int)

cutoff = df.signup.quantile(0.8)
train, test = df[df.signup < cutoff], df[df.signup >= cutoff]  # train on the past, test on the most recent 20%
features = ["plan", "logins_30d", "tickets"]  # not the timestamp itself: the future has dates the model never saw

with Client() as yh:
    res = yh.predict(train[features + ["churned"]], "churned", test[features], task="classify")
    print(f"accuracy on the most recent 20%: {(res.predictions == test.churned).mean():.3f}")
