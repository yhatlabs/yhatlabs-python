import os

import numpy as np
import pandas as pd
import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("YHAT_API_KEY"), reason="set YHAT_API_KEY to run"),
]


def test_live_forecast():
    from yhatlabs import Client

    with Client() as c:
        df = pd.DataFrame(
            {"ds": pd.date_range("2026-01-01", periods=168, freq="h"), "y": np.sin(np.arange(168) * 2 * np.pi / 24)}
        )
        fc = c.forecast(df, h=24)
        assert len(fc.df) == 24 and fc.df["q10"].le(fc.df["q90"]).all()
        assert any(m["id"] == "chakra-ts" for m in c.models())
