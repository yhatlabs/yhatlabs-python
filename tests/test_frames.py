import httpx
import numpy as np
import pandas as pd
import pytest

import yhatlabs
from yhatlabs import Client, DataError, Limits


def _fake(module: str, name: str, **methods):
    """A stand-in for a Spark / Dask / Ray frame: same module name and methods, no library needed."""

    def init(self, df, log=None):
        self.df, self.log = df, log if log is not None else []

    return type(name, (), {"__module__": module, "__init__": init, **methods})


Spark = _fake(
    "pyspark.sql.dataframe",
    "DataFrame",
    columns=property(lambda self: list(self.df.columns)),
    limit=lambda self, n: (self.log.append(("limit", n)), type(self)(self.df.head(n), self.log))[1],
    toPandas=lambda self: (self.log.append("collect"), self.df.copy())[1],
)
Dask = _fake(
    "dask.dataframe.core",
    "DataFrame",
    columns=property(lambda self: self.df.columns),
    compute=lambda self: (self.log.append("collect"), self.df.copy())[1],
    head=lambda self, n, npartitions=1: (self.log.append(("head", n)), self.df.head(n).copy())[1],
)
Ray = _fake(
    "ray.data.dataset",
    "Dataset",
    schema=lambda self: type("Schema", (), {"names": list(self.df.columns)})(),
    limit=lambda self, n: (self.log.append(("limit", n)), type(self)(self.df.head(n), self.log))[1],
    to_pandas=lambda self: (self.log.append("collect"), self.df.copy())[1],
)


@pytest.fixture
def table():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.normal(size=40), "b": rng.integers(0, 5, 40), "y": [0, 1] * 20})


def _small(api, **kw):
    return Client(
        "test-key",
        base_url="https://fake.test",
        limits=Limits(**kw),
        http_client=httpx.Client(transport=httpx.MockTransport(api)),
    )


@pytest.mark.parametrize("kind", [Spark, Dask, Ray])
def test_distributed_frames_in(client, hourly, table, kind):
    fc = client.forecast(kind(hourly), h=2)
    assert len(fc.df) == 4
    res = client.predict(kind(table.iloc[:30]), "y", kind(table.iloc[30:].drop(columns="y")), task="classify")
    assert len(res.predictions) == 10
    m = client.fit(kind(table), "y", task="classify")
    assert len(m.predict(kind(table.drop(columns="y"))).predictions) == 40
    assert yhatlabs.check(kind(hourly)).ok
    assert len(client.detect_anomalies(kind(hourly), h=12, windows=2)) == 48


@pytest.mark.parametrize("kind", [Spark, Dask, Ray])
def test_a_distributed_table_past_the_limit_is_never_collected(api, table, kind):
    c = _small(api, max_tabular_rows=25)
    log: list = []
    with pytest.raises(DataError, match="Sample or aggregate"):
        c.predict(kind(table, log), "y", table.drop(columns="y"), task="classify")
    assert log[0] in (("limit", 26), ("head", 26))  # only the first rows past the limit are read, never the whole table
    with pytest.raises(DataError, match="more than 25 rows"):
        c.fit(kind(table), "y")


def test_the_cell_limit_caps_rows_by_columns(api, table):
    c = _small(api, max_tabular_rows=1000, max_cells_async=60)  # 3 columns: at most 20 rows
    with pytest.raises(DataError, match="more than 20 rows"):
        c.predict(Spark(table), "y", table.drop(columns="y"), task="classify")
    assert len(c.predict(Spark(table.head(20)), "y", table.drop(columns="y"), task="classify").predictions) == 40


def test_polars_lazy_frames(client, api, hourly, table):
    pl = pytest.importorskip("polars")
    pytest.importorskip("pyarrow")
    assert len(client.forecast(pl.from_pandas(hourly).lazy(), h=2).df) == 4
    lazy = pl.from_pandas(table).lazy()
    assert len(client.predict(lazy, "y", pl.from_pandas(table.drop(columns="y")), task="classify").predictions) == 40
    with pytest.raises(DataError, match="polars LazyFrame has more than 25 rows"):
        _small(api, max_tabular_rows=25).predict(lazy, "y", table.drop(columns="y"), task="classify")
