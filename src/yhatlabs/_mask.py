"""mask_columns=True: column names leave this machine as c0, c1, ... and every result comes back under the caller's own names.
Values, category labels and series ids are sent as they are. The models never read names (a renamed table gives the same
predictions), so masking costs nothing in accuracy.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, Optional

import pandas as pd

from ._errors import APIError

_WIRE = re.compile(r"\bc\d+\b")


class Mask:
    def __init__(self, wire: Dict[str, str]):
        self.wire = wire  # real name -> wire name
        self.real = {w: n for n, w in wire.items()}

    @classmethod
    def of(cls, names: Iterable[Any]) -> Mask:
        return cls({n: f"c{i}" for i, n in enumerate(dict.fromkeys(str(x) for x in names))})

    def name(self, n: Any) -> Any:
        return self.wire.get(str(n), n) if n is not None else None

    def unname(self, n: Any) -> Any:
        return self.real.get(n, n) if isinstance(n, str) else n

    def frame(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns=lambda c: self.wire.get(str(c), c))

    def keys(self, d: Dict[str, Any]) -> Dict[str, Any]:
        return {self.name(k): v for k, v in d.items()}

    def unkeys(self, d: Dict[str, Any]) -> Dict[str, Any]:
        return {self.unname(k): v for k, v in d.items()}

    def options(self, extra: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """groups= / time_col= name columns too."""
        if not extra:
            return extra
        return {k: (self.name(v) if k in ("groups", "time_col") else v) for k, v in extra.items()}

    def text(self, s: str) -> str:
        return _WIRE.sub(lambda m: self.real.get(m.group(0), m.group(0)), s)

    @contextmanager
    def errors(self) -> Iterator[None]:
        """An API error that names a masked column names the caller's column instead."""
        try:
            yield
        except APIError as e:
            e.message = self.text(e.message)
            e.args = (e.message,)
            raise


@contextmanager
def masked(mask: Optional[Mask]) -> Iterator[None]:
    if mask is None:
        yield
    else:
        with mask.errors():
            yield
