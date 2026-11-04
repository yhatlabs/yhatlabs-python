"""Tuned ChakraTS: the model fine-tuned on your own series, then used for forecasts like the zero-shot one.

A tune trains on all but the last `h` steps of each series and scores itself there against the zero-shot model; it is used only
when it wins (`used`, with the numbers in `report`), otherwise forecasts with it are the zero-shot forecast. It lives on our side
30 days after its last use (`ttl_days`, up to 365; each forecast renews it) or until you delete it. The handle is picklable and
reattaches to a client with `bind()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ._errors import ConfigurationError


@dataclass(repr=False)
class Tuned:
    id: str
    model: str
    horizon: int
    freq: str
    series: int
    steps: int
    used: bool
    report: Dict[str, Any]
    created_at: int
    expires_at: int
    name: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict, repr=False)  # what the tune cost (tokens, credits)
    _client: Any = field(default=None, repr=False, compare=False)

    @classmethod
    def from_reply(cls, reply: dict, client: Any) -> Tuned:
        t = cls(
            reply["id"],
            reply.get("model", ""),
            int(reply.get("horizon", 0)),
            str(reply.get("freq", "")),
            int(reply.get("series", 0)),
            int(reply.get("steps", 0)),
            bool(reply.get("used")),
            dict(reply.get("report") or {}),
            int(reply.get("created_at", 0)),
            int(reply.get("expires_at", 0)),
            reply.get("name"),
        )
        t.usage = dict(reply.get("usage") or {})
        t._client = client
        return t

    @property
    def holdout(self) -> Dict[str, Any]:
        """The score on your last `horizon` steps: zero-shot vs tuned (loss: scaled pinball, lower is better; wape)."""
        return dict(self.report.get("holdout") or {})

    def __repr__(self) -> str:
        h = self.holdout
        z, t = (h.get("zero_shot") or {}).get("wape"), (h.get("tuned") or {}).get("wape")
        score = f", holdout WAPE {z} -> {t}" if z is not None and t is not None else ""
        return f"Tuned(id={self.id!r}, model={self.model!r}, used={self.used}, series={self.series}, horizon={self.horizon}{score})"

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def bind(self, client: Any) -> Tuned:
        self._client = client
        return self

    def _c(self) -> Any:
        if self._client is None:
            raise ConfigurationError("This tune is not attached to a client; call .bind(Client()).")
        return self._client

    def forecast(self, data: Any, h: Optional[int] = None, **kw: Any) -> Any:
        """Forecast with this tune; the same options as Client.forecast. h defaults to the tune's horizon."""
        return self._c().forecast(data, h or self.horizon, model=self.model, tune=self.id, **kw)

    def delete(self) -> None:
        """Deletes the tune on our side."""
        self._c()._t.request("DELETE", f"/v1/series/tunes/{self.id}")

    def info(self) -> Dict[str, Any]:
        """The tune as the API shows it now (expiry, last use, report)."""
        return self._c()._t.request("GET", f"/v1/series/tunes/{self.id}")

    def __enter__(self) -> Tuned:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.delete()


class AsyncTuned(Tuned):
    async def forecast(self, data: Any, h: Optional[int] = None, **kw: Any) -> Any:  # type: ignore[override]
        return await self._c().forecast(data, h or self.horizon, model=self.model, tune=self.id, **kw)

    async def delete(self) -> None:  # type: ignore[override]
        await self._c()._t.request("DELETE", f"/v1/series/tunes/{self.id}")

    async def info(self) -> Dict[str, Any]:  # type: ignore[override]
        return await self._c()._t.request("GET", f"/v1/series/tunes/{self.id}")

    async def __aenter__(self) -> AsyncTuned:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.delete()
