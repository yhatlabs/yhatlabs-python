import httpx
import pytest

from yhatlabs import (
    AuthenticationError,
    Client,
    ConfigurationError,
    RateLimitError,
    ServiceUnavailableError,
)


def test_missing_key(monkeypatch):
    monkeypatch.delenv("YHAT_API_KEY", raising=False)
    monkeypatch.delenv("YHATLABS_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="YHAT_API_KEY"):
        Client()


def test_env_key_and_base_url(monkeypatch):
    monkeypatch.setenv("YHAT_API_KEY", "k")
    monkeypatch.setenv("YHAT_BASE_URL", "http://localhost:8787/")
    assert Client().base_url == "http://localhost:8787"


def test_auth_error(api):
    c = Client("wrong", base_url="https://fake.test", http_client=httpx.Client(transport=httpx.MockTransport(api)))
    with pytest.raises(AuthenticationError) as e:
        c.models()
    assert e.value.status == 401 and e.value.request_id == "r1" and "bad key" in str(e.value)


def test_retries_transient_then_succeeds(client, api):
    api.script = [(503, {"error": {"type": "model_unavailable", "message": "starting"}}), (502, None)]
    assert client.models()[0]["id"] == "chakra-ts"
    assert len(api.requests) == 3


def test_gives_up_after_max_retries(client, api):
    api.script = [(503, {"error": {"type": "model_unavailable", "message": "starting"}})] * 4
    with pytest.raises(ServiceUnavailableError):
        client.models()
    assert len(api.requests) == 4


def test_quota_not_retried(client, api):
    api.script = [(429, {"error": {"type": "quota_exceeded", "message": "used up"}})]
    with pytest.raises(RateLimitError) as e:
        client.usage()
    assert e.value.type == "quota_exceeded" and len(api.requests) == 1


def test_rate_limit_retried(client, api):
    api.script = [(429, {"error": {"type": "rate_limited", "message": "slow down"}})]
    assert client.usage()["unit"] == "series"
    assert len(api.requests) == 2


def test_connection_error():
    def boom(request):
        raise httpx.ConnectError("no route")

    from yhatlabs import APIConnectionError

    c = Client(
        "k", base_url="https://fake.test", max_retries=1, http_client=httpx.Client(transport=httpx.MockTransport(boom))
    )
    with pytest.raises(APIConnectionError, match="Could not reach"):
        c.models()


def test_streamed_error_in_200_body_is_raised_and_retried(client, api):
    # a long tabular fit answers 200 with keep-alive spaces, so a later failure arrives as {error} in a 200 body
    api.script = [(200, {"error": {"type": "model_unavailable", "message": "starting", "request_id": "r9"}})]
    assert client.usage()["unit"] == "series"
    assert len(api.requests) == 2
    api.script = [(200, {"error": {"type": "invalid_request_error", "message": "bad table", "request_id": "r10"}})]
    from yhatlabs import BadRequestError

    with pytest.raises(BadRequestError) as e:
        client.usage()
    assert e.value.status == 422 and e.value.request_id == "r10"


def test_streamed_error_retried_once_only(client, api):
    api.script = [(200, {"error": {"type": "model_unavailable", "message": "timed out"}})] * 3
    with pytest.raises(ServiceUnavailableError):
        client.usage()
    assert len(api.requests) == 2


def test_standard_library_json_fallback():
    import numpy as np

    from yhatlabs._transport import _dumps, _dumps_std

    body = {"series": [{"id": "s0", "values": [1.0, None, 2.5]}], "horizon": 3, "model": "chakra-ts"}
    assert _dumps_std(body) == b'{"series":[{"id":"s0","values":[1.0,null,2.5]}],"horizon":3,"model":"chakra-ts"}'
    assert __import__("json").loads(_dumps(body)) == __import__("json").loads(_dumps_std(body))
    with pytest.raises(ValueError):
        _dumps_std({"x": float("nan")})
    assert np  # numpy values are converted before encoding, so neither encoder sees them
