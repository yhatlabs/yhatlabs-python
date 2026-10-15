"""Exceptions. Every API error carries the HTTP status, the error type the API returned and the request id, so a
failed call can be traced on our side. Grouped by HTTP status, so callers can catch a family (RateLimitError) or everything (APIError).
"""

from __future__ import annotations

from typing import Any, Optional


class YHatError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(YHatError):
    """The client is missing something it needs, such as an API key."""


class DataError(YHatError, ValueError):
    """The input data cannot be turned into a valid request (raised before anything is sent)."""


class UnsupportedError(YHatError):
    """The request asks for something the chosen model or the API does not support yet."""


class APIConnectionError(YHatError):
    """The API could not be reached (network error, DNS, TLS)."""


class APITimeoutError(APIConnectionError):
    """The request did not finish within the client's timeout."""


class APIError(YHatError):
    """The API answered with an error."""

    def __init__(
        self,
        message: str,
        *,
        status: int,
        type: Optional[str] = None,
        request_id: Optional[str] = None,
        body: Any = None,
    ):
        super().__init__(message)
        self.message = message
        self.status = status
        self.type = type
        self.request_id = request_id
        self.body = body

    def __str__(self) -> str:
        rid = f" (request_id={self.request_id})" if self.request_id else ""
        return f"{self.status} {self.type or 'error'}: {self.message}{rid}"


class BadRequestError(APIError):
    """400 / 422: the request is malformed or the model could not use the data."""


class AuthenticationError(APIError):
    """401: missing or invalid API key."""


class PermissionDeniedError(APIError):
    """403: the key is valid but not allowed to do this."""


class NotFoundError(APIError):
    """404: unknown model or route."""


class RequestTooLargeError(APIError):
    """413: the request exceeds a size limit (the client normally splits batches before this happens)."""


class RateLimitError(APIError):
    """429: too many requests, or the key's quota is used up (`type == "quota_exceeded"`)."""


class InternalServerError(APIError):
    """500 / 502: something went wrong on our side."""


class ServiceUnavailableError(APIError):
    """503: the model is starting up or temporarily unavailable; the client retries these automatically."""


_BY_STATUS = {
    400: BadRequestError,
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    413: RequestTooLargeError,
    422: BadRequestError,
    429: RateLimitError,
    500: InternalServerError,
    502: InternalServerError,
    503: ServiceUnavailableError,
}


def error_from_response(status: int, body: Any, request_id: Optional[str]) -> APIError:
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        message = str(err.get("message") or "The API returned an error.")
        etype = err.get("type")
        request_id = err.get("request_id") or request_id
    else:
        message = f"The API returned HTTP {status}."
        etype = None
    cls = _BY_STATUS.get(status, InternalServerError if status >= 500 else APIError)
    return cls(message, status=status, type=etype, request_id=request_id, body=body)
