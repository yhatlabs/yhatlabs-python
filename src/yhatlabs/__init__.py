"""YHat Labs prediction API client: ChakraTS forecasting and ChakraTab tabular prediction."""

from ._check import DataCheck, check
from ._client import AsyncClient, Client
from ._errors import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConfigurationError,
    DataError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    RequestTooLargeError,
    ServiceUnavailableError,
    UnsupportedError,
    YHatError,
)
from ._fits import AsyncFittedModel, FittedModel, load
from ._forecast import Limits
from ._results import EvaluationResult, ForecastResult, PredictionResult
from ._version import __version__

__all__ = [
    "Client",
    "AsyncClient",
    "FittedModel",
    "check",
    "load",
    "DataCheck",
    "AsyncFittedModel",
    "ForecastResult",
    "PredictionResult",
    "EvaluationResult",
    "Limits",
    "__version__",
    "YHatError",
    "ConfigurationError",
    "DataError",
    "UnsupportedError",
    "APIError",
    "APIConnectionError",
    "APITimeoutError",
    "BadRequestError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "RequestTooLargeError",
    "RateLimitError",
    "InternalServerError",
    "ServiceUnavailableError",
]


def __getattr__(name: str):
    if name in ("ChakraTabClassifier", "ChakraTabRegressor"):
        from . import _sklearn

        return getattr(_sklearn, name)
    raise AttributeError(f"module 'yhatlabs' has no attribute {name!r}")
