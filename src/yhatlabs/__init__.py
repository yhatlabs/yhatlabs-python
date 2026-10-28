"""YHat Labs prediction API client: ChakraTS forecasting and ChakraTab tabular prediction."""

from ._anomaly import AnomalyStream
from ._check import DataCheck, check
from ._client import AsyncClient, Client
from ._compare import Comparison
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
from ._fits import AsyncFitGroup, AsyncFittedModel, FitGroup, FittedModel, GroupPrediction, load
from ._forecast import Limits
from ._jobs import AsyncJob, Job, JobFailedError
from ._profile import outliers
from ._results import EvaluationResult, ForecastResult, PredictionResult
from ._table_eval import TableEvaluation
from ._version import __version__

__all__ = [
    "Client",
    "AsyncClient",
    "FittedModel",
    "check",
    "outliers",
    "load",
    "DataCheck",
    "AsyncFittedModel",
    "FitGroup",
    "AsyncFitGroup",
    "GroupPrediction",
    "TableEvaluation",
    "ForecastResult",
    "PredictionResult",
    "EvaluationResult",
    "Limits",
    "Job",
    "AsyncJob",
    "Comparison",
    "AnomalyStream",
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
    "JobFailedError",
]


def __getattr__(name: str):
    if name in ("ChakraTabClassifier", "ChakraTabRegressor"):
        from . import _sklearn

        return getattr(_sklearn, name)
    raise AttributeError(f"module 'yhatlabs' has no attribute {name!r}")
