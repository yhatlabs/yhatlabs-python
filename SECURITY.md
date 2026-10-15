# Security

Please report vulnerabilities privately to security@yhatlabs.com, not as a public issue. We acknowledge reports within
two business days.

- API keys: the client sends your key only in the `Authorization` header to the configured `base_url`, and never writes it to
  disk or logs (`YHAT_LOG` logs method, path, status, timing and request ids only).
- Data: inputs are sent to the YHat Labs API for processing. A one-shot prediction keeps nothing beyond our request archive
  (configurable per key); a fit keeps your training table until it expires (`ttl_hours`) or you delete it (`model.delete()`).
