# Security

Please report vulnerabilities privately to security@yhatlabs.com, not as a public issue. We acknowledge reports within
two business days.

- API keys: the client sends your key only in the `Authorization` header to the configured `base_url`, and never writes it to
  disk or logs (`YHAT_LOG` logs method, path, status, timing and request ids only).
- Data: inputs are sent to the YHat Labs API for processing. A saved model (a fit) is kept until it expires (`ttl_days`, 1 to
  365 days after its last use) or you delete it (`model.delete()`). How request data is handled is in the
  [privacy policy](https://yhatlabs.com/privacy).
