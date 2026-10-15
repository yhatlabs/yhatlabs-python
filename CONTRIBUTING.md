# Contributing

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/ruff check src tests examples && .venv/bin/ruff format --check src tests examples
.venv/bin/pytest -m "not live"                  # offline, against a fake API
YHAT_API_KEY=... .venv/bin/pytest -m live       # against the real API
```

- Every user-visible change gets a line in `CHANGELOG.md` under "Unreleased" (CI checks this on pull requests).
- Public API docstrings use Google style; ruff checks them.
- Only the public model names `chakra-ts` and `chakra-tab` appear anywhere in this repository.
- Releases: bump `src/yhatlabs/_version.py`, move "Unreleased" to the version in `CHANGELOG.md`, tag `vX.Y.Z` and push the
  tag. The release workflow tests, builds and publishes to PyPI via trusted publishing.
