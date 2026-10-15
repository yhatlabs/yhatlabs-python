"""A one-line activity indicator on stderr while requests are in flight, only in an interactive terminal (never in scripts,
pipes or CI, so output stays clean), and only once a call has run for a couple of seconds. YHAT_PROGRESS=0 turns it off.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Iterator

DELAY_S = 2.0
_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_lock = threading.Lock()
_active: dict = {}  # token -> (label, started)
_thread = None


def enabled() -> bool:
    return os.environ.get("YHAT_PROGRESS", "1") != "0" and hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


def _run() -> None:
    global _thread
    i, shown = 0, False
    while True:
        time.sleep(0.12)
        with _lock:
            if not _active:
                if shown:
                    sys.stderr.write("\r\033[K")
                    sys.stderr.flush()
                _thread = None
                return
            label, started = min(_active.values(), key=lambda v: v[1])
            n = len(_active)
        elapsed = time.monotonic() - started
        if elapsed < DELAY_S:
            continue
        more = f" (+{n - 1} more)" if n > 1 else ""
        sys.stderr.write(f"\r\033[K{_FRAMES[i % len(_FRAMES)]} {label}{more} · {elapsed:.0f}s")
        sys.stderr.flush()
        shown, i = True, i + 1


@contextmanager
def activity(label: str) -> Iterator[None]:
    if not enabled():
        yield
        return
    global _thread
    token = object()
    with _lock:
        _active[token] = (label, time.monotonic())
        if _thread is None:
            _thread = threading.Thread(target=_run, daemon=True)
            _thread.start()
    try:
        yield
    finally:
        with _lock:
            _active.pop(token, None)
