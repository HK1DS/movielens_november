"""Cooperative request pacing for concurrent LLM pipeline stages.

Ported from HK1DS/hk1ds_jwkim_movielens (built after a 2026-07 Luxia run where eight
unpaced workers got immediate 429s/502s). Defaults here start UNPACED, as the
2026-11 Goodreads runs did fine with eight workers (~4.8 calls/s); the first 429
switches every worker to the shared stable interval (IKGR_LLM_429_MIN_INTERVAL_SECONDS,
default 1.0 s) for the rest of the process.

  IKGR_LLM_MIN_INTERVAL_SECONDS      minimum spacing between request starts from the start (default 0)
  IKGR_LLM_429_MIN_INTERVAL_SECONDS  spacing used after the first 429 (default 1.0)
"""

from __future__ import annotations

import email.utils
import os
import random
import threading
import time
from datetime import datetime, timezone
from typing import Optional


class SharedRateLimiter:
    """Serialize request starts and share provider-imposed cooldowns across workers.

    A provider limit applies to the API key, not to an individual worker.  Without
    this small shared gate, workers that all receive a 429 tend to retry together
    and repeatedly create another burst.
    """

    def __init__(self, min_interval_seconds: Optional[float] = None):
        if min_interval_seconds is None:
            min_interval_seconds = float(os.environ.get("IKGR_LLM_MIN_INTERVAL_SECONDS", "0"))
        if min_interval_seconds < 0:
            raise ValueError("IKGR_LLM_MIN_INTERVAL_SECONDS must be non-negative")
        self.min_interval_seconds = min_interval_seconds
        self.fallback_interval_seconds = float(
            os.environ.get("IKGR_LLM_429_MIN_INTERVAL_SECONDS", "1.0")
        )
        if self.fallback_interval_seconds < 0:
            raise ValueError("IKGR_LLM_429_MIN_INTERVAL_SECONDS must be non-negative")
        self._lock = threading.Lock()
        self._next_allowed = 0.0
        self._cooldown_until = 0.0

    def acquire(self) -> None:
        """Wait until this worker may start its next API request."""
        with self._lock:
            now = time.monotonic()
            scheduled = max(now, self._next_allowed, self._cooldown_until)
            self._next_allowed = scheduled + self.min_interval_seconds
        wait = scheduled - now
        if wait > 0:
            time.sleep(wait)

    def defer_all(self, seconds: float) -> bool:
        """Apply a 429 backoff and, when needed, enter the stable pacing mode.

        Returns whether this response changed the request-start interval.  A run
        may begin unpaced to use available quota, but its first 429 switches all
        later worker requests to the conservative fallback interval.
        """
        with self._lock:
            self._cooldown_until = max(self._cooldown_until, time.monotonic() + max(0.0, seconds))
            previous = self.min_interval_seconds
            self.min_interval_seconds = max(
                self.min_interval_seconds, self.fallback_interval_seconds
            )
            return self.min_interval_seconds > previous


def retry_after_seconds(response) -> Optional[float]:
    """Read an HTTP Retry-After header in either seconds or HTTP-date form."""
    if response is None:
        return None
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = email.utils.parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, IndexError):
            return None


def backoff_with_jitter(delay: float) -> float:
    """Avoid synchronized retries while retaining an exponential-backoff floor."""
    return delay * random.uniform(0.8, 1.2)
