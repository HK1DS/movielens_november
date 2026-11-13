"""Cooperative request pacing for concurrent LLM pipeline stages.

Ported from HK1DS/hk1ds_jwkim_movielens (built after a 2026-07 Luxia run where eight
unpaced workers got immediate 429s/502s). Defaults here start UNPACED, as the
2026-11 Goodreads runs did fine with eight workers (~4.8 calls/s); the first 429
switches every worker to the shared stable interval (IKGR_LLM_429_MIN_INTERVAL_SECONDS,
default 1.0 s). Unlike the original (which stayed at that interval for the rest of the
process), the interval relaxes again: after IKGR_LLM_PACE_RECOVER_EVERY consecutive
successes without a 429 it is multiplied by IKGR_LLM_PACE_RECOVER_FACTOR, down to 0
(unpaced); a 429 while already paced multiplies it by 1.5 (capped at 10 s). So a long run
settles near the rate the account actually allows instead of staying at the most
conservative one.

  IKGR_LLM_MIN_INTERVAL_SECONDS      minimum spacing between request starts from the start (default 0)
  IKGR_LLM_429_MIN_INTERVAL_SECONDS  spacing used after the first 429 (default 1.0)
  IKGR_LLM_PACE_RECOVER_EVERY        429-free successes before relaxing (default 200; 0 = never relax)
  IKGR_LLM_PACE_RECOVER_FACTOR       relax multiplier (default 0.75)
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

    One instance per API key (ikgr_core/key_lanes.py): workers using the same key share it.

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
        self.recover_every = int(os.environ.get("IKGR_LLM_PACE_RECOVER_EVERY", "200"))
        self.recover_factor = float(os.environ.get("IKGR_LLM_PACE_RECOVER_FACTOR", "0.75"))
        # Also relax after this many 429-free SECONDS (with at least one success in between): a slow key
        # (e.g. 10 s interval = 0.1 calls/s) would otherwise need ~2,000 s per relax step and stay throttled
        # for hours, e.g. after another key ran out of credit. 0 = count-based only.
        self.recover_seconds = float(os.environ.get("IKGR_LLM_PACE_RECOVER_SECONDS", "60"))
        self.max_interval_seconds = 10.0
        self._lock = threading.Lock()
        self._next_allowed = 0.0
        self._cooldown_until = 0.0
        self._streak = 0
        self._last_change = time.monotonic()      # last 429 escalation or relax step

    def acquire(self) -> None:
        """Wait until this worker may start its next API request."""
        with self._lock:
            now = time.monotonic()
            scheduled = max(now, self._next_allowed, self._cooldown_until)
            self._next_allowed = scheduled + self.min_interval_seconds
        wait = scheduled - now
        if wait > 0:
            time.sleep(wait)

    def ready_at(self) -> float:
        """Monotonic time from which the next request may start (may lie in the past); used to pick
        the key that can serve a request soonest."""
        with self._lock:
            return max(self._next_allowed, self._cooldown_until)

    def defer_all(self, seconds: float) -> bool:
        """Apply a 429 backoff and, when needed, enter the stable pacing mode.

        Returns whether this response changed the request-start interval.  A run
        may begin unpaced to use available quota, but its first 429 switches all
        later worker requests to the conservative fallback interval.
        """
        with self._lock:
            now = time.monotonic()
            # 429s from requests that were already in flight during the current cooldown belong to
            # the same burst: escalate the interval once per burst, not once per worker.
            same_burst = now < self._cooldown_until
            self._cooldown_until = max(self._cooldown_until, now + max(0.0, seconds))
            self._streak = 0
            previous = self.min_interval_seconds
            self._last_change = now
            if previous > 0 and not same_burst:
                self.min_interval_seconds = min(previous * 1.5, self.max_interval_seconds)
            self.min_interval_seconds = max(self.min_interval_seconds, self.fallback_interval_seconds)
            return self.min_interval_seconds > previous

    def note_success(self) -> Optional[float]:
        """Count a successful call; after a 429-free streak relax the pacing interval.
        Returns the new interval when it changed, else None."""
        with self._lock:
            if self.min_interval_seconds <= 0:
                return None
            self._streak += 1
            now = time.monotonic()
            by_count = self.recover_every > 0 and self._streak >= self.recover_every
            by_time = self.recover_seconds > 0 and now - self._last_change >= self.recover_seconds
            if not (by_count or by_time):
                return None
            self._streak = 0
            self._last_change = now
            new = self.min_interval_seconds * self.recover_factor
            self.min_interval_seconds = 0.0 if new < 0.05 else new
            return self.min_interval_seconds


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
