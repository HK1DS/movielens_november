"""
Per-API-key request lanes for the paid LLM stages (step1 / step2).

  IKGR_LLM_KEY_MODE=parallel   (default) the first IKGR_LLM_ACTIVE_KEYS usable keys (default 3) are
                               called at the same time. Each key has its own pacing
                               (rate_limit.SharedRateLimiter: 429 cooldown, AIMD interval) and at most
                               IKGR_LLM_WORKERS_PER_KEY requests in flight (default 8), so the thread
                               pool is workers_per_key x active keys. A 429 slows down only the key
                               that got it. Keys beyond the active ones (e.g. LUXIA_API_KEY_4 / _5)
                               are STANDBY: unused until an active key is exhausted, then the next
                               standby key (in .env order) takes over its share. With no standby
                               left, an exhausted key's requests move to the remaining keys.
                               IKGR_LLM_ACTIVE_KEYS=0 = every key at once.
  IKGR_LLM_KEY_MODE=sequential the earlier behaviour: one key at a time (the first usable one), the
                               next key only after it is exhausted; all workers share its pacing.
  IKGR_STEP1_WORKERS / IKGR_STEP2_WORKERS, when set, fix the TOTAL number of worker threads
                               (parallel: spread evenly over the keys).

Whether parallel keys are faster depends on where the provider applies its rate limit: per key ->
up to n_keys x the throughput; per account (keys of the same account) -> about the same total with
more 429s. The per-key counters in the progress log (`keys` column) and in step*_throughput.json
show which case applies.
"""
import os
import threading
import time
from contextlib import contextmanager

import requests

from .key_pool import AllKeysExhausted, KeyExhausted
from .rate_limit import SharedRateLimiter, backoff_with_jitter, retry_after_seconds


class Lane:
    """One API key: its pacing state, in-flight count and counters."""

    def __init__(self, idx):
        self.idx = idx
        self.limiter = SharedRateLimiter()
        self.inflight = 0
        self.leases = 0
        self.ok = 0
        self.n429 = 0


class KeyLanes:
    def __init__(self, pool, mode=None, workers_per_key=None, total_workers=None, active_keys=None):
        self.pool = pool
        self.mode = (mode or os.environ.get("IKGR_LLM_KEY_MODE", "parallel")).strip().lower()
        if self.mode not in ("parallel", "sequential"):
            raise ValueError(f"IKGR_LLM_KEY_MODE must be 'parallel' or 'sequential', not {self.mode!r}")
        n = len(pool)
        if self.mode == "parallel":
            active = int(active_keys if active_keys is not None
                         else os.environ.get("IKGR_LLM_ACTIVE_KEYS", "3") or 3)
            self.active = n if active <= 0 else min(active, n)
            per_key = workers_per_key or int(os.environ.get("IKGR_LLM_WORKERS_PER_KEY", "8"))
            self.workers = int(total_workers or per_key * self.active)
            self.cap = max(1, -(-self.workers // self.active))   # in-flight requests allowed per key
        else:
            self.active = 1
            self.workers = int(total_workers or 8)
            self.cap = self.workers
        self.lanes = [Lane(i) for i in range(n)]
        self._cv = threading.Condition()
        self._started = {ln.idx for ln in self._active_lanes()}   # keys that have been active so far

    def _active_lanes(self):
        """The usable keys in use now: the first `active` ones in .env order (the rest are standby)."""
        return [self.lanes[i] for i in self.pool.usable()[:self.active]]

    def _announce_takeovers(self, active):
        for ln in active:
            if ln.idx not in self._started:
                self._started.add(ln.idx)
                print(f"\n[llm-keys] standby {self.pool.name(ln.idx)} takes over from an exhausted key", flush=True)

    # ---- choosing a key -------------------------------------------------------------------------
    @contextmanager
    def lease(self):
        """Reserve one in-flight slot on a usable key (raises AllKeysExhausted when none is left).
        parallel: the key whose pacing lets a request start soonest, then the least busy, then the
        least used one (round robin among idle keys)."""
        with self._cv:
            while True:
                now = time.monotonic()
                usable = self._active_lanes()
                if not usable:
                    raise AllKeysExhausted(self.pool.summary())
                self._announce_takeovers(usable)
                free = [ln for ln in usable if ln.inflight < self.cap]
                if free:
                    lane = min(free, key=lambda ln: (max(now, ln.limiter.ready_at()), ln.inflight,
                                                     ln.leases, ln.idx))
                    lane.inflight += 1
                    lane.leases += 1
                    break
                self._cv.wait(timeout=1.0)        # a slot frees up, or a key is marked exhausted
        try:
            yield lane
        finally:
            with self._cv:
                lane.inflight -= 1
                self._cv.notify_all()

    def wake(self):
        with self._cv:
            self._cv.notify_all()

    # ---- reporting ------------------------------------------------------------------------------
    @property
    def min_interval_seconds(self):
        """Largest current pacing interval among the active keys (progress log column paced_s)."""
        return max((ln.limiter.min_interval_seconds for ln in self._active_lanes()), default=0.0)

    def name(self, lane):
        return self.pool.name(lane.idx)

    def describe(self):
        n = len(self.lanes)
        if self.mode == "parallel":
            standby = (f" + {n - self.active} standby key(s) that take over when an active key is exhausted"
                       if n > self.active else "")
            return (f"parallel over {self.active} key(s){standby}: up to {self.cap} request(s) in flight per "
                    f"key, {self.workers} worker threads (IKGR_LLM_KEY_MODE=sequential for one key at a time)")
        return f"sequential: one key at a time ({n} configured), {self.workers} worker threads"

    def status(self):
        """Pool status + per-key pacing and counters, e.g. for the progress log."""
        usable = set(self.pool.usable())
        active = {ln.idx for ln in self._active_lanes()}
        per = []
        for ln in self.lanes:
            if ln.idx in active:
                per.append(f"#{ln.idx + 1} {ln.limiter.min_interval_seconds:.2f}s {ln.ok}ok/{ln.n429}x429")
            elif ln.idx in usable:
                per.append(f"#{ln.idx + 1} standby")
            elif ln.ok or ln.n429:
                per.append(f"#{ln.idx + 1} exhausted {ln.ok}ok/{ln.n429}x429")
        return f"{len(self.lanes)} key(s), {len(self.lanes) - len(usable)} exhausted, {self.mode}: " + " | ".join(per)

    def per_key(self):
        active = {ln.idx for ln in self._active_lanes()}
        return [{"key": f"#{ln.idx + 1}", "ok": ln.ok, "n429": ln.n429,
                 "paced_interval_s": ln.limiter.min_interval_seconds,
                 "exhausted": not self.pool.is_usable(ln.idx),
                 "standby": self.pool.is_usable(ln.idx) and ln.idx not in active} for ln in self.lanes]


def chat_with_retry(llm, lanes, progress, sys_prompt, prompt, max_retries, initial_delay):
    """One LLM call with per-key pacing and retries (shared by step1 / step2).

    429: cooldown + pacing on THAT key only (Retry-After honoured, jitter), then retry (the retry may
    go to another key). Other HTTP / network errors: exponential backoff. Exhausted key: retried at
    once on another key (not counted as an attempt). After max_retries failed attempts the last try's
    error propagates. AllKeysExhausted always propagates."""
    delay = initial_delay
    attempt = 0
    while True:
        last = attempt >= max_retries
        sleep_for = None
        with lanes.lease() as lane:
            try:
                lane.limiter.acquire()
                out = llm.chat(sys_prompt, prompt, key_idx=lane.idx).strip()
                lane.ok += 1
                lane.limiter.note_success()        # lets this key's pacing relax after a 429-free streak
                return out
            except KeyExhausted:
                lanes.wake()                       # the key dropped out: retry right away on another key
                continue
            except AllKeysExhausted:
                raise
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else "unknown"
                progress.error("429" if status == 429 else "http")
                if last:
                    raise
                if status == 429:
                    lane.n429 += 1
                    wait = max(delay, retry_after_seconds(e.response) or 0.0)
                    switched = lane.limiter.defer_all(wait)
                    sleep_for = backoff_with_jitter(wait)
                    print(f"\n[Rate Limit 429] {lanes.name(lane)}: cooldown {wait:.1f}s; retrying in "
                          f"{sleep_for:.1f}s (attempt {attempt + 1}/{max_retries})...")
                    if switched:
                        print(f"[Rate Limit 429] {lanes.name(lane)}: requests on this key now paced at one "
                              f"request start per {lane.limiter.min_interval_seconds:.2f}s")
                else:
                    sleep_for = backoff_with_jitter(delay)
                    print(f"\n[HTTP Error {status}] Retrying in {sleep_for:.1f} seconds...")
            except Exception as e:
                progress.error("other")
                if last:
                    raise
                sleep_for = backoff_with_jitter(delay)
                print(f"\n[Error] {e}. Retrying in {sleep_for:.1f} seconds...")
        time.sleep(sleep_for)                      # outside the lease: the key's slot is free meanwhile
        delay = min(delay * 2, 60)
        attempt += 1
