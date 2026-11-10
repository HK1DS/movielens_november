"""
Run-control helpers shared by step1 / step2 (paid LLM stages).

- Time budget: IKGR_LLM_DEADLINE (epoch seconds, set by `run_pipeline.py --max_hours`).
  After the deadline no new request is started; in-flight calls finish, the cache is
  saved and the step exits with code 5 ("paused"). Re-running the same command resumes.
- Stop file: IKGR_LLM_STOP_FILE (run_pipeline sets <run_dir>/STOP). Creating that file has the
  same effect as reaching the deadline -> a graceful pause from outside (no Ctrl+C needed).
- Progress log: <workdir>/llm_progress_<step>.csv, one row every IKGR_PROGRESS_EVERY_S
  seconds (default 60): cumulative calls per kind, window/average rate, 429 / HTTP / other
  error counts, current pacing interval and key status. Rows of resumed runs are appended
  (column run_start tells the runs apart), so the file is the throughput trend.
- keep_awake(): on Windows, asks the OS not to sleep while this process runs.
"""
import csv
import os
import threading
import time
from datetime import datetime

EXIT_PAUSED = 5


def deadline():
    try:
        return float(os.environ.get("IKGR_LLM_DEADLINE", "0") or 0)
    except ValueError:
        return 0.0


def stop_file():
    return os.environ.get("IKGR_LLM_STOP_FILE", "")


def time_up():
    """True once the time budget is over or the stop file exists (no new call may start)."""
    d = deadline()
    if d and time.time() >= d:
        return True
    f = stop_file()
    return bool(f) and os.path.exists(f)


def keep_awake():
    """Prevent Windows from sleeping while the paid stage runs (no-op elsewhere).
    The request ends automatically when the process exits."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
    except Exception:
        return False


class Progress:
    """Thread-safe counters + a background CSV writer."""
    FIELDS = ["run_start", "time", "elapsed_s", "done_total", "done_user", "done_item", "pending_left",
              "rate_window", "rate_avg", "n_429", "n_http_err", "n_other_err", "paced_s", "keys"]

    def __init__(self):
        self._lock = threading.Lock()
        self.done = {"user": 0, "item": 0}
        self.err = {"429": 0, "http": 0, "other": 0}
        self._thread = None
        self._stop = threading.Event()

    def success(self, kind):
        with self._lock:
            self.done[kind] = self.done.get(kind, 0) + 1

    def error(self, which):
        with self._lock:
            self.err[which] = self.err.get(which, 0) + 1

    def total(self):
        with self._lock:
            return sum(self.done.values())

    def start(self, path, pending, limiter, keys, every_s=None):
        every_s = float(os.environ.get("IKGR_PROGRESS_EVERY_S", "60")) if every_s is None else every_s
        self._path, self._pending, self._limiter, self._keys = path, pending, limiter, keys
        self._t0 = time.time()
        self._run_start = datetime.now().isoformat(timespec="seconds")
        self._last = (self._t0, 0)
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            if new:
                csv.writer(f).writerow(self.FIELDS)
        self._row()

        def loop():
            while not self._stop.wait(every_s):
                self._row()
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def _row(self):
        now = time.time()
        with self._lock:
            du, di = self.done.get("user", 0), self.done.get("item", 0)
            e = dict(self.err)
        tot = du + di
        t_prev, n_prev = self._last
        window = (tot - n_prev) / max(now - t_prev, 1e-9) if now > t_prev else 0.0
        self._last = (now, tot)
        row = [self._run_start, datetime.now().isoformat(timespec="seconds"), round(now - self._t0, 1),
               tot, du, di, self._pending - tot, round(window, 3), round(tot / max(now - self._t0, 1e-9), 3),
               e["429"], e["http"], e["other"], round(getattr(self._limiter, "min_interval_seconds", 0.0), 3),
               self._keys.status() if self._keys is not None else ""]
        with open(self._path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5)
            self._row()
