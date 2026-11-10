"""
API-key pool: which keys exist and which are exhausted (shared by every LLM worker thread).

Keys come from the environment (and .env, loaded by llm_client) for one base
variable name, e.g. LUXIA_API_KEY:

  LUXIA_API_KEY=key-a                 single key (old format, still works)
  LUXIA_API_KEY_1=key-b               numbered keys, used in numeric order
  LUXIA_API_KEY_2=key-c
  LUXIA_API_KEYS=key-d,key-e          comma/whitespace separated list

Order: LUXIA_API_KEY, LUXIA_API_KEY_1, _2, ..., then LUXIA_API_KEYS (duplicates
removed). acquire() hands out the first key that is not marked exhausted (one key
at a time). The paid stages (step1 / step2) instead go through key_lanes.KeyLanes,
which by default calls every usable key in parallel, each with its own pacing, and
pins a request to one key with key(idx); an exhausted key then drops out and the
other keys carry on (KeyExhausted tells the caller to pick another key).

A key is marked exhausted (for the rest of this process) when a call with it
  - returns HTTP 401 / 402 / 403 (invalid, unpaid or blocked key),
  - returns a body whose text matches EXHAUSTED_PATTERN (credit / balance /
    quota / 크레딧 / 잔액 ...), with any status code, including an HTTP 200
    whose JSON carries an error instead of `choices`,
  - keeps returning HTTP 429 without a single success for
    IKGR_LLM_ROTATE_AFTER_429_SEC seconds (default 600; 0 disables). A plain
    rate limit clears within a minute or two; a key that is still refused after
    ten minutes is treated as unusable.
Plain 429s before that threshold are NOT a reason to rotate: the caller's
backoff handles them on the same key.

Luxia does not document its out-of-credit response. Extra patterns can be added
with IKGR_LLM_EXHAUSTED_PATTERN (a regex, OR-ed with the built-in one); every
distinct error body is printed once (llm_client) so the real message can be
copied from the log. Keys are never printed, only "#i/n (...last4)".
Exhausted state is per process: a re-run probes the keys again from the first
one, so a recharged key is picked up automatically.
"""
import os
import re
import threading
import time

EXHAUSTED_PATTERN = (
    r"insufficient|quota|credit|balance|billing|payment|exceeded your|out of (?:credits?|tokens?)"
    r"|not enough|크레딧|잔액|잔여|포인트|충전|결제|소진|부족|사용량 ?한도|이용 ?한도"
)


class AllKeysExhausted(RuntimeError):
    """Every configured API key is exhausted / unusable in this process."""


class KeyExhausted(RuntimeError):
    """The key a request was pinned to is exhausted; retry the request on another key."""

    def __init__(self, idx):
        super().__init__(f"API key #{idx + 1} is exhausted")
        self.idx = idx


def _exhausted_regex():
    extra = os.environ.get("IKGR_LLM_EXHAUSTED_PATTERN", "").strip()
    pattern = EXHAUSTED_PATTERN + (f"|{extra}" if extra else "")
    return re.compile(pattern, re.IGNORECASE)


def classify_failure(status, body):
    """Return 'exhausted', 'rate_limited' or 'other' for a failed call."""
    if status in (401, 402, 403):
        return "exhausted"
    if body and _exhausted_regex().search(body):
        return "exhausted"
    if status == 429:
        return "rate_limited"
    return "other"


def collect_keys(var_name, environ=None):
    """All keys configured for `var_name` (see module docstring), in order of use."""
    env = os.environ if environ is None else environ
    keys = []

    def add(value):
        for k in re.split(r"[,\s]+", value or ""):
            k = k.strip().strip("\"'")
            if k and k not in keys:
                keys.append(k)

    add(env.get(var_name, ""))
    numbered = []
    for name, value in env.items():
        m = re.fullmatch(re.escape(var_name) + r"_(\d+)", name)
        if m:
            numbered.append((int(m.group(1)), value))
    for _, value in sorted(numbered):
        add(value)
    add(env.get(var_name + "S", ""))
    return keys


def mask(key):
    return f"...{key[-4:]}" if len(key) > 8 else "..."


class KeyPool:
    def __init__(self, keys, label="api", rotate_after_429_sec=None):
        if not keys:
            raise ValueError(f"no API key configured for {label}")
        self.keys = list(keys)
        self.label = label
        if rotate_after_429_sec is None:
            rotate_after_429_sec = float(os.environ.get("IKGR_LLM_ROTATE_AFTER_429_SEC", "600"))
        self.rotate_after_429_sec = rotate_after_429_sec
        self._lock = threading.Lock()
        self._exhausted = {}            # idx -> reason
        self._first_429 = {}            # idx -> monotonic time of first 429 since last success

    def __len__(self):
        return len(self.keys)

    def _name(self, idx):
        return f"{self.label} key #{idx + 1}/{len(self.keys)} ({mask(self.keys[idx])})"

    def name(self, idx):
        return self._name(idx)

    def is_usable(self, idx):
        with self._lock:
            return idx not in self._exhausted

    def usable(self):
        """Indices of the keys not marked exhausted, in order of use."""
        with self._lock:
            return [i for i in range(len(self.keys)) if i not in self._exhausted]

    def key(self, idx):
        """The key with index idx; raises KeyExhausted once it is marked exhausted."""
        with self._lock:
            if idx in self._exhausted:
                raise KeyExhausted(idx)
            return self.keys[idx]

    def _current(self):
        for i in range(len(self.keys)):
            if i not in self._exhausted:
                return i
        return None

    def acquire(self):
        """(idx, key) of the key to use now; raises AllKeysExhausted."""
        with self._lock:
            i = self._current()
            if i is None:
                raise AllKeysExhausted(self._summary())
            return i, self.keys[i]

    def mark_exhausted(self, idx, reason):
        with self._lock:
            if idx in self._exhausted:
                return
            self._exhausted[idx] = reason
            left = [i for i in range(len(self.keys)) if i not in self._exhausted]
            where = (f"{len(left)} usable key(s) left, next in line {self._name(left[0])}" if left
                     else "NO keys left")
            print(f"\n[llm-keys] {self._name(idx)} exhausted ({reason}) -> {where}", flush=True)

    def note_rate_limited(self, idx):
        if self.rotate_after_429_sec <= 0:
            return
        with self._lock:
            first = self._first_429.setdefault(idx, time.monotonic())
            waited = time.monotonic() - first
        if waited >= self.rotate_after_429_sec:
            self.mark_exhausted(idx, f"HTTP 429 for {waited:.0f}s without a success")

    def note_success(self, idx):
        if idx in self._first_429:
            with self._lock:
                self._first_429.pop(idx, None)

    def status(self):
        with self._lock:
            i = self._current()
            using = self._name(i) if i is not None else "none usable"
            return f"{len(self.keys)} key(s), {len(self._exhausted)} exhausted, using {using}"

    def summary(self):
        return self._summary()

    def _summary(self):
        lines = [f"all {len(self.keys)} {self.label} API key(s) are exhausted:"]
        lines += [f"  {self._name(i)}: {r}" for i, r in sorted(self._exhausted.items())]
        lines.append("  -> add or recharge keys in .env and re-run the same command (finished calls are cached).")
        return "\n".join(lines)


_POOLS = {}
_POOLS_LOCK = threading.Lock()


def get_pool(label, keys):
    """One shared pool per (label, keys) in this process, so every LocalLLM instance and
    worker thread sees the same exhausted state."""
    sig = (label, tuple(keys))
    with _POOLS_LOCK:
        if sig not in _POOLS:
            _POOLS[sig] = KeyPool(keys, label)
        return _POOLS[sig]
