#!/usr/bin/env python3
"""Tiny OpenAI-compatible chat server for offline smoke tests (NO real LLM).
step1 prompts -> a Python list of intent phrases derived from profile words.
step2 prompts -> a JSON list of option indices sharing a word with the profile.

Per-key behaviour (read from this process's environment at request time; the key is
taken from the Luxia `apikey` header or `Authorization: Bearer`):
  FAKE_LLM_KEY_BUDGET="k1:5,k2:3"   successful calls allowed per key, then the key is "out of credit"
  FAKE_LLM_EXHAUSTED_KEYS="k1,k2"   keys that are out of credit from the start
  FAKE_LLM_EXHAUSTED_STYLE          how "out of credit" looks: 402 | 429_quota | 200_error | 401
  FAKE_LLM_429_KEYS="k1"            keys that always get a plain HTTP 429 (rate limit, empty body)
KEY_LOG records the key of every request (successful or not)."""
import hashlib
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STOP = {"the", "by", "and", "of", "a", "recently", "consumed", "newest", "first", "frequent",
        "creators", "creator", "common", "tags", "story", "about", "number", "adventure", "s"}
CALLS = {"n": 0}
KEY_LOG = []
_KEY_USED = {}
_LOCK = threading.Lock()


def reset():
    with _LOCK:
        CALLS["n"] = 0
        KEY_LOG.clear()
        _KEY_USED.clear()


def _set(name):
    return {k.strip() for k in os.environ.get(name, "").split(",") if k.strip()}


def _exhausted_response():
    style = os.environ.get("FAKE_LLM_EXHAUSTED_STYLE", "402")
    if style == "429_quota":
        return 429, {"error": {"code": "insufficient_quota",
                               "message": "You exceeded your current quota, please check your plan and billing details."}}
    if style == "200_error":
        return 200, {"errorCode": 4001, "message": "크레딧이 부족합니다."}
    if style == "401":
        return 401, {"error": "Invalid apikey"}
    return 402, {"error": "Payment Required"}


def _answer(prompt):
    prof = prompt.split("### Profile:", 1)[-1]
    if "### Available Options" in prompt:
        prof_part, opts = prof.split("### Available Options", 1)
        words = set(re.findall(r"[a-z]+", prof_part.lower())) - STOP
        sel = []
        for line in opts.splitlines():
            m = re.match(r"\s*(\d+):\s*(.+)", line)
            if m and set(re.findall(r"[a-z]+", m.group(2).lower())) & words:
                sel.append(int(m.group(1)))
        return json.dumps(sel[:8])
    words = [w for w in re.findall(r"[a-z\-]+", prof.split("### Answer", 1)[0].lower()) if w not in STOP and len(w) > 3]
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w); out.append(f"{w} stories")
    return "```python\n" + repr(out[:6]) + "\n```"


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        time.sleep(float(os.environ.get("FAKE_LLM_DELAY", "0")))
        prompt = body["messages"][-1]["content"]
        auth = self.headers.get("Authorization", "")
        key = self.headers.get("apikey") or (auth[7:] if auth.startswith("Bearer ") else "")
        budgets = {}
        for part in os.environ.get("FAKE_LLM_KEY_BUDGET", "").split(","):
            if ":" in part:
                k, n = part.rsplit(":", 1)
                budgets[k.strip()] = int(n)
        with _LOCK:
            KEY_LOG.append(key)
            if key in _set("FAKE_LLM_429_KEYS"):
                verdict = "429"
            elif key in _set("FAKE_LLM_EXHAUSTED_KEYS") or (key in budgets and _KEY_USED.get(key, 0) >= budgets[key]):
                verdict = "exhausted"
            else:
                verdict = "ok"
                if key in budgets:
                    _KEY_USED[key] = _KEY_USED.get(key, 0) + 1
        if verdict == "429":
            self.send_response(429); self.send_header("Content-Length", "0"); self.end_headers()
            return
        if verdict == "exhausted":
            code, obj = _exhausted_response()
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
            return
        every = int(os.environ.get("FAKE_LLM_FAIL_EVERY", "0"))   # deterministic persistent 429s
        if every and "### Profile" in prompt and int(hashlib.md5(prompt.encode()).hexdigest(), 16) % every == 0:
            self.send_response(429); self.send_header("Content-Length", "0"); self.end_headers()
            return
        with _LOCK:
            CALLS["n"] += 1
        txt = _answer(prompt)
        data = json.dumps({"choices": [{"message": {"content": txt}}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def log_message(self, *a):
        pass


def start(port=0):
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
