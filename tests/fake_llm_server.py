#!/usr/bin/env python3
"""Tiny OpenAI-compatible chat server for offline smoke tests (NO real LLM).
step1 prompts -> a Python list of intent phrases derived from profile words.
step2 prompts -> a JSON list of option indices sharing a word with the profile."""
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
        every = int(os.environ.get("FAKE_LLM_FAIL_EVERY", "0"))   # deterministic persistent 429s
        if every and "### Profile" in prompt and int(hashlib.md5(prompt.encode()).hexdigest(), 16) % every == 0:
            self.send_response(429); self.send_header("Content-Length", "0"); self.end_headers()
            return
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
