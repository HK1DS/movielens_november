#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parallel API keys (ikgr_core/key_lanes.py): every usable key is called at the same time, each with
its own pacing and in-flight cap; a 429 slows down only its key; an exhausted key drops out while
the others carry on; IKGR_LLM_KEY_MODE=sequential keeps the one-key-at-a-time behaviour.

Run:  python -m pytest -q tests/test_key_lanes.py
"""
import csv
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from contextlib import ExitStack

import pandas as pd
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import fake_llm_server  # noqa: E402
from ikgr_core.key_lanes import KeyLanes  # noqa: E402
from ikgr_core.key_pool import AllKeysExhausted, KeyPool  # noqa: E402

KA, KB, KC = "key-aaaa-1111", "key-bbbb-2222", "key-cccc-3333"
FAKE_ENV = ("FAKE_LLM_KEY_BUDGET", "FAKE_LLM_EXHAUSTED_KEYS", "FAKE_LLM_EXHAUSTED_STYLE", "FAKE_LLM_429_KEYS",
            "FAKE_LLM_DELAY")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("IKGR_LLM_") or name.startswith("TEST_LXK") or name in FAKE_ENV:
            monkeypatch.delenv(name, raising=False)
    yield
    for k in FAKE_ENV:
        os.environ.pop(k, None)


def _lanes(mode=None, total=None, n=3):
    return KeyLanes(KeyPool([KA, KB, KC][:n], "TEST"), mode=mode, total_workers=total)


def _picks(lanes, n):
    out = []
    for _ in range(n):
        with lanes.lease() as ln:
            out.append(ln.idx)
    return out


def test_parallel_caps_in_flight_per_key():
    lanes = _lanes(total=6)
    assert lanes.workers == 6 and lanes.cap == 2
    with ExitStack() as st:
        got = [st.enter_context(lanes.lease()).idx for _ in range(6)]
        assert Counter(got) == {0: 2, 1: 2, 2: 2}                 # spread evenly, two in flight per key
        done = threading.Event()

        def seventh():
            with lanes.lease():
                done.set()
        t = threading.Thread(target=seventh, daemon=True)
        t.start()
        assert not done.wait(0.3)                                 # every key is full -> waits
    assert done.wait(2.0)                                         # a slot freed up -> proceeds
    t.join(2.0)


def test_default_workers_and_mode_from_env(monkeypatch):
    monkeypatch.setenv("IKGR_LLM_WORKERS_PER_KEY", "3")
    lanes = _lanes()
    assert lanes.mode == "parallel" and lanes.workers == 9 and lanes.cap == 3
    assert "parallel over 3 key(s)" in lanes.describe()
    monkeypatch.setenv("IKGR_LLM_KEY_MODE", "sideways")
    with pytest.raises(ValueError):
        _lanes()


def test_429_on_one_key_steers_new_requests_to_the_others():
    lanes = _lanes(total=6)
    lanes.lanes[0].limiter.defer_all(5.0)                         # key #1 got a 429: 5 s cooldown
    assert 0 not in _picks(lanes, 6)
    assert lanes.lanes[0].limiter.min_interval_seconds == 1.0     # only key #1 is paced
    assert lanes.lanes[1].limiter.min_interval_seconds == 0.0 == lanes.lanes[2].limiter.min_interval_seconds
    assert lanes.min_interval_seconds == 1.0


def test_exhausted_key_drops_out_then_all_exhausted():
    lanes = _lanes(total=6)
    assert _picks(lanes, 6) == [0, 1, 2, 0, 1, 2]                 # idle keys: round robin
    lanes.pool.mark_exhausted(1, "test")
    assert _picks(lanes, 4) == [0, 2, 0, 2]
    assert "1 exhausted" in lanes.status() and "#2 exhausted" not in lanes.status()   # unused key: not listed
    lanes.pool.mark_exhausted(0, "test")
    lanes.pool.mark_exhausted(2, "test")
    with pytest.raises(AllKeysExhausted):
        _picks(lanes, 1)


def test_sequential_mode_uses_one_key_at_a_time():
    lanes = _lanes(mode="sequential", total=4)
    assert lanes.workers == 4 and lanes.cap == 4
    with ExitStack() as st:
        assert {st.enter_context(lanes.lease()).idx for _ in range(4)} == {0}
    lanes.pool.mark_exhausted(0, "test")
    assert set(_picks(lanes, 3)) == {1}


# ---- step1 end to end with the offline fake LLM -----------------------------------------------------

@pytest.fixture()
def server():
    from ikgr_core import key_pool
    key_pool._POOLS.clear()
    fake_llm_server.reset()
    srv = fake_llm_server.start()
    yield srv
    srv.shutdown()


def _step1_env(tmp_path, server, n_users=30, n_items=10):
    rows = [(f"u{k}", f"Recently watched: Space Story {k} | Frequent genres: dragons{k}",
             f"i{k % n_items}", f"Dinosaurs Movie {k % n_items} | pirates{k % n_items}") for k in range(n_users)]
    pd.DataFrame(rows, columns=["user_id", "user_profile", "item_id", "item_profile"]).to_csv(
        tmp_path / "profiles.csv", index=False)
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
    cfg["paths"].update({"input_csv": str(tmp_path / "profiles.csv"), "workdir": str(tmp_path / "run"),
                         "step1_output": str(tmp_path / "run" / "step1_intents.csv")})
    cfg["llm"].update({"base_url": f"http://127.0.0.1:{server.server_address[1]}/luxia",
                       "api_key": "${TEST_LXK}", "provider": "luxia"})
    yaml.safe_dump(cfg, open(tmp_path / "config.yaml", "w", encoding="utf-8"), sort_keys=False)
    return dict(os.environ, IKGR_CONFIG=str(tmp_path / "config.yaml"), TEST_LXK_1=KA, TEST_LXK_2=KB,
                TEST_LXK_3=KC, IKGR_LLM_RETRY_DELAY="0.05", IKGR_LLM_WORKERS_PER_KEY="2",
                IKGR_PROGRESS_EVERY_S="0.2", IKGR_LLM_429_MIN_INTERVAL_SECONDS="0.2")


def _run_step1(env):
    r = subprocess.run([sys.executable, "step1.py"], cwd=ROOT, env=env, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def test_step1_calls_all_keys_in_parallel(server, tmp_path):
    os.environ["FAKE_LLM_DELAY"] = "0.05"
    env = _step1_env(tmp_path, server)
    rc, out = _run_step1(env)
    assert rc == 0, out[-3000:]
    assert "parallel over 3 key(s): up to 2 request(s) in flight per key, 6 worker threads" in out
    used = Counter(fake_llm_server.KEY_LOG)
    assert set(used) == {KA, KB, KC} and min(used.values()) >= 8      # 40 calls spread over all keys
    tp = json.load(open(tmp_path / "run" / "step1_throughput.json", encoding="utf-8"))
    assert tp["key_mode"] == "parallel" and tp["workers"] == 6
    assert sum(k["ok"] for k in tp["per_key"]) == tp["calls"] == 40
    prog = list(csv.DictReader(open(tmp_path / "run" / "llm_progress_step1.csv", encoding="utf-8")))
    assert "parallel: #1 " in prog[-1]["keys"] and "#3 " in prog[-1]["keys"]
    assert KA not in out and KB not in out and KC not in out           # keys never printed in full


def test_rate_limited_key_only_slows_itself(server, tmp_path):
    os.environ["FAKE_LLM_DELAY"] = "0.05"
    os.environ["FAKE_LLM_429_KEYS"] = KA                               # key #1 always answers 429
    env = _step1_env(tmp_path, server)
    t0 = time.time()
    rc, out = _run_step1(env)
    assert rc == 0, out[-3000:]
    assert "[Rate Limit 429] TEST_LXK key #1/3" in out
    assert "TEST_LXK key #2/3" not in out.split("[Rate Limit 429]", 1)[-1].split("\n")[0]
    cache = json.load(open(tmp_path / "run" / "step1_cache.json", encoding="utf-8"))
    assert len(cache["user"]) == 30 and len(cache["item"]) == 10       # everything done by keys #2/#3
    used = Counter(fake_llm_server.KEY_LOG)
    assert used[KA] < used[KB] and used[KA] < used[KC]                 # the paced key gets few requests
    tp = json.load(open(tmp_path / "run" / "step1_throughput.json", encoding="utf-8"))
    per = {k["key"]: k for k in tp["per_key"]}
    assert per["#1"]["ok"] == 0 and per["#1"]["n429"] >= 1 and per["#1"]["paced_interval_s"] >= 0.2
    assert per["#2"]["paced_interval_s"] == 0.0 == per["#3"]["paced_interval_s"]
    assert time.time() - t0 < 60


def test_sequential_mode_env_keeps_one_key(server, tmp_path):
    env = dict(_step1_env(tmp_path, server), IKGR_LLM_KEY_MODE="sequential", IKGR_STEP1_WORKERS="4")
    rc, out = _run_step1(env)
    assert rc == 0, out[-3000:]
    assert "sequential: one key at a time (3 configured), 4 worker threads" in out
    assert set(fake_llm_server.KEY_LOG) == {KA}
