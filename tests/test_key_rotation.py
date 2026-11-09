#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multiple LLM API keys: when a key runs out of credit the next one is used; when
every key is gone, step1/step2 stop cleanly (exit 4, cache saved) and the same
command resumes after keys are added. Uses the offline fake LLM server.

Run:  python -m pytest -q tests/test_key_rotation.py
"""
import json
import os
import subprocess
import sys

import pandas as pd
import pytest
import requests
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import fake_llm_server  # noqa: E402
from ikgr_core import key_pool  # noqa: E402
from ikgr_core.key_pool import AllKeysExhausted, KeyPool, classify_failure, collect_keys  # noqa: E402
from ikgr_core.llm_client import LocalLLM  # noqa: E402

FAKE_ENV = ("FAKE_LLM_KEY_BUDGET", "FAKE_LLM_EXHAUSTED_KEYS", "FAKE_LLM_EXHAUSTED_STYLE", "FAKE_LLM_429_KEYS")
KA, KB, KC = "key-aaaa-1111", "key-bbbb-2222", "key-cccc-3333"


@pytest.fixture(scope="module")
def server():
    srv = fake_llm_server.start()
    yield srv
    srv.shutdown()


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    key_pool._POOLS.clear()
    fake_llm_server.reset()
    monkeypatch.setattr(LocalLLM, "_DOTENV_LOADED", True)      # never read a real .env in tests
    for name in list(os.environ):
        if name.startswith("TEST_LXK") or name in FAKE_ENV or name.startswith("IKGR_LLM_"):
            monkeypatch.delenv(name, raising=False)
    yield


def _llm(server, var="TEST_LXK"):
    return LocalLLM(base_url=f"http://127.0.0.1:{server.server_address[1]}/luxia", api_key="${%s}" % var,
                    model="llm", temperature=0.2, top_p=0.95, max_tokens=64, provider="luxia")


PROMPT = "### Profile:\nRecently watched: Space Story\n### Answer:"


def test_collect_keys_order_and_formats():
    env = {"X_KEY": "a", "X_KEY_10": "j", "X_KEY_2": "b", "X_KEY_1": "a", "X_KEYS": "c, d\ne,b",
           "X_KEY_OTHER": "zzz"}
    assert collect_keys("X_KEY", env) == ["a", "b", "j", "c", "d", "e"]
    assert collect_keys("MISSING", env) == []


@pytest.mark.parametrize("status,body,kind", [
    (402, "", "exhausted"), (401, "", "exhausted"), (403, "forbidden", "exhausted"),
    (429, "", "rate_limited"), (429, '{"error":{"code":"insufficient_quota"}}', "exhausted"),
    (200, '{"errorCode":4001,"message":"크레딧이 부족합니다."}', "exhausted"),
    (500, "upstream timeout", "other"), (400, "bad request: max_tokens", "other"),
])
def test_classify(status, body, kind):
    assert classify_failure(status, body) == kind


def test_extra_pattern_from_env(monkeypatch):
    assert classify_failure(400, "LUX-E042 wallet empty") == "other"
    monkeypatch.setenv("IKGR_LLM_EXHAUSTED_PATTERN", r"LUX-E042")
    assert classify_failure(400, "LUX-E042 wallet empty") == "exhausted"


@pytest.mark.parametrize("style", ["402", "429_quota", "200_error", "401"])
def test_rotates_to_next_key(server, monkeypatch, style):
    monkeypatch.setenv("TEST_LXK_1", KA)
    monkeypatch.setenv("TEST_LXK_2", KB)
    monkeypatch.setenv("FAKE_LLM_EXHAUSTED_KEYS", KA)
    monkeypatch.setenv("FAKE_LLM_EXHAUSTED_STYLE", style)
    llm = _llm(server)
    assert "space stories" in llm.chat("sys", PROMPT)
    assert fake_llm_server.KEY_LOG == [KA, KB]
    assert llm.chat("sys", PROMPT)                            # exhausted key is not retried
    assert fake_llm_server.KEY_LOG == [KA, KB, KB]
    assert "using TEST_LXK key #2/2" in llm.keys.status()


def test_plain_rate_limit_does_not_rotate(server, monkeypatch):
    monkeypatch.setenv("TEST_LXK_1", KA)
    monkeypatch.setenv("TEST_LXK_2", KB)
    monkeypatch.setenv("FAKE_LLM_429_KEYS", KA)
    llm = _llm(server)
    with pytest.raises(requests.exceptions.HTTPError):
        llm.chat("sys", PROMPT)                               # caller's backoff handles it
    assert fake_llm_server.KEY_LOG == [KA]
    assert "#1/2" in llm.keys.status()


def test_long_429_streak_rotates(server, monkeypatch):
    monkeypatch.setenv("TEST_LXK_1", KA)
    monkeypatch.setenv("TEST_LXK_2", KB)
    monkeypatch.setenv("FAKE_LLM_429_KEYS", KA)
    monkeypatch.setenv("IKGR_LLM_ROTATE_AFTER_429_SEC", "0.2")
    llm = _llm(server)
    with pytest.raises(requests.exceptions.HTTPError):
        llm.chat("sys", PROMPT)
    import time
    time.sleep(0.25)
    with pytest.raises(requests.exceptions.HTTPError):
        llm.chat("sys", PROMPT)                               # streak >= 0.2 s -> key #1 marked exhausted
    assert llm.chat("sys", PROMPT)                            # retry goes to key #2
    assert fake_llm_server.KEY_LOG == [KA, KA, KB]


def test_all_keys_exhausted(server, monkeypatch):
    monkeypatch.setenv("TEST_LXKS", f"{KA},{KB}")
    monkeypatch.setenv("FAKE_LLM_EXHAUSTED_KEYS", f"{KA},{KB}")
    llm = _llm(server)
    with pytest.raises(AllKeysExhausted) as ei:
        llm.chat("sys", PROMPT)
    msg = str(ei.value)
    assert "all 2" in msg and "#1/2" in msg and "#2/2" in msg
    assert KA not in msg and KB not in msg                    # keys are never printed in full
    with pytest.raises(AllKeysExhausted):
        llm.chat("sys", PROMPT)                               # no further requests are sent
    assert fake_llm_server.KEY_LOG == [KA, KB]


def test_single_key_old_format_still_works(server, monkeypatch):
    monkeypatch.setenv("TEST_LXK", KA)
    llm = _llm(server)
    assert llm.chat("sys", PROMPT) and llm.api_key == KA and len(llm.keys) == 1


def test_missing_key_is_a_clear_error(server):
    with pytest.raises(ValueError, match="TEST_LXK_1"):
        _llm(server)


def test_pool_is_shared_between_clients(server, monkeypatch):
    monkeypatch.setenv("TEST_LXK_1", KA)
    monkeypatch.setenv("TEST_LXK_2", KB)
    monkeypatch.setenv("FAKE_LLM_EXHAUSTED_KEYS", KA)
    a, b = _llm(server), _llm(server)
    a.chat("sys", PROMPT)
    b.chat("sys", PROMPT)
    assert fake_llm_server.KEY_LOG == [KA, KB, KB]


def test_step1_stops_when_all_keys_run_out_and_resumes(server, tmp_path):
    n_users, n_items = 20, 10
    rows = []
    for k in range(max(n_users, n_items)):
        u, i = k % n_users, k % n_items
        rows.append((f"u{u}", f"Recently watched: Space Story {u} | Frequent genres: dragons{u}",
                     f"i{i}", f"Dinosaurs Movie {i} | pirates{i}"))
    pd.DataFrame(rows, columns=["user_id", "user_profile", "item_id", "item_profile"]).to_csv(
        tmp_path / "profiles.csv", index=False)
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
    cfg["paths"].update({"input_csv": str(tmp_path / "profiles.csv"), "workdir": str(tmp_path / "run"),
                         "step1_output": str(tmp_path / "run" / "step1_intents.csv")})
    cfg["llm"].update({"base_url": f"http://127.0.0.1:{server.server_address[1]}/luxia",
                       "api_key": "${TEST_LXK}", "provider": "luxia"})
    yaml.safe_dump(cfg, open(tmp_path / "config.yaml", "w", encoding="utf-8"), sort_keys=False)

    os.environ["FAKE_LLM_KEY_BUDGET"] = f"{KA}:10,{KB}:10,{KC}:100"     # read by the in-process server
    os.environ["FAKE_LLM_EXHAUSTED_STYLE"] = "200_error"
    env = dict(os.environ, IKGR_CONFIG=str(tmp_path / "config.yaml"), TEST_LXK_1=KA, TEST_LXK_2=KB,
               IKGR_LLM_RETRY_DELAY="0.01", IKGR_LLM_MAX_RETRIES="2", IKGR_STEP1_WORKERS="4")
    run = lambda e: subprocess.run([sys.executable, "step1.py"], cwd=ROOT, env=e, capture_output=True, text=True)

    r1 = run(env)
    out1 = r1.stdout + r1.stderr
    assert r1.returncode == 4, out1[-3000:]
    assert "STOPPED: all LLM API keys exhausted" in out1 and "TEST_LXK key #1/2" in out1
    assert KA not in out1 and KB not in out1
    cache = json.load(open(tmp_path / "run" / "step1_cache.json", encoding="utf-8"))
    assert len(cache["user"]) + len(cache["item"]) == 20                # exactly the two budgets

    r2 = run(dict(env, TEST_LXK_3=KC))                                   # add a key, same command
    out2 = r2.stdout + r2.stderr
    assert r2.returncode == 0, out2[-3000:]
    cache = json.load(open(tmp_path / "run" / "step1_cache.json", encoding="utf-8"))
    assert len(cache["user"]) == n_users and len(cache["item"]) == n_items
    df = pd.read_csv(tmp_path / "run" / "step1_intents.csv").fillna("")
    assert (df["user_intents_exact"] != "[]").all() and (df["item_intents_exact"] != "[]").all()
    for k in FAKE_ENV:
        os.environ.pop(k, None)
