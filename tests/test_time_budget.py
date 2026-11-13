#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time budget for the paid stages: after IKGR_LLM_DEADLINE no new call starts, the cache
is saved and the step exits 5 ("paused"); items are processed before users; the per-minute
progress CSV records the trend across resumed runs; run_pipeline --max_hours stops after a
paused B instead of running C.

Run:  python -m pytest -q tests/test_time_budget.py
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import time

import pandas as pd
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import fake_llm_server  # noqa: E402
from make_synthetic_data import dataset_section  # noqa: E402


@pytest.fixture()
def server():
    fake_llm_server.reset()
    srv = fake_llm_server.start()
    yield srv
    srv.shutdown()
    os.environ.pop("FAKE_LLM_DELAY", None)


def _cfg(tmp_path, server):
    n_users, n_items = 30, 10
    rows = [(f"u{k}", f"Recently watched: Space Story {k} | Frequent genres: dragons{k}",
             f"i{k % n_items}", f"Dinosaurs Movie {k % n_items} | pirates{k % n_items}") for k in range(n_users)]
    pd.DataFrame(rows, columns=["user_id", "user_profile", "item_id", "item_profile"]).to_csv(
        tmp_path / "profiles.csv", index=False)
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))
    cfg["paths"].update({"input_csv": str(tmp_path / "profiles.csv"), "workdir": str(tmp_path / "run"),
                         "step1_output": str(tmp_path / "run" / "step1_intents.csv")})
    cfg["llm"].update({"base_url": f"http://127.0.0.1:{server.server_address[1]}/v1", "provider": "openai",
                       "api_key": "offline-test", "model": "fake"})
    yaml.safe_dump(cfg, open(tmp_path / "config.yaml", "w", encoding="utf-8"), sort_keys=False)
    return n_users, n_items


def test_step1_pauses_at_deadline_items_first_and_resumes(server, tmp_path):
    n_users, n_items = _cfg(tmp_path, server)
    os.environ["FAKE_LLM_DELAY"] = "0.1"
    env = dict(os.environ, IKGR_CONFIG=str(tmp_path / "config.yaml"), IKGR_STEP1_WORKERS="2",
               IKGR_PROGRESS_EVERY_S="0.3", IKGR_LLM_DEADLINE=str(time.time() + 1.5))
    r = subprocess.run([sys.executable, "step1.py"], cwd=ROOT, env=env, capture_output=True, text=True)
    out = r.stdout + r.stderr
    assert r.returncode == 5, out[-2000:]
    assert "PAUSED: time budget reached" in out and "time budget: stops starting new calls" in out
    cache = json.load(open(tmp_path / "run" / "step1_cache.json", encoding="utf-8"))
    n1 = len(cache["user"]) + len(cache["item"])
    assert 0 < n1 < n_users + n_items
    if n1 >= n_items:
        assert len(cache["item"]) == n_items                       # items were done first
    prog = list(csv.DictReader(open(tmp_path / "run" / "llm_progress_step1.csv", encoding="utf-8")))
    assert len(prog) >= 3 and int(prog[-1]["done_total"]) == n1
    assert int(prog[-1]["pending_left"]) == n_users + n_items - n1
    assert prog[-1]["keys"].startswith("1 key(s)")

    env.pop("IKGR_LLM_DEADLINE")
    r = subprocess.run([sys.executable, "step1.py"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    cache = json.load(open(tmp_path / "run" / "step1_cache.json", encoding="utf-8"))
    assert len(cache["user"]) == n_users and len(cache["item"]) == n_items
    prog = list(csv.DictReader(open(tmp_path / "run" / "llm_progress_step1.csv", encoding="utf-8")))
    assert len({p["run_start"] for p in prog}) == 2                # both runs in one trend file
    assert int(prog[-1]["done_total"]) == n_users + n_items - n1


def test_run_pipeline_max_hours_pauses_after_B(server, tmp_path):
    wd = tmp_path / "repo"
    wd.mkdir()
    for f in ("prepare_interactions.py", "apply_k_core.py", "make_temporal_inter.py", "build_trainonly_profiles.py",
              "estimate_llm_cost.py", "step1.py", "step2.py", "run_pipeline.py", "config.yaml"):
        shutil.copy(os.path.join(ROOT, f), wd)
    for d in ("ikgr_core", "prompts", "data_adapters"):
        shutil.copytree(os.path.join(ROOT, d), wd / d)
    cfg = yaml.safe_load(open(wd / "config.yaml", encoding="utf-8"))
    cfg["llm"].update({"base_url": f"http://127.0.0.1:{server.server_address[1]}/v1", "provider": "openai",
                       "api_key": "offline-test", "model": "fake"})
    cfg["dataset"] = dataset_section(str(wd))
    yaml.safe_dump(cfg, open(wd / "config.yaml", "w", encoding="utf-8"), sort_keys=False)
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "tests", "fakes"), IKGR_STEP1_WORKERS="2")

    def run(*cmd):
        r = subprocess.run([sys.executable, *cmd], cwd=wd, env=env, capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr

    rc, out = run(os.path.join(ROOT, "tests", "make_synthetic_data.py"), "--out_dir", str(wd))
    assert rc == 0, out
    rc, out = run("run_pipeline.py", "--k", "5", "--split", "TO", "--steps", "0ATP")
    assert rc == 0, out[-2000:]
    os.environ["FAKE_LLM_DELAY"] = "0.05"
    rc, out = run("run_pipeline.py", "--k", "5", "--split", "TO", "--steps", "BC", "--yes", "--max_hours", "0.0005")
    assert rc == 5, out[-2000:]
    assert "[budget] paid stages stop starting new calls" in out and "[PAUSE]" in out
    assert "step2.py" not in out.split("[PAUSE]")[0].split("[budget]")[-1]   # C never started
    assert not os.path.exists(wd / "run_k5_TO" / "step2_related_intents.csv")
    assert not os.path.exists(wd / "run_k5_TO" / ".stage_B.json")              # B not marked complete


def test_pacing_relaxes_after_429_free_streak(monkeypatch):
    monkeypatch.setenv("IKGR_LLM_PACE_RECOVER_EVERY", "5")
    monkeypatch.setenv("IKGR_LLM_MIN_INTERVAL_SECONDS", "0")
    sys.path.insert(0, ROOT)
    from ikgr_core.rate_limit import SharedRateLimiter
    r = SharedRateLimiter()
    assert r.min_interval_seconds == 0                        # starts unpaced
    assert r.defer_all(0) is True and r.min_interval_seconds == 1.0   # first 429 -> 1 s
    for _ in range(5):
        r.note_success()
    assert r.min_interval_seconds == 0.75                     # relaxed after a 429-free streak
    r.defer_all(0)
    assert r.min_interval_seconds == pytest.approx(1.125)     # repeat 429 while paced -> x1.5
    for _ in range(5 * 40):
        r.note_success()
    assert r.min_interval_seconds == 0.0                      # back to unpaced
    monkeypatch.setenv("IKGR_LLM_PACE_RECOVER_EVERY", "0")
    r2 = SharedRateLimiter()
    r2.defer_all(0)
    for _ in range(1000):
        r2.note_success()
    assert r2.min_interval_seconds == 1.0                     # 0 = original behaviour (never relax)


def test_one_escalation_per_429_burst(monkeypatch):
    """Eight in-flight workers hitting 429 together must not multiply the interval eight times."""
    monkeypatch.setenv("IKGR_LLM_MIN_INTERVAL_SECONDS", "0")
    sys.path.insert(0, ROOT)
    from ikgr_core.rate_limit import SharedRateLimiter
    r = SharedRateLimiter()
    r.defer_all(0)
    assert r.min_interval_seconds == 1.0
    for _ in range(8):                                        # same burst: cooldown still running
        r.defer_all(5.0)
    assert r.min_interval_seconds == pytest.approx(1.5)       # escalated once (x1.5), not 1.5**8
    for _ in range(8):
        r.defer_all(5.0)
    assert r.min_interval_seconds == pytest.approx(1.5)


def test_stop_file_pauses_gracefully(server, tmp_path):
    """Creating IKGR_LLM_STOP_FILE (run_pipeline: <run_dir>/STOP) pauses like the deadline: exit 5, cache saved."""
    n_users, n_items = _cfg(tmp_path, server)
    os.environ["FAKE_LLM_DELAY"] = "0.1"
    stop = tmp_path / "STOP"
    env = dict(os.environ, IKGR_CONFIG=str(tmp_path / "config.yaml"), IKGR_STEP1_WORKERS="2",
               IKGR_PROGRESS_EVERY_S="0.3", IKGR_LLM_STOP_FILE=str(stop))
    p = subprocess.Popen([sys.executable, "step1.py"], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    time.sleep(1.5)
    stop.write_text("stop")
    out = p.communicate(timeout=60)[0]
    assert p.returncode == 5, out[-2000:]
    assert "PAUSED: time budget reached or STOP file found" in out
    cache = json.load(open(tmp_path / "run" / "step1_cache.json", encoding="utf-8"))
    assert 0 < len(cache["user"]) + len(cache["item"]) < n_users + n_items


def test_slow_key_relaxes_by_time(monkeypatch):
    """A key stuck at the 10 s cap gets few successes; it must still relax after 429-free time
    (otherwise it stays throttled for hours, e.g. after another key ran out of credit)."""
    monkeypatch.setenv("IKGR_LLM_PACE_RECOVER_EVERY", "200")
    monkeypatch.setenv("IKGR_LLM_PACE_RECOVER_SECONDS", "0.05")
    sys.path.insert(0, ROOT)
    from ikgr_core.rate_limit import SharedRateLimiter
    r = SharedRateLimiter()
    r.min_interval_seconds = 10.0
    r.defer_all(0)                                            # last 429 just now: no relax yet
    assert r.note_success() is None and r.min_interval_seconds == 10.0
    time.sleep(0.06)
    assert r.note_success() == pytest.approx(7.5)             # 429-free for the window -> one step
    assert r.note_success() is None                           # next step needs another window
    time.sleep(0.06)
    assert r.note_success() == pytest.approx(5.625)
