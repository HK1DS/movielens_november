#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pilot_llm.py on synthetic data with the offline fake LLM: measures throughput, projects the
full B/C time, writes a report with sample intents; the full run then reuses the pilot's
step1 results. Also checks the shared 429 pacing (Retry-After honoured, switch to the stable
interval, no key rotation for a plain rate limit).

Run:  python -m pytest -q tests/test_pilot.py
"""
import json
import os
import re
import shutil
import subprocess
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import fake_llm_server  # noqa: E402
from make_synthetic_data import dataset_section  # noqa: E402

pytest.importorskip("recbole")
CODE = ["prepare_interactions.py", "apply_k_core.py", "make_temporal_inter.py", "build_trainonly_profiles.py",
        "estimate_llm_cost.py", "step1.py", "step2.py", "build_kg.py", "build_meta_kg.py", "eval_slices.py",
        "paired_compare.py", "run_pipeline.py", "pilot_llm.py", "config.yaml"]


def test_pilot_projects_time_and_full_run_reuses_it(tmp_path):
    wd = tmp_path / "repo"
    wd.mkdir()
    for f in CODE:
        shutil.copy(os.path.join(ROOT, f), wd)
    for d in ("ikgr_core", "prompts", "data_adapters"):
        shutil.copytree(os.path.join(ROOT, d), wd / d)
    fake_llm_server.reset()
    srv = fake_llm_server.start()
    try:
        cfg = yaml.safe_load(open(wd / "config.yaml", encoding="utf-8"))
        cfg["llm"].update({"base_url": f"http://127.0.0.1:{srv.server_address[1]}/v1", "provider": "openai",
                           "api_key": "offline-test", "model": "fake"})
        cfg["rag"]["knn_k"] = 10
        cfg["dataset"] = dataset_section(str(wd))
        yaml.safe_dump(cfg, open(wd / "config.yaml", "w", encoding="utf-8"), sort_keys=False)
        env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "tests", "fakes"), IKGR_STEP1_WORKERS="4",
                   IKGR_STEP2_WORKERS="4", IKGR_LLM_RETRY_DELAY="0.05", IKGR_LLM_429_MIN_INTERVAL_SECONDS="0.02")

        def run(*cmd, ok=True):
            r = subprocess.run([sys.executable, *cmd], cwd=wd, env=env, capture_output=True, text=True)
            out = r.stdout + r.stderr
            assert (r.returncode == 0) == ok, out[-3000:]
            return out

        run(os.path.join(ROOT, "tests", "make_synthetic_data.py"), "--out_dir", str(wd))
        run("run_pipeline.py", "--k", "5", "--split", "TO", "--steps", "0ATPS")

        # first 3 requests are rate-limited (with Retry-After): shared cooldown + switch to pacing
        os.environ["FAKE_LLM_429_FIRST"] = "3"
        os.environ["FAKE_LLM_RETRY_AFTER"] = "0.1"
        try:
            out = run("pilot_llm.py", "--k", "5", "--split", "TO", "--n_users", "20", "--n_items", "10", "--yes")
        finally:
            os.environ.pop("FAKE_LLM_429_FIRST"); os.environ.pop("FAKE_LLM_RETRY_AFTER")
        assert re.search(r"\[Rate Limit 429\] config key #1/1 \(\.\.\.\w+\): cooldown", out)
        assert re.search(r"requests on this key now paced at one request start per 0\.0\d+s", out)  # 0.02, x1.5 if 429s race
        assert "[llm-keys]" not in out and "0 exhausted" in out           # plain 429 never rotates keys
        pdir = wd / "run_k5_TO_pilot"
        res = json.load(open(pdir / "pilot_result.json", encoding="utf-8"))
        assert res["step1"]["calls"] == 30 and res["step2"]["calls"] == 30
        assert res["step1"]["paced_interval_s"] >= 0.02                 # paced (x1.5 per repeat 429)
        report = open(pdir / "pilot_report.md", encoding="utf-8").read()
        assert "| **total** |" in report and "calls/s" in report
        assert "## Sample users" in report and "exact:" in report and "related:" in report
        full1 = res["full_calls"]["step1"]

        # full run: the pilot's 30 step1 results are seeded, not paid again
        before = fake_llm_server.CALLS["n"]
        out = run("run_pipeline.py", "--k", "5", "--split", "TO", "--steps", "SB", "--yes")
        assert "[cache] seeded 30 step1 entries" in out
        assert fake_llm_server.CALLS["n"] - before == full1 - 30 + 1   # + 1 preflight call
        tp = json.load(open(wd / "run_k5_TO" / "step1_throughput.json", encoding="utf-8"))
        assert tp["calls"] == full1 - 30
    finally:
        srv.shutdown()
