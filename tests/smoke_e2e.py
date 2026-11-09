#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline end-to-end smoke test of the leakage-free pipeline (NO paid LLM, NO GPU).

  synthetic generic-CSV data -> run_pipeline.py stages 0ATPBCDER for TO and TO_GLOBAL
  with a fake OpenAI-compatible LLM server and a fake sentence encoder, tiny model,
  then two negative checks: the leakage guard must abort when (a) a provenance pair
  lies outside the train split, (b) profiles of another split are evaluated.

Usage:  python tests/smoke_e2e.py --workdir /tmp/ikgr_smoke
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
import fake_llm_server  # noqa: E402
from make_synthetic_data import dataset_section  # noqa: E402

CODE = ["prepare_interactions.py", "apply_k_core.py", "make_temporal_inter.py",
        "build_trainonly_profiles.py", "estimate_llm_cost.py", "step1.py", "step2.py",
        "build_kg.py", "build_meta_kg.py", "eval_slices.py",
        "paired_compare.py", "run_pipeline.py", "config.yaml"]


def run(cmd, env, expect_ok=True):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    tail = (r.stdout + r.stderr)[-2500:]
    if expect_ok and r.returncode != 0:
        print(tail); raise SystemExit(f"FAILED: {' '.join(cmd)}")
    if not expect_ok and r.returncode == 0:
        print(tail); raise SystemExit(f"expected failure but succeeded: {' '.join(cmd)}")
    return r.stdout + r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--k", default="5")
    args = ap.parse_args()
    wd = os.path.abspath(args.workdir)
    shutil.rmtree(wd, ignore_errors=True)
    os.makedirs(wd)
    for f in CODE:
        shutil.copy(os.path.join(ROOT, f), wd)
    for d in ("ikgr_core", "prompts", "data_adapters"):
        shutil.copytree(os.path.join(ROOT, d), os.path.join(wd, d))
    srv = fake_llm_server.start()
    port = srv.server_address[1]
    cfg = yaml.safe_load(open(os.path.join(wd, "config.yaml"), encoding="utf-8"))
    cfg["llm"].update({"base_url": f"http://127.0.0.1:{port}/v1", "provider": "openai",
                       "api_key": "offline-test", "model": "fake"})
    cfg["rag"]["knn_k"] = 10
    cfg["dataset"] = dataset_section(wd)
    cfg["recbole"]["embedding_size"] = 32
    yaml.safe_dump(cfg, open(os.path.join(wd, "config.yaml"), "w", encoding="utf-8"), sort_keys=False)

    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "tests", "fakes"),
               IKGR_EVAL_STEP="1", IKGR_STOPPING_STEP="2", IKGR_STEP1_WORKERS="4", IKGR_STEP2_WORKERS="4")
    os.chdir(wd)
    py = sys.executable
    run([py, os.path.join(ROOT, "tests", "make_synthetic_data.py"), "--out_dir", wd], env)

    common = ["--k", args.k, "--yes", "--epochs", "6", "--seeds", "2020,2021", "--pop_grid", "0.5",
              "--specs", "IKGR_kgoff,BPR,LightGCN,IKGR_full_hetero,IKGR_dyn_nointent,IKGR_dyn,"
                         "IKGR_lgcn_dyn_nointent,IKGR_lgcn_dyn,IKGR_full,IKGR_cand_db"]
    out_to = run([py, "run_pipeline.py", "--split", "TO", "--dump_valid"] + common, env)
    calls_to = fake_llm_server.CALLS["n"]
    out_gl = run([py, "run_pipeline.py", "--split", "TO_GLOBAL"] + common, env)
    calls_gl = fake_llm_server.CALLS["n"] - calls_to
    assert "[leakage-guard] OK" in out_to and "[leakage-guard] OK" in out_gl
    assert "[cache] seeded" in out_gl, "TO_GLOBAL should reuse step1 item intents from TO"

    # resume: the same command again must skip B/C/D, make no LLM calls, keep every result
    before = fake_llm_server.CALLS["n"]
    out_re = run([py, "run_pipeline.py", "--split", "TO", "--dump_valid"] + common, env)
    for st in ("B", "C", "D"):
        assert f"[skip] stage {st}" in out_re, f"stage {st} not skipped on resume"
    assert "[invalidate]" not in out_re and "=====" in out_re
    assert fake_llm_server.CALLS["n"] == before, "resume made LLM calls"
    print("[smoke] resume: B/C/D skipped, 0 LLM calls, no result invalidated")

    k = args.k
    res = json.load(open(f"run_k{k}_TO/slice_eval_TO_result.json", encoding="utf-8"))
    for name in ("IKGR_kgoff", "BPR", "LightGCN", "IKGR_dyn", "IKGR_dyn_nointent", "BPR_pop0p5",
                 "IKGR_lgcn_dyn_nointent", "IKGR_lgcn_dyn", "IKGR_full", "IKGR_cand_db"):
        assert name in res and len(res[name]["seeds"]) == 2, name
        s = res[name]["seeds"]["2020"]
        assert s["definitions"]["activity"] == "train interactions"
        assert s["fit"]["best_epoch"] is not None
    assert os.path.exists(f"run_k{k}_TO/per_user_TO/IKGR_dyn__s2020__valid.csv.gz")
    assert os.path.exists(f"run_k{k}_TO/report_TO.md")
    gl = json.load(open(f"run_k{k}_TO_GLOBAL/slice_eval_TO_GLOBAL_result.json", encoding="utf-8"))
    cold0 = gl["IKGR_dyn"]["seeds"]["2020"]["cold_abs_buckets"]["cold0_train"]["n_users"]
    print(f"[smoke] TO_GLOBAL cold0 users = {cold0}; fake LLM calls TO={calls_to} TO_GLOBAL={calls_gl}")

    # ---- negative checks: the guard must stop leaky inputs -------------------
    import pandas as pd
    import numpy as np
    sys.path.insert(0, wd)
    from ikgr_core.temporal_split import assign_phases, read_atomic_inter
    df = read_atomic_inter(f"data/kc_k{k}/ikgr-custom/ikgr-custom.inter")
    ph, _ = assign_phases(df, "TO")
    prov_path = f"data/kc_k{k}/profile_provenance_k{k}_TO.csv.gz"
    backup = prov_path + ".bak"
    shutil.copy(prov_path, backup)
    pv = pd.read_csv(prov_path, dtype=str)
    leak = df[ph == 2][["user_id", "item_id"]].head(1)
    pd.concat([pv, leak]).to_csv(prov_path, index=False, compression="gzip")
    neg_env = dict(env, IKGR_CONFIG=f"run_k{k}_TO/config.yaml", IKGR_SPLIT="TO", IKGR_EPOCHS="1",
                   IKGR_SEEDS="2099", IKGR_SPECS="IKGR_dyn")
    out = run([py, "eval_slices.py"], neg_env, expect_ok=False)
    assert "[leakage-guard] FAILED" in out
    shutil.move(backup, prov_path)
    # profiles of TO evaluated under TO_GLOBAL must be refused
    out = run([py, "eval_slices.py"], dict(neg_env, IKGR_SPLIT="TO_GLOBAL"), expect_ok=False)
    assert "profiles were built for split=TO" in out
    print("[smoke] negative leakage checks OK")
    print(open(f"run_k{k}_TO/report_TO.md", encoding="utf-8").read()[:3000])
    print("SMOKE PASSED")


if __name__ == "__main__":
    main()
