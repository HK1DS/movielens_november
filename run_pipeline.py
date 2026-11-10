#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end leakage-free pipeline, parameterized by k-core value and split
protocol. Dataset-specific reading goes through the adapter configured in the
`dataset:` section of config.yaml (data_adapters/). Every split gets its OWN profiles, intents, KG and results,
because a user profile that is training-only for TO is not training-only for
TO_GLOBAL (and vice versa).

Layout
  data/interactions.csv                       raw interactions (+timestamp)       stage 0
  data/kc_k{K}/interactions_k{K}.csv          k-core interactions (+timestamp)    stage A
  data/kc_k{K}/ikgr-custom/ikgr-custom.inter  RecBole atomic file (clipped ts)    stage T
  data/kc_k{K}/profiles_k{K}_{SPLIT}.csv      train-only user + static item profiles  stage P
  data/kc_k{K}/profile_provenance_k{K}_{SPLIT}.csv.gz (+.meta.json)                stage P
  run_k{K}_{SPLIT}/                           config, LLM caches/outputs, KG packs, results
    config.yaml, step1_*, step2_*, intent_vocab.json, intents_emb.npy,
    kg_pack.pt, meta_kg_pack.pt, slice_eval_{SPLIT}_result.json,
    per_user_{SPLIT}/*.csv.gz, report_{SPLIT}.md

Stages (select with --steps, default 0ATPBCDER)
  0  prepare_interactions    free   raw dataset (adapter) -> data/interactions.csv (skipped if present)
  A  apply_k_core            free   k-core interactions only
  T  make_temporal_inter     free   RecBole .inter with clipped timestamps
  P  build_trainonly_profiles free  profiles from profile-safe TRAIN rows of this split
  S  seed step1 cache + estimate LLM cost only (no calls)    [not in the default]
  B  step1  (LLM)            $$     exact intents       (asks for confirmation unless --yes)
  C  step2  (LLM + RAG)      $$     related intents
  D  build_kg + build_meta_kg free  intent KG pack + metadata KG pack (adapter.meta_relations)
  E  eval_slices             GPU    early-stopped multi-seed evaluation + leakage guard
  R  paired_compare --report free   paired CIs + Pareto table -> report_{SPLIT}.md

Resuming after an interruption: re-run the same command.
  0/A/T/P  skipped when their outputs exist (--force_prep rebuilds)
  B/C      LLM caches (saved every 50 calls and on Ctrl+C) -> only missing profiles are called;
           a finished stage is skipped via run_k{K}_{SPLIT}/.stage_{B,C}.json (content hash of inputs)
  D        skipped via .stage_D.json while its inputs are unchanged (--redo D forces)
  E        finished spec x seed results are kept (fingerprint = code + config + input CONTENT hashes);
           the spec x seed that was running restarts from epoch 0

Examples
  python run_pipeline.py --k 30 --split TO                       # everything
  python run_pipeline.py --k 30 --split TO --steps 0ATP          # free data prep only
  python run_pipeline.py --k 30 --split TO --steps ER --seeds 2020
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
import yaml

from data_adapters import get_adapter

PY = sys.executable
# Current main comparison (Goodreads, 2026-11): CF baselines + metadata-KG/recency facets
# (+/- LLM intent) on the MF and LightGCN backbones. Other specs: eval_slices.build_specs().
DEFAULT_SPECS = ("IKGR_kgoff,BPR,LightGCN,IKGR_dyn_nointent,IKGR_dyn,"
                 "IKGR_lgcn_dyn_nointent,IKGR_lgcn_dyn")


def sh(cmd, env=None):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    p = subprocess.Popen(cmd, env=env)
    try:
        rc = p.wait()
    except KeyboardInterrupt:
        # Ctrl+C reaches the child too; give it time to cancel queued LLM calls and
        # save its cache instead of killing it mid-write (a 2nd Ctrl+C kills it).
        print("\n[interrupt] waiting for the running step to save its state (Ctrl+C again to kill) ...", flush=True)
        try:
            p.wait(timeout=180)
        except (KeyboardInterrupt, subprocess.TimeoutExpired):
            p.kill()
            p.wait()
        raise SystemExit("[interrupt] stopped. Re-run the same command to resume.")
    dt = round(time.time() - t0, 1)
    if rc == 4:
        raise SystemExit(f"[STOP] ({dt}s) all LLM API keys are exhausted: {' '.join(cmd)}\n"
                         f"  -> add/recharge keys in .env (LUXIA_API_KEY_2=..., or LUXIA_API_KEYS=k1,k2) "
                         f"and re-run the same command; finished calls are cached.")
    if rc != 0:
        raise SystemExit(f"[FAIL] ({dt}s) exit={rc}: {' '.join(cmd)}")
    label = next((c for c in cmd[1:] if c.endswith(".py")), cmd[0])
    print(f"[ok] ({dt}s) {label}", flush=True)


def make_config(k, split, base="config.yaml"):
    cfg = yaml.safe_load(open(base, encoding="utf-8"))
    data_dir = f"data/kc_k{k}"
    run_dir = f"run_k{k}_{split}/"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    p = cfg["paths"]
    p["input_csv"] = f"{data_dir}/profiles_k{k}_{split}.csv"
    p["profile_provenance"] = f"{data_dir}/profile_provenance_k{k}_{split}.csv.gz"
    p["inter_file"] = f"{data_dir}/interactions_k{k}.csv"
    p["workdir"] = run_dir
    p["step1_output"] = f"{run_dir}step1_intents.csv"
    p["step2_output"] = f"{run_dir}step2_related_intents.csv"
    p["kg_triples"] = f"{run_dir}kg_triples.txt"
    p["recbole_dump"] = f"{run_dir}recbole/"
    p["kg_pack"] = f"{run_dir}kg_pack.pt"
    p["meta_kg_pack"] = f"{run_dir}meta_kg_pack.pt"
    r = cfg["rag"]
    r["encoding_npy"] = f"{run_dir}intents_emb.npy"
    r["annoy_index"] = f"{run_dir}intents.ann"
    r["vocab_json"] = f"{run_dir}intent_vocab.json"
    cfg_path = f"{run_dir}config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"[config] wrote {cfg_path}")
    return cfg_path, cfg


def seed_step1_cache(run_dir, sources):
    """step1 output is a pure function of the profile TEXT, so entries can be reused
    across splits/k safely (a reused key means the identical train-only text).
    step2 depends on the split-specific intent vocabulary and is never seeded."""
    dst = os.path.join(run_dir, "step1_cache.json")
    cache = json.load(open(dst, encoding="utf-8")) if os.path.exists(dst) else {"user": {}, "item": {}}
    added = 0
    for src_dir in sources:
        src = os.path.join(src_dir, "step1_cache.json")
        if not src_dir or not os.path.exists(src) or os.path.abspath(src) == os.path.abspath(dst):
            continue
        other = json.load(open(src, encoding="utf-8"))
        for kind in ("user", "item"):
            for key, val in other.get(kind, {}).items():
                if key not in cache[kind]:
                    cache[kind][key] = val
                    added += 1
    if added:
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        print(f"[cache] seeded {added} step1 entries into {dst}")


def has_timestamp(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        return "timestamp" in f.readline().strip().split(",")


def file_sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def llm_preflight(cfg):
    """One tiny LLM call before any bulk spending: a wrong key / blocked network / broken
    SSL would otherwise make every one of ~180k jobs retry for minutes."""
    try:
        import ssl  # noqa: F401  (HTTPS needs it; conda-based venvs can miss its DLLs)
        from ikgr_core.llm_client import LocalLLM
        llm = LocalLLM(**cfg["llm"])
        out = llm.chat("You are a helpful assistant that returns ONLY Python lists.",
                       "Return the Python list ['ok'] and nothing else.")
    except Exception as e:
        raise SystemExit(f"[preflight] LLM check failed before spending: {type(e).__name__}: {e}\n"
                         f"  -> check the LUXIA_API_KEY(s) in .env, network, and that 'import ssl' works in this Python.")
    print(f"[preflight] LLM reachable ({cfg['llm'].get('base_url')}): {str(out).strip()[:40]!r} | {llm.keys.status()}")


def stage_signature(inputs, extra=None):
    """Content hash of a stage's inputs (+ settings). A finished stage is skipped on
    re-run only while this signature is unchanged and its outputs exist."""
    h = hashlib.sha256()
    for p in inputs:
        h.update(p.encode("utf-8"))
        h.update(file_sha256(p).encode("ascii") if os.path.exists(p) else b"missing")
    h.update(json.dumps(extra or {}, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--split", default="TO", choices=["TO", "TO_GLOBAL"])
    ap.add_argument("--steps", default="0ATPBCDER")
    ap.add_argument("--interactions_in", default="data/interactions.csv")
    ap.add_argument("--dataset", default="ikgr-custom", help="RecBole dataset (atomic file) name")
    ap.add_argument("--seed-cache-from", nargs="*", default=None,
                    help="run dirs whose step1 cache may be reused (default: the other split's dir + this split's pilot dir)")
    ap.add_argument("--yes", action="store_true", help="do not ask before paid LLM stages")
    ap.add_argument("--max_usd", type=float, default=None,
                    help="abort paid stages if the observed-rate estimate exceeds this")
    ap.add_argument("--redo", default="",
                    help="stage letters to force even if marked complete, e.g. --redo D")
    ap.add_argument("--force_prep", action="store_true",
                    help="rebuild stages 0/A/T/P even if their outputs exist (default: reuse)")
    ap.add_argument("--epochs", default="40", help="max epochs (early stopping on validation)")
    ap.add_argument("--seeds", default="2020,2021,2022")
    ap.add_argument("--specs", default=DEFAULT_SPECS)
    ap.add_argument("--pop_grid", default="0.25,0.5,1.0",
                    help="popularity-penalty rerank lambdas for CF/KG baselines ('' to disable)")
    ap.add_argument("--dump_valid", action="store_true", help="also dump per-user validation metrics")
    args = ap.parse_args()

    k, split = args.k, args.split
    steps = set(args.steps.upper())
    cfg_path, cfg = make_config(k, split)
    adapter = get_adapter(cfg["dataset"])
    data_dir = f"data/kc_k{k}"
    run_dir = f"run_k{k}_{split}/"
    ds = args.dataset
    env = dict(os.environ, IKGR_CONFIG=cfg_path)
    inter_csv = f"{data_dir}/interactions_k{k}.csv"
    atomic = f"{data_dir}/{ds}/{ds}.inter"
    profiles = cfg["paths"]["input_csv"]
    prov = cfg["paths"]["profile_provenance"]
    step2_csv = cfg["paths"]["step2_output"]
    kg_pack = cfg["paths"]["kg_pack"]
    print(f"\n===== run_pipeline k={k} split={split} | steps={''.join(s for s in '0ATPSBCDER' if s in steps)} =====")

    if "0" in steps:
        if (os.path.exists(args.interactions_in) or os.path.exists(inter_csv)) and not args.force_prep:
            print(f"[0] {args.interactions_in} exists -> skip preprocess")
        else:
            sh([PY, "prepare_interactions.py", "--out", args.interactions_in], env=env)

    def done(*paths):
        ok = all(os.path.exists(x) and os.path.getsize(x) > 0 for x in paths) and not args.force_prep
        if ok:
            print(f"[skip] exists: {', '.join(paths)} (use --force_prep to rebuild)")
        return ok

    if "A" in steps and not done(inter_csv):
        sh([PY, "apply_k_core.py", "--interactions_in", args.interactions_in, "--k", str(k),
            "--out_dir", data_dir], env=env)

    if "T" in steps and not done(atomic):
        if not has_timestamp(inter_csv):
            raise SystemExit(f"{inter_csv} has no timestamp column; the temporal splits need one "
                             f"(the dataset adapter must yield timestamps).")
        clip = cfg["dataset"].get("ts_clip") or adapter.ts_clip or ("", "")
        sh([PY, "make_temporal_inter.py", "--inter_ts", inter_csv, "--out", atomic,
            "--clip_lo", str(clip[0] or ""), "--clip_hi", str(clip[1] or "")], env=env)

    if "P" in steps and not done(profiles, prov, prov + ".meta.json"):
        sh([PY, "build_trainonly_profiles.py", "--split", split, "--inter", atomic,
            "--out_profiles", profiles, "--out_provenance", prov], env=env)

    est_all = None
    if "B" in steps or "C" in steps or "S" in steps:
        sources = args.seed_cache_from
        if sources is None:
            other = "TO_GLOBAL" if split == "TO" else "TO"
            sources = [f"run_k{k}_{other}/", f"run_k{k}_{split}_pilot/"]   # + pilot_llm.py step1 results
        seed_step1_cache(run_dir, sources)
        est_json = os.path.join(run_dir, "llm_cost_estimate.json")
        sh([PY, "estimate_llm_cost.py", "--profiles", profiles, "--run_dir", run_dir, "--json_out", est_json], env=env)
        est_all = json.load(open(est_json, encoding="utf-8"))
        est = est_all["total"]
        if est["calls"] > 0 and ("B" in steps or "C" in steps):
            hi = est["usd_observed_range"][1]
            if args.max_usd is not None and hi > args.max_usd:
                raise SystemExit(f"[cost] estimate ${hi} exceeds --max_usd {args.max_usd}; aborting before any call.")
            llm_preflight(cfg)
            if not args.yes:
                ans = input(f"[cost] {est['calls']:,} LLM calls, est. ${est['usd_list_price']} (list) / "
                            f"${est['usd_observed_range'][0]}-{hi} (observed). Proceed? [y/N] ").strip().lower()
                if ans not in ("y", "yes"):
                    raise SystemExit("[cost] aborted by user (nothing was spent).")

    redo = set(args.redo.upper())

    def stage_done(name, sig, outputs, pending_key=None):
        marker = os.path.join(run_dir, f".stage_{name}.json")
        if name in redo or not os.path.exists(marker):
            return False
        try:
            ok = json.load(open(marker, encoding="utf-8")).get("sig") == sig
        except Exception:
            ok = False
        ok = ok and all(os.path.exists(o) and os.path.getsize(o) > 0 for o in outputs)
        missing = (est_all or {}).get(pending_key, {}).get("calls", 0) if pending_key else 0
        if ok and missing:
            print(f"[stage {name}] marked complete but {missing:,} profiles are missing from its LLM cache "
                  f"(earlier API failures) -> re-running for those only")
            ok = False
        if ok:
            print(f"[skip] stage {name} already complete for these inputs (use --redo {name} to force)")
        return ok

    def mark_done(name, sig):
        with open(os.path.join(run_dir, f".stage_{name}.json"), "w", encoding="utf-8") as f:
            json.dump({"sig": sig, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, f)

    llm_sig = {k: cfg["llm"].get(k) for k in ("base_url", "model", "temperature", "top_p", "max_tokens")}
    if "B" in steps:
        sig = stage_signature([profiles, "prompts/step1_intents.txt"], llm_sig)
        if not stage_done("B", sig, [cfg["paths"]["step1_output"]], pending_key="step1"):
            sh([PY, "-u", "step1.py"], env=env)
            mark_done("B", sig)
    if "C" in steps:
        sig = stage_signature([cfg["paths"]["step1_output"], "prompts/step2_related.txt"],
                              dict(llm_sig, rag={k: cfg["rag"].get(k) for k in ("encoder", "knn_k")}))
        if not stage_done("C", sig, [step2_csv, cfg["rag"]["vocab_json"], cfg["rag"]["encoding_npy"]],
                          pending_key="step2"):
            sh([PY, "-u", "step2.py"], env=env)
            mark_done("C", sig)

    d_sig = stage_signature([step2_csv, cfg["rag"]["vocab_json"], cfg["rag"]["encoding_npy"], atomic]
                            + sorted(adapter.source_files()),
                            {"dataset": cfg["dataset"]}) if "D" in steps else None
    if "D" in steps and not stage_done("D", d_sig, [kg_pack, cfg["paths"]["meta_kg_pack"]]):
        sh([PY, "build_kg.py", "--step2_csv", step2_csv, "--vocab", cfg["rag"]["vocab_json"],
            "--emb", cfg["rag"]["encoding_npy"], "--out", kg_pack], env=env)
        sh([PY, "build_meta_kg.py", "--inter", atomic, "--out", cfg["paths"]["meta_kg_pack"]], env=env)
        mark_done("D", d_sig)

    if "E" in steps:
        eenv = dict(env, IKGR_SPLIT=split, IKGR_EPOCHS=args.epochs, IKGR_SEEDS=args.seeds,
                    IKGR_SPECS=args.specs, IKGR_REQUIRE_PROVENANCE="1")
        if args.pop_grid:
            eenv["IKGR_POP_RERANK_GRID"] = args.pop_grid
        if args.dump_valid:
            eenv["IKGR_DUMP_VALID"] = "1"
        sh([PY, "-u", "eval_slices.py"], env=eenv)

    if "R" in steps:
        sh([PY, "paired_compare.py", "--run_dir", run_dir, "--split", split, "--report"], env=env)

    print(f"\n===== run_pipeline k={k} split={split} complete =====")


if __name__ == "__main__":
    main()
