#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM pilot: run step1 + step2 on a small random sample of the REAL train-only profiles
to measure the actual Luxia throughput (with the current rate limits, workers and keys),
project the time of the full stage B/C, and show sample intents for a quality check
before committing to the full run.

Needs stages 0ATPS of the same --k/--split (run_pipeline.py --steps 0ATPS).
Writes run_k{K}_{SPLIT}_pilot/ (its own config, caches, pilot_report.md). The pilot's
step1 cache is reused by the full run (run_pipeline.py seeds it), so those calls are not
paid twice; step2 depends on the full intent vocabulary and is re-run.

  python pilot_llm.py --k 30 --split TO                     # 300 users + 100 items (~800 calls)
  python pilot_llm.py --k 30 --split TO --n_users 1000 --n_items 300 --yes
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

import pandas as pd
import yaml

PY = sys.executable


def _fmt_h(seconds):
    h = seconds / 3600.0
    return f"{h:.1f} h" if h >= 1 else f"{seconds / 60:.0f} min"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--split", default="TO", choices=["TO", "TO_GLOBAL"])
    ap.add_argument("--n_users", type=int, default=300)
    ap.add_argument("--n_items", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=5, help="sample users/items printed in the report")
    ap.add_argument("--yes", action="store_true", help="do not ask before the (small) paid pilot")
    args = ap.parse_args()

    run_dir = f"run_k{args.k}_{args.split}/"
    cfg_path = os.path.join(run_dir, "config.yaml")
    est_path = os.path.join(run_dir, "llm_cost_estimate.json")
    if not (os.path.exists(cfg_path) and os.path.exists(est_path)):
        raise SystemExit(f"[pilot] {cfg_path} / {est_path} missing: run "
                         f"`python run_pipeline.py --k {args.k} --split {args.split} --steps 0ATPS` first.")
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    est = json.load(open(est_path, encoding="utf-8"))
    profiles = pd.read_csv(cfg["paths"]["input_csv"], dtype=str, keep_default_na=False)

    users = profiles.loc[profiles.user_profile.str.strip() != "", ["user_id", "user_profile"]].drop_duplicates("user_id")
    items = profiles.loc[profiles.item_profile.str.strip() != "", ["item_id", "item_profile"]].drop_duplicates("item_id")
    rng = random.Random(args.seed)
    su = users.iloc[sorted(rng.sample(range(len(users)), min(args.n_users, len(users))))]
    si = items.iloc[sorted(rng.sample(range(len(items)), min(args.n_items, len(items))))]
    n = max(len(su), len(si))
    cover = pd.DataFrame({
        "user_id": [su.user_id.iloc[j % len(su)] for j in range(n)],
        "user_profile": [su.user_profile.iloc[j % len(su)] for j in range(n)],
        "item_id": [si.item_id.iloc[j % len(si)] for j in range(n)],
        "item_profile": [si.item_profile.iloc[j % len(si)] for j in range(n)],
    })

    pdir = f"run_k{args.k}_{args.split}_pilot/"
    os.makedirs(pdir, exist_ok=True)
    cover_csv = os.path.join(pdir, "pilot_profiles.csv")
    cover.to_csv(cover_csv, index=False)
    pcfg = json.loads(json.dumps(cfg))
    pcfg["paths"].update({"input_csv": cover_csv, "workdir": pdir,
                          "step1_output": os.path.join(pdir, "step1_intents.csv"),
                          "step2_output": os.path.join(pdir, "step2_related_intents.csv")})
    pcfg["rag"].update({"encoding_npy": os.path.join(pdir, "intents_emb.npy"),
                        "vocab_json": os.path.join(pdir, "intent_vocab.json")})
    pcfg_path = os.path.join(pdir, "config.yaml")
    yaml.safe_dump(pcfg, open(pcfg_path, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)

    calls = 2 * (len(su) + len(si))
    full1, full2 = est["step1"]["calls"], est["step2"]["calls"]
    print(f"[pilot] sample: {len(su)} users + {len(si)} items -> ~{calls} LLM calls "
          f"(~{calls / max(1, full1 + full2) * 100:.2f}% of the full {full1 + full2:,})")
    if not args.yes:
        if input("[pilot] run the pilot now (small paid run)? [y/N] ").strip().lower() not in ("y", "yes"):
            raise SystemExit("[pilot] aborted (nothing was spent).")

    env = dict(os.environ, IKGR_CONFIG=pcfg_path)
    timing = {}
    for step in ("step1", "step2"):
        t0 = time.time()
        rc = subprocess.call([PY, "-u", f"{step}.py"], env=env)
        timing[f"{step}_wall_s"] = round(time.time() - t0, 1)
        if rc != 0:
            raise SystemExit(f"[pilot] {step}.py exited with {rc} (see the log above); "
                             f"re-run the same command to retry only what is missing.")

    def tp(step):
        p = os.path.join(pdir, f"{step}_throughput.json")
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

    t1, t2 = tp("step1"), tp("step2")
    lines = [f"# LLM pilot ({args.split}, k={args.k}) — {time.strftime('%Y-%m-%d %H:%M')}", ""]
    if t1 and t2 and t1["calls"] and t2["calls"]:
        r1, r2 = t1["calls"] / max(t1["seconds"], 1e-3), t2["calls"] / max(t2["seconds"], 1e-3)
        proj = full1 / r1 + full2 / r2
        paced = max(t1.get("paced_interval_s", 0), t2.get("paced_interval_s", 0))
        lines += [
            "| | pilot calls | seconds | calls/s | full calls | projected |",
            "|---|---:|---:|---:|---:|---:|",
            f"| step1 | {t1['calls']} | {t1['seconds']:.0f} | {r1:.2f} | {full1:,} | {_fmt_h(full1 / r1)} |",
            f"| step2 | {t2['calls']} | {t2['seconds']:.0f} | {r2:.2f} | {full2:,} | {_fmt_h(full2 / r2)} |",
            f"| **total** | | | | **{full1 + full2:,}** | **{_fmt_h(proj)}** |", "",
            f"- workers: {t1.get('workers')} | paced after 429: "
            + (f"yes, one request start per {paced:.2f}s (rate-limited)" if paced > 0 else "no (unpaced)"),
            f"- cost estimate for the full run (llm_cost_estimate.json): list ~${est['total']['usd_list_price']}, "
            f"observed-rate ~${est['total']['usd_observed_range'][0]}–{est['total']['usd_observed_range'][1]}",
            "- projection assumes the pilot's rate holds; a long run may hit new 429s (then it slows to the paced rate).",
            "- step2 here picks related intents from the pilot's small vocabulary; the full run uses the full one.", "",
        ]
    else:
        lines += ["(no new calls in this pilot — everything was already cached; delete the pilot folder to re-measure)", ""]

    s1 = pd.read_csv(os.path.join(pdir, "step2_related_intents.csv"), dtype=str, keep_default_na=False)
    lines += ["## Sample users", ""]
    for _, r in s1.drop_duplicates("user_id").head(args.show).iterrows():
        lines += [f"- **{r.user_id}** profile: {r.user_profile[:300]}{'…' if len(r.user_profile) > 300 else ''}",
                  f"  - exact: {r.user_intents_exact}", f"  - related: {r.user_intents_related}"]
    lines += ["", "## Sample items", ""]
    for _, r in s1.drop_duplicates("item_id").head(args.show).iterrows():
        lines += [f"- **{r.item_id}** {r.item_profile}", f"  - exact: {r.item_intents_exact}",
                  f"  - related: {r.item_intents_related}"]
    report = "\n".join(lines) + "\n"
    with open(os.path.join(pdir, "pilot_report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(pdir, "pilot_result.json"), "w", encoding="utf-8") as f:
        json.dump({"step1": t1, "step2": t2, "full_calls": {"step1": full1, "step2": full2}, **timing}, f, indent=2)
    print("\n" + report)
    print(f"[pilot] report -> {os.path.join(pdir, 'pilot_report.md')}")


if __name__ == "__main__":
    main()
