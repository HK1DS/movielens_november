#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate LLM calls / tokens / cost for step1 + step2 BEFORE spending anything.

Counts unique non-empty user and item profiles in the cover CSV, subtracts what
is already in the step1/step2 caches of the run directory, and estimates tokens
from character counts (~4 chars/token). Two price views are printed:
  - list price of the configured model (default gpt-4o-mini: $0.15 in / $0.60 out per 1M)
  - the previous project's observed rate (Goodreads k=30: 26,168 calls ~ $13-16 on Luxia gpt-4o-mini);
    re-calibrate OBSERVED_PER_1K_CALLS after the first real batch on the new dataset

Exit code 0 always; the orchestrator asks for confirmation separately.
"""
import argparse
import json
import os

import pandas as pd

STEP1_TEMPLATE_CHARS = 450
STEP2_TEMPLATE_CHARS = 500
STEP2_OPTIONS_CHARS = 100 * 28      # knn_k=100 candidate intents "idx: text"
STEP1_OUT_TOKENS = 80
STEP2_OUT_TOKENS = 50
OBSERVED_PER_1K_CALLS = (0.50, 0.61)


def _load(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"user": {}, "item": {}}


def estimate(profiles_csv, run_dir, price_in=0.15, price_out=0.60):
    df = pd.read_csv(profiles_csv, dtype=str, keep_default_na=False)
    users = {p.strip() for p in df["user_profile"] if p.strip()}
    items = {p.strip() for p in df["item_profile"] if p.strip()}
    c1 = _load(os.path.join(run_dir, "step1_cache.json"))
    c2 = _load(os.path.join(run_dir, "step2_cache.json"))
    out = {}
    for step, cache, tmpl, extra, out_tok in (
            ("step1", c1, STEP1_TEMPLATE_CHARS, 0, STEP1_OUT_TOKENS),
            ("step2", c2, STEP2_TEMPLATE_CHARS, STEP2_OPTIONS_CHARS, STEP2_OUT_TOKENS)):
        pu = [p for p in users if p not in cache.get("user", {})]
        pi = [p for p in items if p not in cache.get("item", {})]
        calls = len(pu) + len(pi)
        tin = sum((len(p) + tmpl + extra) / 4 for p in pu + pi)
        tout = calls * out_tok
        out[step] = {"pending_user_calls": len(pu), "pending_item_calls": len(pi), "calls": calls,
                     "tokens_in": int(tin), "tokens_out": int(tout),
                     "usd_list_price": round(tin / 1e6 * price_in + tout / 1e6 * price_out, 2)}
    calls = out["step1"]["calls"] + out["step2"]["calls"]
    out["total"] = {
        "calls": calls,
        "usd_list_price": round(out["step1"]["usd_list_price"] + out["step2"]["usd_list_price"], 2),
        "usd_observed_range": [round(calls / 1000 * OBSERVED_PER_1K_CALLS[0], 2),
                               round(calls / 1000 * OBSERVED_PER_1K_CALLS[1], 2)],
        "unique_user_profiles": len(users), "unique_item_profiles": len(items),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--price_in", type=float, default=0.15, help="USD per 1M input tokens")
    ap.add_argument("--price_out", type=float, default=0.60, help="USD per 1M output tokens")
    ap.add_argument("--json_out", default="")
    a = ap.parse_args()
    est = estimate(a.profiles, a.run_dir, a.price_in, a.price_out)
    for k in ("step1", "step2"):
        e = est[k]
        print(f"[cost] {k}: {e['calls']:,} calls (users {e['pending_user_calls']:,} / items "
              f"{e['pending_item_calls']:,}), ~{e['tokens_in']:,} in / {e['tokens_out']:,} out tokens, "
              f"list ~${e['usd_list_price']}")
    t = est["total"]
    print(f"[cost] TOTAL {t['calls']:,} calls | list price ~${t['usd_list_price']} | "
          f"observed-rate ~${t['usd_observed_range'][0]}-{t['usd_observed_range'][1]}")
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump(est, f, indent=2)


if __name__ == "__main__":
    main()
