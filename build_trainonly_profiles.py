#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build LEAKAGE-FREE user profiles (+ static item profiles) for one split protocol.

For a given split (TO or TO_GLOBAL) this script
  1. reads the exact RecBole atomic .inter file used for evaluation,
  2. reproduces RecBole's temporal split (ikgr_core/temporal_split.py),
  3. builds each user's profile ONLY from profile-safe training rows
     (train rows strictly earlier than the user's/global first non-train row),
  4. writes the cover CSV (user_id,user_profile,item_id,item_profile) consumed by
     step1/step2, plus a provenance file listing every (user_id, item_id) that
     contributed text. eval_slices.py verifies the provenance against RecBole's
     real train split and aborts on any violation.

Users without safe history get an EMPTY profile -> no LLM call, no intents.

Item metadata comes from the dataset adapter (data_adapters/, `dataset:` section
of the config). Item profiles are the adapter's static item text (first
--item_max_text_len chars), so item intents are split-independent and their LLM
cache can be shared across splits.

Structured user profile (aggregated over the user's most recent --agg_n safe items):
  "<adapter.recent_label>: <title>; <title>; ... | <facet.label>: v1, v2, ... | ..."
  recent titles = newest --recent_n items; one facet line per adapter.profile_facets entry.
  Facet values present on more than --ubiquitous_df of all items are dropped (no signal).

Usage
  python build_trainonly_profiles.py --split TO \
      --inter data/kc_k30/ikgr-custom/ikgr-custom.inter \
      --out_profiles data/kc_k30/profiles_k30_TO.csv \
      --out_provenance data/kc_k30/profile_provenance_k30_TO.csv.gz
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

from data_adapters import load_adapter
from ikgr_core.temporal_split import SPLITS, assign_phases, read_atomic_inter


def structured_profile(items_newest_first, meta, facets, ubiquitous, recent_label,
                       recent_n, agg_n, max_len):
    """Return (text, contributing_item_ids)."""
    agg = items_newest_first[:agg_n]
    if not agg:
        return "", []
    recent = []
    for iid in agg[:recent_n]:
        m = meta.get(iid)
        if m is not None and m.title:
            recent.append(m.title)
    parts = []
    if recent:
        parts.append(f"{recent_label}: " + "; ".join(recent))
    for fc in facets:
        cnt = Counter()
        for iid in agg:
            m = meta.get(iid)
            if m is None:
                continue
            vals = m.relations.get(fc.relation, [])
            if fc.primary_only:
                vals = vals[:1]
            for v in vals:
                if (fc.relation, v) not in ubiquitous:
                    cnt[v] += 1
        vals = [(v, c) for v, c in cnt.most_common() if c >= fc.min_count][:fc.top_n]
        if vals:
            shown = [f"{v} ({c})" if fc.show_counts else v for v, c in vals]
            parts.append(f"{fc.label}: " + ", ".join(shown))
    text = " | ".join(parts)
    if max_len > 0:
        text = text[:max_len]
    return text, list(agg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=sorted(SPLITS))
    ap.add_argument("--inter", required=True, help="RecBole atomic .inter with timestamp (evaluation source of truth)")
    ap.add_argument("--config", default=None, help="config with the `dataset:` section (default: IKGR_CONFIG)")
    ap.add_argument("--recent_n", type=int, default=10)
    ap.add_argument("--agg_n", type=int, default=50)
    ap.add_argument("--max_text_len", type=int, default=2000)
    ap.add_argument("--item_max_text_len", type=int, default=2000)
    ap.add_argument("--ubiquitous_df", type=float, default=0.5,
                    help="drop facet values present on more than this fraction of items")
    ap.add_argument("--out_profiles", required=True)
    ap.add_argument("--out_provenance", required=True)
    args = ap.parse_args()

    adapter = load_adapter(args.config)
    df = read_atomic_inter(args.inter)
    phase, safe = assign_phases(df, args.split)
    df["phase"] = phase
    df["safe"] = safe
    users = sorted(str(u) for u in df["user_id"].unique())
    items = sorted(str(i) for i in df["item_id"].unique())
    print(f"[profiles] {args.split}: rows={len(df)} users={len(users)} items={len(items)} "
          f"train={int((phase == 0).sum())} valid={int((phase == 1).sum())} test={int((phase == 2).sum())} "
          f"safe_train={int(safe.sum())}", flush=True)

    meta = adapter.load_items(set(items))
    print(f"[profiles] adapter={adapter.name}: metadata for {len(meta)}/{len(items)} items", flush=True)

    facets = list(adapter.profile_facets)
    df_cnt = Counter()
    for m in meta.values():
        for fc in facets:
            for v in set(m.relations.get(fc.relation, [])):
                df_cnt[(fc.relation, v)] += 1
    ubiquitous = {k for k, c in df_cnt.items() if c > args.ubiquitous_df * max(1, len(meta))}
    if ubiquitous:
        print(f"[profiles] ubiquitous facet values dropped from user profiles: {sorted(ubiquitous)}", flush=True)

    tr = df[df["safe"]].copy()
    tr["_row"] = np.arange(len(tr))
    tr = tr.sort_values(["user_id", "timestamp", "_row"], ascending=[True, False, False], kind="mergesort")
    user_profile, prov_rows, n_items_used = {}, [], []
    for uid, g in tr.groupby("user_id", sort=False, observed=True):
        uid = str(uid)
        seq = [str(x) for x in g["item_id"].tolist()]
        text, used = structured_profile(seq, meta, facets, ubiquitous, adapter.recent_label,
                                        args.recent_n, args.agg_n, args.max_text_len)
        if text:
            user_profile[uid] = text
            prov_rows.extend((uid, i) for i in used)
            n_items_used.append(len(used))

    n_empty = len(users) - len(user_profile)
    print(f"[profiles] users with profile={len(user_profile)} empty={n_empty} "
          f"avg contributing items={np.mean(n_items_used) if n_items_used else 0:.1f}", flush=True)

    def item_text(i):
        m = meta.get(i)
        t = m.profile if m is not None else ""
        return t[:args.item_max_text_len] if args.item_max_text_len > 0 else t

    # cover CSV: every user and every item appears at least once
    n = max(len(users), len(items))
    rows = []
    for k in range(n):
        u = users[k % len(users)]
        i = items[k % len(items)]
        rows.append((u, user_profile.get(u, ""), i, item_text(i)))
    os.makedirs(os.path.dirname(os.path.abspath(args.out_profiles)), exist_ok=True)
    pd.DataFrame(rows, columns=["user_id", "user_profile", "item_id", "item_profile"]).to_csv(
        args.out_profiles, index=False)

    prov = pd.DataFrame(prov_rows, columns=["user_id", "item_id"])
    prov.to_csv(args.out_provenance, index=False, compression="gzip")
    meta_out = {
        "split": args.split, "ratios": SPLITS[args.split][0], "group_by_user": SPLITS[args.split][1],
        "inter": os.path.abspath(args.inter), "dataset": adapter.name, "style": "structured",
        "recent_n": args.recent_n, "agg_n": args.agg_n,
        "max_text_len": args.max_text_len, "n_users": len(users), "n_items": len(items),
        "n_users_with_profile": len(user_profile), "n_users_empty_profile": n_empty,
        "n_provenance_pairs": len(prov),
        "phase_counts": {"train": int((phase == 0).sum()), "valid": int((phase == 1).sum()),
                         "test": int((phase == 2).sum())},
    }
    with open(args.out_provenance + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)
    print(f"[profiles] saved -> {args.out_profiles}\n[profiles] provenance -> {args.out_provenance}", flush=True)


if __name__ == "__main__":
    main()
