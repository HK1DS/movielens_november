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

Structured user profile, shaped by the adapter (data_adapters/base.py; CLI flags override):
  "<adapter.recent_label>: <title>; <title>; ... | <facet.label>: v1, v2, ... | ..."
  window  = the user's newest adapter.profile_window safe-train rows (None = whole train history;
            --agg_n, 0 = whole history)
  titles  = first adapter.title_n items of the window (--recent_n) in adapter.title_order (--title_order):
            "recent" newest first, or "rating" highest rated first (ties newest first)
  facets  = one line per adapter.profile_facets entry, counted over the whole window.
  Facet values present on more than --ubiquitous_df of all items are dropped (no signal).
  Provenance = every (user, item) of the window (all of them feed the facet counts).

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


def facet_index(meta, facets, ubiquitous):
    """Per facet: {item_id: facet values that count} (primary_only and ubiquitous filters applied)."""
    out = []
    for fc in facets:
        d = {}
        for iid, m in meta.items():
            vals = m.relations.get(fc.relation, [])
            if fc.primary_only:
                vals = vals[:1]
            vals = tuple(v for v in vals if (fc.relation, v) not in ubiquitous)
            if vals:
                d[iid] = vals
        out.append(d)
    return out


def structured_profile(items_newest_first, meta, facets, ubiquitous, recent_label,
                       recent_n, agg_n, max_len, ratings=None, title_order="recent", fidx=None):
    """Return (text, contributing_item_ids).

    items_newest_first: the user's safe-train items, newest first (`ratings` aligned with it).
    Window = newest agg_n items (None/0 = all). Titles = first recent_n items of the window in
    title_order ("recent": newest first; "rating": highest rating first, ties keep newest first).
    Facets are counted over the whole window."""
    agg = list(items_newest_first[:agg_n] if agg_n else items_newest_first)
    if not agg:
        return "", []
    if title_order == "rating" and ratings is not None:
        r = list(ratings[:len(agg)])
        ranked = [agg[k] for k in sorted(range(len(agg)), key=lambda k: -r[k])]   # stable sort
    elif title_order in ("recent", "rating"):
        ranked = agg
    else:
        raise ValueError(f"unknown title_order {title_order!r} (use 'recent' or 'rating')")
    titles = []
    for iid in ranked[:recent_n]:
        m = meta.get(iid)
        if m is not None and m.title:
            titles.append(m.title)
    parts = []
    if titles:
        parts.append(f"{recent_label}: " + "; ".join(titles))
    if fidx is None:
        fidx = facet_index({i: meta[i] for i in set(agg) if i in meta}, facets, ubiquitous)
    for fc, d in zip(facets, fidx):
        cnt = Counter()
        for iid in agg:
            vals = d.get(iid)
            if vals:
                cnt.update(vals)
        vals = [(v, c) for v, c in cnt.most_common() if c >= fc.min_count][:fc.top_n]
        if vals:
            shown = [f"{v} ({c})" if fc.show_counts else v for v, c in vals]
            parts.append(f"{fc.label}: " + ", ".join(shown))
    text = " | ".join(parts)
    if max_len > 0:
        text = text[:max_len]
    return text, agg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=sorted(SPLITS))
    ap.add_argument("--inter", required=True, help="RecBole atomic .inter with timestamp (evaluation source of truth)")
    ap.add_argument("--config", default=None, help="config with the `dataset:` section (default: IKGR_CONFIG)")
    ap.add_argument("--recent_n", type=int, default=None, help="titles shown (default: adapter.title_n)")
    ap.add_argument("--agg_n", type=int, default=None,
                    help="newest-N safe rows per user (default: adapter.profile_window; 0 = whole train history)")
    ap.add_argument("--title_order", choices=["recent", "rating"], default=None,
                    help="title order (default: adapter.title_order)")
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

    title_n = args.recent_n if args.recent_n is not None else int(getattr(adapter, "title_n", 10))
    window = args.agg_n if args.agg_n is not None else getattr(adapter, "profile_window", 50)
    window = int(window) if window and int(window) > 0 else None
    title_order = args.title_order or getattr(adapter, "title_order", "recent")
    if title_order == "rating" and "rating" not in df.columns:
        print("[profiles] WARNING: no rating column in the .inter -> titles ordered newest first", flush=True)
        title_order = "recent"
    print(f"[profiles] format: '{adapter.recent_label}' = {title_n} titles ({title_order}) | window = "
          f"{'whole train history' if window is None else f'newest {window}'} | facets = "
          f"{[fc.label for fc in facets]}", flush=True)

    tr = df[df["safe"]].copy()
    tr["_row"] = np.arange(len(tr))
    tr = tr.sort_values(["user_id", "timestamp", "_row"], ascending=[True, False, False], kind="mergesort")
    if window is not None:
        tr = tr.groupby("user_id", sort=False, observed=True).head(window)   # keeps the newest-first order
    fidx = facet_index(meta, facets, ubiquitous)
    has_rating = "rating" in tr.columns
    user_profile, empty_users, n_items_used = {}, [], []
    for uid, g in tr.groupby("user_id", sort=False, observed=True):
        seq = [str(x) for x in g["item_id"].tolist()]
        rat = g["rating"].tolist() if has_rating else None
        text, used = structured_profile(seq, meta, facets, ubiquitous, adapter.recent_label, title_n, None,
                                        args.max_text_len, ratings=rat, title_order=title_order, fidx=fidx)
        if text:
            user_profile[str(uid)] = text
            n_items_used.append(len(used))
        else:
            empty_users.append(uid)
    # provenance = every window row of a user who got a profile (built vectorised: up to ~20M rows)
    keep = ~tr["user_id"].isin(empty_users) if empty_users else np.ones(len(tr), dtype=bool)
    prov = tr.loc[keep, ["user_id", "item_id"]]

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

    prov.to_csv(args.out_provenance, index=False, compression={"method": "gzip", "compresslevel": 1})
    meta_out = {
        "split": args.split, "ratios": SPLITS[args.split][0], "group_by_user": SPLITS[args.split][1],
        "inter": os.path.abspath(args.inter), "dataset": adapter.name, "style": "structured",
        "recent_label": adapter.recent_label, "title_order": title_order, "title_n": title_n,
        "profile_window": window, "recent_n": title_n, "agg_n": window,
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
