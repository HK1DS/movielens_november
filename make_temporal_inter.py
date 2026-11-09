#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Write the RecBole atomic .inter (the evaluation source of truth) WITH a
`timestamp:float` field so RecBole can do a temporal (order=TO) split.

Timestamps can optionally be clamped to [--clip_lo, --clip_hi] (YYYY-MM-DD).
The window comes from the dataset adapter (`ts_clip`) or `dataset.ts_clip` in
config.yaml; e.g. Goodreads used 2006-01-01..2017-12-31 because its dates
contained garbage years. Missing/non-positive timestamps become clip_lo when a
window is set; without a window they stay 0.0 (= earliest) and are counted.
"""
import argparse, csv, os
from datetime import datetime


def _epoch(day):
    return datetime.strptime(day, "%Y-%m-%d").timestamp() if day else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inter_ts", required=True, help="CSV user_id,item_id,rating,timestamp")
    ap.add_argument("--out", required=True, help="e.g. data/kc_k30/ikgr-custom/ikgr-custom.inter")
    ap.add_argument("--clip_lo", default="", help="YYYY-MM-DD clamp floor (also used for missing timestamps)")
    ap.add_argument("--clip_hi", default="", help="YYYY-MM-DD clamp ceiling")
    args = ap.parse_args()

    lo, hi = _epoch(args.clip_lo), _epoch(args.clip_hi)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n, clipped, missing = 0, 0, 0
    with open(args.inter_ts, "r", encoding="utf-8") as fin, \
         open(args.out, "w", newline="", encoding="utf-8") as fout:
        r = csv.DictReader(fin)
        if "timestamp" not in (r.fieldnames or []):
            raise SystemExit(f"{args.inter_ts} has no timestamp column (the adapter must provide one).")
        w = csv.writer(fout, delimiter="\t")
        w.writerow(["user_id:token", "item_id:token", "rating:float", "timestamp:float"])
        for row in r:
            n += 1
            try:
                ts = float(row.get("timestamp", 0) or 0)
            except Exception:
                ts = 0.0
            c = ts
            if ts <= 0:
                missing += 1
                c = lo if lo is not None else 0.0
            else:
                if lo is not None:
                    c = max(c, lo)
                if hi is not None:
                    c = min(c, hi)
            if c != ts:
                clipped += 1
            w.writerow([row["user_id"], row["item_id"], float(row["rating"]), c])
    print(f"[temporal .inter] rows={n} clipped={clipped} missing_ts={missing} "
          f"window=[{args.clip_lo or '-'}, {args.clip_hi or '-'}] -> {args.out}")


if __name__ == "__main__":
    main()
