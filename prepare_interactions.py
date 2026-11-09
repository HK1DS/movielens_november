#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 0: raw dataset -> data/interactions.csv (user_id,item_id,rating,timestamp).

Reads through the dataset adapter configured in the `dataset:` section of the
config (IKGR_CONFIG or --config). Optional `dataset.min_rating` keeps only
interactions with rating >= min_rating (e.g. explicit ratings -> implicit positives).

Usage
  python prepare_interactions.py --out data/interactions.csv
"""
import argparse
import csv
import os

import yaml

from data_adapters import get_adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.environ.get("IKGR_CONFIG", "config.yaml"))
    ap.add_argument("--out", default="data/interactions.csv")
    ap.add_argument("--min_rating", type=float, default=None, help="overrides dataset.min_rating")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    ds_cfg = cfg["dataset"]
    adapter = get_adapter(ds_cfg)
    min_rating = args.min_rating if args.min_rating is not None else ds_cfg.get("min_rating")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_in = n_out = n_no_ts = 0
    tmp = args.out + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "item_id", "rating", "timestamp"])
        for uid, iid, rating, ts in adapter.iter_interactions():
            n_in += 1
            if min_rating is not None and rating < float(min_rating):
                continue
            if not ts or ts <= 0:
                n_no_ts += 1
            w.writerow([uid, iid, rating, ts])
            n_out += 1
    os.replace(tmp, args.out)
    print(f"[prepare] adapter={adapter.name} rows in={n_in:,} kept={n_out:,} "
          f"(min_rating={min_rating}) missing_ts={n_no_ts:,} -> {args.out}")


if __name__ == "__main__":
    main()
