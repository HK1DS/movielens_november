#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tiny synthetic dataset in the generic-CSV adapter format, for offline tests.

Writes (in --out_dir/raw):
  interactions.csv   user_id,item_id,rating,timestamp   (epoch seconds; NOT chronological)
  items.csv          item_id,title,profile,genre,creator,tag

Users have a latent topic preference so the models have signal to learn.
Timestamps deliberately include ties, missing values and out-of-range years,
to exercise the RecBole split replica and the clip window. Every item carries
the tag "family" (ubiquitous -> must be dropped from user profiles).

dataset_section(out_dir) returns the matching `dataset:` config section.
"""
import argparse
import csv
import os
import random
from datetime import datetime, timedelta, timezone

TOPICS = ["dragons", "animals", "space", "princess", "dinosaurs", "pirates", "school", "friendship"]


def dataset_section(out_dir):
    raw = os.path.join(out_dir, "raw")
    return {
        "adapter": "csv",
        "options": {
            "name": "synthetic",
            "interactions": os.path.join(raw, "interactions.csv"),
            "items": os.path.join(raw, "items.csv"),
            "relations": [{"name": "genre"}, {"name": "creator"},
                          {"name": "tag", "max_per_item": 15, "min_df": 5}],
            "facets": [{"relation": "creator", "label": "Frequent creators", "top_n": 8,
                        "min_count": 2, "primary_only": True, "show_counts": True},
                       {"relation": "tag", "label": "Common tags", "top_n": 15}],
            "recent_label": "Recently consumed (newest first)",
            "ts_clip": ["2006-01-01", "2017-12-31"],
        },
        "min_rating": None,
        "ts_clip": None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--users", type=int, default=400)
    ap.add_argument("--items", type=int, default=250)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    raw = os.path.join(args.out_dir, "raw")
    os.makedirs(raw, exist_ok=True)

    items = []
    for i in range(args.items):
        topic = TOPICS[i % len(TOPICS)]
        fmt_tag = rng.choice(["short", "long", "series"])
        items.append({
            "item_id": str(50000 + i),
            "title": f"The {topic.title()} Story {i} by Creator {i % 60}",
            # some items without explicit profile text -> adapter builds one from title + relations
            "profile": "" if i % 5 == 0 else f"A story about {topic} and adventure number {i}.",
            "genre": f"{topic}|{'adventure' if i % 3 else 'comedy'}",
            "creator": f"Creator {i % 60}",
            "tag": f"family|{topic}|{fmt_tag}",
        })
    pop = [1.0 / (1 + (i % 97)) ** 0.7 for i in range(args.items)]

    base = datetime(2009, 1, 1, tzinfo=timezone(timedelta(hours=-8)))
    rows = []
    for u in range(args.users):
        uid = f"u{u:05d}"
        fav = rng.sample(TOPICS, 2)
        n = rng.randint(35, 90)
        start = base + timedelta(days=rng.randint(0, 2000))
        chosen = set()
        for k in range(n):
            # 70% from favourite topics, rest popularity-weighted
            if rng.random() < 0.7:
                t = rng.choice(fav)
                cand = [i for i in range(args.items) if TOPICS[i % len(TOPICS)] == t]
                i = rng.choice(cand)
            else:
                i = rng.choices(range(args.items), weights=pop)[0]
            if i in chosen:
                continue
            chosen.add(i)
            day = start + timedelta(days=k * rng.randint(1, 9))
            if rng.random() < 0.2:                       # ties: several items at the same instant
                day = start + timedelta(days=(k // 3) * 5)
            ts = day.timestamp()
            roll = rng.random()
            if roll < 0.02:
                ts = ""                                  # missing -> clipped floor
            elif roll < 0.03:
                ts = day.replace(year=2031).timestamp()  # future garbage -> clipped ceil
            rows.append({"user_id": uid, "item_id": items[i]["item_id"],
                         "rating": rng.choice([1, 3, 4, 5]), "timestamp": ts})
    rng.shuffle(rows)  # raw file order is NOT chronological

    with open(os.path.join(raw, "interactions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["user_id", "item_id", "rating", "timestamp"])
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(raw, "items.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "title", "profile", "genre", "creator", "tag"])
        w.writeheader(); w.writerows(items)
    print(f"[synthetic] users={args.users} items={args.items} interactions={len(rows)} -> {raw}")


if __name__ == "__main__":
    main()
