#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic two-CSV adapter (used by the offline tests; also a quick way to plug in
any dataset after converting it once).

  interactions CSV : user_id,item_id,rating,timestamp     (timestamp = epoch seconds, may be empty)
  items CSV        : item_id,title[,profile][,<relation columns>]
                     relation cells hold several nodes separated by `multi_sep` ("|"),
                     most important first.

config.yaml
  dataset:
    adapter: csv
    options:
      interactions: raw/interactions.csv
      items: raw/items.csv
      relations: [{name: genre}, {name: tag, max_per_item: 15, min_df: 5}]
      facets: [{relation: genre, label: Frequent genres, top_n: 10}]
      recent_label: Recent items (newest first)
      title_order: recent        # or rating (highest rated first, ties newest first)
      title_n: 10
      profile_window: 50         # newest-N train rows per user; null = whole train history
      ts_clip: ["2006-01-01", "2017-12-31"]      # optional
"""
import csv
import os

from .base import DatasetAdapter, ItemMeta, MetaRelation, ProfileFacet


def _clean(text):
    return " ".join(str(text or "").split())


class GenericCSVAdapter(DatasetAdapter):
    def __init__(self, interactions, items, relations=(), facets=(), recent_label=None,
                 ts_clip=None, multi_sep="|", name="csv", title_order=None, title_n=None,
                 profile_window="default"):
        self.name = name
        self.interactions_path = interactions
        self.items_path = items
        self.multi_sep = multi_sep
        self.meta_relations = [r if isinstance(r, MetaRelation) else
                               MetaRelation(**({"name": r} if isinstance(r, str) else r)) for r in relations]
        self.profile_facets = [f if isinstance(f, ProfileFacet) else ProfileFacet(**f) for f in facets]
        if recent_label:
            self.recent_label = recent_label
        if title_order:
            self.title_order = title_order
        if title_n:
            self.title_n = int(title_n)
        if profile_window != "default":
            self.profile_window = None if profile_window in (None, 0) else int(profile_window)
        self.ts_clip = tuple(ts_clip) if ts_clip else None

    def iter_interactions(self):
        with open(self.interactions_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    rating = float(row.get("rating") or 0)
                except ValueError:
                    rating = 0.0
                try:
                    ts = float(row.get("timestamp") or 0)
                except ValueError:
                    ts = 0.0
                yield str(row["user_id"]), str(row["item_id"]), rating, ts

    def load_items(self, item_ids):
        rel_names = {r.name for r in self.meta_relations} | {f.relation for f in self.profile_facets}
        out = {}
        with open(self.items_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                iid = str(row["item_id"])
                if iid not in item_ids:
                    continue
                rels = {}
                for name in rel_names:
                    cell = row.get(name) or ""
                    vals = [_clean(v) for v in cell.split(self.multi_sep)]
                    rels[name] = [v for v in vals if v]
                title = _clean(row.get("title"))
                profile = _clean(row.get("profile"))
                if not profile:
                    profile = " | ".join(p for p in [title] + [", ".join(rels[n]) for n in sorted(rels)] if p)
                out[iid] = ItemMeta(title=title, profile=profile, relations=rels)
        return out

    def source_files(self):
        return [p for p in (self.interactions_path, self.items_path) if p and os.path.exists(p)]
