#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the HETEROGENEOUS metadata KG pack for IKGR (item -- relation -- node),
with NO LLM calls -- purely from the dataset adapter's static item metadata.

Node types = adapter.meta_relations (e.g. Goodreads: author / publisher / shelf;
MovieLens: genre / ...). For every item of the RecBole .inter file, each
relation keeps the first `max_per_item` nodes from ItemMeta.relations[name];
nodes linked to fewer than `min_df` items are dropped.

Output meta_kg_pack.pt = {
  "format": 2,
  "dataset": adapter name,
  "relations": {
     <name>: {"item_nodes": {item_token(str): LongTensor[node ids]},
              "vocab": [node key, ...],          # node id -> key
              "n_nodes": int},
     ...
  },
}
Node ids index per-relation vocabularies (each relation has its own id space).
ikgr_core/model_ikgr.py learns one embedding table + one gate per relation;
ikgr_core/corona_retriever.py uses each relation as a retrieval channel.
"""
import argparse, csv, os
from collections import Counter

import torch

from data_adapters import load_adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inter", required=True, help="RecBole .inter; ALL its items are covered")
    ap.add_argument("--config", default=None, help="config with the `dataset:` section (default: IKGR_CONFIG)")
    ap.add_argument("--out", default="run/meta_kg_pack.pt")
    args = ap.parse_args()

    with open(args.inter, encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        header = [h.split(":")[0] for h in next(r)]
        col = header.index("item_id")
        item_set = {str(row[col]) for row in r if row}
    print(f"[meta] target items: {len(item_set)}")

    adapter = load_adapter(args.config)
    meta = adapter.load_items(item_set)
    print(f"[meta] adapter={adapter.name}: metadata for {len(meta)}/{len(item_set)} items")
    if not adapter.meta_relations:
        print("[meta] WARNING: adapter defines no meta_relations -> empty pack (meta-KG specs add nothing)")

    relations = {}
    for rel in adapter.meta_relations:
        if rel.name in ("intent", "cf"):
            raise SystemExit(f"meta relation name {rel.name!r} is reserved (CORONA retriever channel); rename it")
        raw = {}
        df = Counter()
        for iid, m in meta.items():
            nodes = list(dict.fromkeys(v for v in m.relations.get(rel.name, []) if v))   # dedup, keep order
            if rel.max_per_item:
                nodes = nodes[:rel.max_per_item]
            if nodes:
                raw[iid] = nodes
                df.update(nodes)
        keep = {v for v, c in df.items() if c >= rel.min_df}
        vocab = {}
        item_nodes = {}
        for iid, nodes in raw.items():
            ids = []
            for v in nodes:
                if v in keep:
                    if v not in vocab:
                        vocab[v] = len(vocab)
                    ids.append(vocab[v])
            if ids:
                item_nodes[iid] = torch.tensor(ids, dtype=torch.long)
        inv = [k for k, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
        relations[rel.name] = {"item_nodes": item_nodes, "vocab": inv, "n_nodes": len(inv)}
        edges = sum(len(v) for v in item_nodes.values())
        print(f"[meta] {rel.name}: nodes {len(df)} -> {len(inv)} kept (min_df>={rel.min_df}) | "
              f"items with node={len(item_nodes)} | item-edges={edges}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"format": 2, "dataset": adapter.name, "relations": relations}, args.out)
    print(f"[meta] saved -> {args.out}")


if __name__ == "__main__":
    main()
