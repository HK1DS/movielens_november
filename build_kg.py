#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a compact KG pack for the (KG-on) IKGR model.

Inputs (already produced by step2):
  run/step2_related_intents.csv   user/item -> exact+related intent strings
  run/intent_vocab.json           list of all intent strings (the vocab)
  run/intents_emb.npy             [n_intents, 768] mpnet embeddings (aligned to vocab)

Output:
  run/kg_pack.pt = {
    "intent_emb": FloatTensor[n_intents, 768],   # init embeddings for intent nodes
    "user_intents": { user_token(str): LongTensor[k_u] of intent ids },
    "item_intents": { item_token(str): LongTensor[k_i] of intent ids },
    "n_intents": int,
  }
Intent ids index into intent_emb (== vocab order).
"""
import argparse, ast, json, os
import numpy as np
import pandas as pd
import torch


def _as_list(s):
    if isinstance(s, str) and s.strip().startswith("["):
        try:
            return list(ast.literal_eval(s))
        except Exception:
            return []
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step2_csv", default="run/step2_related_intents.csv")
    ap.add_argument("--vocab", default="run/intent_vocab.json")
    ap.add_argument("--emb", default="run/intents_emb.npy")
    ap.add_argument("--out", default="run/kg_pack.pt")
    args = ap.parse_args()

    vocab = json.load(open(args.vocab, encoding="utf-8"))
    intent2id = {t: i for i, t in enumerate(vocab)}
    emb = np.load(args.emb)
    assert emb.shape[0] == len(vocab), f"emb {emb.shape} vs vocab {len(vocab)}"
    intent_emb = torch.tensor(emb, dtype=torch.float32)

    df = pd.read_csv(args.step2_csv).fillna("")

    def ids_for(cols, row):
        out, seen = [], set()
        for c in cols:
            for it in _as_list(row.get(c, "")):
                j = intent2id.get(it)
                if j is not None and j not in seen:
                    seen.add(j)
                    out.append(j)
        return out

    user_intents, item_intents = {}, {}
    for _, r in df.iterrows():
        ut = str(r.get("user_id", "")).strip()
        it = str(r.get("item_id", "")).strip()
        if ut and ut not in user_intents:
            ids = ids_for(["user_intents_exact", "user_intents_related"], r)
            if ids:
                user_intents[ut] = torch.tensor(ids, dtype=torch.long)
        if it and it not in item_intents:
            ids = ids_for(["item_intents_exact", "item_intents_related"], r)
            if ids:
                item_intents[it] = torch.tensor(ids, dtype=torch.long)

    nnz_u = sum(len(v) for v in user_intents.values())
    nnz_i = sum(len(v) for v in item_intents.values())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(
        {
            "intent_emb": intent_emb,
            "user_intents": user_intents,
            "item_intents": item_intents,
            "n_intents": len(vocab),
        },
        args.out,
    )
    print(f"[kg] n_intents={len(vocab)} users={len(user_intents)} (nnz={nnz_u}) "
          f"items={len(item_intents)} (nnz={nnz_i}) -> {args.out}")


if __name__ == "__main__":
    main()
