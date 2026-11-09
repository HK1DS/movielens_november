#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline replica of RecBole 1.2.0's temporal ratio split.

Why this exists
---------------
User profiles (and therefore LLM user intents) are built OFFLINE, before
RecBole ever splits the data. To guarantee that a profile only uses
training-period interactions, the profile builder must know -- per split
protocol -- which rows RecBole will put into the train partition.

RecBole (eval_args: order=TO, split RS) does, in Dataset.build():
  1. convert the timestamp column to a torch.FloatTensor (float32!)
  2. stable argsort of all rows by that float32 timestamp (file order breaks ties)
  3. split_by_ratio(ratios, group_by):
       group_by='user' -> per-user lists (in sorted order), counts from
                          _calcu_split_ids(len(user_rows), ratios)
       group_by=None   -> one global list, counts from _calcu_split_ids(N, ratios)

This module reproduces exactly that, and additionally exposes a *conservative*
boundary: a training row is "profile-safe" only if its float32 timestamp is
strictly earlier than the first non-train row of the same group. This makes the
guarantee independent of how ties are broken.

The eval harness (eval_slices.py) re-checks the result against RecBole's real
train split at run time, so a mismatch can never silently leak.
"""
import numpy as np
import pandas as pd

SPLITS = {
    # name: (ratios, group_by_user)
    "TO": ([0.8, 0.1, 0.1], True),
    "TO_GLOBAL": ([0.7, 0.1, 0.2], False),
}


def calcu_split_ids(tot, ratios):
    """Verbatim copy of recbole.data.dataset.Dataset._calcu_split_ids (1.2.0)."""
    cnt = [int(ratios[i] * tot) for i in range(len(ratios))]
    cnt[0] = tot - sum(cnt[1:])
    for i in range(1, len(ratios)):
        if cnt[0] <= 1:
            break
        if 0 < ratios[-i] * tot < 1:
            cnt[-i] += 1
            cnt[0] -= 1
    split_ids = np.cumsum(cnt)[:-1]
    return list(split_ids)


def _normalized(ratios):
    tot = sum(ratios)
    return [r / tot for r in ratios]


def read_atomic_inter(path):
    """Read a RecBole atomic .inter file (tab separated, 'name:type' header).

    user_id / item_id are loaded as pandas categoricals (each distinct id string
    stored once) so a multi-million-row file stays small in memory."""
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
    names = [h.split(":")[0] for h in header]
    if "timestamp" not in names:
        raise ValueError(f"{path} has no timestamp column; run make_temporal_inter.py first.")
    dtype = {}
    for h, n in zip(header, names):
        dtype[h] = "category" if n in ("user_id", "item_id") else (np.float64 if n in ("timestamp", "rating") else str)
    df = pd.read_csv(path, sep="\t", dtype=dtype, keep_default_na=False)
    df.columns = names
    return df.reset_index(drop=True)


def _group_codes(col):
    if isinstance(col.dtype, pd.CategoricalDtype):
        return col.cat.codes.to_numpy()
    return pd.factorize(col)[0]


def assign_phases(df, split):
    """Return (phase, safe) arrays aligned with df rows.

    phase: int8, 0=train, 1=valid, 2=test (exactly what RecBole would assign)
    safe : bool, train rows whose float32 timestamp is strictly before the
           first non-train row of their group (user for TO, global for
           TO_GLOBAL). Rows of groups with no non-train rows are safe.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLITS)}")
    ratios, by_user = SPLITS[split]
    ratios = _normalized(ratios)

    ts32 = df["timestamp"].to_numpy(dtype=np.float64).astype(np.float32)
    order = np.argsort(ts32, kind="stable")          # RecBole Interaction.sort
    phase = np.full(len(df), -1, dtype=np.int8)
    safe = np.zeros(len(df), dtype=bool)

    if not by_user:
        tot = len(order)
        sid = calcu_split_ids(tot, ratios)
        bounds = [0] + sid + [tot]
        for p in range(3):
            phase[order[bounds[p]:bounds[p + 1]]] = p
        c0 = bounds[1]
        train_rows = order[:c0]
        if c0 < tot:
            boundary = ts32[order[c0]]
            safe[train_rows] = ts32[train_rows] < boundary
        else:
            safe[train_rows] = True
        return phase, safe

    # group sorted positions by user, preserving sorted order (RecBole _grouped_index)
    codes = _group_codes(df["user_id"])[order]
    grp_order = np.argsort(codes, kind="stable")
    sorted_codes = codes[grp_order]
    starts = np.flatnonzero(np.r_[True, sorted_codes[1:] != sorted_codes[:-1]])
    ends = np.r_[starts[1:], len(sorted_codes)]
    for s, e in zip(starts, ends):
        rows = order[grp_order[s:e]]                  # this user's rows, time-sorted
        tot = e - s
        sid = calcu_split_ids(tot, ratios)
        bounds = [0] + sid + [tot]
        for p in range(3):
            phase[rows[bounds[p]:bounds[p + 1]]] = p
        c0 = bounds[1]
        tr = rows[:c0]
        if c0 < tot:
            safe[tr] = ts32[tr] < ts32[rows[c0]]
        else:
            safe[tr] = True
    assert (phase >= 0).all()
    return phase, safe
