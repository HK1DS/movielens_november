#!/usr/bin/env python3
"""TO_GLOBAL evaluation with early stopping on WARM users (protocol change of 2026-11-16).

Why: the global temporal split (one time cutoff over all interactions, 70/10/20) puts every user whose
history starts after the cutoff into valid/test with no train interaction. On MovieLens k=30 that is 88.6% of
the TO_GLOBAL test users. For them no model can personalise, so the validation score - and with it early
stopping - rewarded barely trained, popularity-like models (BPR / IKGR_kgoff stopped at their first
validation, 2 epochs, NDCG@10 0.137 overall but lower than IKGR_dyn_nointent on users with history).

What changes (TO_GLOBAL only):
  * the validation loader keeps only users with >= 1 train interaction, so early stopping / model selection
    measures personalisation;
  * the test set is unchanged (all users); per-user files carry `activity_train`, so warm and cold users are
    reported separately;
  * the run signature records protocol.valid_users = "warm" and this file's code hash, so results of the old
    TO_GLOBAL protocol are retrained instead of merged.

TO is not touched: this wrapper refuses other splits and eval_slices.py itself is unchanged, so finished TO
results (signature = eval_slices.py code + protocol) stay valid. run_pipeline.py calls this file instead of
eval_slices.py for --split TO_GLOBAL.

Run (normally through run_pipeline.py):  IKGR_SPLIT=TO_GLOBAL python eval_global_warm.py
"""
import hashlib
import json
import os
import sys

import torch

import eval_slices as ev

VALID_USERS = "warm"        # protocol value recorded in the run signature


def warm_only(valid_data, train_data, config):
    """Copy of the full-sort validation loader restricted to users that have train interactions."""
    uid = config["USER_ID_FIELD"]
    train_users = torch.unique(train_data.dataset.inter_feat[uid])
    ds = valid_data._dataset
    keep = torch.isin(ds.inter_feat[uid], train_users)
    n_before = int(torch.unique(ds.inter_feat[uid]).numel())
    n_after = int(torch.unique(ds.inter_feat[uid][keep]).numel())
    loader = type(valid_data)(config, ds.copy(ds.inter_feat[keep]), valid_data._sampler, shuffle=False)
    print(f"  [valid] warm users only: {n_after:,} of {n_before:,} validation users "
          f"({int(keep.sum()):,} of {len(keep):,} interactions)", flush=True)
    return loader


def install():
    """Patch eval_slices for TO_GLOBAL: protocol entry, code hash, warm-user validation loader."""
    if getattr(ev, "_global_warm_installed", False):
        return
    orig_protocol = ev._protocol
    orig_context = ev._experiment_context
    orig_prep = ev.data_preparation

    def _protocol(rb):
        proto = orig_protocol(rb)
        proto["valid_users"] = VALID_USERS
        return proto

    def _experiment_context(name, model_arg, extra, rb, paths, split):
        _, ctx = orig_context(name, model_arg, extra, rb, paths, split)
        here = os.path.abspath(__file__)
        ctx["code"][ev._portable(here)] = ev._code_sha256(here)
        encoded = json.dumps(ctx, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16], ctx

    def data_preparation(config, dataset):
        train_data, valid_data, test_data = orig_prep(config, dataset)
        return train_data, warm_only(valid_data, train_data, config), test_data

    ev._protocol = _protocol
    ev._experiment_context = _experiment_context
    ev.data_preparation = data_preparation
    ev._global_warm_installed = True


def main():
    split = os.environ.get("IKGR_SPLIT", "").upper()
    if split != "TO_GLOBAL":
        sys.exit(f"[eval_global_warm] only for IKGR_SPLIT=TO_GLOBAL (got {split or 'unset'}); "
                 f"use eval_slices.py for other splits")
    install()
    ev.main()


if __name__ == "__main__":
    main()
