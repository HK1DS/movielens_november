#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verifies, against REAL RecBole 1.2.0 on a synthetic generic-CSV dataset
(with timestamp ties, missing dates and out-of-range years), that
  1. ikgr_core/temporal_split.py assigns exactly RecBole's train/valid/test rows
     for TO and TO_GLOBAL, and
  2. build_trainonly_profiles.py only uses profile-safe TRAIN rows: every
     provenance (user, item) is in RecBole's train split and strictly earlier
     than every validation/test interaction of that user (TO) / the global
     cutoff (TO_GLOBAL); users without such rows get empty profiles.

Run:  python -m pytest -q tests/test_temporal_split.py
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
recbole = pytest.importorskip("recbole")

from ikgr_core.temporal_split import assign_phases, read_atomic_inter  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "tests"))
from synthetic_pipeline import prepare  # noqa: E402


def _run(cmd, cwd):
    r = subprocess.run([sys.executable] + cmd, cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("syn")
    prepare(d, k=5)
    return d


def _recbole_phases(data_dir, split):
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    import eval_slices
    os.environ["IKGR_SPLIT"] = split
    rb = {"epochs": 1, "metrics": ["Recall"], "topk": [10], "embedding_size": 8, "dataset": "ikgr-custom"}
    paths = {"inter_file": str(data_dir / "kc" / "interactions_k5.csv"), "workdir": str(data_dir / "run")}
    cd = eval_slices._config(rb, paths, {}, 2020)
    cd["checkpoint_dir"] = str(data_dir / "ckpt")
    cd["show_progress"] = False
    config = Config(model="BPR", dataset="ikgr-custom", config_dict=cd)
    ds = create_dataset(config)
    parts = data_preparation(config, ds)
    out = []
    for p, dl in enumerate(parts):
        inter = dl.dataset.inter_feat
        u = ds.id2token("user_id", inter["user_id"].numpy())
        i = ds.id2token("item_id", inter["item_id"].numpy())
        out.append(pd.DataFrame({"user_id": u, "item_id": i, "phase": p}))
    return pd.concat(out)


@pytest.mark.parametrize("split", ["TO", "TO_GLOBAL"])
def test_phase_assignment_matches_recbole(data_dir, split):
    df = read_atomic_inter(str(data_dir / "kc" / "ikgr-custom" / "ikgr-custom.inter"))
    assert not df.duplicated(["user_id", "item_id"]).any()
    phase, safe = assign_phases(df, split)
    mine = df.assign(phase=phase)[["user_id", "item_id", "phase"]]
    rb = _recbole_phases(data_dir, split)
    m = mine.merge(rb, on=["user_id", "item_id"], how="outer", suffixes=("_mine", "_rb"), indicator=True)
    assert (m["_merge"] == "both").all()
    mism = (m["phase_mine"] != m["phase_rb"]).sum()
    assert mism == 0, f"{mism} rows assigned differently from RecBole ({split})"
    # safe rows are train rows
    assert (phase[safe] == 0).all()


@pytest.mark.parametrize("split", ["TO", "TO_GLOBAL"])
def test_profiles_use_only_safe_train_history(data_dir, split):
    kc = data_dir / "kc"
    prof = kc / f"profiles_{split}.csv"
    prov = kc / f"prov_{split}.csv.gz"
    _run([os.path.join(ROOT, "build_trainonly_profiles.py"), "--split", split,
          "--config", str(data_dir / "config.yaml"),
          "--inter", str(kc / "ikgr-custom" / "ikgr-custom.inter"),
          "--out_profiles", str(prof), "--out_provenance", str(prov)], ROOT)
    df = read_atomic_inter(str(kc / "ikgr-custom" / "ikgr-custom.inter"))
    phase, _ = assign_phases(df, split)
    df["phase"] = phase
    df["ts32"] = df["timestamp"].astype(np.float32)
    rb = _recbole_phases(data_dir, split)
    train_pairs = set(map(tuple, rb[rb.phase == 0][["user_id", "item_id"]].to_numpy()))
    pv = pd.read_csv(prov, dtype=str)
    assert len(pv) > 0
    assert all((u, i) in train_pairs for u, i in pv.to_numpy()), "provenance item outside RecBole train split"
    # strictly earlier than any non-train interaction of the user / global cutoff
    ts = df.set_index(["user_id", "item_id"])["ts32"]
    if split == "TO":
        first_future = df[df.phase > 0].groupby("user_id", observed=True)["ts32"].min()
        for u, i in pv.to_numpy():
            if u in first_future.index:
                assert ts[(u, i)] < first_future[u]
    else:
        cut = df[df.phase > 0]["ts32"].min()
        assert (pv.apply(lambda r: ts[(r.user_id, r.item_id)] < cut, axis=1)).all()
        # users with no train rows must have empty profiles
        cold = set(df.user_id) - set(df[df.phase == 0].user_id)
        p = pd.read_csv(prof, dtype=str, keep_default_na=False)
        assert all(v == "" for u, v in zip(p.user_id, p.user_profile) if u in cold)
    p = pd.read_csv(prof, dtype=str, keep_default_na=False)
    assert set(p.user_id) == set(df.user_id) and set(p.item_id) == set(df.item_id)
    with_profile = {u for u, v in zip(p.user_id, p.user_profile) if v}
    assert with_profile == set(pv.user_id)
    # adapter-driven structure: recent label + facet lines; the ubiquitous tag is dropped
    texts = [v for v in p.user_profile if v]
    assert all(t.startswith("Recently consumed (newest first): ") for t in texts)
    assert any("Frequent creators: " in t for t in texts) and any("Common tags: " in t for t in texts)
    assert not any("family" in t.split("Common tags: ", 1)[-1] for t in texts if "Common tags: " in t)
    assert all(v for v in p.item_profile)
