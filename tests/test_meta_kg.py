#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The metadata KG is defined by the dataset adapter (any number of relations).
Checks: build_meta_kg.py writes one node vocabulary per adapter relation with the
per-relation caps/df filters, IKGR learns one embedding table + gate per relation
and trains with finite loss and gradients on every gate, and the CORONA retriever
exposes one channel per relation.

Run:  python -m pytest -q tests/test_meta_kg.py
"""
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
recbole = pytest.importorskip("recbole")
torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(ROOT, "tests"))
from synthetic_pipeline import prepare  # noqa: E402

RELS = ["creator", "genre", "tag"]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp("syn")
    cfg_path = prepare(d, k=5)
    pack_path = str(d / "meta_kg_pack.pt")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "build_meta_kg.py"), "--config", cfg_path,
                        "--inter", str(d / "kc" / "ikgr-custom" / "ikgr-custom.inter"), "--out", pack_path],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr

    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    import eval_slices
    os.environ["IKGR_SPLIT"] = "TO"
    rb = {"epochs": 1, "metrics": ["Recall"], "topk": [10], "embedding_size": 16, "dataset": "ikgr-custom"}
    paths = {"inter_file": str(d / "kc" / "interactions_k5.csv"), "workdir": str(d / "run")}

    def build(model, extra):
        cd = eval_slices._config(rb, paths, extra, 2020)
        cd.update({"checkpoint_dir": str(d / "ckpt"), "show_progress": False, "device": "cpu", "use_gpu": False})
        config = Config(model=model, dataset="ikgr-custom", config_dict=cd)
        ds = create_dataset(config)
        train, _, _ = data_preparation(config, ds)
        return config, train
    return {"pack": pack_path, "build": build}


def test_pack_has_one_vocab_per_relation(env):
    pack = torch.load(env["pack"], map_location="cpu")
    assert pack["format"] == 2 and pack["dataset"] == "synthetic"
    assert sorted(pack["relations"]) == RELS
    for name, r in pack["relations"].items():
        assert r["n_nodes"] == len(r["vocab"]) > 0
        ids = torch.cat(list(r["item_nodes"].values()))
        assert int(ids.max()) < r["n_nodes"]
    assert "family" in pack["relations"]["tag"]["vocab"]            # ubiquitous only matters for profiles
    assert max(len(v) for v in pack["relations"]["tag"]["item_nodes"].values()) <= 15


def test_model_learns_one_gate_per_relation(env):
    from ikgr_core.model_ikgr import IKGR
    cfg, train = env["build"](IKGR, {"use_kg": False, "use_meta_kg": True, "meta_kg_path": env["pack"],
                                     "use_dynamic": True, "lgcn_layers": 2})
    m = IKGR(cfg, train.dataset)
    assert m.meta_rels == RELS
    n = m.n_users
    m.set_recency(torch.randint(1, m.n_items, (n, 5)), torch.full((n, 5), 0.2))
    batch = next(iter(train))
    loss = m.calculate_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    for k in RELS:
        assert m.meta_alphas[k].grad is not None and m.meta_embeddings[k].weight.grad is not None
    m.eval()
    with torch.no_grad():
        m.eval_cache_items()
        assert torch.isfinite(m.full_sort_predict(batch)).all()
    # CORONA late-fusion path uses the same relations
    cfg_c, train_c = env["build"](IKGR, {"use_kg": False, "use_meta_kg": True, "meta_kg_path": env["pack"],
                                         "use_corona": True})
    mc = IKGR(cfg_c, train_c.dataset)
    assert torch.isfinite(mc.calculate_loss(next(iter(train_c))))


def test_retriever_channel_per_relation(env):
    from ikgr_core.corona_retriever import CoronaRetriever
    cfg, train = env["build"]("BPR", {})
    r = CoronaRetriever(train, cfg, meta_kg_path=env["pack"], weights={"tag": 0.0}, idf=True, pop_norm=0.5)
    assert sorted(r.meta) == RELS
    users = np.arange(1, 9)
    s_all = r._scores(users)
    assert np.isfinite(s_all).all() and s_all.max() > 0
    cand = r.candidates(users, 20)
    assert cand.shape == (8, 20)
    prior = r.normalized_prior(users)
    assert np.isfinite(prior).all() and prior.max() <= 1.0
