#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The IKGR LightGCN backbone (lgcn_layers > 0) must be RecBole's LightGCN when every
facet is switched off: same weights -> identical full-sort scores. Also checks that
lgcn_layers=0 (all pre-existing specs) builds no graph, that the adjacency only uses
TRAIN interactions, and that the facet variant trains and scores with finite values.

Run:  python -m pytest -q tests/test_lgcn_backbone.py
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
recbole = pytest.importorskip("recbole")
torch = pytest.importorskip("torch")


sys.path.insert(0, os.path.join(ROOT, "tests"))
from synthetic_pipeline import prepare  # noqa: E402


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("syn")
    prepare(d, k=5)

    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    import eval_slices
    os.environ["IKGR_SPLIT"] = "TO"
    rb = {"epochs": 1, "metrics": ["Recall"], "topk": [10], "embedding_size": 16, "dataset": "ikgr-custom"}
    paths = {"inter_file": str(d / "kc" / "interactions_k5.csv"), "workdir": str(d / "run")}

    def build(model, extra):
        cd = eval_slices._config(rb, paths, extra, 2020)
        cd["checkpoint_dir"] = str(d / "ckpt")
        cd["show_progress"] = False
        cd["device"] = "cpu"
        cd["use_gpu"] = False
        config = Config(model=model, dataset="ikgr-custom", config_dict=cd)
        ds = create_dataset(config)
        train, valid, test = data_preparation(config, ds)
        return config, ds, train
    return build


def test_backbone_equals_recbole_lightgcn(data):
    from recbole.data.interaction import Interaction
    from recbole.utils import get_model
    from ikgr_core.model_ikgr import IKGR      # imported first, as in eval_slices.py
    LightGCN = get_model("LightGCN")
    cfg_l, _, train_l = data("LightGCN", {})
    cfg_i, _, train_i = data(IKGR, {"use_kg": False, "lgcn_layers": cfg_l["n_layers"]})
    assert cfg_l["n_layers"] == 2
    lgcn = LightGCN(cfg_l, train_l.dataset).eval()
    # regression: with SciPy >= 1.13 the old dok_matrix._update shim left this EMPTY
    assert lgcn.norm_adj_matrix.coalesce()._nnz() == 2 * train_l.dataset.inter_matrix(form="coo").tocsr().nnz
    ikgr = IKGR(cfg_i, train_i.dataset).eval()
    with torch.no_grad():
        ikgr.user_embedding.weight.copy_(lgcn.user_embedding.weight)
        ikgr.item_embedding.weight.copy_(lgcn.item_embedding.weight)
    users = torch.arange(1, lgcn.n_users)
    inter = Interaction({cfg_l["USER_ID_FIELD"]: users})
    with torch.no_grad():
        a = lgcn.full_sort_predict(inter).view(len(users), -1)
        ikgr.eval_cache_items()
        b = ikgr.full_sort_predict(Interaction({cfg_i["USER_ID_FIELD"]: users})).view(len(users), -1)
    assert a.shape == b.shape
    assert torch.allclose(a, b, atol=1e-5), float((a - b).abs().max())


def test_adjacency_is_train_only_and_off_by_default(data):
    from ikgr_core.model_ikgr import IKGR
    cfg, ds, train = data(IKGR, {"use_kg": False, "lgcn_layers": 2})
    m = IKGR(cfg, train.dataset)
    R = train.dataset.inter_matrix(form="coo").tocsr()
    nu = m.n_users
    rows, cols = m._adj_idx.numpy()
    ui = rows < nu                                   # user -> item half
    pairs = set(zip(rows[ui].tolist(), (cols[ui] - nu).tolist()))
    train_pairs = set(zip(*R.nonzero()))
    assert pairs == {(int(u), int(i)) for u, i in train_pairs}
    assert len(train.dataset) < len(ds)              # valid/test rows exist and are excluded
    cfg0, _, train0 = data(IKGR, {"use_kg": False})
    m0 = IKGR(cfg0, train0.dataset)
    assert m0.lgcn_layers == 0 and m0._adj is None and not hasattr(m0, "_adj_idx")


def test_facet_variant_trains(data):
    from ikgr_core.model_ikgr import IKGR
    cfg, _, train = data(IKGR, {"use_kg": False, "use_dynamic": True, "lgcn_layers": 2})
    m = IKGR(cfg, train.dataset)
    n = m.n_users
    m.set_recency(torch.randint(1, m.n_items, (n, 5)), torch.full((n, 5), 0.2))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    batch = next(iter(train))
    loss = m.calculate_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.user_embedding.weight.grad is not None and m.dynamic_alpha.grad is not None
    opt.step()
    m.eval()
    with torch.no_grad():
        m.eval_cache_items()
        s = m.full_sort_predict(batch)
    assert torch.isfinite(s).all()
    cfg_c, _, train_c = data(IKGR, {"use_kg": False, "lgcn_layers": 2, "use_corona": True})
    with pytest.raises(ValueError):
        IKGR(cfg_c, train_c.dataset)
