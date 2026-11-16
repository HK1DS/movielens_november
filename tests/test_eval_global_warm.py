#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TO_GLOBAL warm-user validation (eval_global_warm.py, 2026-11-16): early stopping must only see users with train
interactions, the TO_GLOBAL run signature must change (old-protocol results are retrained, never merged), and
TO must be untouched (eval_slices.py is not edited, the wrapper refuses other splits).

Run:  python -m pytest -q tests/test_eval_global_warm.py
"""
import os
import subprocess
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
pytest.importorskip("recbole")
torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(ROOT, "tests"))
from synthetic_pipeline import prepare  # noqa: E402

import eval_slices as ev  # noqa: E402
import eval_global_warm as gw  # noqa: E402

PATHS = {"kg_pack": "run_k30_TO_GLOBAL/kg_pack.pt", "meta_kg_pack": "run_k30_TO_GLOBAL/meta_kg_pack.pt",
         "profile_provenance": "data/kc_k30/p.csv.gz", "workdir": "run_k30_TO_GLOBAL/",
         "inter_file": "data/kc_k30/interactions_k30.csv"}


@pytest.fixture()
def installed(monkeypatch):
    """install() for one test only: the patched eval_slices attributes are restored afterwards."""
    for name in ("_protocol", "_experiment_context", "data_preparation", "_global_warm_installed"):
        monkeypatch.setattr(ev, name, getattr(ev, name, False), raising=False)
    monkeypatch.setattr(ev, "_global_warm_installed", False, raising=False)
    gw.install()
    yield


def _sig(name, split):
    rb = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))["recbole"]
    model_arg, extra = ev.build_specs(PATHS)[name]
    return ev._experiment_context(name, model_arg, extra, rb, PATHS, split)


def test_signature_records_warm_validation(monkeypatch):
    monkeypatch.chdir(ROOT)
    plain_sig, plain_ctx = _sig("BPR", "TO_GLOBAL")
    assert "valid_users" not in plain_ctx["protocol"]
    for name in ("_protocol", "_experiment_context", "data_preparation"):
        monkeypatch.setattr(ev, name, getattr(ev, name))
    monkeypatch.setattr(ev, "_global_warm_installed", False, raising=False)
    gw.install()
    sig, ctx = _sig("BPR", "TO_GLOBAL")
    assert ctx["protocol"]["valid_users"] == "warm" and "eval_global_warm.py" in ctx["code"]
    assert sig != plain_sig                                   # old TO_GLOBAL results are retrained


def test_to_signature_untouched_and_wrapper_refuses_to(monkeypatch):
    monkeypatch.chdir(ROOT)
    before = _sig("IKGR_dyn", "TO")[0]
    r = subprocess.run([sys.executable, os.path.join(ROOT, "eval_global_warm.py")], cwd=ROOT,
                       env=dict(os.environ, IKGR_SPLIT="TO"), capture_output=True, text=True)
    assert r.returncode != 0 and "only for IKGR_SPLIT=TO_GLOBAL" in (r.stdout + r.stderr)
    assert _sig("IKGR_dyn", "TO")[0] == before
    src = open(os.path.join(ROOT, "run_pipeline.py"), encoding="utf-8").read()
    assert '"eval_global_warm.py" if split == "TO_GLOBAL" else "eval_slices.py"' in src


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    d = tmp_path_factory.mktemp("gw")
    prepare(d, k=5)
    return d


def test_validation_keeps_only_users_with_train_history(syn, monkeypatch, installed):
    from recbole.config import Config
    from recbole.data import create_dataset
    from recbole.trainer import Trainer
    from recbole.utils import get_model, init_seed
    monkeypatch.setenv("IKGR_SPLIT", "TO_GLOBAL")
    monkeypatch.setenv("IKGR_EVAL_USERS", "8")
    rb = {"epochs": 1, "metrics": ["Recall", "NDCG", "MRR"], "topk": [10, 30], "embedding_size": 16,
          "dataset": "ikgr-custom"}
    cd = ev._config(rb, {"inter_file": str(syn / "kc" / "interactions_k5.csv"), "workdir": str(syn / "run")}, {}, 2020)
    cd.update({"checkpoint_dir": str(syn / "ckpt"), "show_progress": False, "device": "cpu", "use_gpu": False})
    config = Config(model="BPR", dataset="ikgr-custom", config_dict=cd)
    init_seed(2020, True)
    dataset = create_dataset(config)
    ev._apply_eval_batch(config, dataset)
    from recbole.data import data_preparation as recbole_prep                # the unpatched RecBole split
    train, valid_all, test = recbole_prep(config, dataset)
    uid = config["USER_ID_FIELD"]
    train_users = set(train.dataset.inter_feat[uid].tolist())
    all_users = set(valid_all._dataset.inter_feat[uid].tolist())
    assert all_users - train_users, "synthetic TO_GLOBAL should have cold validation users"
    warm = gw.warm_only(valid_all, train, config)
    warm_users = set(warm._dataset.inter_feat[uid].tolist())
    assert warm_users and warm_users <= train_users and warm_users == all_users & train_users
    for u in list(warm_users)[:20]:                           # same positives / history masks per kept user
        assert set(warm.uid2positive_item[u].tolist()) == set(valid_all.uid2positive_item[u].tolist())
        assert set(warm.uid2history_item[u].tolist()) == set(valid_all.uid2history_item[u].tolist())
    dataset2 = create_dataset(config)                                        # build() runs once per dataset
    _, valid_patched, test_patched = ev.data_preparation(config, dataset2)    # the patched path
    assert set(valid_patched._dataset.inter_feat[uid].tolist()) == warm_users
    assert len(test_patched._dataset.inter_feat) == len(test._dataset.inter_feat)   # test keeps everyone
    init_seed(2020, True)
    model = get_model("BPR")(config, train.dataset)
    res = Trainer(config, model).evaluate(warm, load_best_model=False, show_progress=False)
    assert set(res) >= {"ndcg@10", "recall@10"}
