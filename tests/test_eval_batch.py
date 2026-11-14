#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-sort evaluation batch (config.yaml recbole.eval_batch_users): RecBole's eval_batch_size counts
user x item pairs, so it is set to users x item_num once the dataset is loaded. The setting is part of the
protocol (run signature) and only changes how many users are scored at once, not the metrics.

Run:  python -m pytest -q tests/test_eval_batch.py
"""
import os
import sys

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
recbole = pytest.importorskip("recbole")
torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(ROOT, "tests"))
from synthetic_pipeline import prepare  # noqa: E402

import eval_slices as ev  # noqa: E402


def test_protocol_and_signature(monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.delenv("IKGR_EVAL_USERS", raising=False)
    rb = yaml.safe_load(open(os.path.join(ROOT, "config.yaml"), encoding="utf-8"))["recbole"]
    assert rb["eval_batch_users"] == 128 and ev._protocol(rb)["eval_batch_users"] == 128
    assert ev._protocol({})["eval_batch_users"] == 1                  # RecBole's old behaviour
    paths = {"kg_pack": "run_k30_TO/kg_pack.pt", "meta_kg_pack": "run_k30_TO/meta_kg_pack.pt",
             "profile_provenance": "data/kc_k30/p.csv.gz", "workdir": "run_k30_TO/",
             "inter_file": "data/kc_k30/interactions_k30.csv"}
    model_arg, extra = ev.build_specs(paths)["BPR"]
    sig128 = ev._experiment_context("BPR", model_arg, extra, rb, paths, "TO")[0]
    monkeypatch.setenv("IKGR_EVAL_USERS", "1")
    sig1 = ev._experiment_context("BPR", model_arg, extra, rb, paths, "TO")[0]
    assert sig1 != sig128                                             # old-protocol results never merge


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    d = tmp_path_factory.mktemp("evb")
    prepare(d, k=5)
    return d


def _valid_metrics(d, users, monkeypatch):
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.trainer import Trainer
    from recbole.utils import get_model, init_seed
    monkeypatch.setenv("IKGR_SPLIT", "TO")
    monkeypatch.setenv("IKGR_EVAL_USERS", str(users))
    rb = {"epochs": 1, "metrics": ["Recall", "NDCG", "MRR"], "topk": [10, 30], "embedding_size": 16,
          "dataset": "ikgr-custom"}
    cd = ev._config(rb, {"inter_file": str(d / "kc" / "interactions_k5.csv"), "workdir": str(d / "run")},
                    {}, 2020)
    cd.update({"checkpoint_dir": str(d / "ckpt"), "show_progress": False, "device": "cpu", "use_gpu": False})
    config = Config(model="BPR", dataset="ikgr-custom", config_dict=cd)
    init_seed(2020, True)
    dataset = create_dataset(config)
    assert ev._apply_eval_batch(config, dataset) == users
    _, valid, _ = data_preparation(config, dataset)
    first = next(iter(valid))[0]
    assert len(first[config["USER_ID_FIELD"]]) == min(users, valid.sample_size)   # users per batch
    init_seed(2020, True)
    model = get_model("BPR")(config, valid.dataset)
    return Trainer(config, model).evaluate(valid, load_best_model=False, show_progress=False)


def test_batch_of_users_gives_the_same_metrics(syn, monkeypatch):
    one = _valid_metrics(syn, 1, monkeypatch)
    many = _valid_metrics(syn, 16, monkeypatch)
    assert set(one) == set(many) and len(one) == 6
    for k in one:
        assert many[k] == pytest.approx(one[k], abs=1e-6), k
